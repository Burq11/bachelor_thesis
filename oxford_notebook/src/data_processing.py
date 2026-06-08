from src import provider
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

## data_processing refactored for Database access

def extract_unique_signal_values(df, origin="LF_Data"):
    """
    Extrahiert eine eindeutige Übersicht über Signal, Label, Value und Value_String
    für einen gegebenen DataOrigin (z. B. 'LF_Data').

    Parameters
    ----------
    df : pd.DataFrame
        Der vollständige Datenframe mit Spalten wie 'Signal', 'Label', 'Value', etc.
    origin : str, optional
        Der Wert der Spalte 'DataOrigin', nach dem gefiltert werden soll (Default: 'LF_Data').

    Returns
    -------
    pd.DataFrame
        Tabelle mit eindeutigen Kombinationen aus 'Signal', 'Label', 'Value', 'Value_String'.
    """
    df_filtered = df.loc[df["DataOrigin"] == origin]
    return (
        df_filtered[["Signal", "Label", "Value", "Value_String"]]
        .drop_duplicates()
        .sort_values("Signal")
        .reset_index(drop=True)
    )




def summarize_chatter_cases_sql(plate_number: str, *, data_origin: str | None = None) -> pd.DataFrame:
    """DuckDB-first summary for the heatmap overlays (no per-slot raw loads).

    Produces a DataFrame compatible with downstream plotting code that expects the
    columns from `summarize_chatter_cases` / `analyze_platte`.

    Notes:
    - Returns *only* slots where chatter signal rows exist (same net effect as the
      legacy summarizer which returns an empty DataFrame for slots without R310).
    - Keeps column names stable to avoid any visual/logic changes.
    """

    meta = provider.slot_metadata_summary(str(plate_number), data_origin=data_origin)
    chatter = provider.slot_chatter_cases_summary(str(plate_number), data_origin=data_origin)

    if chatter is None or chatter.empty:
        return pd.DataFrame(columns=[
            "Chatter",
            "Y_max",
            "Drehzahl",
            "Werkzeugradius",
            "X_Position_Nut",
            "Werkzeug",
            "Nut_ID",
            "Platte",
            "Nut",
        ])

    df = chatter.merge(meta, on="Nut", how="left")
    df["Nut_ID"] = df.get("Nut_ID", df["Nut"])
    df["Platte"] = str(plate_number)
    return df[[
        "Chatter",
        "Y_max",
        "Drehzahl",
        "Werkzeugradius",
        "X_Position_Nut",
        "Werkzeug",
        "Nut_ID",
        "Platte",
        "Nut",
    ]]


def analyze_platte(plate_number: str, summarize_fn=None, plot_fn=None, *, df_kwatgs: dict | None = None):
    """
    SQL-first implementation: produce per-slot summaries for a plate using
    `summarize_chatter_cases_sql` (no per-slot raw DataFrame loading).

    Keeps the original function signature for compatibility but ignores the
    legacy `summarize_fn` when present.

    Returns
    -------
    pd.DataFrame, object
        (df_summary, fig) where `fig` is the result of `plot_fn(df_summary)` if
        `plot_fn` is provided, else `None`.
    """
    # Keep signature compatibility
    df_kwatgs = df_kwatgs or {}

    # Use the SQL-first summarizer
    df_summary = summarize_chatter_cases_sql(str(plate_number))
    if df_summary is None or df_summary.empty:
        return pd.DataFrame(), None

    fig = None
    if plot_fn is not None:
        try:
            fig = plot_fn(df_summary)
        except Exception as e:
            print(f"Could not build plot from df_summary: {e}")

    return df_summary, fig



def filter_unique_gcodes(df, signal_column='Signal', gcode_column='HFBlockEvent_GCode', sort_column='Time'):
    """
    Filtert den DataFrame nach dem ersten Signal in der angegebenen Signal-Spalte, sortiert ihn optional nach einer 
    angegebenen Spalte und entfernt Duplikate basierend auf dem ersten Vorkommen jedes GCode-Wertes.

    Parameters
    ----------
    df : pandas.DataFrame
        Der Eingabe-DataFrame mit den Daten.
    signal_column : str, optional
        Der Name der Spalte, die die Signal-Daten enthält (default ist 'Signal').
    gcode_column : str, optional
        Der Name der Spalte, die die GCode-Daten enthält (default ist 'HFBlockEvent_GCode').
    sort_column : str, optional
        Der Name der Spalte, nach der der DataFrame vor dem Filtern und Entfernen von Duplikaten sortiert werden soll (default ist 'Time').

    Returns
    -------
    pandas.DataFrame
        Ein DataFrame, der sortiert wurde, nur das erste Signal und die ersten Vorkommen jedes GCode-Wertes enthält.

    Examples
    --------
    >>> df_filtered = filter_unique_gcodes(df, sort_column='Duration_Seconds')
    """
    # Optionales Sortieren des DataFrames nach einer angegebenen Spalte
    if sort_column is not None:
        df = df.sort_values(by=sort_column)

    # Filtern des DataFrames für das erste Signal in der Signal-Spalte
    first_signal = df[signal_column].unique()[0]
    df_filtered = df.loc[df[signal_column] == first_signal]

    # Entfernen von Duplikaten, basierend auf dem ersten Vorkommen jedes GCode-Wertes
    df_filtered = df_filtered.drop_duplicates(subset=gcode_column, keep='first').copy()

    return df_filtered



def downsample_dataframe(df: pd.DataFrame, max_points: int = 10_000) -> pd.DataFrame:
    """
    Reduziert die Anzahl der Zeilen eines DataFrames auf maximal `max_points` durch gleichmäßiges Sampling.

    Diese Funktion ist nützlich zur Visualisierung großer Datenmengen,
    z. B. beim Plotten von Zeitreihen, um Performance und Lesbarkeit zu verbessern.

    Parameters
    ----------
    df : pandas.DataFrame
        Das Eingabe-DataFrame, das ggf. reduziert werden soll.
    max_points : int, optional
        Maximale Anzahl der Zeilen im Ergebnis. Standard ist 10.000.

    Returns
    -------
    pandas.DataFrame
        Ein DataFrame mit höchstens `max_points` Zeilen. 
        Wenn `len(df) <= max_points`, wird das Original unverändert zurückgegeben.
    """
    if len(df) <= max_points:
        return df
    else:
        idx = np.linspace(0, len(df) - 1, max_points).astype(int)
        return df.iloc[idx].reset_index(drop=True)

def compute_sampling_rate(time: np.ndarray, method: str = "mean") -> int:
    """
    Berechnet die Samplingrate (Abtastfrequenz) eines Zeitarrays in Hz als ganze Zahl.

    Parameters
    ----------
    time : np.ndarray
        Zeitstempel in Sekunden, z. B. von einem Sensorsignal.
    method : str, optional
        Methode zur Berechnung:
        - 'mean': Mittelwert der Zeitabstände (robust bei geringem Jitter)
        - 'first_diff': Nur erster Abstand (schnell, bei perfekt equidistanter Abtastung)

    Returns
    -------
    int
        Samplingrate in Hz, gerundet auf ganze Zahl.

    Raises
    ------
    ValueError
        Wenn weniger als zwei unterschiedliche Zeitwerte vorhanden sind.
    """
    unique_times = np.unique(time)
    if len(unique_times) < 2:
        raise ValueError("Nicht genügend unterschiedliche Zeitwerte zur Berechnung der Samplingrate.")

    if method == "first_diff":
        delta_t = unique_times[1] - unique_times[0]
    elif method == "mean":
        delta_t = np.mean(np.diff(unique_times))
    else:
        raise ValueError("Ungültige Methode. Erlaubt sind 'mean' oder 'first_diff'.")

    if delta_t <= 0:
        raise ValueError("Ungültiger Zeitabstand: delta_t muss > 0 sein.")

    return int(round(1 / delta_t))



def butter_lowpass_filter_series(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str,
    cutoff: float,
    order: int = 4
) -> pd.Series:
    """
    Wendet einen Butterworth-Tiefpassfilter auf eine Signalspalte eines DataFrames an.

    Die Abtastrate wird automatisch aus der Zeitspalte berechnet.
    Das Ergebnis ist eine gefilterte Series mit gleichem Index wie das Original.

    Parameters
    ----------
    df : pandas.DataFrame
        Eingabedaten mit mindestens einer Zeit- und einer Signalspalte.
    signal_col : str
        Name der Spalte, die das zu filternde Signal enthält.
    time_col : str
        Name der Spalte, die die Zeitinformation (in Sekunden) enthält.
    cutoff : float
        Grenzfrequenz des Filters in Hz.
    order : int, optional
        Ordnung des Filters (Standard: 4).

    Returns
    -------
    pd.Series
        Gefilterte Series mit dem ursprünglichen Index.

    Raises
    ------
    ValueError
        Wenn Spalten fehlen oder die Samplingrate nicht bestimmbar ist.
    """
    if signal_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Die Spalten '{signal_col}' und/oder '{time_col}' fehlen im DataFrame.")

    signal = df[signal_col].values
    time = df[time_col].values

    unique_times = np.unique(time)
    if len(unique_times) < 2:
        raise ValueError("Nicht genügend unterschiedliche Zeitwerte zur Samplingrate-Berechnung.")

    sampling_rate = compute_sampling_rate(df[time_col].values)

    if cutoff >= 0.5 * sampling_rate:
        raise ValueError("Cutoff-Frequenz muss kleiner als die Nyquist-Frequenz (fs/2) sein.")

    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)

    return pd.Series(filtered, index=df.index, name=f"{signal_col}_gefiltert")

def filter_constant_HF_signals(df, signal_col="Signal", value_col="Value", origin_col="DataOrigin", hf_label="HF_Data"):
    """
    Entfernt HF-Signale, deren Werte über die Zeit konstant sind.
    Alle anderen Daten (z. B. LF_Data, Oscilloscope) bleiben unverändert erhalten.

    Parameters
    ----------
    df : pd.DataFrame
        Der Eingabe-DataFrame mit Signalen und Messwerten.
    signal_col : str
        Name der Spalte mit den Signalnamen (Default: 'Signal').
    value_col : str
        Name der Spalte mit den Signalwerten (Default: 'Value').
    origin_col : str
        Name der Spalte mit der Datenquelle (Default: 'DataOrigin').
    hf_label : str
        Kennzeichnung für HF-Daten (Default: 'HF_Data').

    Returns
    -------
    pd.DataFrame
        Gefilterter DataFrame ohne konstante HF-Signale.
    """
    # Teilmenge nur für HF-Daten
    df_hf = df[df[origin_col] == hf_label]

    # Signale mit mehr als einem einzigartigen Wert in HF-Daten
    non_constant_signals = (
        df_hf.groupby(signal_col)[value_col]
        .nunique(dropna=True)
        .loc[lambda x: x > 1]
        .index
    )

    # HF-Daten filtern
    df_hf_filtered = df_hf[df_hf[signal_col].isin(non_constant_signals)]

    # Alle anderen Daten erhalten
    df_other = df[df[origin_col] != hf_label]

    # Kombinieren und zurückgeben
    df_filtered = pd.concat([df_other, df_hf_filtered], ignore_index=True).sort_values(by="Duration_Seconds")
    return df_filtered.reset_index(drop=True).copy()

def count_value_combinations(df, columns, sort_by_count=True, na_rep='NaN'):
    """
    Zählt die Häufigkeit von Kombinationen bestimmter Spaltenwerte in einem DataFrame.

    Parameter
    ----------
    df : DataFrame
        Das Pandas DataFrame mit den zu untersuchenden Daten.

    columns : list of str
        Liste der Spaltennamen, deren Kombinationen gezählt werden sollen.

    sort_by_count : bool, optional (default=True)
        Falls True, wird das Ergebnis nach Häufigkeit (absteigend) sortiert.

    na_rep : str or float, optional (default='NaN')
        Ersatzwert für fehlende Einträge (NaN), bevor gezählt wird.

    Rückgabe
    --------
    DataFrame
        Ein DataFrame mit den eindeutigen Kombinationen und einer Spalte 'Anzahl'.
    """
    # Prüfen, ob alle Spalten vorhanden sind
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht im DataFrame gefunden.")

    # NaNs ersetzen
    df_temp = df[columns].fillna(na_rep)

    # Kombinationen zählen
    counted = df_temp.value_counts().reset_index(name='Anzahl')

    # Optional sortieren
    if sort_by_count:
        counted = counted.sort_values(by='Anzahl', ascending=False)

    return counted

## HEATMAP
# NOTE: legacy per-slot DataFrame-based heatmap helpers were removed in favor
# of DuckDB-first implementations. Use `prepare_equal_bins_heatmap_sql` and
# `materialize_heatmap_cache` for heatmap computations.
        
# legacy parquet-based heatmap aggregation removed; prefer SQL-based helpers

########### PCA ###############

def prepare_df_wide_for_pca(
    df: pd.DataFrame,
    index_col: str = "Duration_Seconds",
    signal_col: str = "Signal",
    value_col: str = "Value",
    pivot_fill_value: float = 0.0,
    origin_filter: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepares a long-format dataframe for PCA or correlation analysis by:
      1. Converting to wide (matrix) format.
      2. Replacing missing signal names with 'EXT' + DataOrigin.
      3. Optionally filtering by DataOrigin.
      4. Cleaning numeric data (remove non-numeric, zero-variance, NaNs).

    Parameters
    ----------
    df : pd.DataFrame
        Original long-format dataframe (e.g. HF_Data or Oscilloscope data).
    index_col : str
        Column to use as the index (default 'Duration_Seconds').
    signal_col : str
        Column containing signal identifiers (default 'Signal').
    value_col : str
        Column containing numeric values (default 'Value').
    pivot_fill_value : float
        Value to fill NaNs in pivot table (default 0.0).
    origin_filter : str, optional
        Filter by DataOrigin (e.g., "HF_Data", "Oscilloscope").
    verbose : bool
        Print diagnostic info on dropped or cleaned columns.

    Returns
    -------
    pd.DataFrame
        Clean wide-format dataframe ready for PCA.
    """

    df_local = df.copy()

    # Replace None/NaN in 'Signal' with 'EXT' + DataOrigin 
    if signal_col in df_local.columns and "DataOrigin" in df_local.columns:
        mask = df_local[signal_col].isna()
        df_local.loc[mask, signal_col] = (
            "EXT" + df_local.loc[mask, "DataOrigin"].astype(str)
        )

    # Optional filter by DataOrigin 
    if origin_filter is not None:
        if "DataOrigin" not in df_local.columns:
            raise ValueError(
                f"'DataOrigin' column not found in DataFrame, but origin_filter='{origin_filter}' was provided."
            )

        available_origins = df_local["DataOrigin"].dropna().unique().tolist()
        if origin_filter not in available_origins:
            raise ValueError(
                f"origin_filter='{origin_filter}' not found in DataOrigin column. "
                f"Available origins: {available_origins}"
            )

        df_local = df_local[df_local["DataOrigin"] == origin_filter].copy()

    # Ensure required columns exist 
    required_cols = [index_col, signal_col, value_col]
    missing = [c for c in required_cols if c not in df_local.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows missing essential values 
    df_local = df_local.dropna(subset=required_cols)

    # Pivot to wide format 
    df_wide = df_local.pivot_table(
        index=index_col,
        columns=signal_col,
        values=value_col,
        aggfunc="mean"
    ).fillna(pivot_fill_value)

    # Sort index for time alignment 
    df_wide = df_wide.sort_index().reset_index()

    # Clean numeric columns for PCA 
    before_cols = df_wide.columns.tolist()
    df_wide = df_wide.select_dtypes(include="number")       # numeric only
    if index_col in df_wide.columns:                        # Drop Duration_Seconds as feature
        df_wide = df_wide.drop(columns=[index_col])
    df_wide = df_wide.loc[:, df_wide.std(axis=0) > 0]       # remove zero-variance
    df_wide = df_wide.fillna(df_wide.mean())                # fill stray NaNs
    after_cols = df_wide.columns.tolist()

    return df_wide


def reduce_redundant_signals(df, corr_threshold=0.9):
    """
    Removes redundant signals using absolute correlation + connected components.
    Keeps one representative per correlated group based on variance.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe (rows = samples, columns = signals).
    corr_threshold : float, default 0.9
        |corr| >= threshold → variables considered redundant.

    Returns
    -------
    df_reduced : pd.DataFrame
        DataFrame with redundant signals removed.
    corr_reduced : pd.DataFrame
        Correlation matrix of df_reduced.
    groups : dict
        Dictionary mapping group_id → list of column names in the group.
    kept : list
        Columns retained after redundancy removal.
    dropped : list
        Columns removed.
    """

    # Compute correlation matrix 
    corr = df.corr(method="pearson").fillna(0.0)
    abs_corr = corr.abs()

    cols = df.columns.to_list()
    n = len(cols)

    # Build adjacency matrix for |corr| >= threshold 
    adjacency = (abs_corr.values >= corr_threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    # Connected components: groups of correlated variables 
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(graph, directed=False)

    # Build groups as dict: group_id → list of colnames
    groups = {}
    for idx, comp_id in enumerate(labels):
        groups.setdefault(comp_id, []).append(cols[idx])

    kept = []
    dropped = []

    # Select representative per group 
    for comp_id, members in groups.items():
        if len(members) == 1:
            # No redundancy → keep the lone variable
            kept.append(members[0])
        else:
            # Choose representative by highest variance
            variances = df[members].var().sort_values(ascending=False)
            rep = variances.index[0]

            kept.append(rep)
            dropped.extend([c for c in members if c != rep])

    # Build reduced dataframe
    df_reduced = df[kept].copy()
    corr_reduced = df_reduced.corr()

    return df_reduced, corr_reduced, groups, kept, dropped

def prepare_equal_bins_heatmap_sql(
    platte,
    bin_size_mm=10,
    target_signal='X',
    target_origin='Oscilloscope',
    *,
    compute_normalized_global: bool = False,
    prefer_cache: bool = True,
):
    """
    Erstellt Heatmap-Daten direkt aus der DuckDB-Datenbank mit SQL.
    - Binning in Y-Richtung (in mm) für jeden Slot.
    - Berechnung des RMS-Wertes pro Bin.
    - Gibt ein DataFrame mit den Ergebnissen zurück.
    """
    if provider.loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    con = provider.loader_global.con
    table = provider.loader_global.table_name

    def _duckdb_table_exists(connection, name: str) -> bool:
        try:
            return bool(
                connection.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'main' AND lower(table_name) = lower(?)
                    LIMIT 1
                    """,
                    [name],
                ).fetchone()
            )
        except Exception:
            return False

    def _duckdb_table_columns(connection, name: str) -> set[str]:
        try:
            rows = connection.execute(f"PRAGMA table_info('{name}')").fetchall()
        except Exception:
            return set()
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        return {str(r[1]) for r in rows}

    # Phase 6: prefer materialized cache table when present.
    cache_table = "heatmap_bins_cache"
    if prefer_cache and _duckdb_table_exists(con, cache_table):
        cols = _duckdb_table_columns(con, cache_table)
        want_normalized = compute_normalized_global and "RMS_normalized_global" in cols
        normalized_select = ", RMS_normalized_global" if want_normalized else ""

        df_cached = con.execute(
            f"""
            SELECT
                Nut,
                Y_min,
                Y_max,
                Y_bin_center,
                RMS_raw
                {normalized_select}
            FROM {cache_table}
            WHERE Platte = ?
              AND Axis = ?
              AND DataOrigin = ?
              AND bin_size_mm = ?
            ORDER BY Nut, Y_min
            """,
            [str(platte), str(target_signal), str(target_origin), float(bin_size_mm)],
        ).fetchdf()

        # Only accept cache hit when it contains what the caller requested.
        if df_cached is not None and not df_cached.empty:
            if not compute_normalized_global or "RMS_normalized_global" in df_cached.columns:
                return df_cached

    # Notes on schema:
    # - Single table with columns incl. Platte (VARCHAR) + Nut (DOUBLE)
    # - We bin WCS_Y_mm into equal bins per Nut starting at 0
    # - For the last bin, we truncate Y_max to the actual slot max(WCS_Y_mm)
    normalized_select = ""
    if compute_normalized_global:
        normalized_select = """
        ,
        CASE
            WHEN (MAX(RMS_raw) OVER () > MIN(RMS_raw) OVER ())
            THEN (RMS_raw - MIN(RMS_raw) OVER ()) / (MAX(RMS_raw) OVER () - MIN(RMS_raw) OVER ())
            ELSE 0.0
        END AS RMS_normalized_global
        """

    query = f"""
    WITH SignalData AS (
        SELECT
            Nut,
            WCS_Y_mm,
            Value
        FROM {table}
        WHERE Platte = ?
          AND Axis = ?
          AND DataOrigin = ?
          AND Nut IS NOT NULL
          AND WCS_Y_mm IS NOT NULL
          AND Value IS NOT NULL
                    AND WCS_Y_mm >= 0
    ),
    SlotMaxY AS (
        SELECT
            Nut,
            MAX(WCS_Y_mm) AS slot_y_max
        FROM SignalData
        GROUP BY Nut
    ),
    BinnedData AS (
        SELECT
            sd.Nut,
            sd.Value,
            floor(sd.WCS_Y_mm / ?) * ? AS y_min,
            smy.slot_y_max AS slot_y_max
        FROM SignalData sd
        JOIN SlotMaxY smy USING (Nut)
        WHERE sd.WCS_Y_mm < smy.slot_y_max
    ),
    HeatmapBins AS (
        SELECT
            Nut,
            y_min AS Y_min,
            least(y_min + ?, slot_y_max) AS Y_max,
            (y_min + least(y_min + ?, slot_y_max)) / 2.0 AS Y_bin_center,
            sqrt(avg(Value * Value)) AS RMS_raw
        FROM BinnedData
        GROUP BY Nut, y_min, slot_y_max
    )
    SELECT
        Nut,
        Y_min,
        Y_max,
        Y_bin_center,
        RMS_raw
        {normalized_select}
    FROM HeatmapBins
    ORDER BY Nut, Y_min
    """

    params = [
        str(platte),
        str(target_signal),
        str(target_origin),
        float(bin_size_mm),
        float(bin_size_mm),
        float(bin_size_mm),
        float(bin_size_mm),
    ]
    return con.execute(query, params).fetchdf()


def materialize_heatmap_cache(
    plates: list[str] | None = None,
    *,
    bin_size_mm: float = 10.0,
    target_signal: str = "X",
    target_origin: str = "Oscilloscope",
    overwrite: bool = True,
    cache_db_path: Path | None = None,
) -> None:
    """Materialize derived tables inside DuckDB for fast interactive lookups.

    This is an opt-in Phase 6 helper. It does NOT change plotting semantics.
    It creates (if needed) and populates `heatmap_bins_cache`.

    Notes
    -----
        - Requires a writable DuckDB connection. If `provider` is initialized with
            `read_only=False`, we reuse that connection. Otherwise, we try to open a
            separate read-write connection.
            Note: DuckDB does not allow opening the same DB file with a different
            configuration inside one process (e.g., RO + RW). In that case, re-init
            the provider with `read_only=False` (or run this in a separate process).
    - Cache invalidation is manual by default: call again with overwrite=True.
    """

    if provider.loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    import duckdb

    if plates is None:
        plates = [str(p) for p in provider.plates()]
    plates = [str(p) for p in plates]

    cache_table = "heatmap_bins_cache"

    con_rw = None
    owns_connection = False

    try:
        if cache_db_path is not None:
            # Materialize into a separate cache DB by computing in Python and
            # inserting into the cache DB to avoid cross-DB ATTACH issues.
            con_rw = duckdb.connect(str(cache_db_path), read_only=False)
            owns_connection = True

            # create cache table if missing
            con_rw.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cache_table} (
                    Platte VARCHAR,
                    Nut DOUBLE,
                    Axis VARCHAR,
                    DataOrigin VARCHAR,
                    bin_size_mm DOUBLE,
                    Y_min DOUBLE,
                    Y_max DOUBLE,
                    Y_bin_center DOUBLE,
                    RMS_raw DOUBLE,
                    RMS_normalized_global DOUBLE
                )
                """
            )

            for plate in plates:
                if overwrite:
                    con_rw.execute(
                        f"""
                        DELETE FROM {cache_table}
                        WHERE Platte = ? AND Axis = ? AND DataOrigin = ? AND bin_size_mm = ?
                        """,
                        [plate, str(target_signal), str(target_origin), float(bin_size_mm)],
                    )

                df_bins = prepare_equal_bins_heatmap_sql(
                    plate,
                    bin_size_mm=bin_size_mm,
                    target_signal=target_signal,
                    target_origin=target_origin,
                    compute_normalized_global=True,
                    prefer_cache=False,
                )
                if df_bins is None or df_bins.empty:
                    continue

                df_insert = df_bins.copy()
                df_insert['Platte'] = plate
                df_insert['Axis'] = str(target_signal)
                df_insert['DataOrigin'] = str(target_origin)
                df_insert['bin_size_mm'] = float(bin_size_mm)

                if 'RMS_normalized_global' not in df_insert.columns:
                    vmin = df_insert['RMS_raw'].min()
                    vmax = df_insert['RMS_raw'].max()
                    if vmax > vmin:
                        df_insert['RMS_normalized_global'] = (df_insert['RMS_raw'] - vmin) / (vmax - vmin)
                    else:
                        df_insert['RMS_normalized_global'] = 0.0

                cols = [
                    'Platte', 'Nut', 'Axis', 'DataOrigin', 'bin_size_mm',
                    'Y_min', 'Y_max', 'Y_bin_center', 'RMS_raw', 'RMS_normalized_global'
                ]

                con_rw.register('tmp_bins', df_insert[cols])
                con_rw.execute(f"INSERT INTO {cache_table} SELECT * FROM tmp_bins")

        else:
            # Materialize in-place in the provider DB using SQL insert (existing path).
            db_path = provider.loader_global.db_path
            source_table = provider.loader_global.table_name

            if getattr(provider.loader_global, "read_only", True) is False:
                con_rw = provider.loader_global.con
            else:
                try:
                    con_rw = duckdb.connect(str(db_path), read_only=False)
                    owns_connection = True
                except duckdb.ConnectionException as e:
                    raise RuntimeError(
                        "Cannot open a read-write DuckDB connection while an existing "
                        "read-only connection is open in this process. "
                        "Re-run with `provider.close(); provider.init(read_only=False)` "
                        "and then call materialize_heatmap_cache(...)."
                    ) from e

            con_rw.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cache_table} (
                    Platte VARCHAR,
                    Nut DOUBLE,
                    Axis VARCHAR,
                    DataOrigin VARCHAR,
                    bin_size_mm DOUBLE,
                    Y_min DOUBLE,
                    Y_max DOUBLE,
                    Y_bin_center DOUBLE,
                    RMS_raw DOUBLE,
                    RMS_normalized_global DOUBLE
                )
                """
            )

            try:
                con_rw.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{cache_table}_lookup ON {cache_table} (Platte, Axis, DataOrigin, bin_size_mm, Nut, Y_min)"
                )
            except Exception:
                pass

            for plate in plates:
                if overwrite:
                    con_rw.execute(
                        f"""
                        DELETE FROM {cache_table}
                        WHERE Platte = ? AND Axis = ? AND DataOrigin = ? AND bin_size_mm = ?
                        """,
                        [plate, str(target_signal), str(target_origin), float(bin_size_mm)],
                    )

                # Insert using the exact same semantics as prepare_equal_bins_heatmap_sql
                con_rw.execute(
                    f"""
                    INSERT INTO {cache_table}
                    WITH SignalData AS (
                        SELECT
                            Nut,
                            WCS_Y_mm,
                            Value
                        FROM {source_table}
                        WHERE Platte = ?
                          AND Axis = ?
                          AND DataOrigin = ?
                          AND Nut IS NOT NULL
                          AND WCS_Y_mm IS NOT NULL
                          AND Value IS NOT NULL
                          AND WCS_Y_mm >= 0
                    ),
                    SlotMaxY AS (
                        SELECT
                            Nut,
                            MAX(WCS_Y_mm) AS slot_y_max
                        FROM SignalData
                        GROUP BY Nut
                    ),
                    BinnedData AS (
                        SELECT
                            sd.Nut,
                            sd.Value,
                            floor(sd.WCS_Y_mm / ?) * ? AS y_min,
                            smy.slot_y_max AS slot_y_max
                        FROM SignalData sd
                        JOIN SlotMaxY smy USING (Nut)
                        WHERE sd.WCS_Y_mm < smy.slot_y_max
                    ),
                    HeatmapBins AS (
                        SELECT
                            Nut,
                            y_min AS Y_min,
                            least(y_min + ?, slot_y_max) AS Y_max,
                            (y_min + least(y_min + ?, slot_y_max)) / 2.0 AS Y_bin_center,
                            sqrt(avg(Value * Value)) AS RMS_raw
                        FROM BinnedData
                        GROUP BY Nut, y_min, slot_y_max
                    )
                    SELECT
                        ? AS Platte,
                        Nut,
                        ? AS Axis,
                        ? AS DataOrigin,
                        ? AS bin_size_mm,
                        Y_min,
                        Y_max,
                        Y_bin_center,
                        RMS_raw,
                        CASE
                            WHEN (MAX(RMS_raw) OVER () > MIN(RMS_raw) OVER ())
                            THEN (RMS_raw - MIN(RMS_raw) OVER ()) / (MAX(RMS_raw) OVER () - MIN(RMS_raw) OVER ())
                            ELSE 0.0
                        END AS RMS_normalized_global
                    FROM HeatmapBins
                    """,
                    [
                        plate,
                        str(target_signal),
                        str(target_origin),
                        float(bin_size_mm),
                        float(bin_size_mm),
                        float(bin_size_mm),
                        float(bin_size_mm),
                    ],
                )

    finally:
        if owns_connection and con_rw is not None:
            con_rw.close()


def get_min_max_amplitudes_sql_from_db(
    platte,
    *,
    bin_size_mm=10,
    target_signal='X',
    target_origin='Oscilloscope',
):
    """Compute true min/max amplitude mapping without pandas extrema scan.

    Semantics are kept consistent with `get_min_max_amplitudes_sql`:
    - Find the heatmap bin with globally smallest RMS_raw and take MIN(Value) in that bin.
    - Find the heatmap bin with globally largest RMS_raw and take MAX(Value) in that bin.

    Tie-breaking matches pandas `idxmin/idxmax` on a DataFrame ordered by (Nut, Y_min):
    - min: ORDER BY RMS_raw ASC, Nut ASC, Y_min ASC
    - max: ORDER BY RMS_raw DESC, Nut ASC, Y_min ASC
    """

    if provider.loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    con = provider.loader_global.con
    table = provider.loader_global.table_name

    query = f"""
    WITH SignalData AS (
        SELECT
            Nut,
            WCS_Y_mm,
            Value
        FROM {table}
        WHERE Platte = ?
          AND Axis = ?
          AND DataOrigin = ?
          AND Nut IS NOT NULL
          AND WCS_Y_mm IS NOT NULL
          AND Value IS NOT NULL
          AND WCS_Y_mm >= 0
    ),
    SlotMaxY AS (
        SELECT
            Nut,
            MAX(WCS_Y_mm) AS slot_y_max
        FROM SignalData
        GROUP BY Nut
    ),
    BinnedData AS (
        SELECT
            sd.Nut,
            sd.Value,
            floor(sd.WCS_Y_mm / ?) * ? AS y_min,
            smy.slot_y_max AS slot_y_max
        FROM SignalData sd
        JOIN SlotMaxY smy USING (Nut)
        WHERE sd.WCS_Y_mm < smy.slot_y_max
    ),
    HeatmapBins AS (
        SELECT
            Nut,
            y_min AS Y_min,
            least(y_min + ?, slot_y_max) AS Y_max,
            sqrt(avg(Value * Value)) AS RMS_raw
        FROM BinnedData
        GROUP BY Nut, y_min, slot_y_max
    ),
    MinBin AS (
        SELECT Nut, Y_min, Y_max
        FROM HeatmapBins
        ORDER BY RMS_raw ASC, Nut ASC, Y_min ASC
        LIMIT 1
    ),
    MaxBin AS (
        SELECT Nut, Y_min, Y_max
        FROM HeatmapBins
        ORDER BY RMS_raw DESC, Nut ASC, Y_min ASC
        LIMIT 1
    )
    SELECT
        MIN(CASE WHEN Nut = (SELECT Nut FROM MinBin)
                  AND WCS_Y_mm BETWEEN (SELECT Y_min FROM MinBin) AND (SELECT Y_max FROM MinBin)
            THEN Value END) AS min_amplitude,
        MAX(CASE WHEN Nut = (SELECT Nut FROM MaxBin)
                  AND WCS_Y_mm BETWEEN (SELECT Y_min FROM MaxBin) AND (SELECT Y_max FROM MaxBin)
            THEN Value END) AS max_amplitude
    FROM {table}
    WHERE Platte = ?
      AND Axis = ?
      AND DataOrigin = ?
      AND (
            (Nut = (SELECT Nut FROM MinBin)
             AND WCS_Y_mm BETWEEN (SELECT Y_min FROM MinBin) AND (SELECT Y_max FROM MinBin))
         OR (Nut = (SELECT Nut FROM MaxBin)
             AND WCS_Y_mm BETWEEN (SELECT Y_min FROM MaxBin) AND (SELECT Y_max FROM MaxBin))
          )
    """

    params = [
        str(platte),
        str(target_signal),
        str(target_origin),
        float(bin_size_mm),
        float(bin_size_mm),
        float(bin_size_mm),
        str(platte),
        str(target_signal),
        str(target_origin),
    ]

    result = con.execute(query, params).fetchone()
    if not result:
        return 0.0, 0.0

    true_min = result[0] if result[0] is not None else 0.0
    true_max = result[1] if result[1] is not None else 0.0
    return true_min, true_max

def get_min_max_amplitudes_sql(df_heatmap, platte):
    """
    Ermittelt die minimalen und maximalen Schwingungsamplituden für die Bins
    mit dem global niedrigsten und höchsten RMS-Wert direkt aus der DuckDB.
    """
    if df_heatmap.empty:
        return 0.0, 0.0

    if provider.loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    con = provider.loader_global.con
    table = provider.loader_global.table_name

    # Finde die Bins mit min/max RMS
    idx_min_rms = df_heatmap['RMS_raw'].idxmin()
    idx_max_rms = df_heatmap['RMS_raw'].idxmax()

    bin_min = df_heatmap.loc[idx_min_rms]
    bin_max = df_heatmap.loc[idx_max_rms]

    # Extrahiere die Werte für die SQL-Abfrage (Nut is DOUBLE in DB)
    slot_min = float(bin_min['Nut'])
    slot_max = float(bin_max['Nut'])
    y_min_min = float(bin_min['Y_min'])
    y_max_min = float(bin_min['Y_max'])
    y_min_max = float(bin_max['Y_min'])
    y_max_max = float(bin_max['Y_max'])

    # Keep original semantics from old widget code:
    # - stable bin -> MIN(Value)
    # - chatter bin -> MAX(Value)
    query = f"""
    SELECT
        MIN(CASE WHEN Nut = ? AND WCS_Y_mm BETWEEN ? AND ? THEN Value ELSE NULL END) AS min_amplitude,
        MAX(CASE WHEN Nut = ? AND WCS_Y_mm BETWEEN ? AND ? THEN Value ELSE NULL END) AS max_amplitude
    FROM {table}
    WHERE Platte = ?
      AND Axis = 'X'
      AND DataOrigin = 'Oscilloscope'
      AND (
            (Nut = ? AND WCS_Y_mm BETWEEN ? AND ?) OR
            (Nut = ? AND WCS_Y_mm BETWEEN ? AND ?)
          )
    """

    params = [
        slot_min, y_min_min, y_max_min,
        slot_max, y_min_max, y_max_max,
        str(platte),
        slot_min, y_min_min, y_max_min,
        slot_max, y_min_max, y_max_max,
    ]
    result = con.execute(query, params).fetchone()
    
    true_min = result[0] if result[0] is not None else 0.0
    true_max = result[1] if result[1] is not None else 0.0
    
    return true_min, true_max
