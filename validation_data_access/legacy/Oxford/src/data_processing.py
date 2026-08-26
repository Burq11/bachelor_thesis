# src/data_processing.py

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


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


def summarize_chatter_cases(df):
    """
    Gibt eine Übersicht über Chatter-Fälle (0 und 1), wenn vorhanden.
    Berücksichtigt nur vorhandene Fälle im Signal 'R310'.
    """

    chatter_signal = "R310"
    ratter_df = df[df["Signal"] == chatter_signal]

    # Prüfen, ob Chatter überhaupt aufgetreten ist
    unique_vals = ratter_df["Value"].dropna().unique()

    # Gemeinsame Parameter
    drehzahl = df.loc[df["Signal"] == "R330", "Value"].min()
    werkzeugradius = df.loc[df["Signal"] == "actToolRadius[u1]", "Value"].min()
    x_pos = df.loc[df["Signal"] == "R321", "Value"].min()
    werkzeug = df.loc[df["Signal"] == "actToolIdent[u1,1]", "Value_String"].unique()[0]
    nut_id = df.loc[df["Signal"] == "R319", "Value"].min()

    result = []

    # Kein Chatter
    if 0.0 in unique_vals:
        y_max_no_chatter = ratter_df.loc[ratter_df["Value"] == 0.0, "WCS_Y_mm"].max()
        result.append(
            {
                "Chatter": 0,
                "Y_max": y_max_no_chatter,
                "Drehzahl": drehzahl,
                "Werkzeugradius": werkzeugradius,
                "X_Position_Nut": x_pos,
                "Werkzeug": werkzeug,
                "Nut_ID": nut_id,
            }
        )

    # Chatter erkannt
    if 1.0 in unique_vals:
        y_max_chatter = df["WCS_Y_mm"].max()
        result.append(
            {
                "Chatter": 1,
                "Y_max": y_max_chatter,
                "Drehzahl": drehzahl,
                "Werkzeugradius": werkzeugradius,
                "X_Position_Nut": x_pos,
                "Werkzeug": werkzeug,
                "Nut_ID": nut_id,
            }
        )

    return pd.DataFrame(result)

def analyze_platte(platte_nummer: int, processed_path: Path, summarize_fn, plot_fn):
    """
    Analysiert alle Nuten einer gegebenen Platte und erstellt eine Digital-Twin-Visualisierung.

    Diese Funktion:
    - Lädt alle Parquet-Dateien der gewählten Platte
    - Führt für jede Nut die übergebene Analysefunktion (summarize_fn) aus
    - Fasst alle Ergebnisse zusammen
    - Gibt eine Visualisierung über plot_fn zurück

    Parameters
    ----------
    platte_nummer : int
        Die Nummer der zu analysierenden Platte (z. B. 4).
    processed_path : Path
        Pfad zum Ordner mit den prozessierten Parquet-Dateien.
    summarize_fn : function
        Analysefunktion, z. B. summarize_chatter_cases(df), die 0, 1 oder 2 Chatter-Zustände pro Nut extrahiert.
    plot_fn : function
        Funktion zur Visualisierung der aggregierten Platte, z. B. plot_digital_twin_segmented(df_summary).

    Returns
    -------
    df_summary : pd.DataFrame
        Zusammenfassung aller Nuten dieser Platte (eine oder zwei Zeilen pro Nut, je nach Ratterzustand).
    fig : plotly.graph_objects.Figure
        Interaktive Plotly-Visualisierung der Platte mit Chatter-Markierungen.
    """
    suchmuster = f"Platte_{platte_nummer}_*Nut_*.parquet"
    parquet_files = sorted(processed_path.glob(suchmuster))

    zusammenfassungen = []
    for path in parquet_files:
        try:
            df = pd.read_parquet(path)
            df_summary = summarize_fn(df)
            zusammenfassungen.append(df_summary)
        except Exception as e:
            print(f"⚠️ Fehler bei {path.name}: {e}")

    if not zusammenfassungen:
        print(
            f"[INFO] Keine gültigen Parquet-Dateien für Platte {platte_nummer} gefunden."
        )
        return pd.DataFrame(), go.Figure()

    df_summary_gesamt = pd.concat(zusammenfassungen, ignore_index=True)
    fig = plot_fn(df_summary_gesamt)
    return df_summary_gesamt, fig


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

import numpy as np
import pandas as pd

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

from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np

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

import numpy as np
import plotly.graph_objects as go
def prepare_equal_bins_heatmap(df, 
                              slot_column='Nut', 
                              y_column='WCS_Y_mm', 
                              signal_column='Signal', 
                              axis_column='Axis', 
                              origin_column='DataOrigin', 
                              value_column='Value',
                              target_signal='X', 
                              target_origin='Oscilloscope',
                              bin_size_mm=20):
    """
    Prepare heatmap data with equal Y-bins (in mm) per slot.
    Each slot is binned independently starting from Y=0 and trimmed to its true Y_max.
    The last bin is shortened if needed.
    """
    import numpy as np
    import pandas as pd

    # Filter for the relevant signal and origin
    df_signal = df[(df[axis_column] == target_signal) & (df[origin_column] == target_origin)].copy()
    if df_signal.empty:
        return pd.DataFrame()

    results = []

    # Process each slot separately
    for slot_id in sorted(df_signal[slot_column].unique()):
        slot_data = df_signal[df_signal[slot_column] == slot_id].copy()
        if slot_data.empty:
            continue

        # Always start from Y=0 and go to the maximum Y value for this slot
        y_max = slot_data[y_column].max()
        
        # Define bin edges starting from 0 and going to y_max
        y_edges = np.arange(0, y_max + bin_size_mm, bin_size_mm)
        # Ensure the last edge doesn't exceed the actual data range
        if y_edges[-1] > y_max:
            y_edges[-1] = y_max

        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Digitize Y positions for this slot
        bin_indices = np.digitize(slot_data[y_column], y_edges) - 1

        for i in range(len(y_centers)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_values = slot_data[value_column].values[mask]
                rms_value = np.sqrt(np.mean(bin_values ** 2))
                results.append({
                    slot_column: slot_id,
                    'Y_bin_center': y_centers[i],
                    'RMS_raw': rms_value,
                    'Y_min': y_edges[i],
                    'Y_max': y_edges[i + 1]
                })

    return pd.DataFrame(results)
        
def analyze_platte_heatmap(platte_nummer, processed_path, summarize_fn, plot_fn):
    suchmuster = f"Platte_{platte_nummer}_*Nut_*.parquet"
    parquet_files = sorted(processed_path.glob(suchmuster))
    
    all_heatmap_data = []
    for path in parquet_files:
        try:
            df = pd.read_parquet(path)
            df_heatmap = summarize_fn(df)
            if not df_heatmap.empty:
                all_heatmap_data.append(df_heatmap)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
    
    if not all_heatmap_data:
        print(f"No valid heatmap data found for plate {platte_nummer}")
        return pd.DataFrame(), go.Figure()
    
    # Combine all heatmap data once
    df_heatmap_combined = pd.concat(all_heatmap_data, ignore_index=True)

    # Single-step global normalization (Normalization method: Min-Max )
    global_min = df_heatmap_combined['RMS_raw'].min()
    global_max = df_heatmap_combined['RMS_raw'].max()
    if global_max > global_min:
        df_heatmap_combined['RMS_normalized_global'] = (
            (df_heatmap_combined['RMS_raw'] - global_min) / (global_max - global_min)
        )
    else:
        df_heatmap_combined['RMS_normalized_global'] = 0
    
    # Get slot-position summary if possible
    try:
        from validation_data_access.legacy.Oxford.src.data_processing import summarize_chatter_cases, analyze_platte
        df_summary, _ = analyze_platte(platte_nummer, processed_path, summarize_chatter_cases, lambda x: None)
    except Exception as e:
        print(f"Could not compute df_summary: {e}")
        df_summary = None
    
    fig = plot_fn(df_heatmap_combined, df_summary=df_summary)
    return df_heatmap_combined, fig

########### PCA ###############
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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

def run_pca(df_wide, n_components=5, return_scaled=False):
    """
    Run PCA on a wide-format dataframe.
    - Standardizes features (mean=0, std=1)
    - Returns explained variance and loadings
    - Optionally returns the standardized dataframe
    """
    # Drop non-numeric just in case
    X = df_wide.select_dtypes(include=["number"])

    # Standardize and keep DataFrame
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=df_wide.index)

    # Run PCA on DataFrame (with feature names)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled_df)

    explained_variance = pca.explained_variance_ratio_

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X.columns,
    )

    if return_scaled:
        return explained_variance, loadings, X_scaled_df, pca
    else:
        return explained_variance, loadings, pca