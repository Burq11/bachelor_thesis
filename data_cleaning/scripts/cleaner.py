from __future__ import annotations
import duckdb
import re
import polars as pl
import os
import json

def import_InsightHub_meta(file_path):
    """
    Liest die HF-Signalliste aus der Excel-Datei, 
    vereinheitlicht die Spaltennamen auf Großbuchstaben am Anfang und benennt Address in Signal um, 
    damit später darauf gejoint werden kann.
    """
    return (
        pl.read_excel(file_path, engine="calamine")
        .rename(lambda col: col.capitalize())
        .rename({"Address": "Signal"}, strict=False)
    )

def import_lf_variables_readable(file_path):
    """Liest die Variablenliste und schluesselt sie ueber den vollen NC-Pfad auf.

    Die Spalte `name` ist nicht eindeutig: `actFeedRateIpo` existiert sowohl unter
    /Channel/State/ als auch unter /Channel/GeometricAxis/. Ein Join auf dem
    gekuerzten Namen trifft daher beide Zeilen und verdoppelt die Messwerte.
    `successful_readings` enthaelt die aufgeloesten Pfade, die auch in der
    metadata.json stehen, und ist als Schluessel eindeutig.
    """
    df = pl.read_csv(
        file_path, separator=";", encoding="utf-8",
        columns=["name", "short_description", "nc_variable_name", "unit",
                 "successful_readings"],
    )

    df = (
        df.with_columns(pl.col("successful_readings").str.split("\n"))
        .explode("successful_readings")
        .with_columns(pl.col("successful_readings").str.strip_chars().alias("Path"))
        .filter(pl.col("Path").is_not_null() & (pl.col("Path") != ""))
        .select(
            pl.col("Path"),
            pl.col("short_description").alias("Description"),
            pl.col("nc_variable_name").alias("Nc_variable_name"),
            pl.col("unit").alias("Unit"),
        )
    )

    dups = df.filter(pl.col("Path").is_duplicated())
    if dups.height:
        raise ValueError(
            f"Mehrdeutige NC-Pfade in {file_path}: "
            f"{dups['Path'].unique().to_list()[:5]}"
        )
    return df

def clean_signal_name(name: str) -> str:
    """
    Kürzt einen NC-Pfad auf das letzte Segment und schreibt r[u1,323,#1] in R323 um.
    """
    if '/' in name:
        name = name.split('/')[-1]
    name = re.sub(r"r\[u1,(\d+),#1\]", r"R\1", name)
    return name

HF_EVENT_NUMERIC_FIELDS = {
    "Channel":    r"#\s*CH(\d+)",
    "Tick":       r"tick=(-?\d+)",
    "SeekOffset": r"seekOffset=(-?\d+)",
    "LaBuf":      r"laBuf=(-?\d+)",
    "ToolNo":     r"tool=(-?\d+)\(",
    "ToolIdent":  r"tool=-?\d+\((-?\d+)\)",
}
 
HF_EVENT_STRING_FIELDS = {
    "IpoGC":   r"ipoGC=(\S+)",
    "NcBlock": r'"(.*)"',
}
 
HF_EVENT_EMPTY_SCHEMA = {
    "Time": pl.Datetime("us"),
    "Cycle": pl.Int64,
    "Signal": pl.String,
    "Value": pl.Float64,
    "Value_String": pl.String,
    "SamplingPeriod": pl.Float64,
    "Value_Type": pl.String,
    "DataOrigin": pl.String,
}

SIGNAL_LABELS = {
    "R301": "Synchronisationsstatus",
    "R310": "Rattererkennung",
    "R319": "Zähler für Nuten",
    "R320": "Anzahl der zu fräsenden Nuten",
    "R321": "X-Position Nut",
    "R322": "X-Versatz für nächste Nut",
    "R323": "Y-Position Nut",
    "R328": "Z-Position Nut",          # Signal wird aktuell nicht aufgezeichnet
    "R330": "Spindeldrehzahl",
    "R331": "Drehzahländerung nach Nut",
    "R333": "Vorschub für Nutfräsen [mm/U]",
}

SCOPE_LABELS: dict[str, str] = {
    # "dataA1CH1": "Kanal A (V)",
    # "dataA1CH2": "?",
    # "dataA2CH1": "Kanal B (V)",
}


def add_labels(df: pl.DataFrame, extra: dict | None = None) -> pl.DataFrame:
    """ 
    Hängt die deutschen Klartextbezeichnungen aus SIGNAL_LABELS/SCOPE_LABELS als Spalte Label an. Signale ohne Eintrag bleiben NULL.
    """
    mapping = {**SIGNAL_LABELS, **SCOPE_LABELS, **(extra or {})}
    lut = pl.DataFrame(
        {"Signal": list(mapping.keys()), "Label": list(mapping.values())},
        schema={"Signal": pl.String, "Label": pl.String},
    )
    return df.join(lut, on="Signal", how="left")

 
def parse_hf_events(df_events: pl.DataFrame) -> pl.DataFrame:
    """
    Zerlegt die HF-Event-Strings per Regex in Einzelfelder.
      Numerische Felder (Channel, Tick, SeekOffset, LaBuf, ToolNo, ToolIdent) landen in Value, 
      textuelle (IpoGC, NcBlock) in Value_String. NcBlock wird zusätzlich am Semikolon in NcCode und NcComment getrennt, 
      Tick wird zu Cycle. Ergebnis ist Long-Format mit Signal = "<Basisname>|<Feld>".
    """
    if df_events.is_empty():
        return pl.DataFrame(schema=HF_EVENT_EMPTY_SCHEMA)
 
    parsed = df_events.with_columns(
        [
            pl.col("RawEvent").str.extract(pattern, 1).cast(pl.Float64).alias(name)
            for name, pattern in HF_EVENT_NUMERIC_FIELDS.items()
        ]
        + [
            pl.col("RawEvent").str.extract(pattern, 1).str.strip_chars().alias(name)
            for name, pattern in HF_EVENT_STRING_FIELDS.items()
        ]
    ).with_columns(
        pl.col("NcBlock").str.replace_all(r"\s+", " ").str.strip_chars().alias("NcBlock")
    ).with_columns(
        pl.col("NcBlock").str.splitn(";", 2).struct.field("field_0").str.strip_chars().alias("NcCode"),
        pl.col("NcBlock").str.splitn(";", 2).struct.field("field_1").str.strip_chars().alias("NcComment"),
    ).with_columns(
        pl.col("Tick").cast(pl.Int64).alias("Cycle")
    )
 
    index_cols = ["Time", "Cycle", "Signal", "SamplingPeriod"]
 
    numeric_long = (
        parsed
        .unpivot(
            index=index_cols,
            on=list(HF_EVENT_NUMERIC_FIELDS.keys()),
            variable_name="Field",
            value_name="Value",
        )
        .with_columns(
            (pl.col("Signal") + pl.lit("|") + pl.col("Field")).alias("Signal"),
            pl.lit(None, dtype=pl.String).alias("Value_String"),
            pl.lit("Double").alias("Value_Type"),
        )
        .drop("Field")
    )
 
    string_long = (
        parsed
        .unpivot(
            index=index_cols,
            on=list(HF_EVENT_STRING_FIELDS.keys()) + ["NcCode", "NcComment", "RawEvent"],
            variable_name="Field",
            value_name="Value_String",
        )
        .with_columns(
            (pl.col("Signal") + pl.lit("|") + pl.col("Field")).alias("Signal"),
            pl.lit(None, dtype=pl.Float64).alias("Value"),
            pl.lit("String").alias("Value_Type"),
        )
        .drop("Field")
    )
 
    return (
        pl.concat([numeric_long, string_long], how="diagonal")
        .filter(pl.col("Value").is_not_null() | pl.col("Value_String").is_not_null())
        .with_columns(pl.lit("HF_Event").alias("DataOrigin"))
    )


def process_machine_data(meta_json_path: str, hf_parquet_path: str, lf_parquet_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Das Herzstück. Baut aus metadata.json die HF- und LF-Signaltabellen, 
    öffnet eine temporäre DuckDB und entpackt damit die verschachtelten Parquet-Strukturen:

    HF-Numerik: records → vals → val-Arrays werden aufgefaltet, 
        jedes Sample bekommt aus seinem Index einen Zeitstempel im 2-ms-Raster, id = '1' liefert den Zyklenzähler.

    HF-Strings: alle Zeilen mit string_val gehen durch parse_hf_events. 
        NcCode, NcComment und IpoGC werden breit gepivotet und per join_asof rückwärts auf den Zyklus der Messdaten gelegt,
        jedes Sample weiß dadurch, welcher NC-Satz gerade aktiv war.

    LF: flacheres Schema, direkt entpackt.
    """
    with open(meta_json_path, 'r') as f:
        meta = json.load(f)
 
    df_hf_raw = pl.DataFrame(meta['sinumerik_signals']['hf_signals'])
    df_lf_raw = pl.DataFrame(meta['sinumerik_signals']['lf_signals'])
 
    df_hf_meta = df_hf_raw.select(
        pl.col("id").cast(pl.String),
        pl.col("name").map_elements(clean_signal_name, return_dtype=pl.String).alias("Signal"),
        pl.lit(None).cast(pl.String).alias("Path"),
        pl.col("acquisitionCycleInMs").cast(pl.Float64).alias("SamplingPeriod"),
        pl.col("dataType").cast(pl.String).alias("Value_Type")
    )
 
    df_lf_meta = df_lf_raw.select(
        pl.col("id").cast(pl.String),
        pl.col("name").map_elements(clean_signal_name, return_dtype=pl.String).alias("Signal"),
        pl.col("name").alias("Path"),
        pl.col("acquisitionCycleInMs").cast(pl.Float64).alias("SamplingPeriod"),
        pl.col("dataType").cast(pl.String).alias("Value_Type")
    )
 
    current_recording_dir = os.path.dirname(hf_parquet_path)
    
    temp_db_path = os.path.join(current_recording_dir, "temp_processing.duckdb")
    temp_dir_path = os.path.join(current_recording_dir, "duckdb_temp")
 
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except OSError:
            pass
 
    con = duckdb.connect(temp_db_path)
    absoluter_temp_pfad = os.path.abspath(temp_dir_path)
    con.execute(f"PRAGMA temp_directory='{absoluter_temp_pfad}'")
 
    # --- HF: numerische Array-Signale ------------------------------------
    hf_query = f"""
        WITH unnested_records AS (
            SELECT unnest(records) AS r FROM read_parquet('{hf_parquet_path}')
        ),
        
        blocks_with_id AS (
            SELECT 
                ROW_NUMBER() OVER () AS block_id,
                r.ts::TIMESTAMP AS block_timestamp,
                r.vals AS vals
            FROM unnested_records
        ),
        
        block_data AS (
            SELECT 
                block_id,
                block_timestamp, 
                unnest(vals) AS v 
            FROM blocks_with_id
        ),
        
        cycle_blocks AS (
            SELECT 
                block_id, 
                v.val AS cycle_array 
            FROM block_data 
            WHERE v.id = '1'
        ),
        
        signal_data AS (
            SELECT 
                block_id,
                block_timestamp,
                v.id AS id,
                generate_subscripts(v.val, 1) AS chunk_index,
                unnest(v.val) AS Value
            FROM block_data
            WHERE v.id != '1'
              AND v.val IS NOT NULL
        )
        
        SELECT 
            s.block_timestamp + ((s.chunk_index - 1) * 2) * INTERVAL 1 MILLISECOND AS Time,
            c.cycle_array[s.chunk_index]::BIGINT AS "Cycle",
            s.id,
            s.Value::DOUBLE AS Value
        FROM signal_data s
        
        LEFT JOIN cycle_blocks c ON s.block_id = c.block_id;
    """
 
    hf_arrow = con.execute(hf_query).arrow() 
    
    df_HF = pl.from_arrow(hf_arrow).lazy()
    df_HF = df_HF.join(df_hf_meta.lazy(), on="id", how="left").drop("id")
    df_HF = df_HF.with_columns(pl.lit("HF_Data").alias("DataOrigin"))
    
    df_HF = df_HF.collect()
 
    # --- HF: String-Events (z.B. id 175 / HF_EVENT) ----------------------
    hf_event_query = f"""
        WITH unnested_records AS (
            SELECT unnest(records) AS r FROM read_parquet('{hf_parquet_path}')
        ),
        block_data AS (
            SELECT 
                r.ts::TIMESTAMP AS Time,
                unnest(r.vals) AS v
            FROM unnested_records
        )
        SELECT 
            Time,
            v.id AS id,
            v.string_val AS RawEvent
        FROM block_data
        WHERE v.string_val IS NOT NULL;
    """
 
    df_hf_events = pl.from_arrow(con.execute(hf_event_query).arrow())
    df_hf_events = (
        df_hf_events
        .join(df_hf_meta.drop(["Path", "Value_Type"]), on="id", how="left")
        .drop("id")
    )
    df_hf_events = parse_hf_events(df_hf_events)

    if not df_hf_events.is_empty():
        events_wide = (
            df_hf_events
            .filter(pl.col("Signal").is_in(
                ["HF_EVENT|NcCode", "HF_EVENT|NcComment", "HF_EVENT|IpoGC"]
            ))
            .pivot(on="Signal", index="Cycle", values="Value_String")
            .rename(lambda c: c.replace("HF_EVENT|", ""))
            .sort("Cycle")
        )

        has_cycle = df_HF.filter(pl.col("Cycle").is_not_null()).sort("Cycle")
        no_cycle = df_HF.filter(pl.col("Cycle").is_null())

        df_HF = pl.concat(
            [
                has_cycle.join_asof(events_wide, on="Cycle", strategy="backward"),
                no_cycle,
                df_hf_events,
            ],
            how="diagonal",
        )
 
    # --- LF: Zahl- und Stringwerte ---------------------------------------
    lf_query = f"""
        SELECT 
            Time,
            v.id AS id,
            v.val::DOUBLE AS Value,
            v.string_val AS Value_String
        FROM (
            SELECT 
                r.ts::TIMESTAMP AS Time, 
                unnest(r.vals) AS v 
            FROM (
                SELECT unnest(records) AS r 
                FROM read_parquet('{lf_parquet_path}')
            )
        );
    """
    lf_arrow = con.execute(lf_query).arrow()
    
    df_LF = pl.from_arrow(lf_arrow).lazy()
    df_LF = df_LF.join(df_lf_meta.lazy(), on="id", how="left").drop("id")
    df_LF = df_LF.with_columns(pl.lit("LF_Data").alias("DataOrigin"))
    df_LF = df_LF.collect()
 
    con.close()
    try:
        os.remove(temp_db_path)
    except OSError:
        pass

    print(df_hf_events.filter(pl.col("Signal") == "HF_EVENT|NcComment")["Cycle"].max(),
        df_HF.filter(pl.col("DataOrigin") == "HF_Data")["Cycle"].max())

 
    return df_HF, df_LF

def merge_signals_with_lfncvar(df: pl.LazyFrame, df_lf_NC_Var: pl.LazyFrame) -> pl.LazyFrame:
    """Joint Beschreibung und Einheit über den vollen NC-Pfad an und führt bestehende mit neuen Werten per coalesce zusammen."""
    df_joined = df.join(df_lf_NC_Var, on="Path", how="left", suffix="_meta")
    joined_cols = df_joined.collect_schema().names()

    return df_joined.with_columns([
        pl.coalesce(["Description", "Description_meta"]).alias("Description")
        if "Description_meta" in joined_cols else pl.col("Description"),
        pl.coalesce(["Unit", "Unit_meta"]).alias("Unit")
        if "Unit_meta" in joined_cols else pl.lit(None).alias("Unit"),
    ]).drop(["Description_meta", "Unit_meta"], strict=False)

def calculate_duration_seconds(df: pl.DataFrame, time_column: str) -> pl.DataFrame:
    """Sekunden seit dem frühesten Zeitstempel der Aufnahme."""
    return df.with_columns(
        ((pl.col(time_column) - pl.col(time_column).min()).dt.total_milliseconds() / 1000.0).alias("Duration_Seconds")
    )

def estimate_hf_lf_offset(
    df: pl.DataFrame,
    sync_signal: str = "SYNC_CYCLE_HF",
    hf_origins: tuple = ("HF_Data", "HF_Event"),
    max_drift_ms: float = 2.0,
    verbose: bool = True,
):
    """
    HF und LF werden von zwei unsynchronisierten Uhren gestempelt, 
    zählen aber denselben Zyklus (Cycle bzw. SYNC_CYCLE_HF). Über den überlappenden 
    Zyklenbereich wird der Median von t_lf − t_hf bestimmt. Liefert zusätzlich die Drift 
    (Vergleich erstes vs. letztes Zehntel) und ein ok-Flag. Wirft, wenn weniger als 10 gemeinsame Zyklen existieren.
    """
    sync = (
        df.filter(pl.col("Signal") == sync_signal)
        .select(
            pl.col("Value").round(0).cast(pl.Int64).alias("Cycle"),
            pl.col("Time").alias("t_lf"),
        )
        .drop_nulls()
        .unique(subset=["Cycle"], keep="first")
        .sort("Cycle")
    )
 
    hf = (
        df.filter(pl.col("DataOrigin").is_in(list(hf_origins)) & pl.col("Cycle").is_not_null())
        .group_by("Cycle")
        .agg(pl.col("Time").min().alias("t_hf"))
        .sort("Cycle")
    )
 
    pair = hf.join(sync, on="Cycle", how="inner").sort("Cycle")
    if pair.height < 10:
        raise ValueError(
            f"Nur {pair.height} gemeinsame Zyklen — Synchronisation nicht möglich. "
            f"HF: {hf['Cycle'].min()}…{hf['Cycle'].max()}, "
            f"{sync_signal}: {sync['Cycle'].min()}…{sync['Cycle'].max()}"
        )
 
    delta_us = (pair["t_lf"] - pair["t_hf"]).dt.total_microseconds()
    offset_us = int(delta_us.median())
 
    n = delta_us.len()
    head = float(delta_us.head(max(1, n // 10)).median())
    tail = float(delta_us.tail(max(1, n // 10)).median())
    drift_ms = (tail - head) / 1000.0
    ok = abs(drift_ms) <= max_drift_ms
  
    return {"offset_us": offset_us, "drift_ms": drift_ms,
            "n_pairs": pair.height, "ok": ok, "sigma_ms": float(delta_us.std()) / 1000}
 
 
def sync_hf_to_lf(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """
    Addiert diesen Versatz als Konstante auf alle HF-Zeitstempel. 
    Die 2-ms-Feinstruktur innerhalb eines Zyklus bleibt dabei erhalten.
    """
    info = estimate_hf_lf_offset(df, **kwargs)
    hf_origins = kwargs.get("hf_origins", ("HF_Data", "HF_Event"))
 
    return df.with_columns(
        pl.when(pl.col("DataOrigin").is_in(list(hf_origins)))
        .then(pl.col("Time") + pl.duration(microseconds=info["offset_us"]))
        .otherwise(pl.col("Time"))
        .alias("Time")
    ).sort("Time")


def calculate_wcs(
    df: pl.DataFrame,
    signal_name: str = "ENC_POS|2",
    new_column: str = "WCS_Y_mm",
    y_zero: float | None = None,
    offset: float = 0.0,
    tolerance: str = "10ms",
) -> pl.DataFrame:
    """
    Rechnet die Y-Achsposition (ENC_POS|2) ins Werkstückkoordinatensystem um. 
    Nullpunkt ist entweder ein fest übergebener absoluter Wert (y_zero) oder der erste Messwert plus Offset. 
    Die Zuordnung auf alle übrigen Zeilen läuft über join_asof mit nearest und 10-ms-Toleranz, 
    damit auch ET200- und LF-Zeilen einen Wert bekommen.
    """
    pos = (
        df.filter((pl.col("Signal") == signal_name) & pl.col("Value").is_not_null())
        .select("Time", "Value")
        .sort("Time")
        .unique(subset=["Time"], keep="first", maintain_order=True)
    )
    if pos.is_empty():
        raise ValueError(f"Signal '{signal_name}' nicht gefunden oder ohne Werte.")
 
    ref = float(y_zero) if y_zero is not None else float(pos["Value"][0]) + float(offset)
 
    pos = pos.select(
        pl.col("Time"),
        (pl.col("Value") - ref).alias(new_column),
    )
 
    out = df.sort("Time").join_asof(
        pos, on="Time", strategy="nearest", tolerance=tolerance
    )
 
    n_null = out.filter(pl.col("DataOrigin") != "ET200_Data")[new_column].null_count()
    if n_null:
        print(f"{new_column}: {n_null:,} von {out.height:,} Zeilen ohne Zuordnung "
              f"(Toleranz {tolerance} zu klein?)")
    return out


def calibrate_y_zero(
    parquet_paths: list[str],
    pos_signal: str = "ENC_POS|2",
    ref_signal: str = "R323",
) -> pl.DataFrame:
    """Diagnose: Gibt es einen konstanten Versatz zwischen Achsposition und R323?
 
    `R323` ist die programmierte Y-Position der Nut im WKS. Ist
    `ENC_POS|2 − R323` über alle Aufnahmen konstant, ist dieser Wert der gesuchte
    `y_zero` und der Nullpunkt damit absolut statt aufnahmerelativ.
 
    Läuft auf bereits erzeugten Parquet-Dateien — also einmal mit `y_zero=None`
    durchlaufen lassen, kalibrieren, dann mit `y_zero=<Wert>` neu erzeugen.
    """
    rows = []
    for p in parquet_paths:
        lf = pl.scan_parquet(p)
        pos = (
            lf.filter((pl.col("Signal") == pos_signal) & pl.col("Value").is_not_null())
            .select("Time", "Value").sort("Time").collect()
        )
        ref = (
            lf.filter((pl.col("Signal") == ref_signal) & pl.col("Value").is_not_null())
            .select("Value").collect()
        )
        if pos.is_empty():
            continue
        rows.append({
            "file": os.path.basename(p),
            "Nut": lf.select("Nut").first().collect().item(),
            f"{pos_signal}_first": pos["Value"][0],
            f"{pos_signal}_min": pos["Value"].min(),
            f"{pos_signal}_max": pos["Value"].max(),
            f"{ref_signal}_median": ref["Value"].median() if not ref.is_empty() else None,
        })
    out = pl.DataFrame(rows)
    if out.is_empty():
        print("Keine Daten gefunden.")
        return out
 
    col_ref = f"{ref_signal}_median"
    if out[col_ref].null_count() < out.height:
        out = out.with_columns(
            (pl.col(f"{pos_signal}_min") - pl.col(col_ref)).alias("delta")
        )
        spread = out["delta"].max() - out["delta"].min()
        print(f"delta = {pos_signal}_min − {ref_signal}: "
              f"Spannweite über {out.height} Aufnahmen = {spread:.4f} mm")
        if spread < 0.1:
            print(f"Konstant ⇒ y_zero = {out['delta'].median():.4f} verwenden")
        else:
            print("Nicht konstant ⇒ Nullpunkt wechselt zwischen Aufnahmen; "
                  "mit dem Datenverantwortlichen klären oder WCS_POS|2 nachfordern")
    return out


def process_et200_data(file_path: str) -> pl.DataFrame:
    """Liest die ET200-Parquet, dreht sie von breit auf lang (unpivot) und rechnet den Epoch-Mikrosekunden-Zeitstempel in ein Datetime um."""
    df_et200 = pl.read_parquet(file_path)

    df_et200 = df_et200.unpivot(
        index="timestamp", 
        variable_name="Signal", 
        value_name="Value"
    )

    df_et200 = df_et200.with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="us").cast(pl.Datetime("us")).alias("Time"),
        
        pl.col("Value").cast(pl.Float64),
        pl.lit("ET200_Data").alias("DataOrigin")
    ]).drop("timestamp") 

    return df_et200

def combine_all_data(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Fügt HF, LF und ET200 diagonal zusammen, also unter Vereinigung der Spaltenmengen."""
    return pl.concat(dfs, how="diagonal")

def cleaner(
    recording_dir: str,
    nut: int,
    platte: int = 1,
    y_zero: float | None = None,
    overwrite: bool = False,
):
    """
    Führt für einen Aufnahmeordner alles zusammen: Pfade bauen, 
    bei vorhandener Ausgabe überspringen, fehlende HF/LF-Dateien abbrechen, entpacken, 
    Metadaten anjoinen, Axisname zu Axis normalisieren, synchronisieren, Duration und WCS berechnen, 
    Labels und die Spalten Nut/Platte anhängen, Parquet schreiben.
    """
    hf_parquet_path = os.path.join(recording_dir, "hf.parquet")
    lf_parquet_path = os.path.join(recording_dir, "lf.parquet")
    et200_parquet_path = os.path.join(recording_dir, "et200.parquet")
 
    session_dir = os.path.dirname(recording_dir)
    meta_json_path = os.path.join(session_dir, "metadata.json")
 
    output_parquet_path = f"./interim/cleaned_data_{platte}_{nut}.parquet"
 
    if os.path.exists(output_parquet_path) and not overwrite:
        print(f"Überspringe (bereits vorhanden): {recording_dir}")
        return output_parquet_path, recording_dir
 
    hf_meta_excel = "./HFmeta_insight_hub.xlsx"
    lf_meta_csv = "./LF_variables_readable.csv"
 
    if not os.path.exists(hf_parquet_path) or not os.path.exists(lf_parquet_path):
        return None, recording_dir
 
    df_HF, df_LF = process_machine_data(meta_json_path, hf_parquet_path, lf_parquet_path)
 
    lazy_dfs = [df_HF.lazy(), df_LF.lazy()]
 
    if os.path.exists(et200_parquet_path):
        df_ET200 = process_et200_data(et200_parquet_path)
        lazy_dfs.append(df_ET200.lazy())
 
    lazy_InsightHub_meta = import_InsightHub_meta(hf_meta_excel).lazy()
    lazy_lf_NC_Var = import_lf_variables_readable(lf_meta_csv).lazy()
 
    df = combine_all_data(lazy_dfs)
    df = df.join(lazy_InsightHub_meta, on="Signal", how="left", coalesce=True)
    df = merge_signals_with_lfncvar(df, lazy_lf_NC_Var)
    df = df.with_columns(
        pl.col("Axisname").str.replace_all(r"\d+", "").alias("Axis")
    ).drop("Axisname")
 
    df = df.collect()
 
    df = sync_hf_to_lf(df)
    df = calculate_duration_seconds(df, "Time")
    df = calculate_wcs(df, signal_name="ENC_POS|2", new_column="WCS_Y_mm", offset=20)
    df = add_labels(df)
    df = df.with_columns([
        pl.lit(nut).alias("Nut"),
        pl.lit(platte).alias("Platte"),
    ])
 
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.write_parquet(output_parquet_path)

    return output_parquet_path, recording_dir
 
