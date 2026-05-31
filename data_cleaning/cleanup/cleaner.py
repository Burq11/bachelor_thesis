import duckdb
import re
import polars as pl
import os
import json

def import_InsightHub_meta(file_path):
    return (
        pl.read_excel(file_path, engine="calamine")
        .rename(lambda col: col.capitalize())
        .rename({"Address": "Signal"}, strict=False)
    )

def import_lf_variables_readable(file_path):
    col_dict = {
        'name': 'Signal', 
        'short_description': 'Description', 
        'nc_variable_name': 'Nc_variable_name', 
        'unit': 'Unit'
    }

    header = pl.read_csv(file_path, separator=";", n_rows=0).columns
    target_cols = [c for c in col_dict.keys() if c in header]

    return (
        pl.read_csv(file_path, separator=";", columns=target_cols)
        .rename(col_dict, strict=False)
    )

def clean_signal_name(name: str) -> str:
    if '/' in name:
        name = name.split('/')[-1]
    name = re.sub(r"r\[u1,(\d+),#1\]", r"R\1", name)
    return name

def process_machine_data(meta_json_path: str, hf_parquet_path: str, lf_parquet_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    
    lf_query = f"""
        SELECT 
            Time,
            v.id AS id,
            v.val::DOUBLE AS Value
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

    return df_HF, df_LF

def combine_hf_lf_data(df_HFData: pl.DataFrame, df_LFData: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([df_HFData, df_LFData], how="diagonal")

def merge_signals_with_lfncvar(df: pl.LazyFrame, df_lf_NC_Var: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        pl.col("Signal").str.split("[").list.get(0).alias("Signal_lookup")
    )
    
    df_joined = df.join(
        df_lf_NC_Var, left_on="Signal_lookup", right_on="Signal", how="left", suffix="_meta"
    )
    
    joined_cols = df_joined.collect_schema().names()
    
    return df_joined.with_columns([
        pl.coalesce(["Description", "Description_meta"]).alias("Description") if "Description_meta" in joined_cols else pl.col("Description"),
        pl.coalesce(["Unit", "Unit_meta"]).alias("Unit") if "Unit_meta" in joined_cols else pl.lit(None).alias("Unit")
    ]).drop(["Signal_lookup", "Description_meta", "Unit_meta"], strict=False)

def calculate_duration_seconds(df: pl.DataFrame, time_column: str) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col(time_column) - pl.col(time_column).min()).dt.total_milliseconds() / 1000.0).alias("Duration_Seconds")
    )
def calculate_wcs(df: pl.DataFrame, signal_name="ENC_POS|2", new_column="WCS_Y_mm", offset=20) -> pl.DataFrame:
    wcs_subset = (
        df.filter(pl.col("Signal") == signal_name)
        .filter(pl.col("Value").is_not_null())
        .unique(subset=["Time"])
        .select([
            pl.col("Time"),
            (pl.col("Value") - pl.col("Value").first() - offset).alias(new_column)
        ])
    )
    df = df.join(wcs_subset, on="Time", how="left")
    return df.sort("Time").with_columns(pl.col(new_column).interpolate())

def process_et200_data(file_path: str) -> pl.DataFrame:
    df_et200 = pl.read_parquet(file_path)

    df_et200 = df_et200.unpivot(
        index="timestamp", 
        variable_name="Signal", 
        value_name="Value"
    )

    df_et200 = df_et200.with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").cast(pl.Datetime("us")).alias("Time"),
        
        pl.col("Value").cast(pl.Float64),
        pl.lit("ET200_Data").alias("DataOrigin")
    ]).drop("timestamp") 

    return df_et200

def combine_all_data(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(dfs, how="diagonal")

def cleaner(recording_dir: str, r_parameter: int):
    hf_parquet_path = os.path.join(recording_dir, "hf.parquet")
    lf_parquet_path = os.path.join(recording_dir, "lf.parquet")
    et200_parquet_path = os.path.join(recording_dir, "et200.parquet")
    
    session_dir = os.path.dirname(recording_dir)
    meta_json_path = os.path.join(session_dir, "metadata.json")
    
    output_parquet_path = os.path.join(recording_dir, "cleaned_data.parquet")
    
    if os.path.exists(output_parquet_path):
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

    df = calculate_duration_seconds(df, "Time") 
    df = calculate_wcs(df, signal_name="ENC_POS|2", new_column="WCS_Y_mm", offset=20)
    df = df.with_columns([
        pl.lit(r_parameter).alias("Nut"),
        pl.lit(1).alias("Platte")
    ])

    df.sink_parquet(output_parquet_path)

    return output_parquet_path, recording_dir