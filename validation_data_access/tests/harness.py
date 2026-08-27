"""Everything the validation notebook needs to drive both analysis stacks.

Sections, in order:

1. Cases          - the plates, slots and parameters used throughout
2. Stacks         - importing the legacy and the new tree into one interpreter
3. Legacy         - the legacy pipelines, lifted out of their widget callbacks
4. Compare        - order-independent frame comparison
5. Figures        - plotly figures reduced to comparable signatures
6. Measure        - timing and peak memory, one child process per measurement
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import platform
import re
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ===========================================================================
# 1. Cases
# ===========================================================================

PLATES = ["14", "22", "24", "25", "26", "27", "28"]

#: Heatmap 
BIN_SIZE_MM = 10.0
TARGET_SIGNAL = "X"
TARGET_ORIGIN = "Oscilloscope"

#: (plate, slot) pairs for the row-level and plot checks.
SLOT_CASES = [("14", 1), ("22", 3), ("24", 8), ("25", 15), ("26", 20), ("27", 30), ("28", 40)]

#: Filtering benchmark.
TARGETED_QUERY = {
    "plate": "22",
    "slot": 3,
    "data_origin": "HF_Data",
    "signals": ["CMD_SPEED|1"],
    "wcs_min": 0.0,
    "wcs_max": 50.0,
}

TARGETED_QUERY_AXIS = {
    "plate": "22",
    "slot": 3,
    "data_origin": "Oscilloscope",
    "axis": "X",
    "wcs_min": 0.0,
    "wcs_max": 50.0,
}

#: Data origin used for the axis-wise plots.
PLOT_ORIGIN = "HF_Data"

#: Matches viz/visualizer.py's max_display_points default.
MAX_DISPLAY_POINTS = 10_000

#: Plate used for the rendered-figure comparison; 
FIGURE_PLATE = "26"

#: Plates and slots timed in the performance sections.
PERF_PLATES = ["26", "22", "27"]
PERF_SLOTS = [("22", 3), ("27", 30)]


# ===========================================================================
# 2. Stacks
# ===========================================================================

# validation_data_access/tests/harness.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
OXFORD_ROOT = REPO_ROOT / "oxford_notebook"
LEGACY_ROOT = REPO_ROOT / "validation_data_access" / "legacy" / "Oxford"

LEGACY_PARQUET_DIR = LEGACY_ROOT / "data" / "2025-05_Oxford_und_Rebecka" / "processed" / "merge_ext"

#: The DuckDB build of the same Parquet files. 
VALIDATION_DB = OXFORD_ROOT / "data" / "DBold.duckdb"

RESULTS_DIR = REPO_ROOT / "validation_data_access" / "results"


def prepare_environment() -> None:
    """Put both trees on sys.path and register the shared plotly template."""
    for path in (REPO_ROOT, OXFORD_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import plotly.io as pio

    pio.templates.default = "plotly_dark"

    from viz import IWF_template

    IWF_template.register_templates()


def data_status() -> dict:
    """Report which inputs are present, so the notebook can stop cleanly if they are not."""
    files = sorted(LEGACY_PARQUET_DIR.glob("*.parquet")) if LEGACY_PARQUET_DIR.is_dir() else []
    return {
        "legacy_parquet_dir": str(LEGACY_PARQUET_DIR),
        "legacy_parquet_files": len(files),
        "validation_db": str(VALIDATION_DB),
        "validation_db_available": VALIDATION_DB.is_file(),
        "ready": bool(files) and VALIDATION_DB.is_file(),
    }


def init_new_stack():
    """Point the provider at the validation database and return the module."""
    prepare_environment()

    if not VALIDATION_DB.is_file():
        raise FileNotFoundError(f"{VALIDATION_DB} not found; see VALIDATION.md.")

    from src import provider

    provider.init(db_path=VALIDATION_DB)

    if provider.table() != "my_table":
        raise RuntimeError(f"Expected table 'my_table', got {provider.table()!r}")

    return provider


def load_stacks() -> dict:
    """Import both stacks and initialise the provider."""
    provider = init_new_stack()

    from validation_data_access.legacy.Oxford.src import data_processing as legacy_dp
    from validation_data_access.legacy.Oxford.viz import visualizer as legacy_viz
    from src import data_processing as new_dp
    from viz import visualizer as new_viz

    return {
        "legacy_dp": legacy_dp,
        "legacy_viz": legacy_viz,
        "provider": provider,
        "new_dp": new_dp,
        "new_viz": new_viz,
    }


# ===========================================================================
# 3. Legacy
# ===========================================================================

def plate_files(plate: str | int, parquet_dir: Path | None = None) -> list[Path]:
    """Legacy file discovery: a glob over a naming convention (`widgets_digital_twin.py:358`)."""
    return sorted((parquet_dir or LEGACY_PARQUET_DIR).glob(f"Platte_{plate}_*Nut_*.parquet"))


def slot_file(plate: str | int, slot: int, parquet_dir: Path | None = None) -> Path:
    return (parquet_dir or LEGACY_PARQUET_DIR) / f"Platte_{plate}_Nut_{slot}_ext.parquet"


def legacy_slot_df(plate: str | int, slot: int, parquet_dir: Path | None = None) -> pd.DataFrame:
    """How the legacy system obtains one slot: read the whole file."""
    return pd.read_parquet(slot_file(plate, slot, parquet_dir))


def legacy_targeted_query(
    plate: str | int,
    slot: int,
    *,
    data_origin: str,
    signals: list[str],
    wcs_min: float,
    wcs_max: float,
    parquet_dir: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load-then-filter: the legacy equivalent of one `provider.df(...)` call.

    Also returns how many rows had to be materialised, which is the whole file.
    """
    df = pd.read_parquet(slot_file(plate, slot, parquet_dir))
    mask = (
        (df["DataOrigin"] == data_origin)
        & (df["Signal"].isin(signals))
        & (df["WCS_Y_mm"] >= wcs_min)
        & (df["WCS_Y_mm"] <= wcs_max)
    )
    return df.loc[mask], len(df)


def legacy_targeted_query_axis(
    plate: str | int,
    slot: int,
    *,
    data_origin: str,
    axis: str,
    wcs_min: float,
    wcs_max: float,
    parquet_dir: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    """The same question against Oscilloscope data, which is keyed by `Axis`."""
    df = pd.read_parquet(slot_file(plate, slot, parquet_dir))
    mask = (
        (df["DataOrigin"] == data_origin)
        & (df["Axis"] == axis)
        & (df["WCS_Y_mm"] >= wcs_min)
        & (df["WCS_Y_mm"] <= wcs_max)
    )
    return df.loc[mask], len(df)


def legacy_heatmap(
    plate: str | int, bin_size_mm: float = BIN_SIZE_MM, parquet_dir: Path | None = None
) -> pd.DataFrame:
    """Per-slot Python binning plus global min-max normalisation.

    Lifted from `widgets_digital_twin.py:358-384`, calling the unmodified legacy
    `prepare_equal_bins_heatmap`.
    """
    from validation_data_access.legacy.Oxford.src.data_processing import prepare_equal_bins_heatmap

    parts = []
    for file_path in plate_files(plate, parquet_dir):
        binned = prepare_equal_bins_heatmap(pd.read_parquet(file_path), bin_size_mm=bin_size_mm)
        if not binned.empty:
            parts.append(binned)

    if not parts:
        return pd.DataFrame()

    df_heatmap = pd.concat(parts, ignore_index=True)

    vmin, vmax = df_heatmap["RMS_raw"].min(), df_heatmap["RMS_raw"].max()
    if vmax > vmin:
        df_heatmap["RMS_normalized_global"] = (df_heatmap["RMS_raw"] - vmin) / (vmax - vmin)
    else:
        df_heatmap["RMS_normalized_global"] = 0.0

    return df_heatmap


def legacy_amplitude_anchors(
    df_heatmap: pd.DataFrame, plate: str | int, parquet_dir: Path | None = None
) -> tuple[float, float]:
    """The two values that anchor the ends of the heatmap colour scale.

    Lifted from `widgets_digital_twin.py:394-473`: locate the bins with the globally
    smallest and largest `RMS_raw`, then take the min / max raw oscilloscope `Value`
    inside those two bins. The legacy version re-reads every Parquet file of the plate
    even though only two slots can contribute; that cost is part of the baseline.
    """
    if df_heatmap.empty:
        return 0.0, 0.0

    slot_col = "Nut" if "Nut" in df_heatmap.columns else "Nut_ID"

    bin_min = df_heatmap.loc[df_heatmap["RMS_raw"].idxmin()]
    bin_max = df_heatmap.loc[df_heatmap["RMS_raw"].idxmax()]

    slot_min, slot_max = int(bin_min[slot_col]), int(bin_max[slot_col])
    y_min_min, y_max_min = float(bin_min["Y_min"]), float(bin_min["Y_max"])
    y_min_max, y_max_max = float(bin_max["Y_min"]), float(bin_max["Y_max"])

    vals_min: list[float] = []
    vals_max: list[float] = []

    for file_path in plate_files(plate, parquet_dir):
        df = pd.read_parquet(file_path)
        slot_here = int(df[slot_col].iloc[0]) if slot_col in df.columns else None
        df_sig = df[(df["Axis"] == "X") & (df["DataOrigin"] == "Oscilloscope")]

        if slot_here == slot_min:
            mask = (df_sig["WCS_Y_mm"] >= y_min_min) & (df_sig["WCS_Y_mm"] <= y_max_min)
            vals_min.extend(df_sig.loc[mask, "Value"].values)

        if slot_here == slot_max:
            mask = (df_sig["WCS_Y_mm"] >= y_min_max) & (df_sig["WCS_Y_mm"] <= y_max_max)
            vals_max.extend(df_sig.loc[mask, "Value"].values)

    return (
        float(np.min(vals_min)) if vals_min else 0.0,
        float(np.max(vals_max)) if vals_max else 0.0,
    )


def legacy_summary(plate: str | int, parquet_dir: Path | None = None) -> pd.DataFrame:
    """Per-slot chatter summary, needed as `df_summary` by the heatmap figure.

    `analyze_platte` (`data_processing.py:89`) reads every Parquet file of the plate and
    runs `summarize_chatter_cases` on each.
    """
    from validation_data_access.legacy.Oxford.src.data_processing import (
        analyze_platte,
        summarize_chatter_cases,
    )

    directory = parquet_dir or LEGACY_PARQUET_DIR
    df_summary, _ = analyze_platte(plate, directory, summarize_chatter_cases, lambda _: None)
    return df_summary


# ===========================================================================
# 4. Compare
# ===========================================================================

def _normalise_keys(df: pd.DataFrame, join_keys: list[str], decimals: int) -> pd.DataFrame:
    """Make join keys comparable across the two stacks.

    Numeric keys become float: the same slot number arrives as `int64` from Parquet and
    as `float64` or `int32` from DuckDB, and pandas will not match those. Floats are then
    rounded, because `Y_min` comes out of `np.arange` on one side and `floor(y/bin)*bin`
    on the other.
    """
    out = df.copy()
    for key in join_keys:
        if key in out.columns and pd.api.types.is_numeric_dtype(out[key]):
            out[key] = pd.to_numeric(out[key], errors="coerce").astype("float64").round(decimals)
    return out


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    join_keys: list[str],
    numeric_cols: list[str],
    atol: float = 1e-9,
    rtol: float = 1e-9,
    key_decimals: int = 6,
    left_name: str = "legacy",
    right_name: str = "new",
) -> dict:
    """Outer-join two frames on `join_keys` and quantify the disagreement."""
    if left.empty and right.empty:
        return {
            "left_rows": 0, "right_rows": 0, "matched_rows": 0,
            f"only_{left_name}_rows": 0, f"only_{right_name}_rows": 0,
            "columns": {}, "max_abs_diff": 0.0, "rows_above_tolerance": 0,
            "mismatch_rows": pd.DataFrame(), "ok": True,
        }

    merged = _normalise_keys(left, join_keys, key_decimals).merge(
        _normalise_keys(right, join_keys, key_decimals),
        on=join_keys, how="outer", suffixes=(f"_{left_name}", f"_{right_name}"), indicator=True,
    )

    only_left = int((merged["_merge"] == "left_only").sum())
    only_right = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]

    columns: dict[str, dict] = {}
    above_tolerance = pd.Series(False, index=both.index)

    for col in numeric_cols:
        lcol, rcol = f"{col}_{left_name}", f"{col}_{right_name}"
        if lcol not in both.columns or rcol not in both.columns:
            columns[col] = {"present": False}
            continue

        a = pd.to_numeric(both[lcol], errors="coerce")
        b = pd.to_numeric(both[rcol], errors="coerce")
        diff = (a - b).abs()

        # NaN on both sides counts as agreement; NaN on one side does not.
        both_nan = a.isna() & b.isna()
        one_nan = a.isna() ^ b.isna()

        exceeds = (diff > (atol + rtol * b.abs())).fillna(False) | one_nan
        above_tolerance |= exceeds

        finite = diff[~both_nan].dropna()
        columns[col] = {
            "present": True,
            "max_abs_diff": float(finite.max()) if len(finite) else 0.0,
            "mean_abs_diff": float(finite.mean()) if len(finite) else 0.0,
            "rows_above_tolerance": int(exceeds.sum()),
            "nan_mismatches": int(one_nan.sum()),
        }

    present = [c for c, info in columns.items() if info.get("present")]
    rows_above = int(above_tolerance.sum())

    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "matched_rows": int((merged["_merge"] == "both").sum()),
        f"only_{left_name}_rows": only_left,
        f"only_{right_name}_rows": only_right,
        "columns": columns,
        "max_abs_diff": max((columns[c]["max_abs_diff"] for c in present), default=0.0),
        "rows_above_tolerance": rows_above,
        "mismatch_rows": both[above_tolerance].drop(columns="_merge"),
        "ok": only_left == 0 and only_right == 0 and rows_above == 0,
    }


def report_row(check: str, report: dict, *, detail: str = "") -> dict:
    """Flatten a report into one row of a results table."""
    return {
        "check": check,
        "passed": bool(report["ok"]),
        "rows_legacy": report["left_rows"],
        "rows_new": report["right_rows"],
        "matched": report["matched_rows"],
        "rows_above_tolerance": report["rows_above_tolerance"],
        "max_abs_diff": report["max_abs_diff"],
        "detail": detail,
    }


def _round_significant(values: np.ndarray, sig_digits: int) -> np.ndarray:
    """Round to a fixed number of significant digits, vectorised.

    Negative zero is folded onto positive zero: Parquet preserves the sign bit of a zero
    and DuckDB does not always return the same one. `-0.0 == 0.0` is true, but the two
    have different bit patterns and hash differently, which would otherwise show up as
    thousands of "differing" rows holding identical values.
    """
    out = np.asarray(values, dtype="float64").copy()
    finite = np.isfinite(out) & (out != 0)
    if finite.any():
        magnitude = np.floor(np.log10(np.abs(out[finite])))
        factor = np.power(10.0, sig_digits - 1 - magnitude)
        out[finite] = np.round(out[finite] * factor) / factor
    out[out == 0] = 0.0
    return out


def _canonical_column(series: pd.Series, *, sig_digits: int = 12) -> pd.Series:
    """Reduce one column to a form that does not depend on how it was stored."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ns]").astype("int64", errors="ignore")
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()
        return pd.Series(_round_significant(numeric, sig_digits), index=series.index)
    return series.astype("string").fillna("<NA>")


def _numeric_like(series: pd.Series) -> bool:
    """True if this column can be read as numeric without losing information.

    An all-missing column counts: that is how an empty integer column arrives from
    Parquet (`object` full of `None`) versus from DuckDB (`Int32` full of `pd.NA`).
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return True
    return bool(series.isna().all())


def _canonical_pair(left: pd.Series, right: pd.Series, *, sig_digits: int) -> tuple[pd.Series, pd.Series]:
    """Canonicalise two versions of the same column the same way.

    The branch must be chosen from both sides at once. Deciding per side would put an
    all-`None` `object` column into the text branch and its all-`NA` `Int32` counterpart
    into the numeric branch, and then every row would appear to differ.
    """
    if _numeric_like(left) and _numeric_like(right):
        return (
            pd.Series(_round_significant(pd.to_numeric(left, errors="coerce").to_numpy(), sig_digits), index=left.index),
            pd.Series(_round_significant(pd.to_numeric(right, errors="coerce").to_numpy(), sig_digits), index=right.index),
        )
    return _canonical_column(left, sig_digits=sig_digits), _canonical_column(right, sig_digits=sig_digits)


def frame_equality(left: pd.DataFrame, right: pd.DataFrame, *, columns: list[str], sig_digits: int = 12) -> dict:
    """Order-independent comparison of two frames that should hold the same records.

    A SQL result set has no inherent order, and sorting by a subset of columns leaves ties
    the two stacks break differently - which shows up as a difference that is not one. So
    both frames are reduced to canonical form, the rows are hashed, and the two multisets
    of row hashes are compared. Per-column counts localise any genuine difference.
    """
    canonical = {c: _canonical_pair(left[c], right[c], sig_digits=sig_digits) for c in columns}
    left_c = pd.DataFrame({c: pair[0] for c, pair in canonical.items()})
    right_c = pd.DataFrame({c: pair[1] for c, pair in canonical.items()})

    result = {
        "rows_legacy": int(len(left_c)),
        "rows_new": int(len(right_c)),
        "row_count_match": len(left_c) == len(right_c),
        "columns": {},
    }

    all_columns_ok = True
    for col in columns:
        aligned = left_c[col].value_counts(dropna=False).align(
            right_c[col].value_counts(dropna=False), fill_value=0
        )
        delta = (aligned[0] - aligned[1]).abs()
        differing = int((delta > 0).sum())
        result["columns"][col] = {"values_with_differing_counts": differing, "rows_affected": int(delta.sum())}
        all_columns_ok &= differing == 0

    # Whole-row multiset equality via a 64-bit hash per row: linear, and fast enough for
    # the million-row slots. A collision masking a genuine difference is negligible here.
    aligned_rows = pd.util.hash_pandas_object(left_c, index=False).value_counts().align(
        pd.util.hash_pandas_object(right_c, index=False).value_counts(), fill_value=0
    )
    row_delta = aligned_rows[0] - aligned_rows[1]

    result["rows_only_legacy"] = int(row_delta[row_delta > 0].sum())
    result["rows_only_new"] = int(-row_delta[row_delta < 0].sum())
    result["columns_ok"] = all_columns_ok
    result["ok"] = bool(result["row_count_match"] and not result["rows_only_legacy"] and not result["rows_only_new"])
    return result


def dtype_table(left: pd.DataFrame, right: pd.DataFrame, *, left_name="legacy", right_name="new") -> pd.DataFrame:
    """Side-by-side dtype and presence map for two frames."""
    return pd.DataFrame(
        [
            {
                "column": c,
                f"{left_name}_dtype": str(left[c].dtype) if c in left.columns else "-",
                f"{right_name}_dtype": str(right[c].dtype) if c in right.columns else "-",
                "in_both": c in left.columns and c in right.columns,
            }
            for c in sorted(set(left.columns) | set(right.columns))
        ]
    )


# ===========================================================================
# 5. Figures
# ===========================================================================

def _numeric_stats(values) -> dict:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return {"n": 0, "min": None, "max": None, "mean": None}
    return {
        "n": int(len(series)),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _trace_signature(trace) -> dict:
    signature = {"type": str(getattr(trace, "type", "")), "name": str(getattr(trace, "name", "") or "")}
    for axis in ("x", "y", "z"):
        source = getattr(trace, axis, None)
        signature[axis] = (
            {"n": 0, "min": None, "max": None, "mean": None}
            if source is None
            else _numeric_stats(np.asarray(source).ravel())
        )
    return signature


def figure_signature(fig: go.Figure) -> dict:
    """Reduce a figure to the numbers that decide what a reader sees."""
    signature = {
        "trace_count": len(fig.data),
        "shape_count": len(fig.layout.shapes or ()),
        "annotation_count": len(fig.layout.annotations or ()),
        "traces": [_trace_signature(t) for t in fig.data],
    }
    payload = json.dumps(signature, sort_keys=True, default=str)
    signature["digest"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return signature


def _pair_traces(left: dict, right: dict) -> tuple[list[tuple], list[str]]:
    """Pair traces by name where possible, falling back to position.

    Trace order follows the order rows arrived in, which the two stacks do not promise to
    agree on. Legend order is not what this validation is about.
    """
    left_names = [t["name"] for t in left["traces"]]
    right_names = [t["name"] for t in right["traces"]]

    nameable = (
        all(left_names) and all(right_names)
        and len(set(left_names)) == len(left_names)
        and len(set(right_names)) == len(right_names)
    )
    if not nameable:
        return list(zip(left["traces"], right["traces"])), []

    right_by_name = {t["name"]: t for t in right["traces"]}
    pairs, unmatched = [], []
    for trace in left["traces"]:
        counterpart = right_by_name.pop(trace["name"], None)
        if counterpart is None:
            unmatched.append(f"trace {trace['name']!r} present in legacy only")
        else:
            pairs.append((trace, counterpart))
    unmatched.extend(f"trace {name!r} present in new only" for name in right_by_name)
    return pairs, unmatched


def compare_signatures(left: dict, right: dict, *, atol: float = 1e-9) -> dict:
    """Structural then numeric comparison of two figure signatures."""
    differences: list[str] = []
    for key in ("trace_count", "shape_count", "annotation_count"):
        if left[key] != right[key]:
            differences.append(f"{key}: legacy={left[key]} new={right[key]}")

    pairs, unmatched = _pair_traces(left, right)
    differences.extend(unmatched)

    max_abs = 0.0
    stats_differing: set[str] = set()

    if left["trace_count"] == right["trace_count"]:
        for i, (lt, rt) in enumerate(pairs):
            if lt["type"] != rt["type"]:
                differences.append(f"trace {i} type: legacy={lt['type']} new={rt['type']}")
            for axis in ("x", "y", "z"):
                la, ra = lt[axis], rt[axis]
                if la["n"] != ra["n"]:
                    differences.append(f"trace {i} {axis}.n: legacy={la['n']} new={ra['n']}")
                    stats_differing.add(f"{axis}.n")
                    continue
                for stat in ("min", "max", "mean"):
                    lv, rv = la[stat], ra[stat]
                    if lv is None and rv is None:
                        continue
                    if lv is None or rv is None:
                        differences.append(f"trace {i} {axis}.{stat}: legacy={lv} new={rv}")
                        stats_differing.add(f"{axis}.{stat}")
                        continue
                    diff = abs(lv - rv)
                    max_abs = max(max_abs, diff)
                    if diff > atol:
                        differences.append(f"trace {i} {axis}.{stat}: legacy={lv:.6g} new={rv:.6g} (diff {diff:.3g})")
                        stats_differing.add(f"{axis}.{stat}")

    return {
        "max_abs_diff": max_abs,
        "differences": differences,
        "stats_differing": stats_differing,
        # A difference confined to the mean, with every count, minimum and maximum
        # intact, means the two figures drew different samples of the same data.
        "sampling_only": bool(stats_differing) and stats_differing <= {"x.mean", "y.mean"},
        "ok": not differences,
    }


_PATH_POINT_RE = re.compile(r"[ML]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def polyline_edges(fig: go.Figure, *, dash: str = "dot") -> set[tuple]:
    """Extract dotted line work as a set of undirected segments.

    The two visualizers encode the Qw iso-lines differently: the legacy one emits a `line`
    shape per segment, the new one coalesces contiguous segments into a single SVG `path`.
    Comparing shape counts would report a difference that does not exist on screen, so
    both encodings are reduced to the segments they draw. Coordinates are rounded to two
    decimals, the precision the new encoding writes its path strings at.
    """

    def point(x, y):
        return (round(float(x), 2), round(float(y), 2))

    edges: set[tuple] = set()
    for shape in fig.layout.shapes or ():
        if getattr(shape.line, "dash", None) != dash:
            continue
        if shape.type == "line":
            edges.add(tuple(sorted((point(shape.x0, shape.y0), point(shape.x1, shape.y1)))))
        elif shape.type == "path":
            points = [point(x, y) for x, y in _PATH_POINT_RE.findall(shape.path or "")]
            for a, b in zip(points, points[1:]):
                edges.add(tuple(sorted((a, b))))
    return edges


def compare_polylines(fig_legacy: go.Figure, fig_new: go.Figure, *, dash: str = "dot") -> dict:
    """Compare the dotted line work of two figures as geometry, not as shape objects."""
    left = polyline_edges(fig_legacy, dash=dash)
    right = polyline_edges(fig_new, dash=dash)
    return {
        "segments_legacy": len(left),
        "segments_new": len(right),
        "shapes_legacy": len(fig_legacy.layout.shapes or ()),
        "shapes_new": len(fig_new.layout.shapes or ()),
        "ok": left == right,
    }


def _label_font(size: int):
    """A real TrueType font if one can be found, otherwise PIL's bitmap default."""
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def save_comparison_png(
    fig_legacy: go.Figure,
    fig_new: go.Figure,
    path,
    *,
    title: str = "",
    width: int = 900,
    height: int = 760,
    scale: int = 2,
) -> dict:
    """Render both figures into a single labelled PNG, legacy left, new right.

    Rendering goes through kaleido, which fails on some plotly/kaleido combinations; the
    result reports whether it succeeded rather than raising, so a notebook run is not lost
    to an image export.
    """
    import io

    from PIL import Image, ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        panels = [
            (label, Image.open(io.BytesIO(fig.to_image(format="png", width=width, height=height, scale=scale))))
            for label, fig in (("Legacy (Parquet)", fig_legacy), ("New (DuckDB access layer)", fig_new))
        ]
    except Exception as exc:
        return {"png": None, "written": False, "error": f"{type(exc).__name__}: {exc}"}

    title_h = 52 * scale if title else 0
    label_h = 40 * scale
    pad = 12 * scale

    panel_w = max(image.width for _, image in panels)
    panel_h = max(image.height for _, image in panels)

    canvas = Image.new(
        "RGB",
        (panel_w * 2 + pad * 3, title_h + label_h + panel_h + pad * 2),
        # Matches the plotly_dark template the figures are rendered with.
        (17, 17, 17),
    )
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((pad, pad), title, fill=(238, 238, 238), font=_label_font(22 * scale))

    for index, (label, image) in enumerate(panels):
        x = pad + index * (panel_w + pad)
        draw.text((x, title_h + pad // 2), label, fill=(238, 238, 238), font=_label_font(17 * scale))
        canvas.paste(image, (x, title_h + label_h))

    canvas.save(path)
    return {"png": str(path), "written": True, "error": None}


# ===========================================================================
# 6. Measure
# ===========================================================================

def peak_rss_bytes() -> int:
    """Peak resident set size of this process. macOS reports bytes, Linux kilobytes."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss * 1024 if platform.system().lower() == "linux" else rss


# Each workload returns (rows_returned, rows_read). `rows_read` is how many rows had to be
# materialised in Python; on the SQL side that equals rows_returned, because filtering
# happened in the engine.


def _legacy_targeted_query(**p) -> tuple[int, int]:
    df, rows_read = legacy_targeted_query(
        p["plate"], p["slot"], data_origin=p["data_origin"], signals=p["signals"],
        wcs_min=p["wcs_min"], wcs_max=p["wcs_max"],
    )
    return len(df), rows_read


def _new_targeted_query(**p) -> tuple[int, int]:
    from src import provider

    df = provider.df(
        p["plate"], p["slot"], data_origin=p["data_origin"], signals=p["signals"],
        wcs_min=p["wcs_min"], wcs_max=p["wcs_max"],
    )
    return len(df), len(df)


def _legacy_targeted_query_axis(**p) -> tuple[int, int]:
    df, rows_read = legacy_targeted_query_axis(
        p["plate"], p["slot"], data_origin=p["data_origin"], axis=p["axis"],
        wcs_min=p["wcs_min"], wcs_max=p["wcs_max"],
    )
    return len(df), rows_read


def _new_targeted_query_axis(**p) -> tuple[int, int]:
    from src import provider

    df = provider.query_df(
        f"""
        SELECT * FROM {provider.table()}
        WHERE Platte = ? AND Nut = ? AND DataOrigin = ? AND Axis = ?
          AND WCS_Y_mm >= ? AND WCS_Y_mm <= ?
        """,
        [p["plate"], p["slot"], p["data_origin"], p["axis"], p["wcs_min"], p["wcs_max"]],
    )
    return len(df), len(df)


def _legacy_plot_df(**p) -> tuple[int, int]:
    """Fetch one slot's data for the axis-wise plots: read the file, drop other origins."""
    df = pd.read_parquet(slot_file(p["plate"], p["slot"]))
    return len(df[df["DataOrigin"] == p["data_origin"]]), len(df)


def _new_plot_df(**p) -> tuple[int, int]:
    from src import provider

    df = provider.axiswise_plot_df(p["plate"], p["slot"], data_origin=p["data_origin"])
    return len(df), len(df)


def _new_plot_df_pushed_down(**p) -> tuple[int, int]:
    from src import provider

    df = provider.axiswise_plot_df(
        p["plate"], p["slot"], data_origin=p["data_origin"],
        max_points_per_signal=p.get("max_points_per_signal", MAX_DISPLAY_POINTS),
    )
    # Reduction happened in the engine, so rows materialised in Python == rows returned,
    # same convention as every other "new_*" workload in this table.
    return len(df), len(df)


def _legacy_plate_rows(plate) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in plate_files(plate))


def _legacy_heatmap(**p) -> tuple[int, int]:
    df = legacy_heatmap(p["plate"], p["bin_size_mm"])
    return len(df), _legacy_plate_rows(p["plate"])


def _new_heatmap(**p) -> tuple[int, int]:
    from src.data_processing import prepare_equal_bins_heatmap_sql

    df = prepare_equal_bins_heatmap_sql(
        p["plate"], bin_size_mm=p["bin_size_mm"],
        target_signal=p["target_signal"], target_origin=p["target_origin"],
        compute_normalized_global=True,
    )
    return len(df), len(df)


WORKLOADS: dict[str, Callable[..., tuple[int, int]]] = {
    "legacy_targeted_query": _legacy_targeted_query,
    "new_targeted_query": _new_targeted_query,
    "legacy_targeted_query_axis": _legacy_targeted_query_axis,
    "new_targeted_query_axis": _new_targeted_query_axis,
    "legacy_plot_df": _legacy_plot_df,
    "new_plot_df": _new_plot_df,
    "new_plot_df_pushed_down": _new_plot_df_pushed_down,
    "legacy_heatmap": _legacy_heatmap,
    "new_heatmap": _new_heatmap,
}


def _child(queue, workload: str, params: dict, repeats: int) -> None:
    try:
        if workload.startswith("new_"):
            init_new_stack()
        else:
            prepare_environment()

        fn = WORKLOADS[workload]

        # Warm-up, excluded from the timings: the first call pays for imports and for
        # DuckDB reading its catalogue.
        rows_returned, rows_read = fn(**params)

        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            rows_returned, rows_read = fn(**params)
            durations.append(time.perf_counter() - start)

        queue.put({
            "workload": workload, "ok": True, "durations": durations,
            "rows_returned": rows_returned, "rows_read": rows_read,
            "peak_rss_bytes": peak_rss_bytes(),
        })
    except Exception as exc:
        import traceback

        queue.put({
            "workload": workload, "ok": False,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })


def measure(workload: str, params: dict, *, repeats: int = 5, timeout: float = 1800.0) -> dict:
    """Run one workload in a fresh process and return timing plus peak RSS."""
    if workload not in WORKLOADS:
        raise KeyError(f"Unknown workload {workload!r}. Known: {sorted(WORKLOADS)}")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_child, args=(queue, workload, params, repeats))
    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception:
        process.terminate()
        process.join()
        return {"workload": workload, "ok": False, "error": f"timed out after {timeout}s"}

    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()

    if not result.get("ok"):
        return result

    durations = sorted(result["durations"])
    result["median_seconds"] = statistics.median(durations)
    result["iqr_seconds"] = (
        statistics.median(durations[len(durations) // 2:])
        - statistics.median(durations[: (len(durations) + 1) // 2])
        if len(durations) > 1
        else 0.0
    )
    result["repeats"] = len(durations)
    result["params"] = params
    return result


def measure_pair(legacy_workload: str, new_workload: str, params: dict, *, label: str, repeats: int = 5) -> dict:
    """Measure the legacy and the new implementation of the same task."""
    row: dict = {"task": label}

    for side, result in (
        ("legacy", measure(legacy_workload, params, repeats=repeats)),
        ("new", measure(new_workload, params, repeats=repeats)),
    ):
        if not result.get("ok"):
            row[f"{side}_error"] = result.get("error")
            continue
        row[f"{side}_seconds"] = result["median_seconds"]
        row[f"{side}_iqr"] = result["iqr_seconds"]
        row[f"{side}_peak_rss_mb"] = result["peak_rss_bytes"] / (1024 * 1024)
        row[f"{side}_rows_returned"] = result["rows_returned"]
        row[f"{side}_rows_read"] = result["rows_read"]

    if row.get("new_seconds"):
        row["speedup"] = row["legacy_seconds"] / row["new_seconds"]
    if row.get("new_peak_rss_mb"):
        row["memory_ratio"] = row["legacy_peak_rss_mb"] / row["new_peak_rss_mb"]
    return row


def results_frame(rows: list[dict]) -> pd.DataFrame:
    """Tidy rows from `measure_pair` into a display-ready table."""
    df = pd.DataFrame(rows)
    preferred = [
        "task", "legacy_seconds", "new_seconds", "speedup",
        "legacy_peak_rss_mb", "new_peak_rss_mb", "memory_ratio",
        "legacy_rows_read", "legacy_rows_returned", "new_rows_returned",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]
