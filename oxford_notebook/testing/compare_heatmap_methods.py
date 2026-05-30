"""Compare heatmap implementations (old Python vs new SQL).

Usage (from repo root):
    python oxford_notebook/tools/compare_heatmap_methods.py --plate 22 --bin-size 10

This script:
- Initializes the DuckDB provider
- Computes heatmap data via:
    1) old path: provider.df(...) per slot + prepare_equal_bins_heatmap
    2) SQL path: prepare_equal_bins_heatmap_sql
- Compares numeric outputs and min/max amplitude mapping
- Writes CSVs + a JSON report into oxford_notebook/results/heatmap_comparison/

Notes
-----
The two implementations can legitimately differ at bin boundaries (e.g., samples
exactly at the slot max Y). The report surfaces such differences explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CompareReport:
    plate: str
    bin_size_mm: float
    target_signal: str
    target_origin: str
    old_rows: int
    sql_rows: int
    old_slots: int
    sql_slots: int
    matched_rows: int
    only_old_rows: int
    only_sql_rows: int
    rms_abs_max: float | None
    rms_abs_mean: float | None
    y_max_abs_max: float | None
    true_min_old: float
    true_max_old: float
    true_min_sql: float
    true_max_sql: float
    seconds_old: float
    seconds_sql: float


def _repo_paths() -> tuple[Path, Path]:
    # .../oxford_notebook/tools/compare_heatmap_methods.py
    tools_dir = Path(__file__).resolve().parent
    oxford_root = tools_dir.parent
    repo_root = oxford_root.parent
    return repo_root, oxford_root


def _init_provider(*, db_path: Path | None, table: str | None) -> None:
    _, oxford_root = _repo_paths()
    sys.path.insert(0, str(oxford_root))

    from src import provider  # noqa: WPS433 (runtime import by design)

    kwargs = {"project_root": oxford_root, "read_only": True}
    if db_path is not None:
        kwargs["db_path"] = Path(db_path)
    if table is not None:
        kwargs["table_name"] = table

    provider.init(**kwargs)


def _compute_old_heatmap(
    plate: str,
    *,
    bin_size_mm: float,
    target_signal: str,
    target_origin: str,
) -> pd.DataFrame:
    from src import provider  # noqa: WPS433
    from src.data_processing import prepare_equal_bins_heatmap  # noqa: WPS433

    slots = provider.slots(plate)
    frames: list[pd.DataFrame] = []

    for slot in slots:
        df_slot = provider.df(
            plate,
            slot,
            fields=["Nut", "WCS_Y_mm", "Value", "Axis", "DataOrigin"],
        )
        if df_slot is None or df_slot.empty:
            continue

        df_heat = prepare_equal_bins_heatmap(
            df_slot,
            bin_size_mm=bin_size_mm,
            target_signal=target_signal,
            target_origin=target_origin,
        )
        if not df_heat.empty:
            frames.append(df_heat)

    if not frames:
        return pd.DataFrame(columns=["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"])

    out = pd.concat(frames, ignore_index=True)
    # Normalize expected column order
    wanted = ["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]
    return out[wanted]


def _compute_sql_heatmap(
    plate: str,
    *,
    bin_size_mm: float,
    target_signal: str,
    target_origin: str,
) -> pd.DataFrame:
    from src.data_processing import prepare_equal_bins_heatmap_sql  # noqa: WPS433

    df = prepare_equal_bins_heatmap_sql(
        plate,
        bin_size_mm=bin_size_mm,
        target_signal=target_signal,
        target_origin=target_origin,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"])

    wanted = ["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]
    return df[wanted]


def _true_min_max_old(df_heatmap: pd.DataFrame, plate: str) -> tuple[float, float]:
    """Reproduce main-branch widget semantics for true amplitude mapping."""
    if df_heatmap.empty:
        return 0.0, 0.0

    from src import provider  # noqa: WPS433

    # Find the bins with globally smallest and largest RMS_raw
    idx_rms_min = df_heatmap["RMS_raw"].idxmin()
    idx_rms_max = df_heatmap["RMS_raw"].idxmax()

    bin_min = df_heatmap.loc[idx_rms_min]
    bin_max = df_heatmap.loc[idx_rms_max]

    slot_min = float(bin_min["Nut"])
    slot_max = float(bin_max["Nut"])

    y_min_min = float(bin_min["Y_min"])
    y_max_min = float(bin_min["Y_max"])
    y_min_max = float(bin_max["Y_min"])
    y_max_max = float(bin_max["Y_max"])

    slots = provider.slots(plate)
    vals_min: list[float] = []
    vals_max: list[float] = []

    for slot in slots:
        df_slot = provider.df(
            plate,
            slot,
            fields=["Nut", "WCS_Y_mm", "Value", "Axis", "DataOrigin"],
        )
        if df_slot is None or df_slot.empty:
            continue

        df_sig = df_slot[(df_slot["Axis"] == "X") & (df_slot["DataOrigin"] == "Oscilloscope")]
        if df_sig.empty:
            continue

        if float(slot) == float(slot_min):
            mask_min = (df_sig["WCS_Y_mm"] >= y_min_min) & (df_sig["WCS_Y_mm"] <= y_max_min)
            vals_min.extend(df_sig.loc[mask_min, "Value"].astype(float).to_list())

        if float(slot) == float(slot_max):
            mask_max = (df_sig["WCS_Y_mm"] >= y_min_max) & (df_sig["WCS_Y_mm"] <= y_max_max)
            vals_max.extend(df_sig.loc[mask_max, "Value"].astype(float).to_list())

    amp_min = float(np.min(vals_min)) if vals_min else 0.0
    amp_max = float(np.max(vals_max)) if vals_max else 0.0
    return amp_min, amp_max


def _true_min_max_sql(df_heatmap_sql: pd.DataFrame, plate: str) -> tuple[float, float]:
    if df_heatmap_sql.empty:
        return 0.0, 0.0

    from src.data_processing import get_min_max_amplitudes_sql  # noqa: WPS433

    return get_min_max_amplitudes_sql(df_heatmap_sql, plate)


def _prep_for_join(df: pd.DataFrame, *, bin_size_mm: float) -> pd.DataFrame:
    out = df.copy()

    # Cast to float for stable merging (DuckDB uses DOUBLE for Nut)
    for col in ["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Key on (Nut, y_min_bin_index) to be robust against float repr noise
    out["y_bin_idx"] = np.round(out["Y_min"] / float(bin_size_mm)).astype("Int64")
    out["Nut"] = out["Nut"].astype(float)

    # Deduplicate defensively
    out = out.dropna(subset=["Nut", "y_bin_idx"]).copy()
    out = out.sort_values(["Nut", "Y_min"], kind="stable")
    out = out.drop_duplicates(subset=["Nut", "y_bin_idx"], keep="first")

    return out


def compare(
    plate: str,
    *,
    bin_size_mm: float,
    target_signal: str,
    target_origin: str,
    db_path: Path | None = None,
    table: str | None = None,
    out_dir: Path | None = None,
) -> CompareReport:
    _init_provider(db_path=db_path, table=table)

    t0 = time.perf_counter()
    df_old = _compute_old_heatmap(
        plate,
        bin_size_mm=bin_size_mm,
        target_signal=target_signal,
        target_origin=target_origin,
    )
    seconds_old = time.perf_counter() - t0

    t1 = time.perf_counter()
    df_sql = _compute_sql_heatmap(
        plate,
        bin_size_mm=bin_size_mm,
        target_signal=target_signal,
        target_origin=target_origin,
    )
    seconds_sql = time.perf_counter() - t1

    df_old_p = _prep_for_join(df_old, bin_size_mm=bin_size_mm)
    df_sql_p = _prep_for_join(df_sql, bin_size_mm=bin_size_mm)

    merged = df_old_p.merge(
        df_sql_p,
        on=["Nut", "y_bin_idx"],
        how="outer",
        suffixes=("_old", "_sql"),
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"].copy()
    only_old = merged[merged["_merge"] == "left_only"].copy()
    only_sql = merged[merged["_merge"] == "right_only"].copy()

    if not matched.empty:
        matched["rms_abs_diff"] = (matched["RMS_raw_old"] - matched["RMS_raw_sql"]).abs()
        matched["y_max_abs_diff"] = (matched["Y_max_old"] - matched["Y_max_sql"]).abs()

        rms_abs_max = float(matched["rms_abs_diff"].max())
        rms_abs_mean = float(matched["rms_abs_diff"].mean())
        y_max_abs_max = float(matched["y_max_abs_diff"].max())
    else:
        rms_abs_max = None
        rms_abs_mean = None
        y_max_abs_max = None

    true_min_old, true_max_old = _true_min_max_old(df_old, plate)
    true_min_sql, true_max_sql = _true_min_max_sql(df_sql, plate)

    report = CompareReport(
        plate=str(plate),
        bin_size_mm=float(bin_size_mm),
        target_signal=str(target_signal),
        target_origin=str(target_origin),
        old_rows=int(len(df_old)),
        sql_rows=int(len(df_sql)),
        old_slots=int(df_old["Nut"].nunique()) if not df_old.empty else 0,
        sql_slots=int(df_sql["Nut"].nunique()) if not df_sql.empty else 0,
        matched_rows=int(len(matched)),
        only_old_rows=int(len(only_old)),
        only_sql_rows=int(len(only_sql)),
        rms_abs_max=rms_abs_max,
        rms_abs_mean=rms_abs_mean,
        y_max_abs_max=y_max_abs_max,
        true_min_old=float(true_min_old),
        true_max_old=float(true_max_old),
        true_min_sql=float(true_min_sql),
        true_max_sql=float(true_max_sql),
        seconds_old=float(seconds_old),
        seconds_sql=float(seconds_sql),
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_old.to_csv(out_dir / "heatmap_old.csv", index=False)
        df_sql.to_csv(out_dir / "heatmap_sql.csv", index=False)

        # Save a focused diff table for quick inspection
        diff_cols = [
            "Nut",
            "y_bin_idx",
            "Y_min_old",
            "Y_max_old",
            "RMS_raw_old",
            "Y_min_sql",
            "Y_max_sql",
            "RMS_raw_sql",
            "_merge",
        ]
        if "rms_abs_diff" in matched.columns:
            diff_cols.append("rms_abs_diff")
        if "y_max_abs_diff" in matched.columns:
            diff_cols.append("y_max_abs_diff")

        merged_view = merged.copy()
        for c in diff_cols:
            if c not in merged_view.columns:
                merged_view[c] = np.nan

        merged_view[diff_cols].to_csv(out_dir / "heatmap_diff.csv", index=False)

        with (out_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plate", required=True, help="Plate identifier (as in Platte column)")
    parser.add_argument("--bin-size", type=float, default=10.0, help="Bin size in mm")
    parser.add_argument("--target-signal", default="X", help="Axis filter (default: X)")
    parser.add_argument("--target-origin", default="Oscilloscope", help="DataOrigin filter")
    parser.add_argument("--db-path", type=str, default=None, help="Optional path to DB.duckdb")
    parser.add_argument("--table", type=str, default=None, help="Optional DuckDB table name")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory. Default: oxford_notebook/results/heatmap_comparison/<timestamp>/",
    )

    args = parser.parse_args()

    repo_root, oxford_root = _repo_paths()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir: Path
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = oxford_root / "results" / "heatmap_comparison" / timestamp

    report = compare(
        args.plate,
        bin_size_mm=args.bin_size,
        target_signal=args.target_signal,
        target_origin=args.target_origin,
        db_path=Path(args.db_path) if args.db_path else None,
        table=args.table,
        out_dir=out_dir,
    )

    print("\n=== Heatmap comparison report ===")
    print(f"Plate: {report.plate} | bin_size_mm={report.bin_size_mm}")
    print(f"Rows (old/sql): {report.old_rows} / {report.sql_rows}")
    print(f"Slots (old/sql): {report.old_slots} / {report.sql_slots}")
    print(f"Matched bins: {report.matched_rows}")
    print(f"Only old bins: {report.only_old_rows} | Only SQL bins: {report.only_sql_rows}")

    if report.rms_abs_max is not None:
        print(f"RMS abs diff max / mean: {report.rms_abs_max:.6g} / {report.rms_abs_mean:.6g}")
    else:
        print("RMS diffs: n/a (no matched bins)")

    if report.y_max_abs_max is not None:
        print(f"Y_max abs diff max: {report.y_max_abs_max:.6g}")

    print(
        "True amplitude mapping (min/max): "
        f"old=({report.true_min_old:.6g}, {report.true_max_old:.6g}) | "
        f"sql=({report.true_min_sql:.6g}, {report.true_max_sql:.6g})"
    )

    speedup = (report.seconds_old / report.seconds_sql) if report.seconds_sql > 0 else float("inf")
    print(f"Runtime seconds (old/sql): {report.seconds_old:.3f} / {report.seconds_sql:.3f}  (x{speedup:.2f})")

    print(f"\nWrote artifacts to: {out_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
