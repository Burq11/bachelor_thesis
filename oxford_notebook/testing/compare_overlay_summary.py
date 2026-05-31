"""Compare old vs DuckDB-first overlay summary (chatter + metadata).

This is a thesis-support utility: the heatmap visualization itself must not
change. We therefore verify that the optimized SQL summary yields the same
per-slot overlay inputs as the legacy Python path that loads raw per-slot frames.

Usage
-----
conda run -n chatterdetect python oxford_notebook/testing/compare_overlay_summary.py --plate 26
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


# Allow running as `python testing/compare_overlay_summary.py`.
OXFORD_ROOT = Path(__file__).resolve().parents[1]
if str(OXFORD_ROOT) not in sys.path:
    sys.path.insert(0, str(OXFORD_ROOT))


def _max_rss() -> int:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    import platform
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system().lower() == "linux":
        return rss * 1024
    return rss


def _assert_close(a: pd.Series, b: pd.Series, *, col: str, atol: float, rtol: float) -> None:
    import numpy as np

    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    diff = (a - b).abs()

    ok = np.isclose(a, b, rtol=rtol, atol=atol) | (a.isna() & b.isna())
    bad = (~ok) & ~(a.isna() & b.isna())

    if bad.any():
        max_abs = float(diff.max(skipna=True))
        raise AssertionError(f"Column {col}: {int(bad.sum())} mismatches (max_abs={max_abs})")


def _compare_one_plate(plate: str, *, atol: float, rtol: float) -> None:
    from src import provider
    from src.data_processing import analyze_platte, summarize_chatter_cases, summarize_chatter_cases_sql

    provider.init(project_root=OXFORD_ROOT, read_only=True)

    t0 = time.perf_counter()
    rss0 = _max_rss()
    df_old, _ = analyze_platte(str(plate), summarize_chatter_cases, lambda x: None)
    t_old = time.perf_counter() - t0
    rss_old = _max_rss() - rss0

    t1 = time.perf_counter()
    rss1 = _max_rss()
    df_new = summarize_chatter_cases_sql(str(plate))
    t_new = time.perf_counter() - t1
    rss_new = _max_rss() - rss1

    keys = ["Nut_ID", "Chatter"]
    merged = df_old.merge(df_new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)

    if (merged["_merge"] != "both").any():
        counts = merged["_merge"].value_counts().to_dict()
        raise AssertionError(f"Row mismatch after merge on {keys}: {counts}")

    for col in ["Y_max", "Drehzahl", "Werkzeugradius", "X_Position_Nut"]:
        _assert_close(
            merged[f"{col}_old"],
            merged[f"{col}_new"],
            col=col,
            atol=atol,
            rtol=rtol,
        )

    # Werkzeug (string) should match; treat NaN==NaN as OK
    tool_old = merged["Werkzeug_old"].fillna("<NA>")
    tool_new = merged["Werkzeug_new"].fillna("<NA>")
    if not (tool_old == tool_new).all():
        raise AssertionError("Werkzeug string mismatch")

    print(
        f"OK plate={plate} rows={len(df_new)} "
        f"time_old={t_old:.3f}s time_new={t_new:.3f}s "
        f"rss_old~={rss_old/1e6:.1f}MB rss_new~={rss_new/1e6:.1f}MB"
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plate", required=True)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args(list(argv) if argv is not None else None)

    _compare_one_plate(str(args.plate), atol=float(args.atol), rtol=float(args.rtol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
