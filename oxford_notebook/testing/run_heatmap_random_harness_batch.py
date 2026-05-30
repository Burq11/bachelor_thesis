"""Batch runner for heatmap parity via branch_harness.

Runs the existing branch harness multiple times for different random plate samples
(seeded) and fails fast if outputs diverge beyond tolerance.

Intended usage (from repo root):
  conda run -n chatterdetect python oxford_notebook/testing/run_heatmap_random_harness_batch.py \
    --plate-counts 5,10,20 \
    --seeds 0,1,7,42 \
    --bin-size-mm 10 \
    --target-signal X \
    --target-origin Oscilloscope
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    # .../oxford_notebook/testing/run_heatmap_random_harness_batch.py
    return Path(__file__).resolve().parents[2]


def _oxford_root(repo_root: Path) -> Path:
    return repo_root / "oxford_notebook"


def _parse_csv_ints(s: str) -> list[int]:
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    return [int(x) for x in items]


def _run_one(
    *,
    repo_root: Path,
    out_dir: Path,
    plate_count: int,
    seed: int,
    bin_size_mm: float,
    target_signal: str,
    target_origin: str,
    atol: float,
    rtol: float,
    db_path: str | None,
    table: str | None,
) -> tuple[bool, Path]:
    harness = repo_root / "oxford_notebook" / "testing" / "branch_harness.py"
    if not harness.exists():
        raise FileNotFoundError(f"Missing harness script: {harness}")

    cmd: list[str] = [
        sys.executable,
        str(harness),
        "--call-main",
        "harness_cases:heatmap_old_random_plates",
        "--call-feature",
        "harness_cases:heatmap_sql_random_plates",
        "--args",
        "[]",
        "--kwargs",
        json.dumps(
            {
                "plate_count": int(plate_count),
                "seed": int(seed),
                "bin_size_mm": float(bin_size_mm),
                "target_signal": str(target_signal),
                "target_origin": str(target_origin),
            }
        ),
        "--derive-y-bin-idx",
        str(int(round(float(bin_size_mm)))),
        "--join-keys",
        "Platte,Nut,y_bin_idx",
        "--atol",
        str(float(atol)),
        "--rtol",
        str(float(rtol)),
        "--out",
        str(out_dir),
    ]

    if db_path:
        cmd.extend(["--db-path", str(db_path)])
    if table:
        cmd.extend(["--table", str(table)])

    p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    # branch_harness.py typically returns 0 even if diffs exist; we rely on compare.json.
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return False, out_dir

    compare_json = out_dir / "compare.json"
    if not compare_json.exists():
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise FileNotFoundError(f"Harness did not produce compare.json at: {compare_json}")

    report = json.loads(compare_json.read_text(encoding="utf-8"))

    ok = True
    if int(report.get("only_main_rows", 0)) != 0:
        ok = False
    if int(report.get("only_feature_rows", 0)) != 0:
        ok = False

    above = report.get("above_tolerance_counts", {}) or {}
    for _, count in above.items():
        if int(count) != 0:
            ok = False
            break

    return ok, out_dir


def _fmt_mb(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "n/a"
    return f"{(float(n_bytes) / (1024 * 1024)):.1f}MB"


def _extract_perf_strings(compare_json: Path) -> tuple[str, str]:
    report = json.loads(compare_json.read_text(encoding="utf-8"))

    speedup = report.get("speedup_main_over_feature")
    speed_s = f"speedup=x{float(speedup):.2f}" if speedup is not None else "speedup=n/a"

    main_rss = report.get("main_max_rss_bytes")
    feat_rss = report.get("feature_max_rss_bytes")
    ratio = report.get("max_rss_ratio_main_over_feature")

    rss_s = f"rss(main/feat)={_fmt_mb(main_rss)}/{_fmt_mb(feat_rss)}"
    rss_s += f" (x{float(ratio):.2f})" if ratio is not None else " (x?)"
    return speed_s, rss_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run branch harness heatmap parity checks.")

    parser.add_argument("--plate-counts", default="10", help="Comma-separated plate counts (e.g. 5,10,20)")
    parser.add_argument("--seeds", default="0", help="Comma-separated random seeds (e.g. 0,1,7,42)")

    parser.add_argument("--bin-size-mm", type=float, default=10.0)
    parser.add_argument("--target-signal", default="X")
    parser.add_argument("--target-origin", default="Oscilloscope")

    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-6)

    parser.add_argument("--db-path", default=None)
    parser.add_argument("--table", default=None)

    parser.add_argument(
        "--out-root",
        default=None,
        help="Root output directory. Default: oxford_notebook/results/branch_harness/batch_<timestamp>/",
    )

    args = parser.parse_args()

    plate_counts = _parse_csv_ints(args.plate_counts)
    seeds = _parse_csv_ints(args.seeds)

    if not plate_counts:
        raise ValueError("--plate-counts must contain at least one integer")
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    repo_root = _repo_root()
    oxford_root = _oxford_root(repo_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (oxford_root / "results" / "branch_harness" / f"batch_{timestamp}")

    total = 0
    for plate_count in plate_counts:
        for seed in seeds:
            total += 1
            out_dir = out_root / f"plates_{plate_count}" / f"seed_{seed}"
            ok, path = _run_one(
                repo_root=repo_root,
                out_dir=out_dir,
                plate_count=plate_count,
                seed=seed,
                bin_size_mm=args.bin_size_mm,
                target_signal=args.target_signal,
                target_origin=args.target_origin,
                atol=args.atol,
                rtol=args.rtol,
                db_path=args.db_path,
                table=args.table,
            )

            compare_json = path / "compare.json"
            speed_s, rss_s = _extract_perf_strings(compare_json)

            if ok:
                print(f"OK  plates={plate_count} seed={seed} {speed_s} {rss_s}")
            else:
                print(f"FAIL plates={plate_count} seed={seed} {speed_s} {rss_s}")
                return 2

    print(f"All OK ({total} runs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
