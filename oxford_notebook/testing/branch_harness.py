"""Branch comparison harness for analysis functions.

Goal
----
Treat `main` as the source-of-truth for analysis. When you optimize code on a
feature branch, use this harness to run the *same* function on both branches,
capture artifacts, and compare outputs numerically (plus timings).

Why a harness?
-------------
Importing two branches in one Python process is error-prone (module name clashes
and `sys.modules` caching). This harness executes each branch in a *separate*
Python subprocess against a dedicated git worktree, then compares artifacts.

Typical usage
-------------
Run a function (module:function) on `main` and on the current working tree:

  conda run -n chatterdetect python oxford_notebook/testing/branch_harness.py \
    --call src.data_processing:prepare_equal_bins_heatmap_sql \
    --args '["22"]' \
    --kwargs '{"bin_size_mm": 10, "target_signal": "X", "target_origin": "Oscilloscope"}' \
    --derive-y-bin-idx 10 \
    --join-keys 'Nut,y_bin_idx'

Notes
-----
- Artifacts are written into `oxford_notebook/results/branch_harness/...`.
- By default, the feature side is your current working tree. Use --feature-branch
  if you want to compare against a different branch via worktree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunMeta:
    branch: str
    commit: str
    call: str
    seconds: float
    rows: int
    cols: int
    columns: list[str]
    dtypes: dict[str, str]


@dataclass
class CompareMeta:
    call: str
    main_branch: str
    feature_branch: str
    join_keys: list[str]
    derive_y_bin_idx: float | None
    main_rows: int
    feature_rows: int
    matched_rows: int
    only_main_rows: int
    only_feature_rows: int
    numeric_columns_compared: list[str]
    max_abs_diff: dict[str, float]
    mean_abs_diff: dict[str, float]
    above_tolerance_counts: dict[str, int]
    atol: float
    rtol: float


def _repo_root() -> Path:
    # .../oxford_notebook/testing/branch_harness.py
    return Path(__file__).resolve().parents[2]


def _oxford_root(repo_root: Path) -> Path:
    return repo_root / "oxford_notebook"


def _default_duckdb_path(repo_root: Path) -> Path | None:
    """Find a DuckDB file in the *current* working tree.

    Worktrees for other branches usually won't contain the DB because data is
    gitignored. For branch-to-branch comparisons we want both sides to run
    against the same DB snapshot.
    """
    data_dir = _oxford_root(repo_root) / "data"
    if not data_dir.exists():
        return None

    candidates = list(data_dir.glob("*.duckdb"))
    if not candidates:
        return None

    # Newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _run_git(repo_root: Path, args: list[str]) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def _current_branch(repo_root: Path) -> str:
    out = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    return out


def _current_commit(repo_root: Path) -> str:
    return _run_git(repo_root, ["rev-parse", "HEAD"]).strip()


def _list_worktrees(repo_root: Path) -> list[dict[str, str]]:
    out = _run_git(repo_root, ["worktree", "list", "--porcelain"])
    blocks = out.strip().split("\n\n") if out.strip() else []
    worktrees: list[dict[str, str]] = []
    for block in blocks:
        entry: dict[str, str] = {}
        for line in block.splitlines():
            if not line.strip():
                continue
            key, _, val = line.partition(" ")
            entry[key] = val.strip()
        if entry:
            worktrees.append(entry)
    return worktrees


def _ensure_worktree(repo_root: Path, branch: str, path: Path) -> Path:
    path = path.resolve()
    existing = _list_worktrees(repo_root)
    for wt in existing:
        if wt.get("worktree") and Path(wt["worktree"]).resolve() == path:
            # Path exists as a worktree already
            return path

    # If directory exists but isn't a worktree, refuse (safety)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(repo_root, ["worktree", "add", str(path), branch])
    return path


def _parse_call(call: str) -> tuple[str, str]:
    if ":" not in call:
        raise ValueError("--call must be in form 'module:function'")
    mod, fn = call.split(":", 1)
    if not mod or not fn:
        raise ValueError("--call must be in form 'module:function'")
    return mod, fn


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _invoke_branch_subprocess(
    *,
    worktree_repo_root: Path,
    branch_label: str,
    call: str,
    harness_support_path: Path,
    args_json: str,
    kwargs_json: str,
    init_provider: bool,
    db_path: str | None,
    table: str | None,
    artifact_dir: Path,
) -> RunMeta:
    """Run target callable inside a subprocess with sys.path set to this worktree."""

    mod, fn = _parse_call(call)

    inline = r"""
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

worktree_repo_root = Path(sys.argv[1]).resolve()
branch_label = sys.argv[2]
call = sys.argv[3]
mod_name = sys.argv[4]
fn_name = sys.argv[5]
harness_support_path = Path(sys.argv[6]).resolve()
args_json = sys.argv[7]
kwargs_json = sys.argv[8]
init_provider = sys.argv[9] == '1'
db_path = sys.argv[10] if sys.argv[10] != 'NONE' else None
table = sys.argv[11] if sys.argv[11] != 'NONE' else None
artifact_dir = Path(sys.argv[12]).resolve()

oxford_root = worktree_repo_root / 'oxford_notebook'
sys.path.insert(0, str(oxford_root))
sys.path.insert(0, str(harness_support_path))

if init_provider:
    from src import provider
    kwargs = {'project_root': oxford_root, 'read_only': True}
    if db_path is not None:
        kwargs['db_path'] = Path(db_path)
    if table is not None:
        kwargs['table_name'] = table
    provider.init(**kwargs)

args = json.loads(args_json) if args_json else []
kwargs = json.loads(kwargs_json) if kwargs_json else {}

module = importlib.import_module(mod_name)
fn = getattr(module, fn_name)

start = time.perf_counter()
result = fn(*args, **kwargs)
seconds = time.perf_counter() - start

artifact_dir.mkdir(parents=True, exist_ok=True)

meta = {
    'branch': branch_label,
    'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=str(worktree_repo_root)).decode().strip(),
    'call': call,
    'seconds': float(seconds),
}

# Persist result (DataFrame-first; otherwise JSON-ish)
if isinstance(result, pd.DataFrame):
    df = result
    df.to_csv(artifact_dir / 'data.csv', index=False)
    # Optional parquet (if engine available)
    try:
        df.to_parquet(artifact_dir / 'data.parquet', index=False)
        meta['parquet'] = True
    except Exception:
        meta['parquet'] = False

    meta.update({
        'rows': int(len(df)),
        'cols': int(df.shape[1]),
        'columns': list(df.columns.astype(str)),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
        'type': 'dataframe',
    })
else:
    # Best-effort JSON serialization
    def _coerce(x):
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, (set,)):
            return sorted(list(x))
        return x

    payload = _coerce(result)
    (artifact_dir / 'data.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    meta.update({'type': type(result).__name__})

(artifact_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(meta))
"""

    # We intentionally run the *same* python environment (user runs harness via conda run)
    cmd = [
        sys.executable,
        "-c",
        inline,
        str(worktree_repo_root),
        branch_label,
        call,
        mod,
        fn,
        str(harness_support_path),
        args_json or "[]",
        kwargs_json or "{}",
        "1" if init_provider else "0",
        str(db_path) if db_path else "NONE",
        str(table) if table else "NONE",
        str(artifact_dir),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            "Branch run failed\n"
            f"branch={branch_label} call={call}\n"
            f"stdout:\n{p.stdout}\n\n"
            f"stderr:\n{p.stderr}\n"
        )

    meta = json.loads(p.stdout.strip().splitlines()[-1])
    return RunMeta(
        branch=str(meta.get("branch")),
        commit=str(meta.get("commit")),
        call=str(meta.get("call")),
        seconds=float(meta.get("seconds", 0.0)),
        rows=int(meta.get("rows", 0)),
        cols=int(meta.get("cols", 0)),
        columns=list(meta.get("columns", [])),
        dtypes=dict(meta.get("dtypes", {})),
    )


def _load_df(artifact_dir: Path) -> pd.DataFrame:
    csv_path = artifact_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing artifact CSV: {csv_path}. "
            "(The callable probably did not return a pandas DataFrame.)"
        )
    return pd.read_csv(csv_path)


def _maybe_derive_y_bin_idx(df: pd.DataFrame, *, bin_size_mm: float) -> pd.DataFrame:
    if "Y_min" not in df.columns:
        return df
    out = df.copy()
    out["Y_min"] = pd.to_numeric(out["Y_min"], errors="coerce")
    out["y_bin_idx"] = np.round(out["Y_min"] / float(bin_size_mm)).astype("Int64")
    return out


def _default_join_keys(df_main: pd.DataFrame, df_feat: pd.DataFrame) -> list[str]:
    # Prefer common heatmap-style identifiers if available.
    for keys in (["Nut", "y_bin_idx"], ["Nut", "Y_min"], ["Nut", "Y_bin_center"]):
        if all(k in df_main.columns for k in keys) and all(k in df_feat.columns for k in keys):
            return list(keys)

    # Fallback: any single shared id-like column
    for k in ["id", "ID", "Slot", "slot", "Nut"]:
        if k in df_main.columns and k in df_feat.columns:
            return [k]

    return []


def _compare_dataframes(
    *,
    df_main: pd.DataFrame,
    df_feat: pd.DataFrame,
    join_keys: list[str],
    derive_y_bin_idx: float | None,
    atol: float,
    rtol: float,
) -> tuple[pd.DataFrame, CompareMeta]:
    if derive_y_bin_idx is not None:
        df_main = _maybe_derive_y_bin_idx(df_main, bin_size_mm=derive_y_bin_idx)
        df_feat = _maybe_derive_y_bin_idx(df_feat, bin_size_mm=derive_y_bin_idx)

    if not join_keys:
        join_keys = _default_join_keys(df_main, df_feat)

    if join_keys:
        merged = df_main.merge(
            df_feat,
            on=join_keys,
            how="outer",
            suffixes=("_main", "_feature"),
            indicator=True,
        )
        matched = merged[merged["_merge"] == "both"].copy()
    else:
        # No keys: compare row-wise after sorting columns. This is weaker but sometimes useful.
        common_cols = sorted(set(df_main.columns) & set(df_feat.columns))
        df_main_s = df_main[common_cols].sort_values(common_cols, kind="stable").reset_index(drop=True)
        df_feat_s = df_feat[common_cols].sort_values(common_cols, kind="stable").reset_index(drop=True)

        n = min(len(df_main_s), len(df_feat_s))
        left = df_main_s.iloc[:n].copy()
        right = df_feat_s.iloc[:n].copy()
        merged = pd.concat(
            [left.add_suffix("_main"), right.add_suffix("_feature")],
            axis=1,
        )
        merged["_merge"] = "both"
        matched = merged

    only_main = merged[merged["_merge"] == "left_only"]
    only_feat = merged[merged["_merge"] == "right_only"]

    # Numeric diffs on shared numeric columns (within matched rows)
    numeric_cols: list[str] = []
    max_abs: dict[str, float] = {}
    mean_abs: dict[str, float] = {}
    above_tol: dict[str, int] = {}

    if not matched.empty:
        # Determine shared columns based on suffix pattern
        main_cols = {c[:-5] for c in matched.columns if c.endswith("_main")}
        feat_cols = {c[:-8] for c in matched.columns if c.endswith("_feature")}
        shared = sorted(main_cols & feat_cols)

        for base in shared:
            a = pd.to_numeric(matched[f"{base}_main"], errors="coerce")
            b = pd.to_numeric(matched[f"{base}_feature"], errors="coerce")

            if a.notna().sum() == 0 or b.notna().sum() == 0:
                continue

            # Treat as numeric only if coercion produced many values
            if (a.notna().sum() + b.notna().sum()) < max(10, 0.1 * len(matched)):
                continue

            diff = (a - b).abs()
            tol = atol + rtol * b.abs()
            numeric_cols.append(base)
            max_abs[base] = float(np.nanmax(diff.values))
            mean_abs[base] = float(np.nanmean(diff.values))
            above_tol[base] = int(np.nansum((diff > tol).values))
            matched[f"{base}_abs_diff"] = diff
            matched[f"{base}_tol"] = tol

    meta = CompareMeta(
        call="",
        main_branch="",
        feature_branch="",
        join_keys=list(join_keys),
        derive_y_bin_idx=float(derive_y_bin_idx) if derive_y_bin_idx is not None else None,
        main_rows=int(len(df_main)),
        feature_rows=int(len(df_feat)),
        matched_rows=int(len(matched)),
        only_main_rows=int(len(only_main)),
        only_feature_rows=int(len(only_feat)),
        numeric_columns_compared=numeric_cols,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        above_tolerance_counts=above_tol,
        atol=float(atol),
        rtol=float(rtol),
    )

    return matched, meta


def main() -> None:
    repo_root = _repo_root()
    harness_support_path = (repo_root / "oxford_notebook" / "testing").resolve()

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--call", help="Callable used for BOTH branches (module:function)")
    group.add_argument("--call-main", help="Callable for main branch (module:function)")

    parser.add_argument("--call-feature", default=None, help="Callable for feature side (module:function). If omitted, uses --call or --call-main")
    parser.add_argument("--args", default="[]", help="JSON array of positional args")
    parser.add_argument("--kwargs", default="{}", help="JSON object of keyword args")

    parser.add_argument("--main-branch", default="main")
    parser.add_argument("--feature-branch", default=None, help="If set, run feature via a worktree for this branch")

    parser.add_argument("--no-init-provider", action="store_true", help="Skip provider.init()")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--table", default=None)

    parser.add_argument("--join-keys", default="", help="Comma-separated join key columns")
    parser.add_argument("--derive-y-bin-idx", type=float, default=None, help="If set, add y_bin_idx = round(Y_min/bin_size)")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-6)

    parser.add_argument("--out", default=None, help="Output directory (default: results/branch_harness/<timestamp>/<call>/)")

    args = parser.parse_args()

    call_main = args.call_main or args.call
    call_feature = args.call_feature or args.call or call_main

    feature_branch = args.feature_branch or _current_branch(repo_root)

    # Prefer an explicit db_path; otherwise use the DB from the working tree.
    # (The main worktree will typically NOT have a DB file.)
    db_path = args.db_path
    if db_path is None:
        auto_db = _default_duckdb_path(repo_root)
        if auto_db is not None:
            db_path = str(auto_db)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    call_slug = _safe_slug(f"main_{call_main}__feature_{call_feature}")

    out_dir = Path(args.out) if args.out else (_oxford_root(repo_root) / "results" / "branch_harness" / timestamp / call_slug)

    worktrees_dir = repo_root / ".worktrees"
    main_wt = _ensure_worktree(repo_root, args.main_branch, worktrees_dir / _safe_slug(args.main_branch))

    # Feature: either current working tree, or its own worktree
    if args.feature_branch:
        feature_wt = _ensure_worktree(repo_root, feature_branch, worktrees_dir / _safe_slug(feature_branch))
        feature_label = feature_branch
    else:
        feature_wt = repo_root
        feature_label = f"working-tree({feature_branch})"

    init_provider = not args.no_init_provider

    # Run both sides
    main_art = out_dir / "artifacts" / "main"
    feat_art = out_dir / "artifacts" / "feature"

    main_meta = _invoke_branch_subprocess(
        worktree_repo_root=main_wt,
        branch_label=args.main_branch,
        call=call_main,
        harness_support_path=harness_support_path,
        args_json=args.args,
        kwargs_json=args.kwargs,
        init_provider=init_provider,
        db_path=db_path,
        table=args.table,
        artifact_dir=main_art,
    )

    feat_meta = _invoke_branch_subprocess(
        worktree_repo_root=feature_wt,
        branch_label=feature_label,
        call=call_feature,
        harness_support_path=harness_support_path,
        args_json=args.args,
        kwargs_json=args.kwargs,
        init_provider=init_provider,
        db_path=db_path,
        table=args.table,
        artifact_dir=feat_art,
    )

    # If the callable didn't return a DataFrame, still keep artifacts + timing,
    # but skip numeric DataFrame diff.
    main_csv = (main_art / "data.csv")
    feat_csv = (feat_art / "data.csv")
    if not main_csv.exists() or not feat_csv.exists():
        _write_json(out_dir / "run_main.json", asdict(main_meta))
        _write_json(out_dir / "run_feature.json", asdict(feat_meta))
        print("\n=== Branch harness report (non-DataFrame output) ===")
        print(f"call(main): {call_main}")
        print(f"call(feat): {call_feature}")
        print(f"main: {args.main_branch} @ {main_meta.commit[:8]}  ({main_meta.seconds:.3f}s)")
        print(f"feat: {feature_label} @ {feat_meta.commit[:8]}  ({feat_meta.seconds:.3f}s)")
        print(f"\nWrote artifacts to: {out_dir.relative_to(repo_root)}")
        return

    df_main = _load_df(main_art)
    df_feat = _load_df(feat_art)

    join_keys = [k.strip() for k in args.join_keys.split(",") if k.strip()]

    matched, cmp_meta = _compare_dataframes(
        df_main=df_main,
        df_feat=df_feat,
        join_keys=join_keys,
        derive_y_bin_idx=args.derive_y_bin_idx,
        atol=args.atol,
        rtol=args.rtol,
    )

    cmp_meta.call = f"main={call_main} | feature={call_feature}"
    cmp_meta.main_branch = args.main_branch
    cmp_meta.feature_branch = feature_label

    # Write outputs
    _write_json(out_dir / "run_main.json", asdict(main_meta))
    _write_json(out_dir / "run_feature.json", asdict(feat_meta))
    _write_json(out_dir / "compare.json", asdict(cmp_meta))

    # Save matched diff view (can be large; keep a smaller head as well)
    matched.to_csv(out_dir / "matched_diff.csv", index=False)
    matched.head(200).to_csv(out_dir / "matched_diff_head.csv", index=False)

    # Print a concise summary
    print("\n=== Branch harness report ===")
    print(f"call(main): {call_main}")
    print(f"call(feat): {call_feature}")
    print(f"main: {args.main_branch} @ {main_meta.commit[:8]}  ({main_meta.seconds:.3f}s)")
    print(f"feat: {feature_label} @ {feat_meta.commit[:8]}  ({feat_meta.seconds:.3f}s)")
    speedup = (main_meta.seconds / feat_meta.seconds) if feat_meta.seconds > 0 else float('inf')
    print(f"speedup (main/feat): x{speedup:.2f}")

    print(f"rows (main/feat): {cmp_meta.main_rows} / {cmp_meta.feature_rows}")
    print(f"matched: {cmp_meta.matched_rows} | only_main: {cmp_meta.only_main_rows} | only_feat: {cmp_meta.only_feature_rows}")
    print(f"join_keys: {cmp_meta.join_keys or 'NONE'}")

    if cmp_meta.numeric_columns_compared:
        # Show up to 6 most divergent numeric columns
        ordered = sorted(cmp_meta.max_abs_diff.items(), key=lambda kv: kv[1], reverse=True)
        print("top numeric diffs (max abs):")
        for k, v in ordered[:6]:
            mean_v = cmp_meta.mean_abs_diff.get(k, float('nan'))
            n_bad = cmp_meta.above_tolerance_counts.get(k, 0)
            print(f"  - {k}: max={v:.6g} mean={mean_v:.6g} above_tol={n_bad}")
    else:
        print("numeric diffs: none computed (no matched numeric columns)")

    print(f"\nWrote report to: {out_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
