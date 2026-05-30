"""Case wrappers for the branch harness.

These helpers are intentionally branch-agnostic and live in the working tree.
The harness injects this directory into `sys.path` for BOTH branch subprocesses.

Add more wrappers here when you need to compare pipelines that don't map
1:1 to a single function name across branches.
"""

from __future__ import annotations

import random

import pandas as pd


def heatmap_old_from_provider(
    plate: str,
    *,
    bin_size_mm: float = 10.0,
    target_signal: str = "X",
    target_origin: str = "Oscilloscope",
) -> pd.DataFrame:
    """Compute the heatmap the "old" way (main source-of-truth style).

    This mirrors the main-branch widget logic:
    - Iterate slots via provider
    - Load each slot DataFrame
    - Bin with prepare_equal_bins_heatmap
    - Concatenate results

    Returns a DataFrame with columns: Nut, Y_min, Y_max, Y_bin_center, RMS_raw
    """
    from src import provider
    from src.data_processing import prepare_equal_bins_heatmap

    slots = provider.slots(str(plate))
    frames: list[pd.DataFrame] = []

    for slot in slots:
        df_slot = provider.df(
            str(plate),
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
        if df_heat is not None and not df_heat.empty:
            frames.append(df_heat)

    if not frames:
        return pd.DataFrame(columns=["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"])

    out = pd.concat(frames, ignore_index=True)
    return out[["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]]


def heatmap_sql(
    plate: str,
    *,
    bin_size_mm: float = 10.0,
    target_signal: str = "X",
    target_origin: str = "Oscilloscope",
) -> pd.DataFrame:
    """Compute the heatmap via SQL (feature-branch optimized path)."""
    from src.data_processing import prepare_equal_bins_heatmap_sql

    df = prepare_equal_bins_heatmap_sql(
        str(plate),
        bin_size_mm=bin_size_mm,
        target_signal=target_signal,
        target_origin=target_origin,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"])

    return df[["Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]]


def _select_random_plates(*, plate_count: int, seed: int | None) -> list[str]:
    from src import provider

    plates = [str(p) for p in provider.plates()]
    plates = sorted(set(plates))

    if plate_count <= 0 or not plates:
        return []

    if plate_count >= len(plates):
        return plates

    rng = random.Random(int(seed) if seed is not None else 0)
    return rng.sample(plates, k=int(plate_count))


def heatmap_old_random_plates(
    *,
    plate_count: int = 10,
    seed: int | None = 0,
    bin_size_mm: float = 10.0,
    target_signal: str = "X",
    target_origin: str = "Oscilloscope",
) -> pd.DataFrame:
    """Compute *old* heatmap outputs for a random sample of plates.

    Returns a single concatenated DataFrame with an extra `Platte` column.
    """

    plates = _select_random_plates(plate_count=plate_count, seed=seed)
    frames: list[pd.DataFrame] = []

    for plate in plates:
        df = heatmap_old_from_provider(
            plate,
            bin_size_mm=bin_size_mm,
            target_signal=target_signal,
            target_origin=target_origin,
        )
        if df is None or df.empty:
            continue

        out = df.copy()
        out.insert(0, "Platte", str(plate))
        frames.append(out)

    cols = ["Platte", "Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]
    if not frames:
        return pd.DataFrame(columns=cols)

    combined = pd.concat(frames, ignore_index=True)
    return combined[cols]


def heatmap_sql_random_plates(
    *,
    plate_count: int = 10,
    seed: int | None = 0,
    bin_size_mm: float = 10.0,
    target_signal: str = "X",
    target_origin: str = "Oscilloscope",
) -> pd.DataFrame:
    """Compute *SQL* heatmap outputs for a random sample of plates.

    Returns a single concatenated DataFrame with an extra `Platte` column.
    """

    plates = _select_random_plates(plate_count=plate_count, seed=seed)
    frames: list[pd.DataFrame] = []

    for plate in plates:
        df = heatmap_sql(
            plate,
            bin_size_mm=bin_size_mm,
            target_signal=target_signal,
            target_origin=target_origin,
        )
        if df is None or df.empty:
            continue

        out = df.copy()
        out.insert(0, "Platte", str(plate))
        frames.append(out)

    cols = ["Platte", "Nut", "Y_min", "Y_max", "Y_bin_center", "RMS_raw"]
    if not frames:
        return pd.DataFrame(columns=cols)

    combined = pd.concat(frames, ignore_index=True)
    return combined[cols]
