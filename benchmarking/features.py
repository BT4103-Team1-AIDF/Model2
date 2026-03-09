"""Feature selection and time-aware splitting utilities."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_HORIZON_PATTERN = re.compile(r"^y_\d+m$")


def select_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    id_columns: tuple[str, ...],
    drop_other_horizon_targets: bool,
) -> list[str]:
    """Select leakage-safe features for one target horizon."""
    exclude = {target_col, *id_columns}
    if drop_other_horizon_targets:
        for col in df.columns:
            if TARGET_HORIZON_PATTERN.match(col):
                exclude.add(col)
    return [c for c in df.columns if c not in exclude]


def split_time_aware(
    df: pd.DataFrame,
    *,
    validation_fraction: float,
    year_col: str = "yyyy",
    month_col: str = "mm",
) -> tuple[np.ndarray, np.ndarray]:
    """Split train data into chronological train/validation sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    if year_col in df.columns and month_col in df.columns:
        periods = (df[year_col].astype(int) * 100 + df[month_col].astype(int)).to_numpy()
    elif year_col in df.columns:
        periods = df[year_col].astype(int).to_numpy()
    else:
        n = len(df)
        split_at = int((1.0 - validation_fraction) * n)
        train_idx = np.arange(0, split_at)
        val_idx = np.arange(split_at, n)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Time-aware fallback split produced empty train or validation.")
        return train_idx, val_idx

    unique_periods = np.sort(pd.unique(periods))
    if unique_periods.size < 2:
        raise ValueError("Need at least two unique periods for time-aware validation split.")

    n_val_periods = max(1, math.ceil(unique_periods.size * validation_fraction))
    n_val_periods = min(n_val_periods, unique_periods.size - 1)
    val_periods = set(unique_periods[-n_val_periods:].tolist())

    train_mask = ~np.isin(periods, list(val_periods))
    val_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Time-aware split produced empty train or validation.")
    return train_idx, val_idx


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
