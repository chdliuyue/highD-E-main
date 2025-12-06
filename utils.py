"""Utility helpers for the highD preprocessing pipeline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def apply_savgol_by_group(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    value_col: str,
    window: int,
    poly: int,
    deriv: int = 0,
) -> pd.Series:
    """Apply Savitzky-Golay smoothing per group."""

    def _smooth_group(sub_df: pd.DataFrame) -> pd.Series:
        if len(sub_df) < poly + 2:
            return sub_df[value_col].astype(float)
        effective_window = min(window, len(sub_df) if len(sub_df) % 2 == 1 else len(sub_df) - 1)
        if effective_window % 2 == 0:
            effective_window -= 1
        effective_window = max(effective_window, poly + 2)
        if effective_window % 2 == 0:
            effective_window += 1
        return pd.Series(
            savgol_filter(sub_df[value_col].to_numpy(), effective_window, poly, deriv=deriv),
            index=sub_df.index,
        )

    return df.groupby(list(group_cols), group_keys=False, sort=False).apply(_smooth_group)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 99.0) -> np.ndarray:
    """Element-wise division with protection against zero denominator."""
    result = np.full_like(numerator, default, dtype=np.float32)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def compute_drac(gap: np.ndarray, rel_vel: np.ndarray, reaction_time: float = 1.0) -> np.ndarray:
    """Compute Deceleration Rate to Avoid Crash (DRAC)."""
    gap_after_reaction = gap - np.maximum(rel_vel, 0) * reaction_time
    gap_after_reaction = np.maximum(gap_after_reaction, 0)
    decel = np.where(gap_after_reaction <= 0, np.inf, (rel_vel ** 2) / (2 * gap_after_reaction))
    decel = np.nan_to_num(decel, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return decel.astype(np.float32)


def ensure_directory(path: str | bytes | "PathLike[str]") -> None:
    """Create directory if it does not exist."""
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)


__all__ = ["apply_savgol_by_group", "compute_drac", "ensure_directory", "safe_divide"]

