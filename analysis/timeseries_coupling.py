"""Time-series alignment between TTC and CO₂/energy rates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_l1(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/data/recording_{rec_id:02d}/L1_master_frame.parquet")
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Missing L1 file: {path}")


def _load_L2_conf(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/events/recording_{rec_id:02d}/L2_conflict_events.parquet")
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Missing L2 conflict file: {path}")


def aggregate_timeseries_over_episodes(
    rec_ids: Sequence[int],
    frame_rate: float = 25.0,
    t_window: Tuple[float, float] = (-5.0, 10.0),
    max_episodes: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Aggregate TTC and CO2 time series across conflict episodes.

    The function aligns each episode at the instant of minimum TTC (t=0),
    interpolates TTC and CO₂ rate (``cpf_co2_rate_gps``) onto a common time
    grid within ``t_window``, and computes mean/quantile envelopes. CI bands
    are based on the 10th/90th percentiles.

    Args:
        rec_ids: Recording IDs to include.
        frame_rate: Frame rate used to convert frames to seconds.
        t_window: Time window around t=0.
        max_episodes: Optional cap on the number of episodes processed.

    Returns:
        Dictionary with aggregated statistics and lag values.
    """
    dt = 1.0 / frame_rate
    t_grid = np.arange(t_window[0], t_window[1] + dt, dt)
    ttc_stack: list[np.ndarray] = []
    co2_stack: list[np.ndarray] = []
    lags: list[float] = []

    for rec_id in rec_ids:
        try:
            df_l1 = _load_l1(rec_id)
            df_L2 = _load_L2_conf(rec_id)
        except FileNotFoundError:
            continue

        for _, row in df_L2.iterrows():
            if max_episodes is not None and len(ttc_stack) >= max_episodes:
                break

            ego_id = int(row.get("ego_id", row.get("trackId", 0)))
            start_frame = int(row.get("start_frame", row.get("conf_start_frame", 0)))
            end_frame = int(row.get("end_frame", row.get("conf_end_frame", 0)))
            df_episode = df_l1[(df_l1["frame"] >= start_frame) & (df_l1["frame"] <= end_frame) & (df_l1["trackId"] == ego_id)]
            if df_episode.empty:
                continue

            ttc_series = df_episode.get("TTC")
            co2_series = df_episode.get("cpf_co2_rate_gps") if "cpf_co2_rate_gps" in df_episode else None
            if co2_series is None:
                co2_series = df_episode.get("co2_rate") if "co2_rate" in df_episode else None
            if ttc_series is None or co2_series is None:
                continue

            idx_min = ttc_series.idxmin()
            t0_frame = df_episode.loc[idx_min, "frame"]
            rel_t = (df_episode["frame"] - t0_frame) / frame_rate

            ttc_interp = np.interp(t_grid, rel_t.to_numpy(), ttc_series.to_numpy(), left=np.nan, right=np.nan)
            co2_interp = np.interp(t_grid, rel_t.to_numpy(), co2_series.to_numpy(), left=np.nan, right=np.nan)

            ttc_stack.append(ttc_interp)
            co2_stack.append(co2_interp)

            peak_idx = int(np.nanargmax(co2_interp)) if np.any(~np.isnan(co2_interp)) else 0
            lags.append(t_grid[peak_idx])

    ttc_arr = np.vstack(ttc_stack) if ttc_stack else np.empty((0, len(t_grid)))
    co2_arr = np.vstack(co2_stack) if co2_stack else np.empty((0, len(t_grid)))

    def _stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if arr.size == 0:
            return np.array([]), np.array([]), np.array([])
        return np.nanmean(arr, axis=0), np.nanquantile(arr, 0.1, axis=0), np.nanquantile(arr, 0.9, axis=0)

    ttc_mean, ttc_low, ttc_up = _stats(ttc_arr)
    co2_mean, co2_low, co2_up = _stats(co2_arr)

    return {
        "t_grid": t_grid,
        "TTC_mean": ttc_mean,
        "TTC_lower": ttc_low,
        "TTC_upper": ttc_up,
        "CO2_mean": co2_mean,
        "CO2_lower": co2_low,
        "CO2_upper": co2_up,
        "lags": np.array(lags),
    }


def plot_mean_timeseries(
    t: np.ndarray,
    TTC_mean: np.ndarray,
    CO2_mean: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot the mean TTC and CO2 time series on two stacked axes.

    This helper keeps the plotting logic used by the experiment entry point
    lightweight: it only draws the mean curves (no CI bands) and optionally
    writes the result to ``save_path``.
    """

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(t, TTC_mean, color="tab:blue")
    axes[0].axvline(0, color="k", linestyle=":", linewidth=1)
    axes[0].set_ylabel("TTC [s]")
    axes[0].set_title("Mean TTC around minimum")

    axes[1].plot(t, CO2_mean, color="tab:red")
    axes[1].axvline(0, color="k", linestyle=":", linewidth=1)
    axes[1].set_ylabel("CO2 rate [g/s]")
    axes[1].set_xlabel("Time aligned to min TTC [s]")
    axes[1].set_title("Mean CO2 response")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ttc_co2_alignment_with_ci(
    t: np.ndarray,
    TTC_mean: np.ndarray,
    TTC_lower: np.ndarray,
    TTC_upper: np.ndarray,
    CO2_mean: np.ndarray,
    CO2_lower: np.ndarray,
    CO2_upper: np.ndarray,
    lag_stats: dict,
    save_path: Path | None = None,
) -> None:
    """
    Plot aligned TTC and CO2 time series with confidence intervals.

    Args:
        t: Time grid.
        TTC_mean: Mean TTC curve.
        TTC_lower: Lower CI bound for TTC.
        TTC_upper: Upper CI bound for TTC.
        CO2_mean: Mean CO2 curve.
        CO2_lower: Lower CI bound for CO2.
        CO2_upper: Upper CI bound for CO2.
        lag_stats: Dictionary with lag statistics (median, mean, quantiles).
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, TTC_mean, color="tab:blue")
    axes[0].fill_between(t, TTC_lower, TTC_upper, color="tab:blue", alpha=0.2)
    axes[0].axvline(0, color="k", linestyle=":", linewidth=1)
    axes[0].set_ylabel("TTC [s]")
    axes[0].set_title("TTC alignment")

    axes[1].plot(t, CO2_mean, color="tab:red")
    axes[1].fill_between(t, CO2_lower, CO2_upper, color="tab:red", alpha=0.2)
    axes[1].axvline(0, color="k", linestyle=":", linewidth=1)
    if "median" in lag_stats:
        axes[1].annotate(
            f"lag={lag_stats['median']:.2f}s",
            xy=(lag_stats["median"], np.nanmax(CO2_mean)),
            xytext=(lag_stats["median"] + 0.5, np.nanmax(CO2_mean) * 0.9),
            arrowprops={"arrowstyle": "->"},
        )
    axes[1].set_ylabel("CO2 rate [g/s]")
    axes[1].set_xlabel("Time aligned to min TTC [s]")
    axes[1].set_title("CO2 response")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
