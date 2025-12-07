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


def safe_nanmean_and_quantiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 arr (shape: n_events x n_time) 做 nanmean 和 10%、90% 分位数聚合。
    若某个时间点全 NaN，则对应 mean/lower/upper 都设为 NaN，避免 RuntimeWarning。
    """

    if arr.size == 0:
        return np.array([]), np.array([]), np.array([])

    n_time = arr.shape[1]
    mean = np.empty(n_time)
    lower = np.empty(n_time)
    upper = np.empty(n_time)

    for j in range(n_time):
        col = arr[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            mean[j] = np.nan
            lower[j] = np.nan
            upper[j] = np.nan
            continue
        mean[j] = np.mean(valid)
        lower[j] = np.quantile(valid, 0.1)
        upper[j] = np.quantile(valid, 0.9)

    return mean, lower, upper


def compute_lag_statistics(lags: np.ndarray) -> dict:
    """Return basic statistics for lag values, handling empty inputs safely."""

    if lags.size == 0:
        return {"mean": np.nan, "median": np.nan, "q10": np.nan, "q90": np.nan, "count": 0}

    return {
        "mean": float(np.nanmean(lags)),
        "median": float(np.nanmedian(lags)),
        "q10": float(np.nanquantile(lags, 0.1)),
        "q90": float(np.nanquantile(lags, 0.9)),
        "count": int(np.sum(~np.isnan(lags))),
    }


def aggregate_timeseries_with_stats(
    rec_ids: Sequence[int],
    frame_rate: float = 25.0,
    t_window: Tuple[float, float] = (-5.0, 10.0),
    max_episodes: int | None = None,
) -> tuple[Dict[str, np.ndarray], dict]:
    """Aggregate time series and compute lag statistics in one call."""

    agg = aggregate_timeseries_over_episodes(
        rec_ids=rec_ids,
        frame_rate=frame_rate,
        t_window=t_window,
        max_episodes=max_episodes,
    )
    lag_stats = compute_lag_statistics(agg.get("lags", np.array([])))
    return agg, lag_stats


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
    min_ttc_list: list[float] = []

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
            min_ttc_list.append(float(np.nanmin(ttc_series)))

            peak_idx = int(np.nanargmax(co2_interp)) if np.any(~np.isnan(co2_interp)) else 0
            lags.append(t_grid[peak_idx])

    ttc_arr = np.vstack(ttc_stack) if ttc_stack else np.empty((0, len(t_grid)))
    co2_arr = np.vstack(co2_stack) if co2_stack else np.empty((0, len(t_grid)))

    ttc_mean, ttc_low, ttc_up = safe_nanmean_and_quantiles(ttc_arr)
    co2_mean, co2_low, co2_up = safe_nanmean_and_quantiles(co2_arr)

    return {
        "t_grid": t_grid,
        "TTC_mean": ttc_mean,
        "TTC_lower": ttc_low,
        "TTC_upper": ttc_up,
        "CO2_mean": co2_mean,
        "CO2_lower": co2_low,
        "CO2_upper": co2_up,
        "lags": np.array(lags),
        "co2_stack": co2_arr,
        "min_ttc_list": np.array(min_ttc_list),
    }


def compute_phase_co2_shares(
    co2_stack: np.ndarray,
    t_grid: np.ndarray,
    min_ttc_list: np.ndarray | None = None,
    frame_rate: float = 25.0,
    t_window: tuple[float, float] = (-5.0, 10.0),
) -> pd.DataFrame:
    """
    基于对齐后的时间序列，计算 pre/core/recovery 三段 CO₂ 贡献占比。

    - pre:     [-5, -1) s
    - core:    [-1, +1] s
    - recovery:(+1, +10] s

    Args:
        co2_stack: shape (n_events, n_time) 对齐到 t=0 的 CO₂ 速率矩阵。
        t_grid: 与 co2_stack 对应的时间网格。
        min_ttc_list: 每个 episode 的 min_TTC_conf，便于分组统计。
        frame_rate: 用于估计时间步长的帧率。
        t_window: 时间窗范围（用于安全截断）。

    Returns:
        DataFrame，其中包含每条 episode 的阶段能耗占比。
    """

    if co2_stack.size == 0 or t_grid.size == 0:
        return pd.DataFrame(columns=["share_pre", "share_core", "share_rec", "episode_idx", "min_TTC_conf"])

    dt = float(np.mean(np.diff(t_grid))) if len(t_grid) > 1 else 1.0 / frame_rate
    t_start, t_end = t_window

    pre_mask = (t_grid >= max(-5.0, t_start)) & (t_grid < -1.0)
    core_mask = (t_grid >= -1.0) & (t_grid <= 1.0)
    rec_mask = (t_grid > 1.0) & (t_grid <= min(10.0, t_end))

    shares = []
    for idx, co2_curve in enumerate(co2_stack):
        co2_curve = np.asarray(co2_curve, dtype=float)
        e_pre = float(np.nansum(co2_curve[pre_mask]) * dt)
        e_core = float(np.nansum(co2_curve[core_mask]) * dt)
        e_rec = float(np.nansum(co2_curve[rec_mask]) * dt)
        total = e_pre + e_core + e_rec
        if total <= 0:
            share_pre = share_core = share_rec = np.nan
        else:
            share_pre = e_pre / total
            share_core = e_core / total
            share_rec = e_rec / total
        shares.append(
            {
                "episode_idx": idx,
                "share_pre": share_pre,
                "share_core": share_core,
                "share_rec": share_rec,
                "min_TTC_conf": float(min_ttc_list[idx]) if min_ttc_list is not None and len(min_ttc_list) > idx else np.nan,
            }
        )

    return pd.DataFrame(shares)


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
    else:
        plt.show()
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
    median_lag = lag_stats.get("median")
    has_peak = np.any(~np.isnan(CO2_mean))
    if median_lag is not None and not np.isnan(median_lag) and has_peak:
        peak_val = np.nanmax(CO2_mean)
        axes[1].annotate(
            f"median lag = {median_lag:.2f}s",
            xy=(median_lag, peak_val),
            xytext=(median_lag + 0.5, peak_val * 0.9),
            arrowprops={"arrowstyle": "->"},
        )
    axes[1].set_ylabel("CO2 rate [g/s]")
    axes[1].set_xlabel("Time aligned to min TTC [s]")
    axes[1].set_title("CO2 response")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)
