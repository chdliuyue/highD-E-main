from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
EVENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"


def load_L1_for_recording(rec_id: int) -> pd.DataFrame:
    """读取 recording_XX 的 L1_master_frame.parquet。"""

    rec_str = f"recording_{rec_id:02d}"
    path = DATA_ROOT / rec_str / "L1_master_frame.parquet"
    return pd.read_parquet(path)


def load_L2_conflicts_for_recording(rec_id: int) -> pd.DataFrame:
    """读取 recording_XX 的 L2_conflict_events.parquet。"""

    rec_str = f"recording_{rec_id:02d}"
    path = EVENTS_ROOT / rec_str / "L2_conflict_events.parquet"
    return pd.read_parquet(path)


def extract_episode_timeseries(
    df_l1: pd.DataFrame,
    event_row: pd.Series,
    t_window: Tuple[float, float] = (-5.0, 10.0),
    frame_rate: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对单个 episode，从 L1 中提取 TTC(t), CO2_rate(t) 的时间序列，并对齐到相对时间 t。

    对齐方法：
      - 在 L2 中，event_row 提供 conf_start_frame/conf_end_frame（核心区间）；
      - 在 L1 中找到该 episode 所在 ego 轨迹的 TTC 时间序列；
      - 在核心区间内找到 TTC 的最小值所在帧 f_min，作为参考帧；
      - 定义 f_min 的时间为 0，根据 t_window = (t_pre, t_post)
        将 frame 区间 [f_min + t_pre*frame_rate, f_min + t_post*frame_rate] 截取出来；
      - 返回:
        t_aligned: np.ndarray (N,), 相对于风险点的时间 [s]
        TTC_t:     np.ndarray (N,)
        CO2_t:     np.ndarray (N,)  # 使用 L1 的 cpf_co2_rate_gps
    若窗口越界，允许用较短序列。
    """

    rec_id = int(event_row["recordingId"])
    track_id = int(event_row.get("ego_id", event_row.get("trackId", -1)))
    conf_start = int(event_row["conf_start_frame"])
    conf_end = int(event_row["conf_end_frame"])

    df_episode = df_l1[(df_l1["recordingId"] == rec_id) & (df_l1["trackId"] == track_id)].sort_values(
        "frame"
    )
    if df_episode.empty:
        return np.array([]), np.array([]), np.array([])

    conf_df = df_episode[(df_episode["frame"] >= conf_start) & (df_episode["frame"] <= conf_end)]
    if conf_df.empty:
        return np.array([]), np.array([]), np.array([])

    ttc_conf = conf_df["TTC"].to_numpy()
    if np.all(np.isnan(ttc_conf)):
        return np.array([]), np.array([]), np.array([])

    f_min = int(conf_df.loc[conf_df["TTC"].idxmin(), "frame"])
    t_pre, t_post = t_window
    frame_pre = int(np.floor(t_pre * frame_rate))
    frame_post = int(np.floor(t_post * frame_rate))

    start_frame = f_min + frame_pre
    end_frame = f_min + frame_post

    window_df = df_episode[(df_episode["frame"] >= start_frame) & (df_episode["frame"] <= end_frame)]
    if window_df.empty:
        return np.array([]), np.array([]), np.array([])

    t_aligned = (window_df["frame"].to_numpy() - f_min) / frame_rate
    ttc_t = window_df["TTC"].to_numpy()
    co2_t = window_df.get("cpf_co2_rate_gps", pd.Series(np.nan, index=window_df.index)).to_numpy()

    return t_aligned, ttc_t, co2_t


def aggregate_timeseries_over_episodes(
    rec_ids: Tuple[int, ...],
    frame_rate: float = 25.0,
    t_window: Tuple[float, float] = (-5.0, 10.0),
    max_episodes: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    对多个 recording 的 L2_conflict_events 进行聚合分析。

    返回一个字典：
      {
        "t_grid": np.ndarray,
        "TTC_mean": np.ndarray,
        "TTC_median": np.ndarray,
        "CO2_mean": np.ndarray,
        "CO2_median": np.ndarray,
        "lags": np.ndarray  # 每条 episode 的风险→排放滞后时间
      }
    """

    all_events: list[pd.Series] = []
    for rec_id in rec_ids:
        df_l2 = load_L2_conflicts_for_recording(rec_id)
        for _, row in df_l2.iterrows():
            all_events.append(row)

    if not all_events:
        return {
            "t_grid": np.array([]),
            "TTC_mean": np.array([]),
            "TTC_median": np.array([]),
            "CO2_mean": np.array([]),
            "CO2_median": np.array([]),
            "lags": np.array([]),
        }

    if len(all_events) > max_episodes:
        rng = np.random.default_rng()
        sampled_indices = rng.choice(len(all_events), size=max_episodes, replace=False)
        selected_events = [all_events[i] for i in sampled_indices]
    else:
        selected_events = all_events

    t_pre, t_post = t_window
    t_grid = np.arange(t_pre, t_post + 1.0 / frame_rate, 1.0 / frame_rate)
    n_time = len(t_grid)

    ttc_all = []
    co2_all = []
    lags = []

    for event_row in selected_events:
        df_l1 = load_L1_for_recording(int(event_row["recordingId"]))
        t_rel, ttc_t, co2_t = extract_episode_timeseries(df_l1, event_row, t_window, frame_rate)
        if t_rel.size == 0:
            continue

        aligned_ttc = np.full(n_time, np.nan)
        aligned_co2 = np.full(n_time, np.nan)
        idx = np.round((t_rel - t_pre) * frame_rate).astype(int)
        valid = (idx >= 0) & (idx < n_time)
        aligned_ttc[idx[valid]] = ttc_t[valid]
        aligned_co2[idx[valid]] = co2_t[valid]

        ttc_all.append(aligned_ttc)
        co2_all.append(aligned_co2)

        try:
            risk_idx = np.nanargmin(ttc_t)
            co2_idx = np.nanargmax(co2_t)
            lag = t_rel[co2_idx] - t_rel[risk_idx]
            lags.append(lag)
        except ValueError:
            continue

    if not ttc_all:
        return {
            "t_grid": t_grid,
            "TTC_mean": np.array([]),
            "TTC_median": np.array([]),
            "CO2_mean": np.array([]),
            "CO2_median": np.array([]),
            "lags": np.array(lags),
        }

    ttc_all_arr = np.vstack(ttc_all)
    co2_all_arr = np.vstack(co2_all)

    ttc_mean = np.nanmean(ttc_all_arr, axis=0)
    ttc_median = np.nanmedian(ttc_all_arr, axis=0)
    co2_mean = np.nanmean(co2_all_arr, axis=0)
    co2_median = np.nanmedian(co2_all_arr, axis=0)

    return {
        "t_grid": t_grid,
        "TTC_mean": ttc_mean,
        "TTC_median": ttc_median,
        "CO2_mean": co2_mean,
        "CO2_median": co2_median,
        "lags": np.array(lags),
    }


def plot_mean_timeseries(
    t: np.ndarray,
    TTC_mean: np.ndarray,
    CO2_mean: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    绘制 TTC_mean(t) 与 CO2_mean(t) 的双 subplot 图。
    """

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, TTC_mean, label="TTC_mean", color="tab:blue")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylabel("TTC [s]")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t, CO2_mean, label="CO2_mean", color="tab:green")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_ylabel("CO2 rate [g/s]")
    axes[1].set_xlabel("Time relative to risk [s]")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
