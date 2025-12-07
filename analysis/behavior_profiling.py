from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
MEC_PATH = PROJECT_ROOT / "data" / "analysis" / "L2_conf_mec_baseline.parquet"
FIG_ROOT = PROJECT_ROOT / "output" / "behavior" / "figs"
TABLE_ROOT = PROJECT_ROOT / "output" / "behavior" / "tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def load_mec_events() -> pd.DataFrame:
    """
    从 MEC_PATH 读取事件级 MEC 数据。

    若文件不存在，则返回空 DataFrame 并提示用户。
    """

    if not MEC_PATH.exists():
        print(f"[behavior_profiling] MEC parquet not found: {MEC_PATH}")
        return pd.DataFrame()
    return pd.read_parquet(MEC_PATH)


def filter_events_for_clustering(df_mec: pd.DataFrame) -> pd.DataFrame:
    """
    根据 min_TTC_conf, conf_duration, MEC 等条件筛选用于行为聚类的事件子集。
    """

    if df_mec.empty:
        return df_mec

    df = df_mec.copy()
    min_ttc_col = "min_TTC_conf" if "min_TTC_conf" in df.columns else None
    conf_dur_col = "conf_duration" if "conf_duration" in df.columns else None
    mec_col = "MEC_CO2_per_km" if "MEC_CO2_per_km" in df.columns else None

    if min_ttc_col is None or conf_dur_col is None:
        print("[behavior_profiling] min_TTC_conf/conf_duration column missing; skip filtering.")
        return df

    filt = (df[min_ttc_col] < 3.0) & (df[conf_dur_col] >= 0.8)
    if mec_col is not None:
        filt &= df[mec_col].notna()
    return df.loc[filt].copy()


def _load_l1_for_recording(rec_id: int) -> pd.DataFrame:
    rec_str = f"{rec_id:02d}"
    parquet_path = L1_ROOT / f"recording_{rec_str}" / "L1_master_frame.parquet"
    if not parquet_path.exists():
        print(f"[behavior_profiling] L1 file missing: {parquet_path}")
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def compute_kinematic_features_from_l1(
    df_l1: pd.DataFrame, df_events: pd.DataFrame
) -> pd.DataFrame:
    """
    对 df_events 中的每条事件在对应的 L1 DataFrame 里计算:
      - max_decel: a_long_smooth 的最小值
      - max_accel: a_long_smooth 的最大值
      - v_drop: v_long_smooth 的最大值 - 最小值

    若缺少必要的 frame/track 信息，则返回 NaN。
    """

    if df_events.empty or df_l1.empty:
        df_events["max_decel"] = np.nan
        df_events["max_accel"] = np.nan
        df_events["v_drop"] = np.nan
        return df_events

    required_cols = {"recordingId", "trackId", "frame", "v_long_smooth", "a_long_smooth"}
    if not required_cols.issubset(df_l1.columns):
        df_events["max_decel"] = np.nan
        df_events["max_accel"] = np.nan
        df_events["v_drop"] = np.nan
        return df_events

    start_cols = ["start_frame", "conf_start_frame", "start"]
    end_cols = ["end_frame", "conf_end_frame", "end"]

    max_decel_list: list[float] = []
    max_accel_list: list[float] = []
    v_drop_list: list[float] = []

    for _, row in df_events.iterrows():
        rec_val = row.get("rec_id", row.get("recordingId", np.nan))
        ego_id = row.get("ego_id", row.get("trackId", np.nan))
        start_frame = next((int(row[c]) for c in start_cols if c in row and pd.notna(row[c])), None)
        end_frame = next((int(row[c]) for c in end_cols if c in row and pd.notna(row[c])), None)

        if pd.isna(rec_val) or pd.isna(ego_id) or start_frame is None or end_frame is None:
            max_decel_list.append(np.nan)
            max_accel_list.append(np.nan)
            v_drop_list.append(np.nan)
            continue

        mask = (
            (df_l1["recordingId"] == int(rec_val))
            & (df_l1["trackId"] == ego_id)
            & (df_l1["frame"] >= start_frame)
            & (df_l1["frame"] <= end_frame)
        )
        df_ep = df_l1.loc[mask]
        if df_ep.empty:
            max_decel_list.append(np.nan)
            max_accel_list.append(np.nan)
            v_drop_list.append(np.nan)
            continue

        v_series = df_ep["v_long_smooth"].astype(float)
        a_series = df_ep["a_long_smooth"].astype(float)
        max_decel_list.append(float(a_series.min()))
        max_accel_list.append(float(a_series.max()))
        v_drop_list.append(float(v_series.max() - v_series.min()))

    df_events = df_events.copy()
    df_events["max_decel"] = max_decel_list
    df_events["max_accel"] = max_accel_list
    df_events["v_drop"] = v_drop_list
    return df_events


def build_feature_matrix(df_events: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    从 df_events 中选取一组数值特征列，构成标准化后的 feature matrix。
    """

    feature_candidates = [
        "min_TTC_conf",
        "conf_duration",
        "MEC_CO2_per_km",
        "E_real_CO2_per_km",
        "max_decel",
        "max_accel",
        "v_drop",
    ]
    available = [c for c in feature_candidates if c in df_events.columns]
    if not available:
        raise ValueError("No available feature columns for clustering.")

    df_feat = df_events[available].copy()
    X = df_feat.to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df_events


def cluster_events(
    df_events: pd.DataFrame, n_clusters: int = 3, random_state: int = 0
) -> pd.DataFrame:
    """
    对 df_events 进行 KMeans 聚类并添加 'cluster' 列。
    """

    if df_events.empty:
        df_events["cluster"] = []
        return df_events

    X, df_aug = build_feature_matrix(df_events)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)
    df_aug = df_aug.copy()
    df_aug["cluster"] = labels
    return df_aug


def _get_min_ttc_frame(event_row: pd.Series) -> int | None:
    for key in ["min_ttc_frame", "frame_min_ttc", "conf_min_ttc_frame", "ttc_min_frame"]:
        if key in event_row and pd.notna(event_row[key]):
            return int(event_row[key])
    return None


def _extract_episode_timeseries(
    df_l1: pd.DataFrame,
    event_row: pd.Series,
    frame_rate: float,
    t_window: Tuple[float, float],
) -> dict[str, np.ndarray]:
    rec_val = event_row.get("rec_id", event_row.get("recordingId", np.nan))
    ego_id = event_row.get("ego_id", event_row.get("trackId", np.nan))

    start_frame = None
    for key in ["start_frame", "conf_start_frame", "start"]:
        if key in event_row and pd.notna(event_row[key]):
            start_frame = int(event_row[key])
            break

    end_frame = None
    for key in ["end_frame", "conf_end_frame", "end"]:
        if key in event_row and pd.notna(event_row[key]):
            end_frame = int(event_row[key])
            break

    if pd.isna(rec_val) or pd.isna(ego_id) or start_frame is None or end_frame is None:
        return {}

    mask = (
        (df_l1["recordingId"] == int(rec_val))
        & (df_l1["trackId"] == ego_id)
        & (df_l1["frame"] >= start_frame)
        & (df_l1["frame"] <= end_frame)
    )
    df_ep = df_l1.loc[mask].copy()
    if df_ep.empty:
        return {}

    min_ttc_frame = _get_min_ttc_frame(event_row)
    if min_ttc_frame is None and "TTC" in df_ep.columns:
        ttc_series = df_ep["TTC"]
        if ttc_series.notna().any():
            min_ttc_frame = int(df_ep.loc[ttc_series.idxmin(), "frame"])

    if min_ttc_frame is None:
        min_ttc_frame = start_frame

    df_ep["t_rel"] = (df_ep["frame"] - min_ttc_frame) / frame_rate
    df_ep = df_ep[(df_ep["t_rel"] >= t_window[0]) & (df_ep["t_rel"] <= t_window[1])]
    if df_ep.empty:
        return {}

    df_ep = df_ep.sort_values("t_rel")
    t_vals = df_ep["t_rel"].to_numpy()
    v_vals = df_ep["v_long_smooth"].to_numpy() if "v_long_smooth" in df_ep.columns else None
    a_vals = df_ep["a_long_smooth"].to_numpy() if "a_long_smooth" in df_ep.columns else None
    co2_vals = df_ep["cpf_co2_rate_gps"].to_numpy() if "cpf_co2_rate_gps" in df_ep.columns else None

    t_grid = np.arange(t_window[0], t_window[1] + 1e-6, 1.0 / frame_rate)

    def _interp(values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.full_like(t_grid, np.nan, dtype=float)
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return np.full_like(t_grid, np.nan, dtype=float)
        t_valid = t_vals[valid_mask]
        v_valid = values[valid_mask]
        if len(t_valid) == 1:
            return np.where(
                (t_grid >= t_valid[0] - (1.0 / frame_rate)) & (t_grid <= t_valid[0] + (1.0 / frame_rate)),
                v_valid[0],
                np.nan,
            )
        return np.interp(t_grid, t_valid, v_valid, left=np.nan, right=np.nan)

    return {
        "t": t_grid,
        "v": _interp(v_vals),
        "a": _interp(a_vals),
        "co2": _interp(co2_vals),
    }


def compute_cluster_centroid_timeseries(
    df_mec_clustered: pd.DataFrame,
    rec_ids: Sequence[int] | None,
    frame_rate: float = 25.0,
    t_window: Tuple[float, float] = (-5.0, 8.0),
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    对每个 cluster 构建 v/a/CO2 时间序列平均曲线。
    """

    centroids: Dict[int, Dict[str, np.ndarray]] = {}
    if df_mec_clustered.empty or "cluster" not in df_mec_clustered.columns:
        return centroids

    if rec_ids is None:
        rec_ids = sorted(df_mec_clustered.get("rec_id", df_mec_clustered.get("recordingId", [])).dropna().unique())

    l1_cache: dict[int, pd.DataFrame] = {}
    t_grid = np.arange(t_window[0], t_window[1] + 1e-6, 1.0 / frame_rate)

    for cluster_id, df_cluster in df_mec_clustered.groupby("cluster"):
        v_stack = []
        a_stack = []
        co2_stack = []
        for rec_id in rec_ids:
            l1_cache.setdefault(rec_id, _load_l1_for_recording(int(rec_id)))
        for _, event_row in df_cluster.iterrows():
            rec_val = event_row.get("rec_id", event_row.get("recordingId", np.nan))
            if pd.isna(rec_val):
                continue
            rec_id = int(rec_val)
            df_l1 = l1_cache.get(rec_id)
            if df_l1 is None or df_l1.empty:
                continue
            ts = _extract_episode_timeseries(df_l1, event_row, frame_rate, t_window)
            if not ts:
                continue
            v_stack.append(ts["v"])
            a_stack.append(ts["a"])
            co2_stack.append(ts["co2"])

        if v_stack:
            v_mean = np.nanmean(np.vstack(v_stack), axis=0)
            a_mean = np.nanmean(np.vstack(a_stack), axis=0)
            co2_mean = np.nanmean(np.vstack(co2_stack), axis=0)
            centroids[cluster_id] = {
                "t": t_grid,
                "v_mean": v_mean,
                "a_mean": a_mean,
                "co2_mean": co2_mean,
            }
    return centroids


def plot_cluster_centroids(
    centroids: Dict[int, Dict[str, np.ndarray]],
    save_path: Path | None = None,
) -> None:
    """
    绘制 cluster-level 时间序列画像图：三行子图 (v/a/CO2)。
    """

    if not centroids:
        print("[behavior_profiling] No centroids to plot.")
        return

    if save_path is None:
        save_path = FIG_ROOT / "01_behavior_clusters_centroid.png"

    clusters = sorted(centroids.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 3)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for idx, cluster_id in enumerate(clusters):
        c = colors[idx % len(colors)]
        cent = centroids[cluster_id]
        t = cent["t"]
        axes[0].plot(t, cent["v_mean"], label=f"Cluster {cluster_id}", color=c)
        axes[1].plot(t, cent["a_mean"], label=f"Cluster {cluster_id}", color=c)
        axes[2].plot(t, cent["co2_mean"], label=f"Cluster {cluster_id}", color=c)

    axes[0].set_ylabel("Speed v [m/s]")
    axes[1].set_ylabel("Longitudinal a [m/s^2]")
    axes[2].set_ylabel("CO₂ rate [g/s]")
    axes[2].set_xlabel("Time aligned to min TTC [s]")

    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.6)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def summarize_clusters(df_clustered: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    if df_clustered.empty or "cluster" not in df_clustered.columns:
        summary = pd.DataFrame()
        summary.to_csv(save_path, index=False)
        return summary

    summary_rows = []
    for cluster_id, df_c in df_clustered.groupby("cluster"):
        summary_rows.append(
            {
                "cluster": cluster_id,
                "count": len(df_c),
                "min_TTC_conf_median": df_c.get("min_TTC_conf", pd.Series(dtype=float)).median(),
                "conf_duration_median": df_c.get("conf_duration", pd.Series(dtype=float)).median(),
                "MEC_CO2_per_km_median": df_c.get("MEC_CO2_per_km", pd.Series(dtype=float)).median(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_path, index=False)
    return summary_df
