"""Behavior clustering based on conflict MEC features."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_behavior_features(df_mec: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for behavioral clustering from MEC data.

    Features include min TTC, conflict duration, acceleration extremes,
    velocity drop, and MEC metrics when available. Missing columns are
    filled with NaN and later imputed.
    """
    feature_cols = [
        "min_TTC_conf",
        "conf_duration",
        "a_min",
        "a_max",
        "v_drop",
        "MEC_CO2_per_km",
    ]

    df = df_mec.copy()
    for col in feature_cols:
        if col not in df:
            df[col] = np.nan

    df_features = df[["recordingId", "ego_id"] + feature_cols].copy()

    for col in ["a_min", "a_max", "v_drop", "MEC_CO2_per_km"]:
        if df_features[col].isna().all():
            df_features[col] = 0.0
        else:
            df_features[col].fillna(df_features[col].median(), inplace=True)

    return df_features


def cluster_behaviors(features_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 0) -> pd.DataFrame:
    """Cluster episodes using KMeans on standardized features."""
    feature_cols = [col for col in features_df.columns if col not in {"recordingId", "ego_id"}]
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)
    features_df = features_df.copy()
    features_df["cluster"] = labels
    return features_df


def _extract_episode_timeseries(
    df_l1_all: pd.DataFrame,
    rec_id: int,
    ego_id: int,
    start_frame: int,
    end_frame: int,
    frame_rate: float,
    t_window: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to extract aligned timeseries for a single episode."""
    df_episode = df_l1_all[
        (df_l1_all["recordingId"] == rec_id)
        & (df_l1_all["trackId"] == ego_id)
        & (df_l1_all["frame"] >= start_frame)
        & (df_l1_all["frame"] <= end_frame)
    ]
    if df_episode.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    t0 = df_episode.loc[df_episode["TTC"].idxmin(), "frame"]
    rel_t = (df_episode["frame"] - t0) / frame_rate
    t_grid = np.arange(t_window[0], t_window[1] + 1.0 / frame_rate, 1.0 / frame_rate)

    v_series = df_episode.get("v_long_smooth", df_episode.get("v_x", pd.Series(dtype=float)))
    a_series = df_episode.get("a_long_smooth", df_episode.get("ax", pd.Series(dtype=float)))
    co2_series = df_episode.get("cpf_co2_rate_gps", df_episode.get("co2_rate", pd.Series(dtype=float)))

    v_interp = np.interp(t_grid, rel_t, v_series, left=np.nan, right=np.nan)
    a_interp = np.interp(t_grid, rel_t, a_series, left=np.nan, right=np.nan)
    co2_interp = np.interp(t_grid, rel_t, co2_series, left=np.nan, right=np.nan)
    return t_grid, v_interp, a_interp, co2_interp


def plot_cluster_centroid_timeseries(
    df_mec: pd.DataFrame,
    df_l1_all: pd.DataFrame,
    features_with_cluster: pd.DataFrame,
    frame_rate: float = 25.0,
    t_window: tuple[float, float] = (-5.0, 10.0),
    save_dir: Path | None = None,
) -> None:
    """
    Plot cluster-average velocity, acceleration, and CO2 rate time series.

    Args:
        df_mec: MEC dataframe with episode metadata.
        df_l1_all: Concatenated L1 dataframe for selected recordings.
        features_with_cluster: Feature dataframe with ``cluster`` labels.
        frame_rate: Frame rate (Hz).
        t_window: Time window around min TTC for alignment.
        save_dir: Output directory.
    """
    save_dir = save_dir or Path("figs")
    save_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, feats in features_with_cluster.groupby("cluster"):
        t_list: list[np.ndarray] = []
        v_list: list[np.ndarray] = []
        a_list: list[np.ndarray] = []
        co2_list: list[np.ndarray] = []

        for _, row in feats.iterrows():
            meta = df_mec[
                (df_mec.get("recordingId") == row["recordingId"]) & (df_mec.get("ego_id") == row["ego_id"])
            ]
            if meta.empty:
                continue
            meta_row = meta.iloc[0]
            t, v, a, co2 = _extract_episode_timeseries(
                df_l1_all,
                rec_id=int(meta_row.get("recordingId", 0)),
                ego_id=int(meta_row.get("ego_id", 0)),
                start_frame=int(meta_row.get("start_frame", meta_row.get("conf_start_frame", 0))),
                end_frame=int(meta_row.get("end_frame", meta_row.get("conf_end_frame", 0))),
                frame_rate=frame_rate,
                t_window=t_window,
            )
            if t.size == 0:
                continue
            t_list.append(t)
            v_list.append(v)
            a_list.append(a)
            co2_list.append(co2)

        if not t_list:
            continue

        t_ref = t_list[0]
        v_mean = np.nanmean(np.vstack(v_list), axis=0)
        a_mean = np.nanmean(np.vstack(a_list), axis=0)
        co2_mean = np.nanmean(np.vstack(co2_list), axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t_ref, v_mean, color="tab:blue")
        axes[0].set_ylabel("v [m/s]")
        axes[0].set_title(f"Cluster {cluster_id} centroid dynamics")

        axes[1].plot(t_ref, a_mean, color="tab:orange")
        axes[1].set_ylabel("a [m/s²]")

        axes[2].plot(t_ref, co2_mean, color="tab:red")
        axes[2].set_ylabel("CO2 [g/s]")
        axes[2].set_xlabel("Time aligned to min TTC [s]")

        fig.tight_layout()
        fig.savefig(save_dir / f"behavior_clusters_cluster{cluster_id}.png", dpi=200)
        plt.close(fig)
