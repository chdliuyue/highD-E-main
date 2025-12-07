from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "output" / "mec"


def load_conflict_mec(base_dir: Path | None = None) -> pd.DataFrame:
    """
    读取在阶段 2 中生成的 MEC 数据。
    """

    root = base_dir or ANALYSIS_ROOT
    parquet_path = root / "L2_conf_mec_baseline.parquet"
    csv_path = root / "L2_conf_mec_baseline.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def build_feature_matrix(df_mec: pd.DataFrame) -> pd.DataFrame:
    """
    从 MEC DataFrame 中构建行为特征向量 DataFrame。
    """

    if df_mec.empty:
        return pd.DataFrame()

    features = pd.DataFrame()
    features["event_id"] = df_mec.get("event_id")
    features["rec_id"] = df_mec.get("rec_id", df_mec.get("recordingId"))
    features["min_TTC_conf"] = df_mec.get("min_TTC_conf")
    features["conf_duration"] = df_mec.get("conf_duration", df_mec.get("duration"))
    features["a_min"] = df_mec.get("a_min", np.nan)
    features["a_max"] = df_mec.get("a_max", np.nan)
    features["v_drop"] = df_mec.get("v_drop", np.nan)
    features["v_recovery"] = df_mec.get("v_recovery", np.nan)
    features["MEC_CO2_per_s"] = df_mec.get("MEC_CO2_per_s")

    return features


def _initialize_centers(data: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    indices = rng.choice(data.shape[0], size=n_clusters, replace=False)
    return data[indices]


def simple_behavior_clustering(
    features_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    使用简单的 k-means（基于 numpy）在特征空间中做聚类。
    """

    if features_df.empty:
        return features_df

    feature_cols = [
        col
        for col in [
            "min_TTC_conf",
            "conf_duration",
            "a_min",
            "a_max",
            "v_drop",
            "v_recovery",
            "MEC_CO2_per_s",
        ]
        if col in features_df.columns
    ]
    if not feature_cols:
        return features_df
    data = features_df[feature_cols].to_numpy(dtype=float)
    n_samples = data.shape[0]
    n_clusters = min(n_clusters, n_samples) if n_samples > 0 else 0
    if n_clusters == 0:
        return features_df
    col_means = np.nanmean(data, axis=0)
    col_stds = np.nanstd(data, axis=0)
    col_stds[col_stds == 0] = 1.0
    data_std = (data - col_means) / col_stds
    data_std = np.nan_to_num(data_std)

    centers = _initialize_centers(data_std, n_clusters, random_state)
    for _ in range(10):
        distances = np.linalg.norm(data_std[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centers = np.array(
            [
                data_std[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ]
        )
        centers = new_centers

    result = features_df.copy()
    result["cluster"] = labels
    return result
