from pathlib import Path

import pandas as pd

from analysis.behavior_profiling import (
    build_feature_matrix,
    load_conflict_mec,
    simple_behavior_clustering,
)


def run_behavior_profiling_experiment(output_root: Path | str = "output") -> None:
    """
    行为谱系实验主入口。
    """

    df_mec = load_conflict_mec(base_dir=Path(output_root) / "mec")
    if df_mec.empty:
        print("No MEC data available. Please run MEC experiment first.")
        return

    features_df = build_feature_matrix(df_mec)
    clustered_df = simple_behavior_clustering(features_df)

    if clustered_df.empty:
        print("No features available for clustering.")
        return

    cluster_stats = clustered_df.groupby("cluster").agg(
        size=("event_id", "count"),
        min_TTC_conf_mean=("min_TTC_conf", "mean"),
        conf_duration_mean=("conf_duration", "mean"),
        MEC_CO2_per_s_median=("MEC_CO2_per_s", "median"),
    )
    print("Cluster stats:")
    print(cluster_stats)

    out_root = Path(output_root) / "behavior_profiling"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "L2_conf_mec_behavior_clusters.parquet"
    clustered_df.to_parquet(out_path, index=False)
    print(f"Saved clustered dataframe to {out_path}")
