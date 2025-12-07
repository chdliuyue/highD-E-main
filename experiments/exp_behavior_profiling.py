from __future__ import annotations

from pathlib import Path
from typing import Sequence

from analysis.behavior_profiling import (
    cluster_events,
    compute_cluster_centroid_timeseries,
    filter_events_for_clustering,
    load_mec_events,
    plot_cluster_centroids,
    summarize_clusters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = PROJECT_ROOT / "output" / "behavior" / "figs"
TABLE_ROOT = PROJECT_ROOT / "output" / "behavior" / "tables"
ANALYSIS_OUT_PATH = PROJECT_ROOT / "data" / "analysis" / "L2_conf_mec_clusters.parquet"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def run_behavior_profiling(
    rec_ids: Sequence[int] | None = None,
    n_clusters: int = 3,
) -> None:
    """
    驾驶行为谱系阶段的统一入口。

    步骤：
      1) 读取并筛选 MEC 事件；
      2) 对事件进行 KMeans 聚类；
      3) 输出 cluster 摘要、时序画像以及带标签的 parquet。
    """

    df_mec = load_mec_events()
    if df_mec.empty:
        print("[behavior_profiling] MEC data is empty; aborting.")
        return

    df_filtered = filter_events_for_clustering(df_mec)
    if df_filtered.empty:
        print("[behavior_profiling] No events remain after filtering.")
        return

    df_clustered = cluster_events(df_filtered, n_clusters=n_clusters)
    df_clustered.to_parquet(ANALYSIS_OUT_PATH, index=False)

    summary_path = TABLE_ROOT / "cluster_summary.csv"
    summary_df = summarize_clusters(df_clustered, summary_path)
    if not summary_df.empty:
        print("Cluster summary:")
        print(summary_df)

    centroids = compute_cluster_centroid_timeseries(df_clustered, rec_ids=rec_ids)
    plot_cluster_centroids(centroids, save_path=FIG_ROOT / "01_behavior_clusters_centroid.png")

    print(f"[behavior_profiling] Clustered MEC events saved to: {ANALYSIS_OUT_PATH}")
    print(f"[behavior_profiling] Cluster summary saved to: {summary_path}")


if __name__ == "__main__":
    run_behavior_profiling()
