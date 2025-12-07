"""Dispatcher for conflict–energy analysis experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.ghost_car import plot_ghost_car_validation, select_severe_conflicts, simulate_ghost_car
from analysis.timeseries_coupling import aggregate_timeseries_over_episodes, plot_ttc_co2_alignment_with_ci
from analysis.phase_plane import build_phase_plane_samples, plot_safety_energy_phase_plane
from analysis.mec_plots import add_severity_bins, build_mec_summary_table, load_mec_data, plot_mec_distributions
from analysis.behavior_clustering import (
    build_behavior_features,
    cluster_behaviors,
    plot_cluster_centroid_timeseries,
)
from analysis.map_visualization import plot_trajectories_on_map


def _load_l1(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/data/recording_{rec_id:02d}/L1_master_frame.parquet")
    return pd.read_parquet(path)


def _load_L2_conf(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/events/recording_{rec_id:02d}/L2_conflict_events.parquet")
    return pd.read_parquet(path)


def run_experiment(task: str, recordings: Sequence[int], output_root: Path | str = "output") -> None:
    """
    Dispatch experiment tasks for conflict–energy analysis.

    - "ghost_car": Ghost car validation plot.
    - "timeseries": TTC–CO2 alignment and lag statistics.
    - "phase_plane": Safety–energy phase plane visualization.
    - "mec": MEC distribution and summary table.
    - "clusters": Behavior clustering and centroid trajectories.
    - "map": Map-based trajectory visualization.
    - "all": Run all tasks sequentially.
    """
    task = task.lower()
    output_root = Path(output_root)
    fig_dir = output_root / "conflict_energy"
    mec_dir = output_root / "mec"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mec_dir.mkdir(parents=True, exist_ok=True)
    if task in {"ghost_car", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        severe = select_severe_conflicts(df_L2)
        if severe.empty:
            print("No severe conflicts found for ghost car validation.")
        else:
            episode = severe.iloc[0]
            idm_params = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.5, "b": 2.0}
            data = simulate_ghost_car(df_l1, episode, idm_params)
            save_path = fig_dir / f"ghost_car_validation_rec{rec_id:02d}_event{episode.name}.png"
            plot_ghost_car_validation(data, save_path)
            print(f"Ghost car validation saved to {save_path}")

    if task in {"timeseries", "all"}:
        agg = aggregate_timeseries_over_episodes(recordings)
        if agg["t_grid"].size == 0:
            print("No episodes available for time-series aggregation.")
        else:
            lags = agg.get("lags", np.array([]))
            lag_stats = {
                "mean": float(np.nanmean(lags)) if lags.size > 0 else np.nan,
                "median": float(np.nanmedian(lags)) if lags.size > 0 else np.nan,
                "q10": float(np.nanquantile(lags, 0.1)) if lags.size > 0 else np.nan,
                "q90": float(np.nanquantile(lags, 0.9)) if lags.size > 0 else np.nan,
            }
            print("Lag statistics:", lag_stats)
            plot_ttc_co2_alignment_with_ci(
                agg["t_grid"],
                agg["TTC_mean"],
                agg["TTC_lower"],
                agg["TTC_upper"],
                agg["CO2_mean"],
                agg["CO2_lower"],
                agg["CO2_upper"],
                lag_stats,
                save_path=fig_dir / "timeseries_coupling.png",
            )
            print("Saved TTC–CO2 coupling plot.")

    if task in {"phase_plane", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        samples = build_phase_plane_samples(df_l1, df_L2)
        plot_safety_energy_phase_plane(
            samples["TTC_all"],
            samples["energy_all"],
            samples["example_trajectories"],
            save_path=fig_dir / "phase_plane_hysteresis.png",
        )
        print("Saved safety–energy phase plane plot.")

    if task in {"mec", "all"}:
        try:
            df_mec = load_mec_data(path=mec_dir / "L2_conf_mec_baseline.parquet")
        except FileNotFoundError as exc:
            print(str(exc))
            df_mec = pd.DataFrame()
        if df_mec.empty:
            print("MEC data unavailable.")
        else:
            df_mec = add_severity_bins(df_mec)
            plot_mec_distributions(df_mec, save_path=fig_dir / "mec_distributions.png")
            summary = build_mec_summary_table(df_mec)
            out_path = mec_dir / "mec_summary.csv"
            summary.to_csv(out_path, index=False)
            print(summary)
            print(f"Saved MEC summary to {out_path}")

    if task in {"clusters", "all"}:
        try:
            df_mec = load_mec_data(path=mec_dir / "L2_conf_mec_baseline.parquet")
        except FileNotFoundError as exc:
            print(str(exc))
            df_mec = pd.DataFrame()
        if df_mec.empty:
            print("MEC data unavailable for clustering.")
        else:
            feature_df = build_behavior_features(df_mec)
            clustered = cluster_behaviors(feature_df)
            cluster_path = mec_dir / "L2_conf_mec_clusters.parquet"
            clustered.to_parquet(cluster_path, index=False)
            print(f"Saved cluster assignments to {cluster_path}")

            rec_subset = recordings[:3]
            df_l1_all = pd.concat([_load_l1(rid).assign(recordingId=rid) for rid in rec_subset], ignore_index=True)
            plot_cluster_centroid_timeseries(
                df_mec,
                df_l1_all,
                clustered,
                save_dir=fig_dir,
            )

    if task in {"map", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        plot_trajectories_on_map(
            df_l1,
            df_L2,
            rec_id=rec_id,
            save_path=fig_dir / f"map_trajectories_rec{rec_id:02d}.png",
        )
        print("Saved trajectory map plot.")
