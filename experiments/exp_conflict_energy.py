"""Dispatcher for conflict–energy analysis experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.ghost_car import plot_ghost_car_validation, select_severe_conflicts, simulate_ghost_car
from analysis.timeseries_coupling import (
    aggregate_timeseries_with_stats,
    compute_phase_co2_shares,
    plot_ttc_co2_alignment_with_ci,
)
from analysis.phase_plane import build_phase_plane_samples, plot_safety_energy_phase_plane
from analysis.mec_plots import (
    add_severity_bins,
    build_mec_summary_table,
    compute_mec_per_km,
    load_mec_data,
    plot_mec_distributions,
)
from analysis.behavior_clustering import (
    build_behavior_features,
    cluster_behaviors,
    plot_cluster_centroid_timeseries,
)
from analysis.map_visualization import plot_trajectories_on_map
from analysis.mec_baseline import build_and_save_mec, DEFAULT_MEC_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = PROJECT_ROOT / "output" / "conflict_energy" / "figs"
TABLE_ROOT = PROJECT_ROOT / "output" / "conflict_energy" / "tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def _load_l1(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/data/recording_{rec_id:02d}/L1_master_frame.parquet")
    return pd.read_parquet(path)


def _load_L2_conf(rec_id: int) -> pd.DataFrame:
    path = Path(f"data/processed/highD/events/recording_{rec_id:02d}/L2_conflict_events.parquet")
    return pd.read_parquet(path)


def _resolve_output_roots(output_root: Path | str) -> tuple[Path, Path]:
    base = Path(output_root)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    fig_dir = base / "conflict_energy" / "figs"
    table_dir = base / "conflict_energy" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def run_experiment(task: str, recordings: Sequence[int], output_root: Path | str = PROJECT_ROOT / "output") -> None:
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
    fig_dir, table_dir = _resolve_output_roots(output_root)
    if task in {"ghost_car", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        severe = select_severe_conflicts(df_L2)
        if severe.empty:
            print("No severe conflicts found for ghost car validation.")
        else:
            # Pick the most severe episode (lowest TTC) to emphasize contrast
            episode = severe.sort_values("min_TTC_conf").iloc[0]
            idm_params = {"v0": 32.0, "T": 1.6, "s0": 2.5, "a_max": 1.2, "b_comf": 3.0}
            data = simulate_ghost_car(df_l1, episode, idm_params)
            save_path = fig_dir / f"01_ghost_car_validation_rec{rec_id:02d}_event{episode.name}.png"
            plot_ghost_car_validation(data, save_path)
            print(f"Ghost car validation saved to {save_path}")

    if task in {"timeseries", "all"}:
        agg, lag_stats = aggregate_timeseries_with_stats(recordings)
        if agg["t_grid"].size == 0:
            print("No episodes available for time-series aggregation.")
        else:
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
                save_path=fig_dir / "02_timeseries_alignment.png",
            )
            print("Saved TTC–CO2 coupling plot.")

            phase_df = compute_phase_co2_shares(
                agg.get("co2_stack", np.array([])), agg["t_grid"], agg.get("min_ttc_list")
            )
            if not phase_df.empty:
                bins = [(-np.inf, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, np.inf)]
                labels = ["<2", "2-3", "3-4", ">=4"]
                phase_df["severity_bin"] = pd.cut(
                    phase_df["min_TTC_conf"], bins=[b[0] for b in bins] + [bins[-1][1]], labels=labels
                )
                table_rows = []
                for label in labels:
                    subset = phase_df[phase_df["severity_bin"] == label]
                    table_rows.append(
                        {
                            "severity_bin": label,
                            "mean_share_pre": subset["share_pre"].mean(),
                            "mean_share_core": subset["share_core"].mean(),
                            "mean_share_rec": subset["share_rec"].mean(),
                            "median_share_rec": subset["share_rec"].median(),
                            "n_events": len(subset),
                        }
                    )
                phase_summary = pd.DataFrame(table_rows)
                out_path = table_dir / "table02_phase_co2_shares.csv"
                phase_summary.to_csv(out_path, index=False)
                print("Phase CO2 shares (mean):\n", phase_summary)
                print(f"Saved phase share table to {out_path}")

    if task in {"phase_plane", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        samples = build_phase_plane_samples(df_l1, df_L2)
        plot_safety_energy_phase_plane(
            samples["TTC_all"],
            samples["energy_all"],
            samples["example_trajectories"],
            save_path=fig_dir / "03_phase_plane_hysteresis.png",
        )
        print("Saved safety–energy phase plane plot.")

    if task in {"mec", "all"}:
        default_path = DEFAULT_MEC_PATH
        if default_path.exists():
            df_mec = load_mec_data(path=default_path)
        else:
            df_mec = build_and_save_mec(recordings, output_path=default_path)
        if df_mec.empty:
            print("MEC data unavailable.")
        else:
            df_mec = compute_mec_per_km(add_severity_bins(df_mec))
            plot_mec_distributions(df_mec, save_path=fig_dir / "04_mec_distributions.png")
            summary = build_mec_summary_table(df_mec)
            out_path = table_dir / "table01_mec_summary.csv"
            summary.to_csv(out_path, index=False)
            print(summary)
            print(f"Saved MEC summary to {out_path}")

    if task in {"clusters", "all"}:
        try:
            df_mec = load_mec_data(path=DEFAULT_MEC_PATH)
        except FileNotFoundError as exc:
            print(str(exc))
            df_mec = pd.DataFrame()
        if df_mec.empty:
            print("MEC data unavailable for clustering.")
        else:
            df_mec = compute_mec_per_km(df_mec)
            feature_df = build_behavior_features(df_mec)
            clustered = cluster_behaviors(feature_df)
            cluster_path = PROJECT_ROOT / "data" / "analysis" / "L2_conf_mec_clusters.parquet"
            cluster_path.parent.mkdir(parents=True, exist_ok=True)
            clustered.to_parquet(cluster_path, index=False)
            print(f"Saved cluster assignments to {cluster_path}")

            for cid, group in clustered.groupby("cluster"):
                stats = {
                    "n": len(group),
                    "median_min_TTC_conf": group["min_TTC_conf"].median(),
                    "median_conf_duration": group["conf_duration"].median(),
                    "median_a_min": group["a_min"].median(),
                    "median_MEC_CO2_per_km": group.get("MEC_CO2_per_km", pd.Series(dtype=float)).median(),
                }
                print(f"Cluster {cid} stats: {stats}")

            rec_subset = recordings[:3]
            df_l1_all = pd.concat([_load_l1(rid).assign(recordingId=rid) for rid in rec_subset], ignore_index=True)
            plot_cluster_centroid_timeseries(
                df_mec,
                df_l1_all,
                clustered,
                save_path=fig_dir / "05_behavior_clusters.png",
            )

    if task in {"map", "all"}:
        rec_id = recordings[0]
        df_l1 = _load_l1(rec_id)
        df_L2 = _load_L2_conf(rec_id)
        plot_trajectories_on_map(
            df_l1,
            df_L2,
            rec_id=rec_id,
            save_path=fig_dir / f"06_map_trajectories_rec{rec_id:02d}.png",
        )
        print("Saved trajectory map plot.")
