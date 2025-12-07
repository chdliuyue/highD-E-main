"""Unified entrypoint for highD-E analysis experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.analysis_config import (
    AnalysisCliDefaults,
    AnalysisRunConfig,
    CONFLICT_TASK_CHOICES,
    config_from_args,
)
from experiments.exp_behavior_profiling import run_behavior_profiling_experiment
from experiments.exp_conflict_energy import run_experiment as run_conflict_energy
from experiments.exp_mec_baseline import run_mec_baseline_experiment
from experiments.exp_timeseries_coupling import run_timeseries_coupling_experiment


def _build_parser() -> argparse.ArgumentParser:
    defaults = AnalysisCliDefaults()
    parser = argparse.ArgumentParser(
        description="Unified Stage 3-5 analysis entrypoint (conflict, coupling, MEC, behavior)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=defaults.tasks,
        help=(
            "Comma-separated tasks to run: conflict_energy, timeseries_coupling, "
            "mec_baseline, behavior_profiling, or all."
        ),
    )
    parser.add_argument(
        "--conflict-task",
        type=str,
        default=defaults.conflict_task,
        choices=sorted(CONFLICT_TASK_CHOICES),
        help=(
            "Conflict-energy subtask: ghost_car, timeseries, phase_plane, mec, "
            "clusters, map, or all."
        ),
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default=defaults.recordings,
        help="Recording ids (e.g., '01,02') or 'all'. Applies to relevant tasks.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=defaults.frame_rate,
        help="Frame rate for time-series coupling analysis.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=defaults.max_episodes,
        help="Maximum episodes to aggregate for time-series coupling.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults.output_root,
        help="Root directory for all analysis artifacts (figures, tables, parquet files).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_config: AnalysisRunConfig = config_from_args(args)

    for task in run_config.tasks:
        if task == "conflict_energy":
            run_conflict_energy(
                task=run_config.conflict_task,
                recordings=run_config.recordings,
                output_root=run_config.output_root,
            )
        elif task == "timeseries_coupling":
            run_timeseries_coupling_experiment(
                recordings=run_config.recordings,
                frame_rate=run_config.frame_rate,
                max_episodes=run_config.max_episodes,
                output_root=run_config.output_root,
            )
        elif task == "mec_baseline":
            run_mec_baseline_experiment(
                recordings=run_config.mec_recordings,
                output_root=run_config.output_root,
            )
        elif task == "behavior_profiling":
            run_behavior_profiling_experiment(output_root=run_config.output_root)


if __name__ == "__main__":
    main()
