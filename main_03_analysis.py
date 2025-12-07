"""Unified entrypoint for highD-E analysis experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from experiments.exp_behavior_profiling import run_behavior_profiling_experiment
from experiments.exp_conflict_energy import run_experiment as run_conflict_energy
from experiments.exp_mec_baseline import run_mec_baseline_experiment
from experiments.exp_timeseries_coupling import run_timeseries_coupling_experiment

TASK_CHOICES = {
    "conflict_energy",
    "timeseries_coupling",
    "mec_baseline",
    "behavior_profiling",
}

CONFLICT_TASK_CHOICES = {
    "ghost_car",
    "timeseries",
    "phase_plane",
    "mec",
    "clusters",
    "map",
    "all",
}


def _parse_recording_list(rec_arg: str) -> list[int]:
    if rec_arg.lower() == "all":
        return list(range(1, 61))
    return [int(x) for x in rec_arg.split(",") if x.strip()]


def _parse_optional_recordings(rec_arg: str) -> list[int] | None:
    if rec_arg.lower() == "all":
        return None
    return [int(x) for x in rec_arg.split(",") if x.strip()]


def _parse_tasks(task_arg: str) -> List[str]:
    requested = {t.strip().lower() for t in task_arg.split(",") if t.strip()}
    if not requested or "all" in requested:
        return sorted(TASK_CHOICES)
    unknown = requested.difference(TASK_CHOICES)
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown))}")
    return sorted(requested)


def _parse_conflict_task(task: str) -> str:
    task = task.lower()
    if task not in CONFLICT_TASK_CHOICES:
        raise ValueError(
            "Unknown conflict-energy subtask: "
            f"{task}. Choose from {', '.join(sorted(CONFLICT_TASK_CHOICES))}."
        )
    return task


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Stage 3-5 analysis entrypoint (conflict, coupling, MEC, behavior)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated tasks to run: conflict_energy, timeseries_coupling, "
            "mec_baseline, behavior_profiling, or all."
        ),
    )
    parser.add_argument(
        "--conflict-task",
        type=str,
        default="all",
        choices=sorted(CONFLICT_TASK_CHOICES),
        help=(
            "Conflict-energy subtask: ghost_car, timeseries, phase_plane, mec, "
            "clusters, map, or all."
        ),
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="Recording ids (e.g., '01,02') or 'all'. Applies to relevant tasks.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate for time-series coupling analysis.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum episodes to aggregate for time-series coupling.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for all analysis artifacts (figures, tables, parquet files).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    recordings_list = _parse_recording_list(args.recordings)
    conflict_task = _parse_conflict_task(args.conflict_task)
    mec_recordings: Sequence[int] | None = _parse_optional_recordings(args.recordings)

    for task in tasks:
        if task == "conflict_energy":
            run_conflict_energy(
                task=conflict_task,
                recordings=recordings_list,
                output_root=args.output_root,
            )
        elif task == "timeseries_coupling":
            run_timeseries_coupling_experiment(
                recordings=recordings_list,
                frame_rate=args.frame_rate,
                max_episodes=args.max_episodes,
                output_root=args.output_root,
            )
        elif task == "mec_baseline":
            run_mec_baseline_experiment(
                recordings=mec_recordings,
                output_root=args.output_root,
            )
        elif task == "behavior_profiling":
            run_behavior_profiling_experiment(output_root=args.output_root)


if __name__ == "__main__":
    main()
