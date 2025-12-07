"""Standalone entrypoint for conflict–energy analysis figures and tables."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.exp_conflict_energy import run_experiment

TASK_CHOICES = {"ghost_car", "timeseries", "phase_plane", "mec", "clusters", "map", "all"}


def _parse_recordings(rec_arg: str) -> Sequence[int]:
    if rec_arg.lower() == "all":
        return list(range(1, 61))
    return [int(x) for x in rec_arg.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conflict–energy analysis entrypoint")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Subtask to run: ghost_car, timeseries, phase_plane, mec, clusters, map, or all.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="Recording ids (e.g., '01,02') or 'all'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for figure and table exports.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    task = args.task.lower()
    if task not in TASK_CHOICES:
        raise ValueError(f"Unknown task: {task}")

    recordings = _parse_recordings(args.recordings)
    run_experiment(task=task, recordings=recordings, output_root=args.output_root)


if __name__ == "__main__":
    main()
