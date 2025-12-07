"""Entry point for Stage 3 conflict–energy analysis."""
from __future__ import annotations

import argparse

from experiments.exp_conflict_energy import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 03: Conflict–Energy analysis (TTC–CO2, MEC, behavior)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="要运行的分析任务: ghost_car, timeseries, phase_plane, mec, clusters, map, 或 all.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="01",
        help="录制编号列表，逗号分隔，如 '01,02,03' 或 'all'.",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    run_experiment(task=args.task, recordings=rec_ids)


if __name__ == "__main__":
    main()
