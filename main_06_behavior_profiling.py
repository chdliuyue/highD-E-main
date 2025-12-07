from __future__ import annotations

import argparse

from experiments.exp_behavior_profiling import run_behavior_profiling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 06: Behavioral profiling and time-series portraits based on MEC."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="要用于时序画像的 recordings, 如 '01,02,03' 或 'all' (默认).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="聚类数目 (默认为 3).",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    run_behavior_profiling(rec_ids=rec_ids, n_clusters=args.n_clusters)


if __name__ == "__main__":
    main()
