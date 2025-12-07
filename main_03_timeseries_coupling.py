import argparse

from experiments.exp_timeseries_coupling import run_timeseries_coupling_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 03: TTC–CO2 spatio-temporal coupling analysis."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="01",
        help="Comma-separated recording IDs (e.g., '01,02,03') or 'all'.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate in Hz (default 25 for highD).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Max number of episodes to sample for aggregation.",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    run_timeseries_coupling_experiment(
        recordings=rec_ids,
        frame_rate=args.frame_rate,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
