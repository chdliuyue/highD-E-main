import argparse

from experiments.exp_mec_baseline import run_mec_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 04: Baseline-matched MEC analysis for high-interaction episodes."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="Comma-separated recording IDs or 'all'.",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = None
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    run_mec_baseline_experiment(recordings=rec_ids)


if __name__ == "__main__":
    main()
