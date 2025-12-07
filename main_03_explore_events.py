import argparse
from experiments.exp_explore_events import run_explore_experiment


def main() -> None:
    print("[Stage 03] Exploring L1/L2 data base...")
    parser = argparse.ArgumentParser(
        description="Stage 3: Explore L1/L2 event-level data base."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="Recordings to explore, comma-separated like '01,02,03' or 'all'.",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    for i, rec_id in enumerate(rec_ids, start=1):
        print(f"  [Stage 03] Processing recording {rec_id:02d} ({i}/{len(rec_ids)})")

    run_explore_experiment(rec_ids)
    print("[Stage 03] Exploration finished.")


if __name__ == "__main__":
    main()
