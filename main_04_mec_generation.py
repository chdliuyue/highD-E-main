import argparse

from experiments.exp_mec_generation import run_mec_generation

def main() -> None:
    print("[Stage 04] Generating MEC data...")
    parser = argparse.ArgumentParser(
        description="Stage 4: Generate MEC data for conflict episodes (baseline matching)."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="录制编号，如 '01,02,03' 或 'all'. 默认 all 表示 1..60.",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    for i, rec_id in enumerate(rec_ids, start=1):
        print(f"  [Stage 04] Processing recording {rec_id:02d} ({i}/{len(rec_ids)})")

    run_mec_generation(rec_ids)
    print("[Stage 04] MEC generation finished.")


if __name__ == "__main__":
    main()
