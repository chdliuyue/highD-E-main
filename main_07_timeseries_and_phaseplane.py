import argparse

from experiments.exp_timeseries_phaseplane import run_timeseries_and_phaseplane


def main() -> None:
    print("[Stage 07] TTC–CO2 time-series alignment & Safety–Energy phase plane...")
    parser = argparse.ArgumentParser(
        description="Stage 07: TTC–CO2 time-series alignment and Safety–Energy phase plane analysis."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="要用于分析的 recordings, 如 '01,02,03' 或 'all' (默认).",
    )
    args = parser.parse_args()

    if args.recordings.lower() == "all":
        rec_ids = list(range(1, 61))
    else:
        rec_ids = [int(x) for x in args.recordings.split(",") if x.strip()]

    for i, rec_id in enumerate(rec_ids, start=1):
        print(f"  [Stage 07] Will process recording {rec_id:02d} ({i}/{len(rec_ids)})")

    print(f"[Stage 07] Running time-series alignment and phase-plane for rec_ids={rec_ids}")
    run_timeseries_and_phaseplane(rec_ids)
    print("[Stage 07] Stage finished.")


if __name__ == "__main__":
    main()
