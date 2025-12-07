"""Primary CLI entrypoint for highD L1 preprocessing."""
from __future__ import annotations

import argparse

from data_preproc.preprocessing import parse_recording_ids_arg, run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build L1 master tables with VT-CPFM power/fuel/CO2 estimation for highD recordings."
        )
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="all",
        help="Comma-separated recording ids (e.g., '01,02,03') or 'all'.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for the number of workers used during preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording_ids = parse_recording_ids_arg(args.recordings)
    run_preprocessing(recording_ids=recording_ids, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
