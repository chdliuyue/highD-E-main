"""Project entrypoint for highD-E-main."""
from __future__ import annotations

import argparse

from data_preproc.preprocessing import parse_recording_ids_arg, run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="highD processing and experimentation entrypoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="build_data",
        help="Workflow mode: build_data | train | eval | analyze (train/eval/analyze are placeholders)",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated recording ids (e.g., '1,2,3') or 'all'. Overrides TEST_MODE defaults when provided.",
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

    if args.mode == "build_data":
        recording_ids = parse_recording_ids_arg(args.recordings)
        run_preprocessing(recording_ids=recording_ids, num_workers=args.num_workers)
    else:
        # TODO: add support for additional modes such as train, eval, and analyze
        raise ValueError(f"Unsupported mode '{args.mode}'. Available: build_data")


if __name__ == "__main__":
    main()
