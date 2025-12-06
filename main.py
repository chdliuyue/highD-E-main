"""Project entrypoint for highD-E-main."""
from __future__ import annotations

import argparse

import preprocess_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="highD processing and experimentation entrypoint")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "build_data":
        preprocess_main.main_build_data(args.recordings)
    else:
        # TODO: add support for additional modes such as train, eval, and analyze
        raise ValueError(f"Unsupported mode '{args.mode}'. Available: build_data")


if __name__ == "__main__":
    main()
