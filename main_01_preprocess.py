"""Convenience wrapper for L1 preprocessing entry point."""
from __future__ import annotations

import argparse

import preprocess_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L1 preprocessing wrapper")
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated recording ids (e.g., '01,02,03') or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_main.main_build_data(args.recordings)


if __name__ == "__main__":
    main()
