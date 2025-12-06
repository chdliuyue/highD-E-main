"""Entry point for constructing L2 events across recordings."""
from __future__ import annotations

import argparse
from typing import List

import config
from scripts.build_L2_events import build_for_recording


def parse_recordings(arg: str | None) -> List[int]:
    """Parse recording list argument such as "01,02" or "all"."""

    if arg is None:
        return []
    arg = arg.strip().lower()
    if arg == "all":
        return list(range(1, config.MAX_RECORDING_ID + 1))
    return [int(x) for x in arg.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L2 event construction entrypoint")
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated recording ids (e.g., '01,02,03') or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec_ids = parse_recordings(args.recordings)
    if not rec_ids:
        rec_ids = config.TEST_RECORDINGS if config.TEST_MODE else [1]

    for rec_id in rec_ids:
        build_for_recording(int(rec_id))


if __name__ == "__main__":
    main()
