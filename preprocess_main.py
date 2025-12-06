"""Data preprocessing entrypoint for highD-E-main."""
from __future__ import annotations

import argparse
from typing import List, Optional

import config
from data_preproc.builder import HighDDataBuilder


def parse_recordings(arg: Optional[str]) -> List[int]:
    """Parse the recordings argument from CLI."""
    if arg is None:
        return []
    arg = arg.strip().lower()
    if arg == "all":
        return list(range(1, config.MAX_RECORDING_ID + 1))
    if not arg:
        return []
    return [int(x) for x in arg.split(",") if x.strip()]


def main_build_data(recordings: Optional[str] = None) -> None:
    """Run the build_data workflow for selected recordings."""
    cli_recs = parse_recordings(recordings)
    if cli_recs:
        recording_ids = cli_recs
    elif config.TEST_MODE:
        print(f"TEST MODE ON. Only processing recordings: {config.TEST_RECORDINGS}")
        recording_ids = config.TEST_RECORDINGS
    else:
        recording_ids = list(range(1, config.MAX_RECORDING_ID + 1))

    builder = HighDDataBuilder(
        raw_data_dir=config.RAW_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
        num_workers=config.NUM_WORKERS,
    )
    builder.process_all_recordings(recording_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="highD preprocessing entrypoint")
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated recording ids (e.g., '1,2,3') or 'all'. Overrides TEST_MODE defaults when provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_build_data(args.recordings)


if __name__ == "__main__":
    main()

