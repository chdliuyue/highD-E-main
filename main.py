"""Project entrypoint for highD-E-main."""
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


def main() -> None:
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
    args = parser.parse_args()

    if args.mode == "build_data":
        cli_recs = parse_recordings(args.recordings)
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
    else:
        raise ValueError(f"Unsupported mode '{args.mode}'. Available: build_data")


if __name__ == "__main__":
    main()
