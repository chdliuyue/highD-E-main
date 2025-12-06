"""Entry point for constructing L2 events across recordings."""
from __future__ import annotations

import argparse
from typing import List, Optional

import config
from scripts.build_L2_events import build_for_recording


def parse_recordings(arg: str | None) -> Optional[List[int]]:
    """Parse recording list argument such as "01,02" or "all"."""

    if arg is None:
        return None

    cleaned = arg.strip()
    if not cleaned:
        return None

    if cleaned.lower() == "all":
        return list(range(1, config.MAX_RECORDING_ID + 1))

    return [int(x) for x in cleaned.split(",") if x.strip()]


def resolve_recordings(recording_ids: Optional[List[int]]) -> List[int]:
    """Determine which recordings to process based on overrides and config."""

    if recording_ids:
        return list(recording_ids)

    if config.TEST_MODE:
        print(f"TEST MODE ON. Only processing recordings: {config.TEST_RECORDINGS}")
        return list(config.TEST_RECORDINGS)

    return list(range(1, config.MAX_RECORDING_ID + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct L2 conflict and baseline events using VT-CPFM-derived emissions "
            "from the L1 master tables."
        )
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated recording ids (e.g., '01,02,03') or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec_ids = resolve_recordings(parse_recordings(args.recordings))

    for rec_id in rec_ids:
        build_for_recording(int(rec_id))


if __name__ == "__main__":
    main()
