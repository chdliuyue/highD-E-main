"""Utilities for running the highD L1 preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import config
from data_preproc.l1_builder import L1Builder


def parse_recording_ids_arg(arg: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated recordings argument from the CLI."""

    if arg is None:
        return None

    cleaned = arg.strip()
    if not cleaned:
        return None

    if cleaned.lower() == "all":
        return list(range(1, config.MAX_RECORDING_ID + 1))

    return [int(value) for value in cleaned.split(",") if value.strip()]


def _resolve_recording_ids(recording_ids: Optional[Sequence[int]]) -> List[int]:
    """Determine which recordings to process based on overrides and config."""

    if recording_ids:
        return list(recording_ids)

    if config.TEST_MODE:
        print(f"TEST MODE ON. Only processing recordings: {config.TEST_RECORDINGS}")
        return list(config.TEST_RECORDINGS)

    return list(range(1, config.MAX_RECORDING_ID + 1))


def run_preprocessing(
    recording_ids: Optional[Sequence[int]] = None, num_workers: Optional[int] = None
) -> None:
    """Build master tables and derived L1 artifacts for the selected recordings."""

    resolved_ids = _resolve_recording_ids(recording_ids)
    workers = config.NUM_WORKERS if num_workers is None else num_workers

    raw_data_dir = Path(config.RAW_DATA_DIR)
    output_dir = Path(config.PROCESSED_DATA_DIR)

    builder = L1Builder(raw_data_dir=raw_data_dir, processed_data_dir=output_dir, num_workers=workers)
    builder.build_many(resolved_ids, num_workers=workers)
