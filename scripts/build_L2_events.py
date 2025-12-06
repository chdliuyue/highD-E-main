"""Build L2 high-interaction (conflict) and baseline events for recordings."""
from __future__ import annotations

import argparse
from typing import Iterable

from config import FRAME_RATE_DEFAULT, PROJECT_ROOT
from data_preproc import events


L1_DIR = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
EVENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"


def _iter_recordings(recordings: Iterable[int]) -> Iterable[int]:
    for rec_id in recordings:
        yield int(rec_id)


def build_for_recording(rec_id: int, frame_rate: float = FRAME_RATE_DEFAULT) -> None:
    """Build and save conflict/baseline events for a single recording id."""

    l1_path = L1_DIR / f"recording_{rec_id:02d}" / "L1_master_frame.parquet"
    out_dir = EVENTS_ROOT / f"recording_{rec_id:02d}"

    events.build_events_for_recording(
        rec_id=rec_id,
        l1_path=l1_path,
        events_dir=out_dir,
        frame_rate=frame_rate,
    )

    print(f"Recording {rec_id:02d}: wrote L2 conflict/baseline events to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L2 events for specified recordings")
    parser.add_argument(
        "--recordings",
        type=str,
        default="1",
        help="Comma-separated recording ids to process (e.g., '1,2,3')",
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=FRAME_RATE_DEFAULT,
        help="Frame rate to use for time conversion (Hz).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec_ids = [int(r.strip()) for r in args.recordings.split(",") if r.strip()]
    for rec_id in _iter_recordings(rec_ids):
        build_for_recording(rec_id, frame_rate=args.frame_rate)


if __name__ == "__main__":
    main()
