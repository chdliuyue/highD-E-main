"""Build L2 high-interaction (conflict) and baseline events for recordings."""
from __future__ import annotations

import argparse

import pandas as pd

from config import (
    FRAME_RATE_DEFAULT,
    PROJECT_ROOT,
)
from data_preproc.events import build_baseline_events, build_conflict_events


def build_for_recording(rec_id: int, frame_rate: float = FRAME_RATE_DEFAULT) -> None:
    """Build L2 events for a single recording id using the high-interaction definition."""

    l1_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "highD"
        / "data"
        / f"recording_{rec_id:02d}"
        / "L1_master_frame.parquet"
    )
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 master frame not found for recording {rec_id:02d} at {l1_path}")

    df_l1 = pd.read_parquet(l1_path)
    df_rec = df_l1[df_l1["recordingId"] == rec_id].copy()

    df_conf = build_conflict_events(df_rec, frame_rate)
    df_base = build_baseline_events(df_rec, frame_rate)

    out_dir = PROJECT_ROOT / "data" / "processed" / "highD" / "events" / f"recording_{rec_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_conf.to_parquet(out_dir / "L2_conflict_events.parquet", index=False)
    df_base.to_parquet(out_dir / "L2_baseline_events.parquet", index=False)

    print(
        f"Recording {rec_id:02d}: wrote {len(df_conf)} conflict events and {len(df_base)} baseline events to {out_dir}"
    )


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
    for rec_id in rec_ids:
        build_for_recording(rec_id, frame_rate=args.frame_rate)


if __name__ == "__main__":
    main()
