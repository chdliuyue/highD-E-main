"""Compatibility wrapper around :class:`data_preproc.l1_builder.L1Builder`."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import config
from data_preproc.l1_builder import L1Builder


class HighDDataBuilder(L1Builder):
    """Backward-compatible entrypoint for building L1 master tables."""

    def __init__(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
        frame_rate: float = config.FRAME_RATE_DEFAULT,
    ) -> None:
        super().__init__(
            raw_data_dir=raw_data_dir,
            processed_data_dir=output_dir,
            frame_rate=frame_rate,
            num_workers=num_workers,
        )

    def process_one_recording(self, rec_id: int) -> None:  # pragma: no cover - compatibility shim
        self.build_one(rec_id)

    def process_all_recordings(self, recording_ids: Optional[Iterable[int]] = None) -> None:
        if recording_ids is None:
            recording_ids = (
                config.TEST_RECORDINGS if config.TEST_MODE else list(range(1, config.MAX_RECORDING_ID + 1))
            )
        self.build_many(recording_ids, num_workers=self.num_workers)


def _process_single_recording_entry(args: tuple) -> None:  # pragma: no cover - compatibility shim
    raw_dir, out_dir, rec_id, num_workers = args
    builder = HighDDataBuilder(Path(raw_dir), Path(out_dir), num_workers=num_workers)
    builder.process_one_recording(rec_id)


__all__ = ["HighDDataBuilder", "_process_single_recording_entry"]
