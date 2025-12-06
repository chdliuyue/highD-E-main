"""Summarize generated L2 event tables across recordings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Ensure the repository root is on the path so ``config`` can be imported when the
# script is executed directly (for example via ``python scripts/summarize_L2_events.py``).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PROJECT_ROOT


EVENT_DIR = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
SUMMARY_COLS = ["min_TTC_conf", "conf_duration", "E_cpf_CO2"]


def iter_recording_dirs(base_dir: Path) -> Iterable[Path]:
    """Yield recording directories under ``base_dir`` sorted by name."""
    return sorted([p for p in base_dir.glob("recording_*") if p.is_dir()])


def load_event_table(path: Path) -> pd.DataFrame | None:
    """Load a parquet file if it exists, otherwise return ``None``."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def summarize_recording(rec_dir: Path) -> dict:
    """Summarize conflict/baseline events for a single recording directory."""
    rec_name = rec_dir.name
    try:
        rec_id = int(rec_name.split("_")[1])
    except (IndexError, ValueError):
        rec_id = rec_name

    conf_df = load_event_table(rec_dir / "L2_conflict_events.parquet")
    base_df = load_event_table(rec_dir / "L2_baseline_events.parquet")

    row: dict[str, object] = {
        "recording": rec_id,
        "conflict_events": len(conf_df) if conf_df is not None else 0,
        "baseline_events": len(base_df) if base_df is not None else 0,
    }

    if conf_df is not None and not conf_df.empty:
        for col in SUMMARY_COLS:
            if col in conf_df.columns:
                row[f"{col}_median"] = conf_df[col].median()
                row[f"{col}_sum"] = conf_df[col].sum()

    return row


def build_summary(events_root: Path = EVENT_DIR) -> pd.DataFrame:
    """Build a consolidated summary DataFrame for all available recordings."""
    if not events_root.exists():
        return pd.DataFrame()

    rows: List[dict[str, object]] = []
    for rec_dir in iter_recording_dirs(events_root):
        rows.append(summarize_recording(rec_dir))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize L2 event tables across recordings.")
    parser.parse_args()

    summary = build_summary()
    if summary.empty:
        print(f"No recordings found under {EVENT_DIR}")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
