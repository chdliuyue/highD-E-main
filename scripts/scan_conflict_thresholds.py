"""Parameter sweep over TTC/duration thresholds for conflict event detection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median
from typing import Iterable, List

import pandas as pd

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from config import FRAME_RATE_DEFAULT, POST_EVENT_TIME, PRE_EVENT_TIME, PROJECT_ROOT
from data_preproc.events import extract_high_interaction_events

L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"

TTC_THRESHOLDS = [2.0, 3.0, 4.0]
DURATION_THRESHOLDS = [0.4, 0.6, 0.8, 1.0]


def _safe_median(values: List[float]) -> float:
    return median(values) if values else float("nan")


def _iter_recording_paths(recording_ids: Iterable[int]) -> Iterable[tuple[int, Path]]:
    for rec_id in recording_ids:
        yield rec_id, L1_ROOT / f"recording_{rec_id:02d}" / "L1_master_frame.parquet"


def scan_thresholds(recording_ids: Iterable[int]) -> pd.DataFrame:
    """Run grid search over TTC and duration thresholds."""

    results = []

    for ttc_thr in TTC_THRESHOLDS:
        for dur_thr in DURATION_THRESHOLDS:
            total_conf_events = 0
            all_min_TTC_conf: List[float] = []
            all_conf_durations: List[float] = []
            all_E_cpf_CO2: List[float] = []

            for rec_id, l1_path in _iter_recording_paths(recording_ids):
                if not l1_path.exists():
                    print(f"Warning: L1 file not found for recording {rec_id:02d} at {l1_path}")
                    continue

                df_l1 = pd.read_parquet(l1_path)
                df_rec = df_l1[df_l1["recordingId"] == rec_id].copy()

                df_conf = extract_high_interaction_events(
                    df_rec,
                    frame_rate=FRAME_RATE_DEFAULT,
                    ttc_upper=ttc_thr,
                    min_conf_duration=dur_thr,
                    pre_event_time=PRE_EVENT_TIME,
                    post_event_time=POST_EVENT_TIME,
                )

                total_conf_events += len(df_conf)

                if not df_conf.empty:
                    all_min_TTC_conf.extend(df_conf["min_TTC_conf"].tolist())
                    all_conf_durations.extend(df_conf["conf_duration"].tolist())
                    if "E_cpf_CO2" in df_conf.columns:
                        all_E_cpf_CO2.extend(df_conf["E_cpf_CO2"].tolist())

            results.append(
                {
                    "TTC_thr": ttc_thr,
                    "dur_thr": dur_thr,
                    "n_conf": total_conf_events,
                    "med_min_TTC_conf": _safe_median(all_min_TTC_conf),
                    "med_conf_dur": _safe_median(all_conf_durations),
                    "med_E_CO2": _safe_median(all_E_cpf_CO2),
                }
            )

    df_results = pd.DataFrame(results)
    return df_results.sort_values(["TTC_thr", "dur_thr"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan TTC/duration thresholds for conflict detection")
    parser.add_argument(
        "--recordings",
        type=str,
        default="1-60",
        help="Recording id range or comma list (e.g., '1-3,5')",
    )
    return parser.parse_args()


def parse_recording_range(spec: str) -> List[int]:
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            ids.extend(list(range(int(start), int(end) + 1)))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def main() -> None:
    args = parse_args()
    rec_ids = parse_recording_range(args.recordings)
    df_results = scan_thresholds(rec_ids)
    output_path = PROJECT_ROOT / "data" / "processed" / "highD" / "threshold_scan_conflicts.csv"
    df_results.to_csv(output_path, index=False)
    print(df_results)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
