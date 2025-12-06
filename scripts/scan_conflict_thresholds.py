"""Scan TTC and duration thresholds for conflict event detection.

This script iterates over combinations of TTC and conflict duration thresholds,
constructs conflict events per recording using the parameterized builder, and
summarizes aggregate statistics for each configuration.
"""

from pathlib import Path
from statistics import median
from typing import Iterable, List

import pandas as pd

from config import FRAME_RATE_DEFAULT, PROJECT_ROOT
from data_preproc.events import build_conflict_events_param

L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"

# Threshold grids
TTC_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
DURATION_THRESHOLDS = [0.4, 0.8, 1.0]


def _safe_median(values: List[float]) -> float:
    """Return median of a list or NaN if the list is empty."""

    return median(values) if values else float("nan")


def _iter_recording_paths(recording_ids: Iterable[int]) -> Iterable[tuple[int, Path]]:
    """Yield recording id and expected L1 parquet path."""

    for rec_id in recording_ids:
        yield rec_id, L1_ROOT / f"recording_{rec_id:02d}" / "L1_master_frame.parquet"


def scan_thresholds() -> pd.DataFrame:
    """Run grid search over TTC and duration thresholds.

    Returns
    -------
    pd.DataFrame
        Summary table with counts and median metrics for each threshold pair.
    """

    results = []

    for ttc_thr in TTC_THRESHOLDS:
        for dur_thr in DURATION_THRESHOLDS:
            total_conf_events = 0
            all_min_TTC_conf: List[float] = []
            all_conf_durations: List[float] = []
            all_E_cpf_CO2: List[float] = []

            for rec_id, l1_path in _iter_recording_paths(range(1, 61)):
                if not l1_path.exists():
                    print(f"Warning: L1 file not found for recording {rec_id:02d} at {l1_path}")
                    continue

                df_l1 = pd.read_parquet(l1_path)
                df_rec = df_l1[df_l1["recordingId"] == rec_id].copy()

                df_conf = build_conflict_events_param(
                    df_rec,
                    frame_rate=FRAME_RATE_DEFAULT,
                    ttc_conf_thresh=ttc_thr,
                    min_conf_dur=dur_thr,
                    pre_event_time=3.0,
                    post_event_time=5.0,
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


def main() -> None:
    """Entry point for threshold scanning."""

    df_results = scan_thresholds()
    output_path = PROJECT_ROOT / "data" / "processed" / "highD" / "threshold_scan_conflicts.csv"
    df_results.to_csv(output_path, index=False)
    print(df_results)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
