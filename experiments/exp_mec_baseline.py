from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.mec_baseline import (
    compute_event_level_metrics,
    compute_mec_from_matching,
    load_all_baseline_events,
    load_all_conflict_events,
    match_baseline_for_conflicts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "analysis"


def _print_group_stats(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"{label}: no samples")
        return
    print(label)
    print(df["MEC_CO2_per_s"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    print("-")


def run_mec_baseline_experiment(
    recordings: Sequence[int] | None = None,
) -> None:
    """
    事件级 MEC (data-driven baseline) 实验主入口。
    """

    df_conf = load_all_conflict_events(recordings)
    df_base = load_all_baseline_events(recordings)

    df_conf = compute_event_level_metrics(df_conf)
    df_base = compute_event_level_metrics(df_base)

    df_matched = match_baseline_for_conflicts(df_conf, df_base, max_candidates=len(df_conf))
    df_mec = compute_mec_from_matching(df_matched)

    if df_mec.empty:
        print("No matched MEC samples available.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "L2_conf_mec_baseline.parquet"
    df_mec.to_parquet(out_path, index=False)
    print(f"Saved MEC dataframe to {out_path}")

    _print_group_stats(df_mec, "All samples")

    bins = [0, 2, 3, 4, np.inf]
    labels = ["<2", "2-3", "3-4", ">=4"]
    df_mec["min_TTC_group"] = pd.cut(df_mec.get("min_TTC_conf"), bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        subgroup = df_mec[df_mec["min_TTC_group"] == label]
        _print_group_stats(subgroup, f"min_TTC_conf {label}")

    dur_bins = [0.0, 0.4, 0.8, 1.0, np.inf]
    dur_labels = [">=0", ">=0.4", ">=0.8", ">=1.0"]
    df_mec["duration_group"] = pd.cut(df_mec.get("conf_duration"), bins=dur_bins, labels=dur_labels, include_lowest=True)
    for label in dur_labels:
        subgroup = df_mec[df_mec["duration_group"] == label]
        _print_group_stats(subgroup, f"conf_duration {label}")
