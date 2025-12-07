from pathlib import Path
from typing import Sequence

from analysis.inspect_l1 import summarize_l1_overall
from analysis.inspect_l2 import summarize_l2_overall

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_explore_experiment(rec_ids: Sequence[int]) -> None:
    """
    Stage 3: unified entrypoint for exploring L1/L2 data.
    """

    print("Running Stage 3 exploration experiment...")
    print(f"Recordings: {[f'{r:02d}' for r in rec_ids]}")

    l1_summary = summarize_l1_overall(rec_ids)
    l2_summary = summarize_l2_overall(rec_ids)

    total_frames = l1_summary["n_rows"].sum() if not l1_summary.empty else 0
    mean_frames = l1_summary["n_rows"].mean() if not l1_summary.empty else 0
    total_conf = l2_summary["n_conf"].sum() if not l2_summary.empty else 0
    total_base = l2_summary["n_base"].sum() if not l2_summary.empty else 0

    print("\nSummary highlights:")
    print(f"- Total frames: {total_frames}")
    print(f"- Mean frames per recording: {mean_frames:.2f}")
    print(f"- Total conflict events: {total_conf}")
    print(f"- Total baseline events: {total_base}")
    if not l2_summary.empty:
        print(
            "- Median conf duration (overall): "
            f"{l2_summary['median_conf_duration'].median():.4f}"
        )
        print(
            "- Median min TTC conf (overall): "
            f"{l2_summary['median_min_TTC_conf'].median():.4f}"
        )
        print(
            "- Median E_cpf_CO2 conf (overall): "
            f"{l2_summary['median_E_cpf_CO2_conf'].median():.4f}"
        )

    print("\nExploration complete. Outputs saved under output/explore/.")
