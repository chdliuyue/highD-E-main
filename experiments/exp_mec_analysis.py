"""Experiment runner for MEC distribution and heterogeneity analysis."""
from pathlib import Path

from analysis.mec_analysis import (
    add_duration_bins,
    add_severity_bins,
    basic_mec_stats,
    build_mec_summary_table,
    load_mec_data,
    plot_mec_distributions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = PROJECT_ROOT / "output" / "mec" / "figs"
TABLE_ROOT = PROJECT_ROOT / "output" / "mec" / "tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def run_mec_analysis() -> None:
    """
    Unified entry point for MEC distribution and heterogeneity analysis.
    """

    df = load_mec_data()
    if df.empty:
        print("[WARN] MEC DataFrame is empty; aborting analysis.")
        return

    df = add_severity_bins(df)
    df = add_duration_bins(df)

    stats_df = basic_mec_stats(df)
    if not stats_df.empty:
        print("\nSaved basic statistics to", TABLE_ROOT / "mec_basic_stats.csv")

    plot_mec_distributions(df, save_path=FIG_ROOT / "01_mec_distributions.png")

    summary_df = build_mec_summary_table(df)
    if not summary_df.empty:
        print("Summary table head:\n", summary_df.head())


__all__ = ["run_mec_analysis"]
