"""MEC distribution and heterogeneity analysis utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEC_PATH = PROJECT_ROOT / "data" / "analysis" / "L2_conf_mec_baseline.parquet"
FIG_ROOT = PROJECT_ROOT / "output" / "mec" / "figs"
TABLE_ROOT = PROJECT_ROOT / "output" / "mec" / "tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist, creating fallbacks where possible."""

    rename_map = {
        "vehicle_class": "veh_class",
        "vehicleType": "veh_class",
        "ttc_min": "min_TTC_conf",
        "ttc_conf_min": "min_TTC_conf",
        "conf_dur": "conf_duration",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "event_id" not in df.columns:
        df = df.copy()
        df["event_id"] = df.index

    if "flow_state" not in df.columns:
        df = df.copy()
        df["flow_state"] = "unknown"

    if "veh_class" in df.columns:
        df = df.copy()
        df["veh_class"] = df["veh_class"].map({1: "Car", 2: "Truck"}).fillna(df["veh_class"])

    return df


def load_mec_data() -> pd.DataFrame:
    """
    Load MEC data from the default parquet location.

    Returns an empty DataFrame with a warning if the file is missing.
    """

    if not MEC_PATH.exists():
        print(f"[WARN] MEC file not found at {MEC_PATH}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_parquet(MEC_PATH)
    df = _normalize_columns(df)
    return df


def add_severity_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add severity_bin column based on min_TTC_conf.

    Bin edges are [0, 2], (2, 3], (3, 4], (4, inf). Lower TTC implies higher severity.
    """

    if "min_TTC_conf" not in df.columns:
        print("[WARN] min_TTC_conf not found; severity_bin will be set to 'unknown'.")
        df = df.copy()
        df["severity_bin"] = "unknown"
        return df

    edges = [0, 2, 3, 4, np.inf]
    labels = ["<=2s", "(2,3]s", "(3,4]s", ">4s"]
    df = df.copy()
    df["severity_bin"] = pd.cut(df["min_TTC_conf"], bins=edges, labels=labels, include_lowest=True)
    df["severity_bin"] = df["severity_bin"].cat.add_categories(["unknown"]).fillna("unknown")
    return df


def add_duration_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally add duration_bin based on conf_duration.

    Bin edges are [0, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0], (1.0, inf).
    """

    if "conf_duration" not in df.columns:
        print("[WARN] conf_duration not found; duration_bin will be set to 'unknown'.")
        df = df.copy()
        df["duration_bin"] = "unknown"
        return df

    edges = [0, 0.4, 0.6, 0.8, 1.0, np.inf]
    labels = ["<=0.4s", "(0.4,0.6]s", "(0.6,0.8]s", "(0.8,1.0]s", ">1.0s"]
    df = df.copy()
    df["duration_bin"] = pd.cut(df["conf_duration"], bins=edges, labels=labels, include_lowest=True)
    df["duration_bin"] = df["duration_bin"].cat.add_categories(["unknown"]).fillna("unknown")
    return df


def _quantiles(series: pd.Series, qs: list[float]) -> list[float]:
    return series.quantile(qs).tolist()


def basic_mec_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall and grouped MEC statistics.
    """

    if df.empty:
        print("[WARN] Empty DataFrame received in basic_mec_stats.")
        return pd.DataFrame()

    mec_col = "MEC_CO2_per_km"
    if mec_col not in df.columns:
        print("[WARN] MEC_CO2_per_km missing; cannot compute statistics.")
        return pd.DataFrame()

    stats = {
        "count": df[mec_col].count(),
        "mean": df[mec_col].mean(),
        "median": df[mec_col].median(),
        "min": df[mec_col].min(),
        "max": df[mec_col].max(),
    }
    q10, q90 = _quantiles(df[mec_col], [0.1, 0.9])
    stats["q10"], stats["q90"] = q10, q90

    overall_df = pd.DataFrame(stats, index=["overall"])

    grouped = (
        df.groupby(["severity_bin", "veh_class"], dropna=False)[mec_col]
        .agg(median="median", q10=lambda s: s.quantile(0.1), q90=lambda s: s.quantile(0.9))
        .reset_index()
    )

    print("MEC_CO2_per_km overall stats:")
    print(overall_df)
    print("\nMedian MEC_CO2_per_km by severity_bin and veh_class:")
    print(grouped)

    output = pd.concat({"overall": overall_df, "by_group": grouped.set_index(["severity_bin", "veh_class"])}, axis=1)
    output.to_csv(TABLE_ROOT / "mec_basic_stats.csv")
    return output


def plot_mec_distributions(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Plot MEC distributions split by severity and vehicle class.
    """

    mec_col = "MEC_CO2_per_km"
    if df.empty or mec_col not in df.columns:
        print("[WARN] Data unavailable for plotting.")
        return

    plot_df = df.copy()
    if "severity_bin" not in plot_df.columns:
        plot_df["severity_bin"] = "unknown"
    if "veh_class" not in plot_df.columns:
        plot_df["veh_class"] = "unknown"

    order = ["<=2s", "(2,3]s", "(3,4]s", ">4s", "unknown"]
    plot_df["severity_bin"] = pd.Categorical(plot_df["severity_bin"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("severity_bin")

    severity_levels = [lvl for lvl in order if lvl in plot_df["severity_bin"].unique()]
    veh_classes = ["Car", "Truck"]

    positions = np.arange(len(severity_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, vclass in enumerate(veh_classes):
        subset = plot_df[plot_df["veh_class"] == vclass]
        data = [subset[subset["severity_bin"] == sev][mec_col].dropna() for sev in severity_levels]
        offset = (idx - 0.5) * width
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.9,
            patch_artist=True,
            labels=[""] * len(severity_levels),
        )
        color = "#4C72B0" if vclass == "Car" else "#DD8452"
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(severity_levels)
    ax.set_xlabel("Severity (by min_TTC_conf)")
    ax.set_ylabel("MEC_CO2_per_km (g/km)")
    ax.legend(["Car", "Truck"], title="Vehicle class")
    ax.set_title("MEC distribution by severity and vehicle class")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved MEC distribution plot to {save_path}")
    plt.close(fig)


def build_mec_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary table for MEC analysis (Table 1).
    """

    if df.empty:
        print("[WARN] Empty DataFrame received in build_mec_summary_table.")
        return pd.DataFrame()

    required_cols = [
        "severity_bin",
        "flow_state",
        "duration_real",
        "E_real_CO2_per_km",
        "E_base_CO2_per_km",
        "MEC_CO2_per_km",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] Column {col} missing; filling with NaN or default where appropriate.")
            df[col] = np.nan

    group_cols = ["severity_bin", "flow_state"]
    grouped = df.groupby(group_cols, dropna=False)

    summary = grouped.agg(
        n_events=("event_id", "count"),
        mean_duration_real=("duration_real", "mean"),
        mean_E_real_CO2_per_km=("E_real_CO2_per_km", "mean"),
        mean_E_base_CO2_per_km=("E_base_CO2_per_km", "mean"),
        mean_MEC_CO2_per_km=("MEC_CO2_per_km", "mean"),
    ).reset_index()

    summary["MEC_share_pct"] = (
        summary["mean_MEC_CO2_per_km"] / summary["mean_E_real_CO2_per_km"] * 100
    )

    output_path = TABLE_ROOT / "table01_mec_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved MEC summary table to {output_path}")
    return summary


__all__ = [
    "load_mec_data",
    "add_severity_bins",
    "add_duration_bins",
    "basic_mec_stats",
    "plot_mec_distributions",
    "build_mec_summary_table",
]
