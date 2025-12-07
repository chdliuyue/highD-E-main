"""MEC distribution plotting and summary utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEC_PATH = PROJECT_ROOT / "output" / "mec" / "L2_conf_mec_baseline.parquet"


def load_mec_data(path: Path | None = None) -> pd.DataFrame:
    """Read MEC data parquet file from ``path`` or the default output location."""

    target = path or MEC_PATH
    if target.exists():
        return pd.read_parquet(target)
    raise FileNotFoundError(f"MEC data file not found at {target}")


def add_severity_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical ``severity_bin`` based on ``min_TTC_conf``.

    Bins used: ``<1.5``, ``1.5-2.5``, ``2.5-3.5``, ``>=3.5``.
    """
    bins = [-np.inf, 1.5, 2.5, 3.5, np.inf]
    labels = ["<1.5", "1.5-2.5", "2.5-3.5", ">=3.5"]
    df = df.copy()
    df["severity_bin"] = pd.cut(df["min_TTC_conf"], bins=bins, labels=labels)
    return df


def plot_mec_distributions(df_mec: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot MEC distribution by severity and vehicle class."""
    if df_mec.empty:
        return

    if "MEC_CO2_per_km" not in df_mec.columns:
        print("MEC_CO2_per_km column missing; skipping MEC distribution plot.")
        return

    if "severity_bin" not in df_mec:
        df_mec = add_severity_bins(df_mec)

    fig, ax = plt.subplots(figsize=(9, 6))
    severity_order = ["<1.5", "1.5-2.5", "2.5-3.5", ">=3.5"]
    severity_bins = [b for b in severity_order if b in df_mec["severity_bin"].astype(str).unique()]
    veh_classes = sorted(df_mec["veh_class"].unique()) if "veh_class" in df_mec else ["All"]
    colors = plt.get_cmap("tab10")(range(len(veh_classes)))

    positions = np.arange(len(severity_bins))
    width = 0.35 if len(veh_classes) > 1 else 0.5
    for i, veh in enumerate(veh_classes):
        subset = df_mec[df_mec.get("veh_class", veh) == veh]
        data = [subset.loc[subset["severity_bin"] == b, "MEC_CO2_per_km"].dropna() for b in severity_bins]
        box = ax.boxplot(
            data,
            positions=positions + (i - (len(veh_classes) - 1) / 2) * width,
            widths=width,
            patch_artist=True,
            boxprops={"facecolor": colors[i], "alpha": 0.55},
            medianprops={"color": "black"},
        )
        ax.plot([], [], color=colors[i], label=str(veh))

    ax.set_xticks(positions)
    ax.set_xticklabels(severity_bins)
    ax.set_ylabel("MEC_CO2_per_km")
    ax.set_xlabel("Severity bin (min TTC during conflict)")
    ax.set_title("MEC distributions by severity and vehicle class")
    if len(veh_classes) > 1:
        ax.legend(title="Vehicle class")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)


def build_mec_summary_table(df_mec: pd.DataFrame) -> pd.DataFrame:
    """Generate MEC summary grouped by severity and flow state."""

    df = df_mec.copy()
    if "severity_bin" not in df:
        df = add_severity_bins(df)
    if "flow_state" not in df:
        df["flow_state"] = "unknown"

    duration_col = next((c for c in ["conf_duration", "duration"] if c in df.columns), None)
    real_col = next((c for c in ["E_real_CO2_per_km", "Fuel_real_CO2_per_km", "Fuel_real"] if c in df.columns), None)
    base_col = next((c for c in ["E_base_CO2_per_km", "Fuel_base_CO2_per_km", "Fuel_base"] if c in df.columns), None)

    group_cols = ["severity_bin", "flow_state"]
    grouped = df.groupby(group_cols)
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "n_events": len(g),
                "mean_duration": g[duration_col].mean() if duration_col else np.nan,
                "mean_Fuel_real": g[real_col].mean() if real_col else np.nan,
                "mean_Fuel_base": g[base_col].mean() if base_col else np.nan,
                "mean_MEC_CO2_per_km": g.get("MEC_CO2_per_km", pd.Series(dtype=float)).mean(),
            }
        )
    ).reset_index()

    summary["MEC_share_pct"] = (summary["mean_MEC_CO2_per_km"] / summary["mean_Fuel_real"]) * 100
    severity_order = pd.Categorical(summary["severity_bin"], ["<1.5", "1.5-2.5", "2.5-3.5", ">=3.5"])
    summary["severity_bin"] = severity_order
    summary.sort_values(["severity_bin", "flow_state"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary
