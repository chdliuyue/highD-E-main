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

    fig, ax = plt.subplots(figsize=(8, 6))
    severity_bins = df_mec["severity_bin"].cat.categories if df_mec["severity_bin"].dtype.name == "category" else sorted(df_mec["severity_bin"].unique())
    veh_classes = sorted(df_mec["veh_class"].unique()) if "veh_class" in df_mec else ["All"]

    positions = np.arange(len(severity_bins))
    width = 0.35
    for i, veh in enumerate(veh_classes):
        subset = df_mec[df_mec.get("veh_class", veh) == veh]
        data = [subset.loc[subset["severity_bin"] == b, "MEC_CO2_per_km"].dropna() for b in severity_bins]
        box = ax.boxplot(
            data,
            positions=positions + (i - 0.5) * width,
            widths=width,
            patch_artist=True,
            boxprops={"facecolor": "C" + str(i)},
            medianprops={"color": "black"},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.6)
        ax.plot([], [], color="C" + str(i), label=str(veh))

    ax.set_xticks(positions)
    ax.set_xticklabels(severity_bins)
    ax.set_ylabel("MEC_CO2_per_km")
    ax.set_xlabel("Severity bin (min TTC during conflict)")
    ax.set_title("MEC distributions by severity and vehicle class")
    if len(veh_classes) > 1:
        ax.legend()

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def build_mec_summary_table(df_mec: pd.DataFrame) -> pd.DataFrame:
    """Generate MEC summary grouped by severity and flow state."""
    df = df_mec.copy()
    if "flow_state" not in df:
        df["flow_state"] = "unknown"

    group_cols = ["severity_bin", "flow_state"]
    grouped = df.groupby(group_cols)
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "n_events": len(g),
                "mean_duration": g.get("duration", pd.Series(dtype=float)).mean(),
                "mean_Fuel_real": g.get("E_real_CO2_per_km", pd.Series(dtype=float)).mean(),
                "mean_Fuel_base": g.get("E_base_CO2_per_km", pd.Series(dtype=float)).mean(),
                "mean_MEC": g.get("MEC_CO2_per_km", pd.Series(dtype=float)).mean(),
            }
        )
    ).reset_index()
    summary["MEC_share"] = summary["mean_MEC"] / summary["mean_Fuel_real"]
    return summary
