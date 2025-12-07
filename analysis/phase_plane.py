"""Safety–Energy phase plane utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _select_energy_column(df: pd.DataFrame) -> str:
    for col in ["cpf_power_kw", "cpf_fuel_rate_lps", "cpf_co2_rate_gps", "power"]:
        if col in df.columns:
            return col
    raise KeyError("No energy-related column found in dataframe.")


def build_phase_plane_samples(
    df_l1: pd.DataFrame,
    df_L2_conf: pd.DataFrame,
    n_examples: int = 10,
    frame_rate: float = 25.0,
) -> Dict[str, np.ndarray | list[Tuple[np.ndarray, np.ndarray]]]:
    """
    Construct samples for plotting the safety–energy phase plane.

    Args:
        df_l1: Frame-level dataframe.
        df_L2_conf: Conflict episode dataframe.
        n_examples: Number of example trajectories to keep.
        frame_rate: Frame rate for converting frames to time.

    Returns:
        Dictionary containing flattened TTC and energy arrays and a list of
        example trajectories for hysteresis visualization.
    """
    energy_col = _select_energy_column(df_l1)
    ttc_all: list[np.ndarray] = []
    energy_all: list[np.ndarray] = []
    example_trajs: list[Tuple[np.ndarray, np.ndarray]] = []

    for idx, row in df_L2_conf.iterrows():
        ego_id = int(row.get("ego_id", row.get("trackId", 0)))
        start_frame = int(row.get("start_frame", row.get("conf_start_frame", 0)))
        end_frame = int(row.get("end_frame", row.get("conf_end_frame", 0)))
        df_episode = df_l1[(df_l1["frame"] >= start_frame) & (df_l1["frame"] <= end_frame) & (df_l1["trackId"] == ego_id)]
        if df_episode.empty or "TTC" not in df_episode:
            continue

        ttc_series = df_episode["TTC"].to_numpy()
        energy_series = df_episode[energy_col].to_numpy()
        ttc_all.append(ttc_series)
        energy_all.append(energy_series)

        if len(example_trajs) < n_examples:
            example_trajs.append((ttc_series, energy_series))

    ttc_concat = np.concatenate(ttc_all) if ttc_all else np.array([])
    energy_concat = np.concatenate(energy_all) if energy_all else np.array([])

    return {
        "TTC_all": ttc_concat,
        "energy_all": energy_concat,
        "example_trajectories": example_trajs,
    }


def plot_safety_energy_phase_plane(
    TTC_all: np.ndarray,
    energy_all: np.ndarray,
    example_trajs: List[Tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> None:
    """
    Plot safety–energy phase plane with hexbin density and example loops.

    Args:
        TTC_all: Flattened TTC samples.
        energy_all: Flattened energy metric samples.
        example_trajs: Trajectories to overlay for hysteresis demonstration.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if TTC_all.size > 0 and energy_all.size > 0:
        hb = ax.hexbin(TTC_all, energy_all, gridsize=40, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Density")

    for traj in example_trajs:
        ax.plot(traj[0], traj[1], alpha=0.7, linewidth=1.5)

    ax.set_xlabel("TTC [s]")
    ax.set_ylabel("Energy metric")
    ax.set_title("Safety–Energy Phase Plane")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
