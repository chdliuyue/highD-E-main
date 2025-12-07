"""Trajectory visualization on highway background maps."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread


def load_highway_background(rec_id: int) -> np.ndarray:
    """Read highway background image for the given recording."""
    img_path = Path(f"data/raw/highD/data/{rec_id:02d}_highway.jpg")
    if not img_path.exists():
        raise FileNotFoundError(f"Background image not found: {img_path}")
    return imread(img_path)


def _select_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_col = next((c for c in ["x_img", "x", "x_center", "x_raw"] if c in df.columns), None)
    y_col = next((c for c in ["y_img", "y", "y_center", "y_raw"] if c in df.columns), None)
    if x_col is None or y_col is None:
        return np.array([]), np.array([])
    return df[x_col].to_numpy(), df[y_col].to_numpy()


def plot_trajectories_on_map(df_l1: pd.DataFrame, df_L2_conf: pd.DataFrame, rec_id: int, save_path: Path | None = None) -> None:
    """
    Plot conflict episode trajectories on a highway background.

    Args:
        df_l1: Frame-level data for the recording.
        df_L2_conf: Conflict event dataframe.
        rec_id: Recording identifier.
        save_path: Optional path to save the plot.
    """
    try:
        background = load_highway_background(rec_id)
    except FileNotFoundError:
        background = None

    fig, ax = plt.subplots(figsize=(10, 6))
    if background is not None:
        ax.imshow(background, origin="lower")

    cmap = plt.get_cmap("coolwarm")
    scatter = None
    for _, row in df_L2_conf.iterrows():
        ego_id = int(row.get("ego_id", row.get("trackId", 0)))
        start_frame = int(row.get("start_frame", row.get("conf_start_frame", 0)))
        end_frame = int(row.get("end_frame", row.get("conf_end_frame", 0)))
        df_episode = df_l1[(df_l1["frame"] >= start_frame) & (df_l1["frame"] <= end_frame) & (df_l1["trackId"] == ego_id)]
        if df_episode.empty:
            continue
        x, y = _select_xy(df_episode)
        if x.size == 0:
            continue
        co2 = df_episode.get("cpf_co2_rate_gps", pd.Series(np.zeros_like(x)))
        ttc = df_episode.get("TTC", pd.Series(np.ones_like(x)))
        sizes = np.clip(1 / np.maximum(ttc, 1e-2), 0, None) * 10
        scatter = ax.scatter(x, y, c=co2, s=sizes, cmap=cmap, alpha=0.7, edgecolors="none")

    ax.set_title(f"Recording {rec_id:02d}: CO2 + 1/TTC hotspots")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if background is None:
        ax.invert_yaxis()
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("CO2 rate [g/s]")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
