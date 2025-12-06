"""Visualization helpers for L1_master_frame outputs."""
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def visualize_frame(
    recording_id: int,
    frame: int,
    processed_dir: Path,
    highway_image_dir: Path,
    color_by: str = "cpf_co2_rate_gps",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Visualize one frame on top of the corresponding highway background image."""
    parquet_path = processed_dir / f"recording_{recording_id:02d}" / "L1_master_frame.parquet"
    df = pd.read_parquet(parquet_path)
    frame_df = df[df["frame"] == frame]

    img_path = highway_image_dir / f"{recording_id:02d}_highway.png"
    img = plt.imread(img_path)

    plt.figure(figsize=figsize)
    plt.imshow(img)
    sc = plt.scatter(frame_df["x_img"], frame_df["y_img"], c=frame_df[color_by], cmap="viridis", s=20)
    plt.colorbar(sc, label=color_by)
    plt.title(f"Recording {recording_id:02d} Frame {frame}")
    plt.xlabel("x_img (pixels)")
    plt.ylabel("y_img (pixels)")
    plt.tight_layout()
    plt.show()


__all__ = ["visualize_frame"]
