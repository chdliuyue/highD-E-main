from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
L2_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
OUT_ROOT = PROJECT_ROOT / "output" / "phase_plane"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_l1(rec_id: int) -> pd.DataFrame:
    rec_str = f"{rec_id:02d}"
    path = L1_ROOT / f"recording_{rec_str}" / "L1_master_frame.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_l2_conflicts(rec_id: int) -> pd.DataFrame:
    rec_str = f"{rec_id:02d}"
    path = L2_ROOT / f"recording_{rec_str}" / "L2_conflict_events.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _extract_episode_frames(
    df_l1: pd.DataFrame, event_row: pd.Series, energy_field: str
) -> dict[str, np.ndarray]:
    required_cols = {"recordingId", "trackId", "frame"}
    if not required_cols.issubset(df_l1.columns):
        return {}

    rec_val = event_row.get("rec_id", event_row.get("recordingId", np.nan))
    ego_id = event_row.get("ego_id", event_row.get("trackId", np.nan))

    if pd.isna(rec_val) or pd.isna(ego_id):
        return {}

    start_frame = None
    for key in ["start_frame", "conf_start_frame", "start"]:
        if key in event_row and pd.notna(event_row[key]):
            start_frame = int(event_row[key])
            break

    end_frame = None
    for key in ["end_frame", "conf_end_frame", "end"]:
        if key in event_row and pd.notna(event_row[key]):
            end_frame = int(event_row[key])
            break

    mask = (df_l1["recordingId"] == int(rec_val)) & (df_l1["trackId"] == ego_id)
    if start_frame is not None:
        mask &= df_l1["frame"] >= start_frame
    if end_frame is not None:
        mask &= df_l1["frame"] <= end_frame

    df_ep = df_l1.loc[mask].copy()
    if df_ep.empty:
        return {}

    ttc_vals = df_ep["TTC"].to_numpy() if "TTC" in df_ep.columns else None
    energy_vals = df_ep[energy_field].to_numpy() if energy_field in df_ep.columns else None

    if ttc_vals is None or energy_vals is None:
        return {}

    ttc_vals = np.clip(ttc_vals, 0, 60)
    return {"TTC": ttc_vals, "energy": energy_vals}


def build_phase_plane_samples(
    rec_ids: Sequence[int],
    frame_rate: float = 25.0,
    n_example_events: int = 5,
) -> Dict[str, object]:
    _ = frame_rate  # reserved for potential resampling logic
    energy_field = "cpf_co2_rate_gps"
    ttc_all: list[float] = []
    energy_all: list[float] = []
    severe_events: list[dict[str, np.ndarray]] = []

    total = len(rec_ids)
    for idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[Stage 07] ({idx}/{total}) Building phase-plane samples for rec {rec_id:02d}...")
        df_l1 = _load_l1(rec_id)
        df_l2 = _load_l2_conflicts(rec_id)
        if df_l1.empty or df_l2.empty:
            print(f"  [Stage 07] Skipping recording {rec_id:02d}: missing L1/L2 data")
            continue

        df_l2_sorted = df_l2.sort_values("min_TTC_conf") if "min_TTC_conf" in df_l2.columns else df_l2

        for _, row in df_l2_sorted.iterrows():
            ep = _extract_episode_frames(df_l1, row, energy_field)
            if not ep:
                continue
            ttc_all.extend(ep["TTC"].tolist())
            energy_all.extend(ep["energy"].tolist())
            is_severe = True
            if "min_TTC_conf" in row.index:
                is_severe &= row["min_TTC_conf"] < 2.0
            if "conf_duration" in row.index:
                is_severe &= row["conf_duration"] >= 0.8
            if is_severe and len(severe_events) < n_example_events:
                severe_events.append(ep)
        print(
            f"  [Stage 07] Recording {rec_id:02d} contributed {len(ttc_all)} TTC samples so far"
        )

    ttc_arr = np.asarray(ttc_all, dtype=float)
    energy_arr = np.asarray(energy_all, dtype=float)

    if len(severe_events) > n_example_events:
        severe_events = severe_events[:n_example_events]

    return {
        "ttc_all": ttc_arr,
        "energy_all": energy_arr,
        "example_trajs": severe_events,
    }


def plot_phase_plane(
    ttc_all: np.ndarray,
    energy_all: np.ndarray,
    example_trajs: list[Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    if len(ttc_all) > 0 and len(energy_all) > 0:
        hb = ax.hexbin(ttc_all, energy_all, gridsize=60, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="counts")

    colors = plt.cm.tab10.colors
    for idx, traj in enumerate(example_trajs):
        ax.plot(traj["TTC"], traj["energy"], color=colors[idx % len(colors)], alpha=0.8, label=f"traj {idx + 1}")
        if len(traj["TTC"]) >= 2:
            ax.annotate(
                "start",
                xy=(traj["TTC"][0], traj["energy"][0]),
                xytext=(5, 5),
                textcoords="offset points",
                color=colors[idx % len(colors)],
                fontsize=9,
            )
            ax.annotate(
                "end",
                xy=(traj["TTC"][-1], traj["energy"][-1]),
                xytext=(5, -10),
                textcoords="offset points",
                color=colors[idx % len(colors)],
                fontsize=9,
            )

    ax.set_xlim(0, 60)
    ax.set_xlabel("TTC [s]")
    ax.set_ylabel("CO2 rate [g/s]")
    ax.set_title("Safety–Energy Phase Plane (TTC vs CO2 rate)")
    if example_trajs:
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


__all__ = ["build_phase_plane_samples", "plot_phase_plane"]
