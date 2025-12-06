"""Plot velocity/acceleration/CO2 traces for selected L2 events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import FRAME_RATE_DEFAULT, PROCESSED_DATA_DIR, PROJECT_ROOT

EVENTS_DIR = PROJECT_ROOT / "data" / "processed" / "highD" / "events"


def load_l1_master_frame(recording_id: int, processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load the L1 master frame parquet for a single recording."""

    parquet_path = processed_dir / f"recording_{recording_id:02d}" / "L1_master_frame.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"L1 parquet not found at {parquet_path}")

    return pd.read_parquet(parquet_path)


def load_l2_events(recording_id: int, events_dir: Path = EVENTS_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load conflict and baseline L2 events for a recording."""

    rec_dir = events_dir / f"recording_{recording_id:02d}"
    conf_path = rec_dir / "L2_conflict_events.parquet"
    base_path = rec_dir / "L2_baseline_events.parquet"

    if not conf_path.exists():
        raise FileNotFoundError(f"Missing conflict events parquet at {conf_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Missing baseline events parquet at {base_path}")

    return pd.read_parquet(conf_path), pd.read_parquet(base_path)


def plot_event_time_series(
    l1_df: pd.DataFrame,
    event: pd.Series,
    event_type: str,
    save_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    frame_rate: float = FRAME_RATE_DEFAULT,
) -> Path | None:
    """Plot velocity, acceleration, and CO2 rate traces for a single L2 event."""

    track_df = l1_df[l1_df["trackId"] == event["ego_id"]].sort_values("frame")
    window_mask = (track_df["frame"] >= event["start_frame"]) & (track_df["frame"] <= event["end_frame"])
    window_df = track_df.loc[window_mask]

    if window_df.empty:
        raise ValueError(
            "No L1 data found for track "
            f"{event['ego_id']} between frames {event['start_frame']} and {event['end_frame']}"
        )

    if "time" in window_df.columns and window_df["time"].notna().any():
        time_series = window_df["time"].to_numpy()
        start_time = time_series[0]
        x_vals = time_series - start_time
        time_lookup = dict(zip(window_df["frame"].to_numpy(), time_series - start_time))
    else:
        x_vals = (window_df["frame"] - event["start_frame"]) / frame_rate
        time_lookup = dict(zip(window_df["frame"].to_numpy(), x_vals))

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(x_vals, window_df["v_long_smooth"], color="C0")
    axes[0].set_ylabel("v_long_smooth [m/s]")

    axes[1].plot(x_vals, window_df["a_long_smooth"], color="C1")
    axes[1].set_ylabel("a_long_smooth [m/s^2]")

    axes[2].plot(x_vals, window_df["cpf_co2_rate_gps"], color="C2")
    axes[2].set_ylabel("cpf_co2_rate_gps [g/s]")
    axes[2].set_xlabel("Time since window start [s]")

    if "conf_start_frame" in event and pd.notna(event.get("conf_start_frame")):
        conf_start = time_lookup.get(int(event["conf_start_frame"]))
        conf_end = time_lookup.get(int(event["conf_end_frame"]))
        if conf_start is not None and conf_end is not None:
            for ax in axes:
                ax.axvspan(conf_start, conf_end, color="red", alpha=0.15, label="Conflict core")
            axes[0].legend(loc="upper right")

    fig.suptitle(
        f"Recording {int(event['recordingId']):02d} | {event_type.title()} event {int(event['event_id'])} | "
        f"Track {int(event['ego_id'])}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"rec{int(event['recordingId']):02d}_{event_type}_event{int(event['event_id']):03d}_"
            f"track{int(event['ego_id'])}_frames{int(event['start_frame'])}-{int(event['end_frame'])}.png"
        )
        out_path = save_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    plt.show()
    return None


def sample_events(df: pd.DataFrame, n_events: int) -> pd.DataFrame:
    if df.empty or n_events <= 0:
        return df.iloc[0:0]
    n = min(n_events, len(df))
    return df.sample(n=n, random_state=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot time-series slices for conflict/baseline events.")
    parser.add_argument("--recording", type=int, required=True, help="Recording ID to visualize (e.g., 1)")
    parser.add_argument("--n-conf", type=int, default=2, help="Number of conflict events to sample")
    parser.add_argument("--n-base", type=int, default=2, help="Number of baseline events to sample")
    parser.add_argument(
        "--save-dir", type=Path, default=None, help="Optional directory to save plots instead of showing them"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l1_df = load_l1_master_frame(args.recording)
    df_conf, df_base = load_l2_events(args.recording)

    sampled_conf = sample_events(df_conf, args.n_conf)
    sampled_base = sample_events(df_base, args.n_base)

    if args.save_dir is not None:
        save_root = Path(args.save_dir)
        conf_dir = save_root / "conflict"
        base_dir = save_root / "baseline"
    else:
        conf_dir = base_dir = None

    for _, event in sampled_conf.iterrows():
        plot_event_time_series(l1_df, event, "conflict", save_dir=conf_dir)

    for _, event in sampled_base.iterrows():
        plot_event_time_series(l1_df, event, "baseline", save_dir=base_dir)


if __name__ == "__main__":
    main()
