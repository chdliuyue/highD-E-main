"""Ghost car validation utilities using a simplified IDM baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.idm import IDM


def select_severe_conflicts(
    df_L2: pd.DataFrame,
    ttc_thresh: float = 2.0,
    min_dur: float = 0.8,
    accel_thresh: float = 2.0,
) -> pd.DataFrame:
    """Return a subset of severe conflict events with strong kinematic response.

    Args:
        df_L2: Conflict events table.
        ttc_thresh: Severity threshold for ``min_TTC_conf``.
        min_dur: Minimum conflict duration in seconds.
        accel_thresh: Minimum absolute acceleration (if available) to qualify
            as a severe event. Columns searched include ``a_min`` and
            ``a_long_min``.

    Returns:
        Filtered DataFrame containing severe episodes only.
    """
    mask = (df_L2.get("min_TTC_conf", pd.Series(dtype=float)) < ttc_thresh) & (
        df_L2.get("conf_duration", pd.Series(dtype=float)) >= min_dur
    )

    accel_cols = ["a_min", "a_long_min", "a_long_smooth_min"]
    for col in accel_cols:
        if col in df_L2:
            mask &= df_L2[col].abs() > accel_thresh
            break

    return df_L2.loc[mask].copy()


def _extract_position_speed(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract longitudinal position and speed arrays with fallbacks."""
    pos_col_candidates = ["s", "x_center", "x"]
    speed_col_candidates = ["v_long_smooth", "v_x", "vx", "speed"]
    pos_col = next((c for c in pos_col_candidates if c in df.columns), None)
    speed_col = next((c for c in speed_col_candidates if c in df.columns), None)
    pos = df[pos_col].to_numpy() if pos_col else np.zeros(len(df))
    speed = df[speed_col].to_numpy() if speed_col else np.zeros(len(df))
    return pos.astype(float), speed.astype(float)


def _extract_leader_id(df_episode: pd.DataFrame) -> int | None:
    """Infer the leader track ID from episode frames if available."""
    for col in ["precedingId", "leader_id", "leaderId"]:
        if col in df_episode.columns:
            non_zero = df_episode[col].replace({0: np.nan}).dropna()
            if not non_zero.empty:
                return int(non_zero.iloc[0])
    return None


def _build_time_grid(df_episode: pd.DataFrame, frame_rate: float) -> np.ndarray:
    if "time" in df_episode.columns:
        t = df_episode["time"].to_numpy(dtype=float)
        t = t - t.min()
    else:
        t = (df_episode["frame"] - df_episode["frame"].min()).to_numpy(dtype=float) / frame_rate
    return t


def _align_leader_to_time(df_leader: pd.DataFrame, t_ref: np.ndarray, frame_ref: np.ndarray, frame_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate leader position and speed onto the ego time grid."""
    if df_leader.empty:
        return np.full_like(t_ref, np.nan, dtype=float), np.full_like(t_ref, np.nan, dtype=float)

    leader_frames = df_leader["frame"].to_numpy(dtype=float)
    leader_t = (leader_frames - frame_ref.min()) / frame_rate
    s_raw, v_raw = _extract_position_speed(df_leader)

    # Use edge values outside leader support to avoid NaNs at boundaries
    s_interp = np.interp(t_ref, leader_t, s_raw, left=np.nan, right=np.nan)
    v_interp = np.interp(t_ref, leader_t, v_raw, left=np.nan, right=np.nan)

    # Fill potential NaNs (e.g., partial tracking) with nearest observed values
    s_series = pd.Series(s_interp).ffill().bfill()
    v_series = pd.Series(v_interp).ffill().bfill()

    return s_series.to_numpy(dtype=float), v_series.to_numpy(dtype=float)


def simulate_ghost_car(
    df_l1: pd.DataFrame,
    event_row: pd.Series,
    idm_params: Dict[str, float],
    frame_rate: float = 25.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate ghost car baseline trajectory for a conflict episode.

    Args:
        df_l1: Frame-level data for a recording.
        event_row: Selected episode row from the L2 table.
        idm_params: Parameters to instantiate ``IDM``.
        frame_rate: Frame rate (Hz) for time grid construction.

    Returns:
        A dictionary containing time grid, real and ghost states.
    """
    ego_id = int(event_row.get("ego_id", event_row.get("trackId", 0)))
    start_frame = int(event_row.get("start_frame", event_row.get("conf_start_frame", 0)))
    end_frame = int(event_row.get("end_frame", event_row.get("conf_end_frame", 0)))

    df_episode = df_l1[(df_l1["frame"] >= start_frame) & (df_l1["frame"] <= end_frame) & (df_l1["trackId"] == ego_id)]
    df_episode = df_episode.sort_values("frame")
    if df_episode.empty:
        return {}

    leader_id = event_row.get("leader_id") or _extract_leader_id(df_episode)
    df_leader = pd.DataFrame()
    if leader_id is not None:
        df_leader = df_l1[(df_l1["frame"].isin(df_episode["frame"])) & (df_l1["trackId"] == leader_id)].sort_values("frame")

    t = _build_time_grid(df_episode, frame_rate)
    s_real, v_real = _extract_position_speed(df_episode)

    s_leader, v_leader = _align_leader_to_time(df_leader, t, df_episode["frame"].to_numpy(), frame_rate)
    if np.all(np.isnan(s_leader)):
        fallback_gap = float(event_row.get("gap_init", 15.0))
        s_leader = s_real + fallback_gap
        v_leader = v_real.copy()

    s0_init = float(event_row.get("s0_init", max(s_leader[0] - s_real[0], 2.0)))
    v_init = float(v_real[0]) if len(v_real) > 0 else 0.0

    idm = IDM(**idm_params)
    s_ghost, v_ghost = idm.simulate(t, s_leader, v_leader, s0_init=s0_init, v_init=v_init)

    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.0
    a_real = np.concatenate([[0.0], np.diff(v_real) / dt]) if dt > 0 else np.zeros_like(v_real)
    a_ghost = np.concatenate([[0.0], np.diff(v_ghost) / dt]) if dt > 0 else np.zeros_like(v_ghost)

    # Vehicle lengths are not available in L1; gap is approximated by the raw
    # longitudinal spacing. If lengths are added later, subtract half-lengths
    # of leader/ego here.
    gap_real = s_leader - s_real
    gap_ghost = s_leader - s_ghost

    return {
        "t": t,
        "s_leader": s_leader,
        "s_real": s_real,
        "s_ghost": s_ghost,
        "gap_real": gap_real,
        "gap_ghost": gap_ghost,
        "v_real": v_real,
        "v_ghost": v_ghost,
        "a_real": a_real,
        "a_ghost": a_ghost,
    }


def plot_ghost_car_validation(data: Dict[str, np.ndarray], save_path: Path | None = None) -> None:
    """
    Plot real vs ghost car trajectories and kinematics.

    Args:
        data: Dictionary returned by :func:`simulate_ghost_car`.
        save_path: Optional path to save the figure. If ``None``, the figure is
            shown interactively.
    """
    if not data:
        return

    t = data["t"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(data["gap_real"], t, label="Gap real", color="tab:blue")
    axes[0].plot(data["gap_ghost"], t, label="Gap ghost (IDM)", color="tab:orange", linestyle="--")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Gap [m]")
    axes[0].set_ylabel("Time [s]")
    axes[0].set_title("Gap–t (headway over time)")
    axes[0].legend()

    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(t, data["v_real"], label="v real", color="tab:blue")
    ax1.plot(t, data["v_ghost"], label="v ghost", color="tab:orange", linestyle="--")
    ax2.plot(t, data["a_real"], label="a real", color="tab:red")
    ax2.plot(t, data["a_ghost"], label="a ghost", color="tab:red", linestyle=":")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Speed [m/s]", color="tab:blue")
    ax2.set_ylabel("Acceleration [m/s²]", color="tab:red")
    ax1.set_title("Kinematics (severe episode)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)
