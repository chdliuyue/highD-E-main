"""L2 event construction for conflict and baseline segments using VT-CPFM rates."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config


def find_contiguous_segments(mask: np.ndarray, frames: np.ndarray) -> List[Tuple[int, int]]:
    """Return start/end frames for contiguous ``True`` segments.

    Parameters
    ----------
    mask:
        Boolean array indicating frames that satisfy a condition (e.g., TTC below
        a threshold).
    frames:
        Array of frame numbers aligned with ``mask``.

    Returns
    -------
    List[Tuple[int, int]]
        List of ``(start_frame, end_frame)`` tuples for each contiguous ``True``
        run. Frames are inclusive.
    """

    mask = np.asarray(mask, dtype=bool)
    frames = np.asarray(frames)

    if mask.size == 0:
        return []

    segments: List[Tuple[int, int]] = []
    in_segment = False
    start_frame = frames[0]
    last_frame = frames[0]

    for is_true, frame in zip(mask, frames):
        if is_true and not in_segment:
            start_frame = frame
            last_frame = frame
            in_segment = True
        elif is_true and in_segment:
            last_frame = frame
        elif not is_true and in_segment:
            segments.append((int(start_frame), int(last_frame)))
            in_segment = False

    if in_segment:
        segments.append((int(start_frame), int(last_frame)))

    return segments


def _integrate_rate(series: pd.Series, frame_rate: float) -> float:
    """Integrate an instantaneous rate time-series using frame rate."""

    if series.empty:
        return float("nan")
    return float(series.astype(float).sum() / frame_rate)


def _count_lane_changes(lanes: pd.Series) -> int:
    """Count lane changes within a window based on laneId_raw transitions."""

    if lanes.empty:
        return 0
    changes = (lanes != lanes.shift()).sum() - 1
    return int(max(changes, 0))


def _build_conflict_events_internal(
    df_l1: pd.DataFrame,
    frame_rate: float,
    ttc_conf_thresh: float,
    min_conf_dur: float,
    pre_event_time: float,
    post_event_time: float,
) -> pd.DataFrame:
    """Shared implementation for constructing conflict events.

    This helper is used by both :func:`build_conflict_events` (config-driven) and
    :func:`build_conflict_events_param` (fully parameterized) to avoid code
    duplication while supporting different threshold sources.
    """

    events: List[Dict] = []
    event_counter = 1

    for (rec_id, track_id), df_grp in df_l1.groupby(["recordingId", "trackId"]):
        df_grp = df_grp.sort_values("frame")
        frames = df_grp["frame"].to_numpy()
        ttc = df_grp["TTC"].to_numpy()
        mask_conflict = ttc < ttc_conf_thresh
        segments = find_contiguous_segments(mask_conflict, frames)

        if not segments:
            continue

        min_frame, max_frame = frames.min(), frames.max()
        pre_frames = int(round(pre_event_time * frame_rate))
        post_frames = int(round(post_event_time * frame_rate))

        for conf_start, conf_end in segments:
            conf_duration = (conf_end - conf_start + 1) / frame_rate
            if conf_duration < min_conf_dur:
                continue

            start_frame = max(min_frame, conf_start - pre_frames)
            end_frame = min(max_frame, conf_end + post_frames)

            window_mask = (df_grp["frame"] >= start_frame) & (df_grp["frame"] <= end_frame)
            window_df = df_grp.loc[window_mask]
            conf_df = df_grp[(df_grp["frame"] >= conf_start) & (df_grp["frame"] <= conf_end)]

            start_time = float(window_df.iloc[0]["time"]) if not window_df.empty else start_frame / frame_rate
            end_time = float(window_df.iloc[-1]["time"]) if not window_df.empty else end_frame / frame_rate

            events.append(
                {
                    "event_id": event_counter,
                    "recordingId": int(rec_id),
                    "ego_id": int(track_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration": float((end_frame - start_frame + 1) / frame_rate),
                    "conf_start_frame": int(conf_start),
                    "conf_end_frame": int(conf_end),
                    "conf_duration": float(conf_duration),
                    "min_TTC": float(window_df["TTC"].min()) if not window_df.empty else float("nan"),
                    "min_TTC_conf": float(conf_df["TTC"].min()) if not conf_df.empty else float("nan"),
                    "max_DRAC": float(conf_df["DRAC"].max()) if "DRAC" in conf_df else float("nan"),
                    "veh_class": df_grp.iloc[0].get("veh_class"),
                    "mean_speed": float(window_df["v_long_smooth"].mean()) if not window_df.empty else float("nan"),
                    "max_decel": float(window_df["a_long_smooth"].min()) if not window_df.empty else float("nan"),
                    "max_accel": float(window_df["a_long_smooth"].max()) if not window_df.empty else float("nan"),
                    "num_lane_changes": _count_lane_changes(window_df.get("laneId_raw", pd.Series(dtype=float))),
                    "E_cpf_CO2": _integrate_rate(
                        window_df.get("cpf_co2_rate_gps", pd.Series(dtype=float)), frame_rate
                    ),
                    "E_cpf_fuel": _integrate_rate(
                        window_df.get("cpf_fuel_rate_lps", pd.Series(dtype=float)), frame_rate
                    ),
                    "E_vsp_CO2": _integrate_rate(window_df.get("vsp_co2_rate", pd.Series(dtype=float)), frame_rate),
                    "E_vsp_NOx": _integrate_rate(window_df.get("vsp_nox_rate", pd.Series(dtype=float)), frame_rate),
                }
            )
            event_counter += 1

    dtype_map = {
        "event_id": "int64",
        "recordingId": "int32",
        "ego_id": "int32",
        "start_frame": "int32",
        "end_frame": "int32",
        "conf_start_frame": "int32",
        "conf_end_frame": "int32",
        "num_lane_changes": "int32",
    }

    df_events = pd.DataFrame(events)
    if df_events.empty:
        return pd.DataFrame(columns=dtype_map.keys())

    return df_events.astype(dtype_map)


def build_conflict_events_param(
    df_l1: pd.DataFrame,
    frame_rate: float,
    ttc_conf_thresh: float,
    min_conf_dur: float,
    pre_event_time: float,
    post_event_time: float,
) -> pd.DataFrame:
    """Construct L2 conflict events with parameterized thresholds.

    This variant mirrors :func:`build_conflict_events` but accepts thresholds and
    buffer durations explicitly, making it suitable for grid searches.
    """

    return _build_conflict_events_internal(
        df_l1=df_l1,
        frame_rate=frame_rate,
        ttc_conf_thresh=ttc_conf_thresh,
        min_conf_dur=min_conf_dur,
        pre_event_time=pre_event_time,
        post_event_time=post_event_time,
    )


def build_conflict_events(df_l1: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Construct L2 conflict events using TTC-based segments.

    Parameters
    ----------
    df_l1:
        L1 master frame dataframe for a single recording.
    frame_rate:
        Frame rate (Hz) used to convert frames to seconds.

    Returns
    -------
    pd.DataFrame
        Conflict events table with expanded pre/post windows and summary metrics.
    """

    return _build_conflict_events_internal(
        df_l1=df_l1,
        frame_rate=frame_rate,
        ttc_conf_thresh=config.TTC_CONF_THRESH,
        min_conf_dur=config.MIN_CONFLICT_DURATION,
        pre_event_time=config.PRE_EVENT_TIME,
        post_event_time=config.POST_EVENT_TIME,
    )


def build_baseline_events(df_l1: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Construct baseline events using sliding windows on trajectories."""

    events: List[Dict] = []
    event_counter = 1

    window_size = int(round(config.BASELINE_WINDOW_TIME * frame_rate))
    step_size = int(round(config.BASELINE_WINDOW_STEP_TIME * frame_rate))
    if window_size <= 0:
        return pd.DataFrame()

    for (rec_id, track_id), df_grp in df_l1.groupby(["recordingId", "trackId"]):
        df_grp = df_grp.sort_values("frame")
        frames = df_grp["frame"].to_numpy()
        num_frames = len(frames)
        if num_frames < window_size:
            continue

        for start_idx in range(0, num_frames - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_df = df_grp.iloc[start_idx:end_idx]

            if window_df["TTC"].min() <= config.TTC_BASE_MIN:
                continue
            if window_df["a_long_smooth"].abs().max() >= config.ACC_SMOOTH_THRESH:
                continue
            if window_df["laneId_raw"].nunique(dropna=True) > 1:
                continue
            if (window_df["precedingId"] <= 0).any():
                continue

            start_frame = int(window_df.iloc[0]["frame"])
            end_frame = int(window_df.iloc[-1]["frame"])
            start_time = float(window_df.iloc[0]["time"])
            end_time = float(window_df.iloc[-1]["time"])

            events.append(
                {
                    "event_id": event_counter,
                    "recordingId": int(rec_id),
                    "ego_id": int(track_id),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": float((end_frame - start_frame + 1) / frame_rate),
                    "veh_class": df_grp.iloc[0].get("veh_class"),
                    "min_TTC": float(window_df["TTC"].min()),
                    "mean_TTC": float(window_df["TTC"].mean()),
                    "mean_speed": float(window_df["v_long_smooth"].mean()),
                    "max_decel": float(window_df["a_long_smooth"].min()),
                    "max_accel": float(window_df["a_long_smooth"].max()),
                    "num_lane_changes": _count_lane_changes(window_df.get("laneId_raw", pd.Series(dtype=float))),
                    "E_cpf_CO2": _integrate_rate(
                        window_df.get("cpf_co2_rate_gps", pd.Series(dtype=float)), frame_rate
                    ),
                    "E_cpf_fuel": _integrate_rate(
                        window_df.get("cpf_fuel_rate_lps", pd.Series(dtype=float)), frame_rate
                    ),
                    "E_vsp_CO2": _integrate_rate(window_df.get("vsp_co2_rate", pd.Series(dtype=float)), frame_rate),
                    "E_vsp_NOx": _integrate_rate(window_df.get("vsp_nox_rate", pd.Series(dtype=float)), frame_rate),
                }
            )
            event_counter += 1

    return pd.DataFrame(events).astype(
        {
            "event_id": "int64",
            "recordingId": "int32",
            "ego_id": "int32",
            "start_frame": "int32",
            "end_frame": "int32",
            "num_lane_changes": "int32",
        }
    )


__all__ = [
    "find_contiguous_segments",
    "build_conflict_events",
    "build_conflict_events_param",
    "build_baseline_events",
]
