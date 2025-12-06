"""Event extraction utilities for L2 processing.

This module centralizes the logic for detecting high-interaction (conflict)
segments and baseline episodes from L1 master tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config


def find_contiguous_segments(mask: np.ndarray, frames: np.ndarray) -> List[Tuple[int, int]]:
    """Return start/end frames for contiguous ``True`` segments."""

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


def _duration_seconds(df: pd.DataFrame, frame_rate: float) -> float:
    if df.empty:
        return float("nan")
    if "dt" in df.columns and df["dt"].notna().any():
        return float(df["dt"].fillna(1.0 / frame_rate).sum())
    frames = df["frame"].to_numpy()
    return float((frames.max() - frames.min() + 1) / frame_rate)


def _energy_from_rate(df: pd.DataFrame, rate_col: str, frame_rate: float) -> float:
    if df.empty or rate_col not in df.columns:
        return float("nan")
    rates = df[rate_col].astype(float)
    if "dt" in df.columns and df["dt"].notna().any():
        dt = df["dt"].fillna(1.0 / frame_rate)
        return float((rates * dt).sum())
    return float(rates.sum() / frame_rate)


def _count_lane_changes(lanes: pd.Series) -> int:
    if lanes.empty:
        return 0
    changes = (lanes != lanes.shift()).sum() - 1
    return int(max(changes, 0))


def extract_high_interaction_events(
    df_l1: pd.DataFrame,
    frame_rate: float,
    ttc_upper: float = 4.0,
    min_conf_duration: float = 0.4,
    pre_event_time: float = config.PRE_EVENT_TIME,
    post_event_time: float = config.POST_EVENT_TIME,
) -> pd.DataFrame:
    """Detect high-interaction (conflict) events from an L1 dataframe."""

    events: List[Dict] = []
    event_counter = 1

    for (rec_id, track_id), df_grp in df_l1.groupby(["recordingId", "trackId"]):
        df_grp = df_grp.sort_values("frame")
        frames = df_grp["frame"].to_numpy()
        ttc = df_grp["TTC"].to_numpy()
        mask_conflict = ttc < ttc_upper
        segments = find_contiguous_segments(mask_conflict, frames)

        if not segments:
            continue

        min_frame, max_frame = frames.min(), frames.max()
        pre_frames = int(round(pre_event_time * frame_rate))
        post_frames = int(round(post_event_time * frame_rate))

        for conf_start, conf_end in segments:
            conf_df = df_grp[(df_grp["frame"] >= conf_start) & (df_grp["frame"] <= conf_end)]
            conf_duration = _duration_seconds(conf_df, frame_rate)
            if conf_duration < min_conf_duration:
                continue

            start_frame = max(min_frame, conf_start - pre_frames)
            end_frame = min(max_frame, conf_end + post_frames)

            window_df = df_grp[(df_grp["frame"] >= start_frame) & (df_grp["frame"] <= end_frame)]
            if window_df.empty:
                continue

            start_time = float(window_df.iloc[0].get("time", start_frame / frame_rate))
            end_time = float(window_df.iloc[-1].get("time", end_frame / frame_rate))

            events.append(
                {
                    "event_id": event_counter,
                    "recordingId": int(rec_id),
                    "ego_id": int(track_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": _duration_seconds(window_df, frame_rate),
                    "conf_start_frame": int(conf_start),
                    "conf_end_frame": int(conf_end),
                    "conf_duration": conf_duration,
                    "min_TTC": float(window_df["TTC"].min()),
                    "min_TTC_conf": float(conf_df["TTC"].min()),
                    "veh_class": window_df.iloc[0].get("veh_class"),
                    "num_lane_changes": _count_lane_changes(window_df.get("laneId_raw", pd.Series(dtype=float))),
                    "E_cpf_CO2": _energy_from_rate(window_df, "cpf_co2_rate_gps", frame_rate),
                    "E_cpf_fuel": _energy_from_rate(window_df, "cpf_fuel_rate_lps", frame_rate),
                    "E_vsp_CO2": _energy_from_rate(window_df, "vsp_co2_rate", frame_rate),
                    "E_vsp_NOx": _energy_from_rate(window_df, "vsp_nox_rate", frame_rate),
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
    if not df_events.empty:
        df_events = df_events[list(df_events.columns)].astype({k: v for k, v in dtype_map.items() if k in df_events.columns})
    return df_events


def extract_baseline_events(
    df_l1: pd.DataFrame,
    frame_rate: float,
    ttc_min: float = config.TTC_BASE_MIN,
    acc_thresh: float = config.ACC_SMOOTH_THRESH,
    window_time: float = config.BASELINE_WINDOW_TIME,
    step_time: float = config.BASELINE_WINDOW_STEP_TIME,
) -> pd.DataFrame:
    """Construct baseline events using sliding windows on trajectories."""

    events: List[Dict] = []
    event_counter = 1

    window_size = int(round(window_time * frame_rate))
    step_size = int(round(step_time * frame_rate))
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

            if window_df["TTC"].min() <= ttc_min:
                continue
            if window_df["a_long_smooth"].abs().max() >= acc_thresh:
                continue
            if window_df["laneId_raw"].nunique(dropna=True) > 1:
                continue

            start_frame = int(window_df.iloc[0]["frame"])
            end_frame = int(window_df.iloc[-1]["frame"])
            start_time = float(window_df.iloc[0].get("time", start_frame / frame_rate))
            end_time = float(window_df.iloc[-1].get("time", end_frame / frame_rate))

            events.append(
                {
                    "event_id": event_counter,
                    "recordingId": int(rec_id),
                    "ego_id": int(track_id),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": _duration_seconds(window_df, frame_rate),
                    "veh_class": window_df.iloc[0].get("veh_class"),
                    "min_TTC": float(window_df["TTC"].min()),
                    "mean_TTC": float(window_df["TTC"].mean()),
                    "num_lane_changes": _count_lane_changes(window_df.get("laneId_raw", pd.Series(dtype=float))),
                    "E_cpf_CO2": _energy_from_rate(window_df, "cpf_co2_rate_gps", frame_rate),
                    "E_cpf_fuel": _energy_from_rate(window_df, "cpf_fuel_rate_lps", frame_rate),
                    "E_vsp_CO2": _energy_from_rate(window_df, "vsp_co2_rate", frame_rate),
                    "E_vsp_NOx": _energy_from_rate(window_df, "vsp_nox_rate", frame_rate),
                }
            )
            event_counter += 1

    df_events = pd.DataFrame(events)
    if not df_events.empty:
        df_events = df_events.astype(
            {
                "event_id": "int64",
                "recordingId": "int32",
                "ego_id": "int32",
                "start_frame": "int32",
                "end_frame": "int32",
                "num_lane_changes": "int32",
            }
        )
    return df_events


# ----------------------------------------------------------------------
# Convenience wrappers used by scripts/CLI
# ----------------------------------------------------------------------

def build_conflict_events(df_l1: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    return extract_high_interaction_events(
        df_l1,
        frame_rate,
        ttc_upper=config.TTC_CONF_THRESH,
        min_conf_duration=config.MIN_CONFLICT_DURATION,
        pre_event_time=config.PRE_EVENT_TIME,
        post_event_time=config.POST_EVENT_TIME,
    )


def build_conflict_events_param(
    df_l1: pd.DataFrame,
    frame_rate: float,
    ttc_conf_thresh: float,
    min_conf_dur: float,
    pre_event_time: float,
    post_event_time: float,
) -> pd.DataFrame:
    return extract_high_interaction_events(
        df_l1,
        frame_rate,
        ttc_upper=ttc_conf_thresh,
        min_conf_duration=min_conf_dur,
        pre_event_time=pre_event_time,
        post_event_time=post_event_time,
    )


def build_baseline_events(df_l1: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    return extract_baseline_events(df_l1, frame_rate)


def build_events_for_recording(
    rec_id: int,
    l1_path: Path,
    events_dir: Path,
    frame_rate: float,
    ttc_conf_thresh: float | None = None,
    min_conf_dur: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load L1 data, build conflict/baseline events, and persist to Parquet."""

    if not l1_path.exists():
        raise FileNotFoundError(f"L1 master frame not found for recording {rec_id:02d} at {l1_path}")

    df_l1 = pd.read_parquet(l1_path)
    df_rec = df_l1[df_l1["recordingId"] == rec_id].copy()

    df_conf = extract_high_interaction_events(
        df_rec,
        frame_rate,
        ttc_upper=config.TTC_CONF_THRESH if ttc_conf_thresh is None else ttc_conf_thresh,
        min_conf_duration=config.MIN_CONFLICT_DURATION if min_conf_dur is None else min_conf_dur,
        pre_event_time=config.PRE_EVENT_TIME,
        post_event_time=config.POST_EVENT_TIME,
    )
    df_base = extract_baseline_events(df_rec, frame_rate)

    events_dir.mkdir(parents=True, exist_ok=True)
    df_conf.to_parquet(events_dir / "L2_conflict_events.parquet", index=False)
    df_base.to_parquet(events_dir / "L2_baseline_events.parquet", index=False)

    return df_conf, df_base


__all__ = [
    "find_contiguous_segments",
    "extract_high_interaction_events",
    "extract_baseline_events",
    "build_conflict_events",
    "build_conflict_events_param",
    "build_baseline_events",
    "build_events_for_recording",
]
