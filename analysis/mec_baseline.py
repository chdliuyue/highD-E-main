from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
L2_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT = PROJECT_ROOT / "output" / "mec"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_l1_for_recording(rec_id: int) -> pd.DataFrame:
    """读取 recording_XX 的 L1_master_frame.parquet。"""

    rec_str = f"{rec_id:02d}"
    parquet_path = L1_ROOT / f"recording_{rec_str}" / "L1_master_frame.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def load_l2_for_recording(rec_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取 recording_XX 的 L2_conflict_events 和 L2_baseline_events。
    若某个 parquet 不存在，则返回空 DataFrame（shape=(0,0)）。
    """

    rec_str = f"{rec_id:02d}"
    conf_path = L2_ROOT / f"recording_{rec_str}" / "L2_conflict_events.parquet"
    base_path = L2_ROOT / f"recording_{rec_str}" / "L2_baseline_events.parquet"

    df_conf = pd.read_parquet(conf_path) if conf_path.exists() else pd.DataFrame()
    df_base = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
    return df_conf, df_base


def _get_dt(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.notna().any():
        return series.fillna(0)
    return pd.Series(np.zeros(len(series)), index=series.index)


def _compute_duration_from_frames(df_episode: pd.DataFrame) -> float:
    if "time" in df_episode.columns:
        time_vals = df_episode["time"].astype(float)
        return float(time_vals.max() - time_vals.min()) if not df_episode.empty else float("nan")
    if "dt" in df_episode.columns:
        return float(df_episode["dt"].sum()) if not df_episode.empty else float("nan")
    if "frame" in df_episode.columns and not df_episode.empty:
        frame_span = df_episode["frame"].max() - df_episode["frame"].min()
        return float(frame_span * 0.04)
    return float("nan")


def compute_energy_for_episode(df_l1: pd.DataFrame, event_row: pd.Series) -> dict:
    """
    根据 L1 帧级数据，对单个事件（冲突或 baseline）计算:
      - E_CO2 [g]，通过 ∑ cpf_co2_rate_gps * dt；
      - E_fuel [L]，通过 ∑ cpf_fuel_rate_lps * dt（若字段存在，否则设 NaN）；
      - dist_m [m]，通过 s_long_end - s_long_start；
      - duration [s]，可以用 event_row['duration'] 或根据时间/帧计算。
    返回一个 dict，字段名如:
      {
        "E_CO2": ...,
        "E_fuel": ...,
        "dist_m": ...,
        "duration": ...
      }
    """

    required_cols = {"recordingId", "trackId", "frame"}
    if df_l1.empty or not required_cols.issubset(df_l1.columns):
        return {
            "E_CO2": np.nan,
            "E_fuel": np.nan,
            "dist_m": np.nan,
            "duration": float(event_row.get("duration", np.nan)),
            "mean_v": np.nan,
        }

    rec_val = event_row.get("recordingId", event_row.get("rec_id", np.nan))
    rec_id = int(rec_val) if pd.notna(rec_val) else -1
    ego_id = event_row.get("ego_id", event_row.get("trackId"))
    start_val = event_row.get("start_frame", event_row.get("conf_start_frame", np.nan))
    end_val = event_row.get("end_frame", event_row.get("conf_end_frame", np.nan))
    start_f = int(start_val) if pd.notna(start_val) else -1
    end_f = int(end_val) if pd.notna(end_val) else -1

    if start_f < 0 or end_f < 0:
        return {
            "E_CO2": np.nan,
            "E_fuel": np.nan,
            "dist_m": np.nan,
            "duration": float(event_row.get("duration", np.nan)),
            "mean_v": np.nan,
        }

    mask = (
        (df_l1["recordingId"] == rec_id)
        & (df_l1["trackId"] == ego_id)
        & (df_l1["frame"] >= start_f)
        & (df_l1["frame"] <= end_f)
    )
    df_episode = df_l1.loc[mask].copy()

    if df_episode.empty:
        return {
            "E_CO2": np.nan,
            "E_fuel": np.nan,
            "dist_m": np.nan,
            "duration": float(event_row.get("duration", np.nan)),
            "mean_v": np.nan,
        }

    dt = _get_dt(df_episode["dt"]) if "dt" in df_episode.columns else None
    if dt is None:
        if "time" in df_episode.columns:
            dt = df_episode["time"].diff().fillna(method="bfill").fillna(0)
        else:
            dt = pd.Series(np.zeros(len(df_episode)), index=df_episode.index)

    co2_rate = df_episode["cpf_co2_rate_gps"] if "cpf_co2_rate_gps" in df_episode.columns else None
    fuel_rate = df_episode["cpf_fuel_rate_lps"] if "cpf_fuel_rate_lps" in df_episode.columns else None
    E_CO2 = float((co2_rate * dt).sum()) if co2_rate is not None else np.nan
    E_fuel = float((fuel_rate * dt).sum()) if fuel_rate is not None else np.nan

    if "s_long" in df_episode.columns:
        dist_m = float(df_episode["s_long"].iloc[-1] - df_episode["s_long"].iloc[0]) if len(df_episode) > 1 else float("nan")
    elif "s" in df_episode.columns:
        dist_m = float(df_episode["s"].iloc[-1] - df_episode["s"].iloc[0]) if len(df_episode) > 1 else float("nan")
    else:
        dist_m = np.nan

    duration = float(event_row.get("duration", _compute_duration_from_frames(df_episode)))
    mean_v = float(df_episode["v_long_smooth"].mean()) if "v_long_smooth" in df_episode.columns and not df_episode.empty else np.nan

    return {
        "E_CO2": E_CO2,
        "E_fuel": E_fuel,
        "dist_m": dist_m,
        "duration": duration,
        "mean_v": mean_v,
    }


def _per_km(value: float, dist_m: float) -> float:
    if dist_m is None or np.isnan(dist_m) or dist_m <= 1e-6:
        return np.nan
    return value / (dist_m / 1000.0)


def _per_s(value: float, duration: float) -> float:
    if duration is None or np.isnan(duration) or duration <= 0:
        return np.nan
    return value / duration


def _bucket_min_ttc(min_ttc: float) -> str:
    if np.isnan(min_ttc):
        return "unknown"
    if min_ttc < 2:
        return "<2"
    if min_ttc < 3:
        return "2-3"
    if min_ttc < 4:
        return "3-4"
    return ">=4"


def match_baseline_for_conflicts(
    df_conf: pd.DataFrame, df_base: pd.DataFrame, df_l1: pd.DataFrame
) -> pd.DataFrame:
    """
    对单个 recording 的冲突事件与 baseline 事件进行匹配，并计算 MEC 所需指标。

    若某个冲突事件找不到候选 baseline，则对应 baseline、MEC 相关指标设为 NaN。

    返回：
      - 组合好的 DataFrame（针对该 recording 的所有冲突事件）。
    """

    if df_conf.empty:
        return pd.DataFrame()

    # 预先缓存 baseline 事件的能耗指标，避免在每个冲突事件匹配时重复计算。
    base_metrics_cache: dict[int, dict] = {}
    if not df_base.empty:
        for base_idx, base_row in df_base.iterrows():
            base_metrics_cache[base_idx] = compute_energy_for_episode(df_l1, base_row)

    records = []
    for event_id, row in df_conf.reset_index().iterrows():
        rec_id = int(row.get("recordingId", row.get("rec_id", -1)))
        ego_id = row.get("ego_id", row.get("trackId"))
        veh_class = row.get("veh_class", row.get("class", "unknown"))
        flow_state = row.get("flow_state", "unknown")

        conf_energy = compute_energy_for_episode(df_l1, row)
        duration_conf = conf_energy["duration"]
        mean_v_conf = conf_energy["mean_v"]

        df_candidates = df_base
        if not df_candidates.empty and "veh_class" in df_candidates.columns:
            df_candidates = df_candidates[df_candidates["veh_class"] == veh_class]
        if not df_candidates.empty and not np.isnan(duration_conf):
            duration_series = df_candidates.index.to_series().map(
                lambda idx: base_metrics_cache.get(idx, {}).get("duration", np.nan)
            )
            df_candidates = df_candidates[np.abs(duration_series - duration_conf) <= 2.0]

        best_candidate = None
        best_dist = None

        for base_idx, base_row in df_candidates.iterrows():
            base_energy = base_metrics_cache.get(base_idx)
            if base_energy is None:
                base_energy = compute_energy_for_episode(df_l1, base_row)
                base_metrics_cache[base_idx] = base_energy

            duration_base = base_energy["duration"]
            mean_v_base = base_energy["mean_v"]

            delta_t = duration_conf - duration_base if not np.isnan(duration_base) else np.nan
            delta_v = mean_v_conf - mean_v_base if not np.isnan(mean_v_base) else np.nan
            if np.isnan(delta_t) or np.isnan(delta_v):
                continue
            dist = (delta_t / 2.0) ** 2 + (delta_v / 5.0) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_candidate = (base_row, base_energy)

        base_energy = {
            "E_CO2": np.nan,
            "E_fuel": np.nan,
            "dist_m": np.nan,
            "duration": np.nan,
            "mean_v": np.nan,
        }
        base_row_info = {}
        if best_candidate is not None:
            base_row, base_energy = best_candidate
            base_row_info = base_row.to_dict()

        E_real_CO2 = conf_energy["E_CO2"]
        E_base_CO2 = base_energy["E_CO2"]

        dist_real = conf_energy["dist_m"]
        dist_base = base_energy["dist_m"]
        duration_base = base_energy.get("duration", np.nan)

        record = {
            "rec_id": rec_id,
            "event_id": event_id,
            "ego_id": ego_id,
            "veh_class": veh_class,
            "flow_state": flow_state,
            "min_TTC_conf": row.get("min_TTC_conf", np.nan),
            "conf_duration": row.get("conf_duration", np.nan),
            "duration": conf_energy.get("duration", row.get("duration", np.nan)),
            "duration_base": duration_base,
            "dist_real_m": dist_real,
            "dist_base_m": dist_base,
            "E_real_CO2": E_real_CO2,
            "E_real_fuel": conf_energy.get("E_fuel", np.nan),
            "E_base_CO2": E_base_CO2,
            "E_base_fuel": base_energy.get("E_fuel", np.nan),
            "mean_v_conf": mean_v_conf,
            "mean_v_base": base_energy.get("mean_v", np.nan),
        }

        record["E_real_CO2_per_km"] = _per_km(record["E_real_CO2"], dist_real)
        record["E_base_CO2_per_km"] = _per_km(record["E_base_CO2"], dist_base)
        record["E_real_CO2_per_s"] = _per_s(record["E_real_CO2"], record["duration"])
        record["E_base_CO2_per_s"] = _per_s(record["E_base_CO2"], duration_base)

        record["MEC_CO2_per_km"] = (
            record["E_real_CO2_per_km"] - record["E_base_CO2_per_km"]
        )
        record["MEC_CO2_per_event"] = record["E_real_CO2"] - record["E_base_CO2"]
        record["MEC_bucket"] = _bucket_min_ttc(record["min_TTC_conf"])

        record.update({f"base_{k}": v for k, v in base_row_info.items()})
        records.append(record)

    return pd.DataFrame(records)


def _debug_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    stats = {
        "n_events": len(df),
        "mec_per_km_mean": df["MEC_CO2_per_km"].mean(),
        "mec_per_km_median": df["MEC_CO2_per_km"].median(),
        "mec_per_km_min": df["MEC_CO2_per_km"].min(),
        "mec_per_km_max": df["MEC_CO2_per_km"].max(),
        "mec_per_event_mean": df["MEC_CO2_per_event"].mean(),
        "mec_per_event_median": df["MEC_CO2_per_event"].median(),
    }
    df_summary = pd.DataFrame([stats])

    buckets = df.groupby("MEC_bucket")["MEC_CO2_per_km"].median().rename("median_MEC_CO2_per_km")
    bucket_df = buckets.reset_index().rename(columns={"MEC_bucket": "min_TTC_bucket"})
    return pd.concat([df_summary, bucket_df], axis=0, ignore_index=True)


def generate_mec_for_recordings(rec_ids: Sequence[int]) -> pd.DataFrame:
    """
    对给定 rec_ids（例如 [1,2,3,...,60]）：
      - 对每个 rec_id:
          * 读取 L1/L2；
          * 调用 match_baseline_for_conflicts 生成该 rec 的 MEC 数据 DataFrame；
      - 将各 rec 的结果 concat 成一个总的 MEC DataFrame；
      - 写入 parquet：
          ANALYSIS_ROOT / "L2_conf_mec_baseline.parquet"
      - 同时生成一个粗略的 debug summary CSV：
          OUT_ROOT / "mec_debug_summary.csv"
        例如包括：
          - 总事件数
          - MEC_CO2_per_km 和 MEC_CO2_per_event 的统计量（mean/median/min/max）
          - 按 min_TTC_conf 桶（<2, 2–3, 3–4）分组的 median MEC 等。
    返回总的 MEC DataFrame（方便 exp 层进一步使用）。
    """

    all_results = []
    total = len(rec_ids)
    for idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[Stage 04] ({idx}/{total}) Matching baseline for recording {rec_id:02d}...")
        df_l1 = load_l1_for_recording(rec_id)
        df_conf, df_base = load_l2_for_recording(rec_id)
        rec_result = match_baseline_for_conflicts(df_conf, df_base, df_l1)
        all_results.append(rec_result)

    non_empty = [df for df in all_results if not df.empty]
    if non_empty:
        df_mec = pd.concat(non_empty, ignore_index=True)
    else:
        df_mec = pd.DataFrame()
    out_parquet = ANALYSIS_ROOT / "L2_conf_mec_baseline.parquet"
    df_mec.to_parquet(out_parquet, index=False)

    debug_df = _debug_summary(df_mec)
    debug_path = OUT_ROOT / "mec_debug_summary.csv"
    debug_df.to_csv(debug_path, index=False)

    return df_mec
