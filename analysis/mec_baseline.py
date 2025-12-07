from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"


def _default_rec_ids(rec_ids: Sequence[int] | None) -> Sequence[int]:
    if rec_ids is None:
        return list(range(1, 61))
    return rec_ids


def load_all_conflict_events(rec_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """
    合并指定 rec_ids（或默认 1..60）的所有 L2_conflict_events.parquet，添加 rec_id 列。
    """

    rec_list = _default_rec_ids(rec_ids)
    dfs = []
    for rec_id in rec_list:
        rec_str = f"recording_{rec_id:02d}"
        path = EVENTS_ROOT / rec_str / "L2_conflict_events.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["rec_id"] = rec_id
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_all_baseline_events(rec_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """
    合并指定 rec_ids 的所有 L2_baseline_events.parquet，添加 rec_id 列。
    """

    rec_list = _default_rec_ids(rec_ids)
    dfs = []
    for rec_id in rec_list:
        rec_str = f"recording_{rec_id:02d}"
        path = EVENTS_ROOT / rec_str / "L2_baseline_events.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["rec_id"] = rec_id
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_event_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 L2 事件 DataFrame 上添加衍生指标:
      - E_CO2_per_s = E_cpf_CO2 / duration
      - E_fuel_per_s = E_cpf_fuel / duration
    返回扩展后的 DataFrame。
    """

    if df.empty:
        return df.copy()

    df_ext = df.copy()
    duration = df_ext.get("duration")
    with np.errstate(divide="ignore", invalid="ignore"):
        df_ext["E_CO2_per_s"] = df_ext.get("E_cpf_CO2", np.nan) / duration
        df_ext["E_fuel_per_s"] = df_ext.get("E_cpf_fuel", np.nan) / duration
    return df_ext


def _select_closest_baseline(
    conflict_row: pd.Series,
    df_base: pd.DataFrame,
    dur_tolerance: float = 2.0,
) -> pd.Series | None:
    veh_class = conflict_row.get("veh_class")
    dur_conf = float(conflict_row.get("duration", np.nan))
    rec_id = int(conflict_row.get("rec_id", conflict_row.get("recordingId", -1)))

    candidates = df_base.copy()
    if not candidates.empty:
        candidates = candidates[candidates.get("veh_class") == veh_class]
    if candidates.empty:
        return None

    if not np.isnan(dur_conf):
        candidates = candidates[candidates["duration"].sub(dur_conf).abs() <= dur_tolerance]
    if candidates.empty:
        candidates = df_base[df_base.get("veh_class") == veh_class]
        if candidates.empty():
            return None

    same_rec = candidates[candidates["rec_id"] == rec_id]
    if not same_rec.empty:
        candidates = same_rec

    candidates = candidates.assign(distance=candidates["duration"].sub(dur_conf).abs())
    best_idx = candidates["distance"].idxmin()
    return candidates.loc[best_idx]


def match_baseline_for_conflicts(
    df_conf: pd.DataFrame,
    df_base: pd.DataFrame,
    max_candidates: int = 10,
) -> pd.DataFrame:
    """
    为每条 high-interaction episode 选择一个 baseline 匹配样本。
    """

    if df_conf.empty or df_base.empty:
        return pd.DataFrame()

    matched_rows: list[Dict] = []
    for _, conf_row in df_conf.iterrows():
        base_row = _select_closest_baseline(conf_row, df_base)
        if base_row is None:
            continue

        row_dict = conf_row.to_dict()
        row_dict["baseline_rec_id"] = base_row.get("rec_id")
        row_dict["baseline_event_id"] = base_row.get("event_id")
        row_dict["E_CO2_per_s_base"] = base_row.get("E_CO2_per_s")
        row_dict["E_fuel_per_s_base"] = base_row.get("E_fuel_per_s")
        row_dict["duration_base"] = base_row.get("duration")
        row_dict["veh_class_base"] = base_row.get("veh_class")
        matched_rows.append(row_dict)

        if len(matched_rows) >= max_candidates and max_candidates > 0:
            break
    if not matched_rows:
        return pd.DataFrame()
    return pd.DataFrame(matched_rows)


def compute_mec_from_matching(
    df_matched: pd.DataFrame,
) -> pd.DataFrame:
    """
    在匹配后的 DataFrame 上计算 MEC。
    """

    if df_matched.empty:
        return df_matched.copy()

    df = df_matched.copy()
    df["MEC_CO2_per_s"] = df.get("E_CO2_per_s") - df.get("E_CO2_per_s_base")
    df["MEC_fuel_per_s"] = df.get("E_fuel_per_s") - df.get("E_fuel_per_s_base")
    return df
