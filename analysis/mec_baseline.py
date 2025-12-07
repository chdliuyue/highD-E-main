from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
DEFAULT_MEC_PATH = PROJECT_ROOT / "data" / "analysis" / "L2_conf_mec_baseline.parquet"


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
    duration = df_ext.get("duration", df_ext.get("conf_duration"))
    with np.errstate(divide="ignore", invalid="ignore"):
        df_ext["E_CO2_per_s"] = df_ext.get("E_cpf_CO2", np.nan) / duration
        df_ext["E_fuel_per_s"] = df_ext.get("E_cpf_fuel", np.nan) / duration
    return df_ext


def _extract_duration(row: pd.Series) -> float:
    for key in ["duration", "conf_duration"]:
        if key in row and not pd.isna(row[key]):
            return float(row[key])
    return np.nan


def _extract_mean_speed(row: pd.Series) -> float:
    for key in ["mean_speed", "v_mean", "avg_speed", "mean_v"]:
        if key in row and not pd.isna(row[key]):
            return float(row[key])
    return np.nan


def _winsorize_series(series: pd.Series, whisker: float = 3.0) -> pd.Series:
    if series.empty:
        return series
    q1, q3 = np.nanquantile(series, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    return series.clip(lower, upper)


def _score_candidates(conflict_row: pd.Series, candidates: pd.DataFrame, dur_tolerance: float, speed_tolerance: float) -> pd.DataFrame:
    dur_conf = _extract_duration(conflict_row)
    speed_conf = _extract_mean_speed(conflict_row)
    flow_conf = conflict_row.get("flow_state", "unknown")
    rec_conf = int(conflict_row.get("rec_id", conflict_row.get("recordingId", -1)))

    scored = candidates.copy()
    if not np.isnan(dur_conf):
        scored = scored.assign(duration_diff=scored["duration"].sub(dur_conf).abs())
    else:
        scored = scored.assign(duration_diff=np.inf)

    speed_base = scored.apply(_extract_mean_speed, axis=1)
    speed_diff = np.abs(speed_base - speed_conf) if not np.isnan(speed_conf) else np.full(len(scored), np.nan)
    scored = scored.assign(speed_diff=speed_diff)

    flow_penalty = np.where(scored.get("flow_state", "unknown") == flow_conf, 0.0, 2.0)
    rec_penalty = np.where(scored.get("rec_id", -1) == rec_conf, -0.5, 0.0)

    dur_term = scored["duration_diff"].fillna(dur_tolerance * 2)
    speed_term = scored["speed_diff"].fillna(speed_tolerance * 2)
    scored = scored.assign(score=np.sqrt(dur_term**2 + (speed_term / max(speed_tolerance, 1e-3)) ** 2) + flow_penalty + rec_penalty)
    return scored


def _select_baseline_candidates(conflict_row: pd.Series, df_base: pd.DataFrame, dur_tolerance: float) -> pd.DataFrame:
    veh_class = conflict_row.get("veh_class")
    dur_conf = _extract_duration(conflict_row)
    rec_id = int(conflict_row.get("rec_id", conflict_row.get("recordingId", -1)))

    candidates = df_base.copy()
    if veh_class is not None:
        candidates = candidates[candidates.get("veh_class") == veh_class]
    if candidates.empty:
        return candidates

    if not np.isnan(dur_conf):
        dur_mask = candidates["duration"].sub(dur_conf).abs() <= dur_tolerance
        filtered = candidates[dur_mask]
        if not filtered.empty:
            candidates = filtered

    same_rec = candidates[candidates.get("rec_id") == rec_id]
    if not same_rec.empty:
        return same_rec

    flow_state = conflict_row.get("flow_state")
    if flow_state is not None and "flow_state" in candidates.columns:
        same_flow = candidates[candidates.get("flow_state") == flow_state]
        if not same_flow.empty:
            return same_flow

    return candidates


def match_baseline_for_conflicts(
    df_conf: pd.DataFrame,
    df_base: pd.DataFrame,
    max_candidates: int = 10,
    k_neighbors: int = 3,
    dur_tolerance: float = 2.0,
    speed_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    为每条 high-interaction episode 选择一个或多个 baseline 匹配样本。
    """

    if df_conf.empty or df_base.empty:
        return pd.DataFrame()

    matched_rows: list[Dict] = []
    df_base = df_base.copy()
    df_base["duration"] = df_base.get("duration", df_base.get("conf_duration"))

    for _, conf_row in df_conf.iterrows():
        candidates = _select_baseline_candidates(conf_row, df_base, dur_tolerance=dur_tolerance)
        if candidates.empty:
            continue

        scored = _score_candidates(conf_row, candidates, dur_tolerance=dur_tolerance, speed_tolerance=speed_tolerance)
        top = scored.nsmallest(k_neighbors, "score")

        row_dict = conf_row.to_dict()
        row_dict["baseline_rec_ids"] = top.get("rec_id", pd.Series(dtype=float)).tolist()
        row_dict["baseline_event_ids"] = top.get("event_id", pd.Series(dtype=float)).tolist()
        row_dict["E_CO2_per_s_base"] = float(np.nanmean(top.get("E_CO2_per_s", np.nan)))
        row_dict["E_fuel_per_s_base"] = float(np.nanmean(top.get("E_fuel_per_s", np.nan)))
        row_dict["duration_base"] = float(np.nanmean(top.get("duration", np.nan)))
        row_dict["veh_class_base"] = top.get("veh_class").iloc[0] if "veh_class" in top else np.nan
        matched_rows.append(row_dict)

        if len(matched_rows) >= max_candidates and max_candidates > 0:
            break

    if not matched_rows:
        return pd.DataFrame()
    return pd.DataFrame(matched_rows)


def compute_mec_from_matching(df_matched: pd.DataFrame) -> pd.DataFrame:
    """
    在匹配后的 DataFrame 上计算 MEC，并进行稳健剪裁。
    """

    if df_matched.empty:
        return df_matched.copy()

    df = df_matched.copy()
    duration_conf = df.get("conf_duration", df.get("duration"))

    with np.errstate(divide="ignore", invalid="ignore"):
        df["E_real_CO2_per_s"] = df.get("E_cpf_CO2") / duration_conf
        df["E_base_CO2_per_s"] = df.get("E_CO2_per_s_base")
        df["MEC_CO2_per_s"] = df["E_real_CO2_per_s"] - df["E_base_CO2_per_s"]

    df["MEC_CO2_per_s"] = _winsorize_series(df["MEC_CO2_per_s"])
    return df


def build_and_save_mec(rec_ids: Sequence[int] | None = None, output_path: Path | None = None) -> pd.DataFrame:
    """
    生成 MEC baseline 匹配结果并写入统一路径。
    """

    out_path = Path(output_path) if output_path is not None else DEFAULT_MEC_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_conf = compute_event_level_metrics(load_all_conflict_events(rec_ids))
    df_base = compute_event_level_metrics(load_all_baseline_events(rec_ids))

    df_matched = match_baseline_for_conflicts(df_conf, df_base, max_candidates=len(df_conf))
    df_mec = compute_mec_from_matching(df_matched)

    if df_mec.empty:
        return df_mec

    df_mec.to_parquet(out_path, index=False)
    return df_mec
