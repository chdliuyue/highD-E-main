from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
L2_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
OUT_ROOT = PROJECT_ROOT / "output" / "timeseries"
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


def _get_min_ttc_frame(event_row: pd.Series) -> int | None:
    for key in ["min_ttc_frame", "frame_min_ttc", "conf_min_ttc_frame", "ttc_min_frame"]:
        if key in event_row and pd.notna(event_row[key]):
            return int(event_row[key])
    return None


def _extract_episode_series(
    df_l1: pd.DataFrame,
    event_row: pd.Series,
    frame_rate: float,
    t_window: Tuple[float, float],
) -> dict[str, np.ndarray]:
    required_cols = {"recordingId", "trackId", "frame"}
    if not required_cols.issubset(df_l1.columns):
        return {}

    rec_val = event_row.get("rec_id", event_row.get("recordingId", np.nan))
    ego_id = event_row.get("ego_id", event_row.get("trackId", np.nan))

    if pd.isna(rec_val) or pd.isna(ego_id):
        return {}

    min_ttc_frame = _get_min_ttc_frame(event_row)

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

    if min_ttc_frame is None and "TTC" in df_ep.columns:
        ttc_series = df_ep["TTC"]
        if ttc_series.notna().any():
            min_ttc_frame = int(df_ep.loc[ttc_series.idxmin(), "frame"])

    if min_ttc_frame is None:
        return {}

    df_ep["t_rel"] = (df_ep["frame"] - min_ttc_frame) / frame_rate
    df_ep = df_ep[(df_ep["t_rel"] >= t_window[0]) & (df_ep["t_rel"] <= t_window[1])]
    if df_ep.empty:
        return {}

    df_ep = df_ep.sort_values("t_rel")
    t_vals = df_ep["t_rel"].to_numpy()
    ttc_vals = df_ep["TTC"].to_numpy() if "TTC" in df_ep.columns else None
    co2_vals = df_ep["cpf_co2_rate_gps"].to_numpy() if "cpf_co2_rate_gps" in df_ep.columns else None

    t_grid = np.arange(t_window[0], t_window[1] + 1e-9, 1.0 / frame_rate)

    def _interp(values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.full_like(t_grid, np.nan, dtype=float)
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return np.full_like(t_grid, np.nan, dtype=float)
        t_valid = t_vals[valid_mask]
        v_valid = values[valid_mask]
        if len(t_valid) == 1:
            return np.where(
                (t_grid >= t_valid[0] - (1.0 / frame_rate)) & (t_grid <= t_valid[0] + (1.0 / frame_rate)),
                v_valid[0],
                np.nan,
            )
        return np.interp(t_grid, t_valid, v_valid, left=np.nan, right=np.nan)

    lag = np.nan
    if co2_vals is not None:
        valid_mask = ~np.isnan(co2_vals)
        if valid_mask.any():
            lag = float(t_vals[valid_mask][np.argmax(co2_vals[valid_mask])])

    return {"t": t_grid, "ttc": _interp(ttc_vals), "co2": _interp(co2_vals), "lag": lag}


def aggregate_ttc_co2_timeseries(
    rec_ids: Sequence[int],
    frame_rate: float = 25.0,
    t_window: Tuple[float, float] = (-5.0, 10.0),
    max_episodes: int | None = None,
) -> Dict[str, np.ndarray]:
    t_grid = np.arange(t_window[0], t_window[1] + 1e-9, 1.0 / frame_rate)
    ttc_stack: list[np.ndarray] = []
    co2_stack: list[np.ndarray] = []
    lags: list[float] = []
    episode_count = 0

    total = len(rec_ids)
    for idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[Stage 07] ({idx}/{total}) Loading data for recording {rec_id:02d}...")
        df_l1 = _load_l1(rec_id)
        df_l2 = _load_l2_conflicts(rec_id)
        if df_l1.empty or df_l2.empty:
            print(f"  [Stage 07] Skipping recording {rec_id:02d}: missing L1/L2 data")
            continue

        for _, row in df_l2.iterrows():
            if max_episodes is not None and episode_count >= max_episodes:
                break
            ts = _extract_episode_series(df_l1, row, frame_rate, t_window)
            if not ts:
                continue
            ttc_stack.append(ts["ttc"])
            co2_stack.append(ts["co2"])
            lags.append(ts["lag"])
            episode_count += 1
        print(
            f"  [Stage 07] Processed recording {rec_id:02d} | episodes so far: {episode_count}"
        )

    if not ttc_stack:
        empty = np.full_like(t_grid, np.nan, dtype=float)
        return {
            "t_grid": t_grid,
            "TTC_mean": empty,
            "TTC_p10": empty,
            "TTC_p90": empty,
            "CO2_mean": empty,
            "CO2_p10": empty,
            "CO2_p90": empty,
            "lags": np.array([], dtype=float),
            "n_episodes": 0,
        }

    ttc_arr = np.vstack(ttc_stack)
    co2_arr = np.vstack(co2_stack)

    ttc_mean = np.nanmean(ttc_arr, axis=0)
    ttc_p10 = np.nanpercentile(ttc_arr, 10, axis=0)
    ttc_p90 = np.nanpercentile(ttc_arr, 90, axis=0)

    co2_mean = np.nanmean(co2_arr, axis=0)
    co2_p10 = np.nanpercentile(co2_arr, 10, axis=0)
    co2_p90 = np.nanpercentile(co2_arr, 90, axis=0)

    return {
        "t_grid": t_grid,
        "TTC_mean": ttc_mean,
        "TTC_p10": ttc_p10,
        "TTC_p90": ttc_p90,
        "CO2_mean": co2_mean,
        "CO2_p10": co2_p10,
        "CO2_p90": co2_p90,
        "lags": np.asarray(lags, dtype=float),
        "n_episodes": len(ttc_stack),
    }


def plot_ttc_co2_alignment(
    t: np.ndarray,
    ttc_mean: np.ndarray,
    ttc_p10: np.ndarray,
    ttc_p90: np.ndarray,
    co2_mean: np.ndarray,
    co2_p10: np.ndarray,
    co2_p90: np.ndarray,
    lag_stats: Dict[str, float],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, ttc_mean, label="TTC mean", color="C0")
    axes[0].fill_between(t, ttc_p10, ttc_p90, color="C0", alpha=0.2, label="p10-p90")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1, label="t_risk")
    axes[0].set_ylabel("TTC [s]")
    axes[0].set_title("TTC aligned at minimum")
    axes[0].legend()

    axes[1].plot(t, co2_mean, label="CO2 mean", color="C2")
    axes[1].fill_between(t, co2_p10, co2_p90, color="C2", alpha=0.2, label="p10-p90")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1, label="t_risk")
    axes[1].set_xlabel("Time from TTC min [s]")
    axes[1].set_ylabel("CO2 rate [g/s]")
    axes[1].set_title("CO2 aligned to TTC minimum")

    median_lag = lag_stats.get("median", np.nan)
    if not np.isnan(median_lag):
        axes[1].axvline(median_lag, color="C3", linestyle=":", linewidth=1.2, label=f"lag median={median_lag:.2f}s")
    axes[1].legend()

    fig.suptitle("TTC–CO2 Time-series Alignment")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


__all__ = [
    "aggregate_ttc_co2_timeseries",
    "plot_ttc_co2_alignment",
]
