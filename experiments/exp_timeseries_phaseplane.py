from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.timeseries_coupling import aggregate_ttc_co2_timeseries, plot_ttc_co2_alignment
from analysis.phase_plane import build_phase_plane_samples, plot_phase_plane

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_OUT = PROJECT_ROOT / "output" / "timeseries"
PP_OUT = PROJECT_ROOT / "output" / "phase_plane"
TS_OUT.mkdir(parents=True, exist_ok=True)
PP_OUT.mkdir(parents=True, exist_ok=True)


def run_timeseries_and_phaseplane(rec_ids: Sequence[int]) -> None:
    """
    Stage 07 的实验入口：
      1) 运行 TTC–CO2 时序对齐：
         - 调用 aggregate_ttc_co2_timeseries(rec_ids)
         - 计算 lag 的 mean/median/q10/q90
         - 调用 plot_ttc_co2_alignment 保存为
           output/timeseries/01_timeseries_alignment.png
      2) 运行 Safety–Energy 相位图：
         - 调用 build_phase_plane_samples(rec_ids)
         - 调用 plot_phase_plane 保存为
           output/phase_plane/01_phase_plane_overall.png
      在控制台打印：
         - 使用的 episode 数量
         - lag 统计
         - 相位平面采样点数量
    """

    print(f"[Stage 07] Aggregating TTC–CO2 alignment for {len(rec_ids)} recordings...")
    agg = aggregate_ttc_co2_timeseries(rec_ids)

    lag_values = agg["lags"]
    lag_stats = {
        "mean": float(np.nanmean(lag_values)) if lag_values.size > 0 else np.nan,
        "median": float(np.nanmedian(lag_values)) if lag_values.size > 0 else np.nan,
        "p10": float(np.nanpercentile(lag_values, 10)) if lag_values.size > 0 else np.nan,
        "p90": float(np.nanpercentile(lag_values, 90)) if lag_values.size > 0 else np.nan,
    }

    print(
        f"  [Stage 07] Episodes used: {agg['n_episodes']} | "
        f"Lag median: {lag_stats['median']:.3f} (mean={lag_stats['mean']:.3f}, "
        f"p10={lag_stats['p10']:.3f}, p90={lag_stats['p90']:.3f})"
    )

    plot_ttc_co2_alignment(
        agg["t_grid"],
        agg["TTC_mean"],
        agg["TTC_p10"],
        agg["TTC_p90"],
        agg["CO2_mean"],
        agg["CO2_p10"],
        agg["CO2_p90"],
        lag_stats,
        save_path=TS_OUT / "01_timeseries_alignment.png",
    )
    print(f"[Stage 07] Saved time-series alignment plot to {TS_OUT / '01_timeseries_alignment.png'}")

    print(f"[Stage 07] Building phase-plane samples for {len(rec_ids)} recordings...")
    samples = build_phase_plane_samples(rec_ids)
    print(
        f"  [Stage 07] Phase-plane samples: n_points={len(samples['ttc_all'])}, "
        f"n_trajs={len(samples['example_trajs'])}"
    )
    plot_phase_plane(
        samples["ttc_all"],
        samples["energy_all"],
        samples["example_trajs"],
        save_path=PP_OUT / "01_phase_plane_overall.png",
    )
    print(f"[Stage 07] Saved phase-plane plot to {PP_OUT / '01_phase_plane_overall.png'}")


__all__ = ["run_timeseries_and_phaseplane"]
