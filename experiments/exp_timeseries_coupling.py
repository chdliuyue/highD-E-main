from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.timeseries_coupling import (
    aggregate_timeseries_over_episodes,
    plot_mean_timeseries,
)

def run_timeseries_coupling_experiment(
    recordings: Sequence[int],
    frame_rate: float = 25.0,
    t_window: tuple[float, float] = (-5.0, 10.0),
    max_episodes: int = 1000,
    output_root: Path | str = "output",
) -> None:
    """
    核心实验入口：
      - 调用 aggregate_timeseries_over_episodes(rec_ids, frame_rate, t_window, max_episodes)
      - 打印使用的 episode 数与 lag 分布统计
      - 调用 plot_mean_timeseries 绘制平均曲线图，保存在 FIG_ROOT 下。
    """

    rec_ids = tuple(int(r) for r in recordings)
    fig_root = Path(output_root) / "timeseries_coupling"
    fig_root.mkdir(parents=True, exist_ok=True)
    agg_result = aggregate_timeseries_over_episodes(
        rec_ids=rec_ids,
        frame_rate=frame_rate,
        t_window=t_window,
        max_episodes=max_episodes,
    )

    t = agg_result["t_grid"]
    lags = agg_result.get("lags", np.array([]))

    print(f"Aggregated recordings: {rec_ids}")
    print(f"Episode used: {len(lags)}")
    if lags.size > 0:
        lag_series = pd.Series(lags)
        print("Lag stats [s]:")
        print(lag_series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    if t.size == 0:
        print("No timeseries available for plotting.")
        return

    fig_path = fig_root / "timeseries_mean.png"
    plot_mean_timeseries(
        t=t,
        TTC_mean=agg_result.get("TTC_mean", np.array([])),
        CO2_mean=agg_result.get("CO2_mean", np.array([])),
        save_path=fig_path,
    )
    print(f"Saved mean curves to {fig_path}")
