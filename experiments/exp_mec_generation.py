from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.mec_baseline import generate_mec_for_recordings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_mec_generation(rec_ids: Sequence[int]) -> None:
    """
    MEC 生成阶段的实验入口。

    步骤：
      - 调用 generate_mec_for_recordings(rec_ids) 生成 MEC DataFrame 并写 parquet；
      - 在控制台打印：
          * recordings 列表；
          * MEC 样本总数；
          * MEC_CO2_per_km 的 median / [10%, 90%] 分位数；
          * 按 min_TTC_conf 桶（例如 <2, 2-3, 3-4）分组后的 median MEC。
    """

    df_mec = generate_mec_for_recordings(rec_ids)

    print("MEC generation finished")
    print(f"Recordings: {rec_ids}")
    print(f"Total MEC events: {len(df_mec)}")

    if df_mec.empty:
        print("No MEC data generated.")
        return

    mec_per_km = df_mec["MEC_CO2_per_km"].replace([np.inf, -np.inf], np.nan).dropna()
    if not mec_per_km.empty:
        median = mec_per_km.median()
        p10 = mec_per_km.quantile(0.1)
        p90 = mec_per_km.quantile(0.9)
        print(f"MEC_CO2_per_km median: {median:.4f} [p10={p10:.4f}, p90={p90:.4f}]")

    grouped = df_mec.groupby("MEC_bucket")["MEC_CO2_per_km"].median().sort_index()
    print("Median MEC_CO2_per_km by min_TTC_conf bucket:")
    for bucket, value in grouped.items():
        print(f"  {bucket}: {value:.4f}")
