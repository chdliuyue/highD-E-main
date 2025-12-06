"""Quick checks for generated L2 event tables."""
from __future__ import annotations

import pandas as pd

from config import PROJECT_ROOT


def load_tables(rec_id: int):
    base_dir = PROJECT_ROOT / "data" / "processed" / "highD" / "events" / f"recording_{rec_id:02d}"
    conf_path = base_dir / "L2_conflict_events.parquet"
    base_path = base_dir / "L2_baseline_events.parquet"

    if not conf_path.exists() or not base_path.exists():
        raise FileNotFoundError(f"Missing L2 outputs under {base_dir}")

    return pd.read_parquet(conf_path), pd.read_parquet(base_path)


def describe_events(rec_id: int = 1) -> None:
    df_conf, df_base = load_tables(rec_id)

    print(f"Recording {rec_id:02d}: conflict events = {len(df_conf)}, baseline events = {len(df_base)}")
    if not df_conf.empty:
        print("\nConflict event stats:")
        print(df_conf[["conf_duration", "min_TTC", "E_cpf_CO2"]].describe())
        print("\nConflict events by veh_class:")
        print(df_conf["veh_class"].value_counts(dropna=False))
    if not df_base.empty:
        print("\nBaseline event stats:")
        print(df_base[["min_TTC", "E_cpf_CO2"]].describe())
        print("\nBaseline events by veh_class:")
        print(df_base["veh_class"].value_counts(dropna=False))


def main() -> None:
    describe_events(1)


if __name__ == "__main__":
    main()
