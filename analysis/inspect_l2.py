from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L2_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "events"
OUT_ROOT = PROJECT_ROOT / "output" / "explore"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _describe_fields(df: pd.DataFrame, fields: Sequence[str]) -> str:
    present_fields = [f for f in fields if f in df.columns]
    if not present_fields:
        return "No target fields present."
    desc = df[present_fields].describe()
    return desc.to_string()


def summarize_l2_for_recording(rec_id: int) -> dict:
    """
    Explore L2 conflict and baseline events for a single recording.
    """

    rec_str = f"{rec_id:02d}"
    conf_path = L2_ROOT / f"recording_{rec_str}" / "L2_conflict_events.parquet"
    base_path = L2_ROOT / f"recording_{rec_str}" / "L2_baseline_events.parquet"
    info_path = OUT_ROOT / f"l2_rec{rec_str}_info.txt"

    n_conf = n_base = 0
    conf_df = base_df = None
    warnings = []

    if conf_path.exists():
        conf_df = pd.read_parquet(conf_path)
        n_conf = len(conf_df)
    else:
        warnings.append(f"Conflict parquet missing: {conf_path}")

    if base_path.exists():
        base_df = pd.read_parquet(base_path)
        n_base = len(base_df)
    else:
        warnings.append(f"Baseline parquet missing: {base_path}")

    lines = [
        f"Recording {rec_str}",
        f"Conflict file: {conf_path} ({n_conf} rows)",
        f"Baseline file: {base_path} ({n_base} rows)",
    ]

    if warnings:
        lines.append("Warnings:")
        lines.extend([f"- {w}" for w in warnings])
        lines.append("")

    if conf_df is not None:
        conf_fields = ["conf_duration", "min_TTC_conf", "duration", "E_cpf_CO2", "E_cpf_fuel"]
        lines.append("Conflict describe:")
        lines.append(_describe_fields(conf_df, conf_fields))
        lines.append("")
        if "veh_class" in conf_df.columns:
            lines.append("veh_class counts (conflict):")
            lines.append(conf_df["veh_class"].value_counts().to_string())
            lines.append("")

    if base_df is not None:
        base_fields = ["duration", "E_cpf_CO2", "E_cpf_fuel"]
        lines.append("Baseline describe:")
        lines.append(_describe_fields(base_df, base_fields))
        lines.append("")
        if "veh_class" in base_df.columns:
            lines.append("veh_class counts (baseline):")
            lines.append(base_df["veh_class"].value_counts().to_string())
            lines.append("")

    info_path.write_text("\n".join(lines))

    summary = {
        "rec_id": rec_id,
        "n_conf": n_conf,
        "n_base": n_base,
        "median_conf_duration": float(conf_df["conf_duration"].median()) if conf_df is not None and "conf_duration" in conf_df else float("nan"),
        "median_min_TTC_conf": float(conf_df["min_TTC_conf"].median()) if conf_df is not None and "min_TTC_conf" in conf_df else float("nan"),
        "median_E_cpf_CO2_conf": float(conf_df["E_cpf_CO2"].median()) if conf_df is not None and "E_cpf_CO2" in conf_df else float("nan"),
    }

    return summary


def summarize_l2_overall(rec_ids: Sequence[int]) -> pd.DataFrame:
    """
    Summarize conflict and baseline events across recordings.
    """

    summaries = []
    total = len(rec_ids)
    for idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[{idx}/{total}] Summarizing L2 for recording {rec_id:02d}...")
        summaries.append(summarize_l2_for_recording(rec_id))

    summary_df = pd.DataFrame(summaries)
    out_path = OUT_ROOT / "l2_overall_summary.csv"
    summary_df.to_csv(out_path, index=False)

    total_conf = summary_df["n_conf"].sum()
    total_base = summary_df["n_base"].sum()

    print("L2 Overall Summary")
    print(f"- Recordings: {rec_ids}")
    print(f"- Total conflict events: {total_conf}")
    print(f"- Total baseline events: {total_base}")
    print(f"- Median conf duration: {summary_df['median_conf_duration'].median():.4f}")
    print(f"- Median min TTC conf: {summary_df['median_min_TTC_conf'].median():.4f}")
    print(f"- Median E_cpf_CO2 conf: {summary_df['median_E_cpf_CO2_conf'].median():.4f}")
    print(f"- Saved to: {out_path}")

    return summary_df
