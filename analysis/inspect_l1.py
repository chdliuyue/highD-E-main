from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
OUT_ROOT = PROJECT_ROOT / "output" / "explore"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _notnull_ratio(series: pd.Series) -> float:
    """Return ratio of non-null values; 0.0 if series is None."""
    if series is None:
        return 0.0
    if len(series) == 0:
        return 0.0
    return series.notna().mean()


def summarize_l1_for_recording(rec_id: int, max_rows: int = 5) -> dict:
    """
    Explore a single L1 master frame parquet for a recording.

    Reads the parquet, gathers schema info, computes key non-null ratios,
    saves a sample CSV and text summary, and returns key metrics as a dict.
    """

    rec_str = f"{rec_id:02d}"
    parquet_path = L1_ROOT / f"recording_{rec_str}" / "L1_master_frame.parquet"
    info_path = OUT_ROOT / f"l1_rec{rec_str}_info.txt"
    sample_path = OUT_ROOT / f"l1_rec{rec_str}_sample.csv"

    if not parquet_path.exists():
        warning = f"Parquet not found: {parquet_path}"
        info_path.write_text(warning)
        return {
            "rec_id": rec_id,
            "n_rows": 0,
            "n_cols": 0,
            "ttc_notnull_ratio": 0.0,
            "co2_notnull_ratio": 0.0,
            "fuel_notnull_ratio": 0.0,
        }

    df = pd.read_parquet(parquet_path)
    n_rows, n_cols = df.shape

    key_cols = [
        "recordingId",
        "frame",
        "trackId",
        "TTC",
        "v_long_smooth",
        "a_long_smooth",
        "cpf_co2_rate_gps",
        "cpf_fuel_rate_lps",
        "dt",
    ]

    column_info = [f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    missing_cols = [col for col in key_cols if col not in df.columns]

    ratios = {
        "ttc_notnull_ratio": _notnull_ratio(df["TTC"]) if "TTC" in df else float("nan"),
        "co2_notnull_ratio": _notnull_ratio(df["cpf_co2_rate_gps"]) if "cpf_co2_rate_gps" in df else float("nan"),
        "fuel_notnull_ratio": _notnull_ratio(df["cpf_fuel_rate_lps"]) if "cpf_fuel_rate_lps" in df else float("nan"),
    }

    df.head(max_rows).to_csv(sample_path, index=False)

    lines = [
        f"Recording {rec_str}",
        f"File: {parquet_path}",
        f"Rows: {n_rows}",
        f"Columns: {n_cols}",
        "",
        "Column schema:",
        *column_info,
        "",
        "Key column presence:",
    ]
    for col in key_cols:
        presence = "present" if col in df.columns else "missing"
        lines.append(f"- {col}: {presence}")
    lines.append("")
    lines.append("Non-null ratios:")
    for key, value in ratios.items():
        lines.append(f"- {key}: {value:.4f}" if pd.notna(value) else f"- {key}: missing")
    if missing_cols:
        lines.append("")
        lines.append("Missing key columns: " + ", ".join(missing_cols))

    info_path.write_text("\n".join(lines))

    return {
        "rec_id": rec_id,
        "n_rows": n_rows,
        "n_cols": n_cols,
        **ratios,
    }


def summarize_l1_overall(rec_ids: Sequence[int]) -> pd.DataFrame:
    """
    Summarize multiple recordings' L1 data and save an overall CSV summary.
    """

    summaries = []
    total = len(rec_ids)
    for idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[{idx}/{total}] Summarizing L1 for recording {rec_id:02d}...")
        summaries.append(summarize_l1_for_recording(rec_id))

    summary_df = pd.DataFrame(summaries)
    out_path = OUT_ROOT / "l1_overall_summary.csv"
    summary_df.to_csv(out_path, index=False)

    total_frames = summary_df["n_rows"].sum()
    mean_frames = summary_df["n_rows"].mean() if not summary_df.empty else 0

    print("L1 Overall Summary")
    print(f"- Recordings: {rec_ids}")
    print(f"- Total frames: {total_frames}")
    print(f"- Mean frames per recording: {mean_frames:.2f}")
    print(f"- Saved to: {out_path}")

    return summary_df
