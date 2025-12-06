"""Command-line sanity report for the L1 master frame table."""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.master_table_checks import (
    PROCESSED_PARQUET,
    RAW_DATA_DIR,
    align_raw_and_processed,
    check_monotonic_s_long,
    compute_nan_ratios,
    emission_field_checks,
    headway_comparison,
    load_master_table,
    load_raw_tracks,
    sample_track_ids,
    summarize_numeric,
    ttc_quantiles,
)


def print_header(title: str) -> None:
    """Print a section header with separators for readability."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_column(series: pd.Series, label: str) -> None:
    """Print min/mean/max stats for a numeric series."""
    stats = summarize_numeric(series)
    print(f"{label:<20} min={stats['min']:.3f} mean={stats['mean']:.3f} max={stats['max']:.3f}")


def report_structure(df: pd.DataFrame) -> None:
    """Report basic shape, columns, and NaN ratios."""
    print_header("Structure & completeness")
    print(f"Rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")

    key_cols = ["recordingId", "frame", "trackId", "s_long", "v_long_smooth", "veh_class", "drivingDirection"]
    ratios = compute_nan_ratios(df, key_cols)
    if not ratios.empty:
        print("\nNaN ratios (key fields):")
        for col, ratio in ratios.items():
            print(f"  {col:<18} {ratio:8.4e}")

    print("\nNaN ratios (first 20 columns):")
    for col in df.columns[:20]:
        ratio = float(df[col].isna().mean())
        print(f"  {col:<18} {ratio:8.4e}")


def report_physics(df: pd.DataFrame, random_state: int = 42, sample_n: int = 5) -> None:
    """Run physics sanity checks and print descriptive outputs."""
    print_header("Physics sanity checks")
    if "v_long_smooth" in df.columns:
        describe_column(df["v_long_smooth"].dropna(), "v_long_smooth")
    if "a_long_smooth" in df.columns:
        describe_column(df["a_long_smooth"].dropna(), "a_long_smooth")

    track_ids = sample_track_ids(df, n=sample_n, random_state=random_state)
    if track_ids:
        results = check_monotonic_s_long(df, track_ids)
        print("\nMonotonicity of s_long (sample tracks):")
        for res in results:
            print(
                f"  track {res['trackId']:>4}: negative steps {res['negative_count']} / {res['total_steps']}"
                f" ({res['negative_ratio']:.3%})"
            )

    pos_ratio, mean_abs_diff = headway_comparison(df)
    print("\nHeadway comparison (precedingId>0):")
    print(f"  Positive dist_headway_raw ratio: {pos_ratio}")
    print(f"  Mean |dist_headway - dist_headway_raw|: {mean_abs_diff}")

    q05, q95 = ttc_quantiles(df)
    print("\nTTC quantiles (5% / 95%):", q05, q95)


def report_emissions(df: pd.DataFrame) -> None:
    """Print emission/energy field statistics."""
    print_header("Emission & energy metrics")
    fields = ["cpf_fuel_rate_lps", "cpf_co2_rate_gps", "vsp_co2_rate", "vsp_nox_rate"]
    summary = emission_field_checks(df, fields)
    if not summary:
        print("No emission fields found.")
        return
    for field, stats in summary.items():
        print(
            f"{field:<15} min={stats['min']:.4f} mean={stats['mean']:.4f} max={stats['max']:.4f}"
            f" neg_ratio={stats['neg_ratio']:.4%}"
        )


def report_raw_alignment(df: pd.DataFrame, raw_df: Optional[pd.DataFrame]) -> None:
    """Print comparisons between processed data and raw recording 01."""
    print_header("Consistency with raw recording 01")
    if raw_df is None:
        print("Raw recording 01 not available; skipping alignment checks.")
        return
    processed_rows = len(df[df["recordingId"] == 1])
    print(f"Processed rows for recording 01: {processed_rows:,}")
    print(f"Raw 01_tracks rows: {len(raw_df):,}")
    if len(raw_df) > 0:
        diff_ratio = abs(processed_rows - len(raw_df)) / len(raw_df)
        print(f"Row count difference ratio: {diff_ratio:.2%}")

    if {"x_raw", "y_raw", "v_long_raw", "drivingDirection"}.issubset(df.columns):
        results = align_raw_and_processed(df, raw_df, sample_size=5, random_state=0)
        if not results:
            print("No overlapping samples for alignment check.")
        else:
            print("\nSample alignment errors (abs diff):")
            for res in results:
                print(
                    f"  track {res['trackId']:>4} frame {res['frame']:>5}:"
                    f" x_diff={res['x_diff']:.2e} y_diff={res['y_diff']:.2e}"
                    f" v_long_diff={res['v_long_diff']:.2e}"
                )
    else:
        print("Processed table missing raw alignment columns; skipping sample comparisons.")


def plot_track_profile(df: pd.DataFrame, track_id: int, output_path: Optional[str] = None) -> None:
    """Plot s_long and longitudinal speeds for a single track.

    This helper is not used in the automated flow but can be called manually for
    visual inspection. matplotlib is imported inside the function to avoid
    introducing a hard dependency unless plotting is requested.
    """
    import matplotlib.pyplot as plt

    track = df[df["trackId"] == track_id].sort_values("frame")
    if track.empty:
        print(f"Track {track_id} not found for plotting.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(track["frame"], track["s_long"], label="s_long", color="tab:blue")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("s_long [m]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if "v_long_smooth" in track.columns:
        ax2 = ax1.twinx()
        ax2.plot(track["frame"], track["v_long_smooth"], label="v_long_smooth", color="tab:orange", alpha=0.7)
        ax2.set_ylabel("v_long_smooth [m/s]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"Track {track_id} profile")
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    """Entry point for generating the sanity report."""
    if not PROCESSED_PARQUET.exists():
        print(f"Processed parquet not found at {PROCESSED_PARQUET}")
        return

    df = load_master_table()
    report_structure(df)
    report_physics(df)
    report_emissions(df)

    raw_df: Optional[pd.DataFrame]
    raw_tracks_path = RAW_DATA_DIR / "01_tracks.csv"
    if raw_tracks_path.exists():
        raw_df = load_raw_tracks(1)
    else:
        raw_df = None
    report_raw_alignment(df, raw_df)


if __name__ == "__main__":
    main()
