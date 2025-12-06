import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

REQUIRED_COLUMNS = {
    "recordingId",
    "frame",
    "trackId",
    "s_long",
    "v_long_smooth",
    "veh_class",
    "drivingDirection",
}


@pytest.fixture(scope="session")
def master_df():
    if not PROCESSED_PARQUET.exists():
        pytest.skip(f"Processed parquet missing at {PROCESSED_PARQUET}")
    df = load_master_table()
    if df.empty:
        pytest.skip("Processed master table is empty")
    return df


@pytest.fixture(scope="session")
def raw_tracks():
    fname = RAW_DATA_DIR / "01_tracks.csv"
    if not fname.exists():
        pytest.skip(f"Raw tracks missing at {fname}")
    return load_raw_tracks(1)


@pytest.mark.structure
def test_can_load_master_table(master_df):
    assert len(master_df) > 0, "Master table should contain rows"


@pytest.mark.structure
def test_required_columns_present(master_df):
    missing = REQUIRED_COLUMNS - set(master_df.columns)
    assert not missing, f"Missing required columns: {missing}"
    has_vtm = any(col.startswith("vtm_") for col in master_df.columns)
    has_vsp = any(col.startswith("vsp_") for col in master_df.columns)
    assert has_vtm, "VT-Micro fields (vtm_*) should exist"
    assert has_vsp, "VSP fields (vsp_*) should exist"


@pytest.mark.structure
def test_key_dtypes(master_df):
    int_cols = ["recordingId", "frame", "trackId"]
    float_cols = ["s_long", "v_long_smooth", "a_long_smooth"]
    for col in int_cols:
        assert col in master_df.columns
        assert pd.api.types.is_integer_dtype(master_df[col]), f"{col} should be integer"
    for col in float_cols:
        if col not in master_df.columns:
            warnings.warn(f"Column {col} missing; dtype check skipped")
            continue
        assert pd.api.types.is_float_dtype(master_df[col])

    if "veh_class" in master_df.columns:
        assert pd.api.types.is_numeric_dtype(master_df["veh_class"]), "veh_class should be numeric"


@pytest.mark.structure
def test_nan_ratios(master_df):
    key_cols = ["recordingId", "frame", "trackId", "s_long", "v_long_smooth", "veh_class", "drivingDirection"]
    ratios = compute_nan_ratios(master_df, key_cols)
    if ratios.empty:
        pytest.skip("None of the key columns are present")
    assert (ratios < 1e-6).all(), f"NaN ratios too high: {ratios[ratios >= 1e-6]}"

    optional_cols = [
        "TTC",
        "DRAC",
        "risk_level",
        "vtm_co2_rate",
        "vtm_nox_rate",
        "vsp_co2_rate",
        "vsp_nox_rate",
    ]
    optional_ratios = compute_nan_ratios(master_df, optional_cols)
    if not optional_ratios.empty:
        warnings.warn(f"Optional metric NaN ratios:\n{optional_ratios}")


@pytest.mark.physics
def test_s_long_monotonicity(master_df):
    track_ids = sample_track_ids(master_df, n=5, random_state=42)
    if not track_ids:
        pytest.skip("No trackIds available for monotonicity check")
    results = check_monotonic_s_long(master_df, track_ids)
    for res in results:
        assert res["negative_count"] < 5 or res["negative_ratio"] < 0.01, (
            f"Track {res['trackId']} has too many negative ds: {res}"
        )


@pytest.mark.physics
def test_velocity_and_acceleration_ranges(master_df):
    v = master_df["v_long_smooth"].dropna()
    assert (v < 0).mean() < 0.01, "Too many negative speeds"
    v_stats = summarize_numeric(v)
    if v_stats["max"] > 60:
        warnings.warn(f"Max v_long_smooth unusually high: {v_stats['max']:.2f} m/s")

    if "a_long_smooth" not in master_df.columns:
        pytest.skip("a_long_smooth missing")
    a = master_df["a_long_smooth"].dropna()
    inside = (a.abs() <= 10).mean() if len(a) else 1.0
    assert inside > 0.98, f"Acceleration magnitude exceeds bounds too often: {inside:.3f} within [-10,10]"


@pytest.mark.physics
def test_headway_and_ttc(master_df):
    required = {"precedingId", "dist_headway_raw", "dist_headway"}
    if not required.issubset(master_df.columns):
        pytest.skip("Headway columns missing")
    pos_ratio, mean_abs_diff = headway_comparison(master_df)
    assert np.isnan(pos_ratio) or pos_ratio >= 0.95, "dist_headway_raw should be positive for most samples"
    if not np.isnan(mean_abs_diff):
        assert mean_abs_diff < 5, f"Mean abs diff between headways too large: {mean_abs_diff:.2f} m"

    q05, q95 = ttc_quantiles(master_df)
    if np.isnan(q05) or np.isnan(q95):
        pytest.skip("Insufficient data for TTC quantiles")
    assert q05 > -0.5, f"TTC lower quantile too negative: {q05:.2f}"
    if q95 > 25:
        warnings.warn(f"TTC 95th percentile high: {q95:.2f}s")


@pytest.mark.physics
def test_emissions_non_negative(master_df):
    fields = ["vtm_co2_rate", "vtm_nox_rate", "vsp_co2_rate", "vsp_nox_rate"]
    summary = emission_field_checks(master_df, fields)
    if not summary:
        pytest.skip("No emission fields present")
    for field, stats in summary.items():
        assert stats["neg_ratio"] <= 1e-3, f"Too many negative values in {field}: {stats['neg_ratio']:.4f}"
        assert not stats.get("exceeds_max", False), f"Extreme values in {field}: {stats['max']:.2e}"


@pytest.mark.raw_alignment
def test_recording_one_rowcount(master_df, raw_tracks):
    processed_rows = len(master_df[master_df["recordingId"] == 1])
    raw_rows = len(raw_tracks)
    assert raw_rows > 0
    diff_ratio = abs(processed_rows - raw_rows) / raw_rows
    assert diff_ratio < 0.2, (
        f"Processed rows for recording 1 differ too much from raw tracks (diff_ratio={diff_ratio:.2%})"
    )


@pytest.mark.raw_alignment
def test_sample_alignment_with_raw(master_df, raw_tracks):
    required_cols = {"x_raw", "y_raw", "v_long_raw", "drivingDirection"}
    if not required_cols.issubset(master_df.columns):
        pytest.skip("Raw alignment columns missing in processed table")
    results = align_raw_and_processed(master_df, raw_tracks, sample_size=5, random_state=0)
    if not results:
        pytest.skip("No matching samples found for alignment check")
    for res in results:
        assert res["x_diff"] < 1e-3, f"x mismatch too large: {res}"
        assert res["y_diff"] < 1e-3, f"y mismatch too large: {res}"
        assert res["v_long_diff"] < 1e-3, f"v_long mismatch too large: {res}"
