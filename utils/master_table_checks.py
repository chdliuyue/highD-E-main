"""
Utility functions for sanity checks of the L1 master frame table derived from highD data.

The helpers in this module are shared by pytest tests and ad-hoc scripts so
that the logic for checking structure, physics sanity, and raw-data alignment
remains consistent.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def _resolve_processed_parquet() -> Path:
    """Return path to the processed parquet, accounting for per-recording folders.

    Older pipelines produced a single parquet at ``data/processed/highD/data`` while
    newer runs place one parquet per recording inside ``recording_XX`` directories.
    This helper finds the first existing parquet so tests and scripts can run
    without manual path tweaks.
    """

    default_path = PROJECT_ROOT / "data/processed/highD/data/L1_master_frame.parquet"
    if default_path.exists():
        return default_path

    # Fall back to the first parquet inside a recording_* folder, if any exist.
    for candidate in sorted(
        PROJECT_ROOT.glob("data/processed/highD/data/recording_*/L1_master_frame.parquet")
    ):
        return candidate

    # Nothing found; return the default location so error messages remain familiar.
    return default_path


def _resolve_raw_data_dir() -> Path:
    """Return path to raw data, handling optional recording subdirectories."""

    default_dir = PROJECT_ROOT / "data/raw/highD/data"
    if default_dir.exists():
        return default_dir

    for candidate in sorted(PROJECT_ROOT.glob("data/raw/highD/data/recording_*")):
        if candidate.is_dir():
            return candidate

    return default_dir


PROCESSED_PARQUET = _resolve_processed_parquet()
RAW_DATA_DIR = _resolve_raw_data_dir()


def load_master_table(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load the processed L1 master frame parquet file.

    Parameters
    ----------
    columns : sequence of str, optional
        Subset of columns to read for efficiency. If None, load all columns.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe. If the parquet file does not exist, raises FileNotFoundError.
    """
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError(f"Processed parquet not found: {PROCESSED_PARQUET}")
    return pd.read_parquet(PROCESSED_PARQUET, columns=columns)


def load_raw_tracks(recording_id: int = 1) -> pd.DataFrame:
    """Load raw highD track CSV for a given recording ID."""
    fname = RAW_DATA_DIR / f"{recording_id:02d}_tracks.csv"
    if not fname.exists():
        raise FileNotFoundError(f"Raw tracks CSV not found: {fname}")
    return pd.read_csv(fname)


def existing_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    """Return a list of columns from candidates that are present in df."""
    return [c for c in candidates if c in df.columns]


def compute_nan_ratios(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Compute NaN ratios for selected columns."""
    cols = existing_columns(df, columns)
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].isna().mean()


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    """Return min/mean/max summary for a numeric series while ignoring NaNs."""
    clean = series.dropna()
    if clean.empty:
        return {"min": np.nan, "mean": np.nan, "max": np.nan}
    return {"min": float(clean.min()), "mean": float(clean.mean()), "max": float(clean.max())}


def sample_track_ids(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> List[int]:
    """Sample up to n distinct trackIds from the dataframe."""
    unique_ids = df["trackId"].dropna().unique()
    if len(unique_ids) == 0:
        return []
    rng = np.random.default_rng(random_state)
    sample_size = min(n, len(unique_ids))
    return rng.choice(unique_ids, size=sample_size, replace=False).tolist()


def check_monotonic_s_long(df: pd.DataFrame, track_ids: Sequence[int]) -> List[Dict[str, float]]:
    """Check s_long monotonicity for selected tracks.

    Returns a list of dictionaries summarizing negative step counts and ratios
    for each track.
    """
    results: List[Dict[str, float]] = []
    for tid in track_ids:
        track_df = df[df["trackId"] == tid].sort_values("frame")
        if track_df.empty:
            continue
        s = track_df["s_long"].astype(float)
        ds = s.diff().dropna()
        negative_count = (ds < 0).sum()
        total = len(ds)
        ratio = float(negative_count / total) if total else 0.0
        results.append({
            "trackId": int(tid),
            "negative_count": int(negative_count),
            "total_steps": int(total),
            "negative_ratio": ratio,
        })
    return results


def emission_field_checks(df: pd.DataFrame, fields: Sequence[str], allowed_negative_ratio: float = 1e-3,
                          max_threshold: float = 1e6) -> Dict[str, Dict[str, float]]:
    """Compute emission sanity statistics for provided fields.

    Returns a dictionary mapping field name to stats (min/mean/max, negative ratio).
    """
    summary: Dict[str, Dict[str, float]] = {}
    for field in existing_columns(df, fields):
        series = df[field].dropna()
        if series.empty:
            summary[field] = {"min": np.nan, "mean": np.nan, "max": np.nan, "neg_ratio": 0.0}
            continue
        neg_ratio = float((series < 0).mean()) if len(series) else 0.0
        summary[field] = {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "max": float(series.max()),
            "neg_ratio": neg_ratio,
            "exceeds_max": float(series.max()) > max_threshold,
            "neg_ratio_ok": neg_ratio <= allowed_negative_ratio,
        }
    return summary


def headway_comparison(df: pd.DataFrame) -> Tuple[float, float]:
    """Return proportion of positive raw headways and mean absolute diff between raw and recomputed headways."""
    if "precedingId" not in df.columns:
        return np.nan, np.nan
    mask = df["precedingId"] > 0
    if mask.sum() == 0:
        return np.nan, np.nan
    subset = df.loc[mask]
    pos_ratio = (subset.get("dist_headway_raw", pd.Series(dtype=float)) > 0).mean()
    if "dist_headway_raw" in subset and "dist_headway" in subset:
        diff = (subset["dist_headway"] - subset["dist_headway_raw"]).dropna().abs()
        mean_abs_diff = float(diff.mean()) if not diff.empty else np.nan
    else:
        mean_abs_diff = np.nan
    return float(pos_ratio), mean_abs_diff


def ttc_quantiles(df: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> Tuple[float, float]:
    """Compute TTC quantiles for samples with positive headway and relative speed."""
    required_cols = {"precedingId", "rel_velocity", "dist_headway", "TTC"}
    if not required_cols.issubset(df.columns):
        return np.nan, np.nan
    subset = df[(df["precedingId"] > 0) & (df["rel_velocity"] > 0) & (df["dist_headway"] > 0)]
    if subset.empty:
        return np.nan, np.nan
    ttc_series = subset["TTC"].dropna()
    if ttc_series.empty:
        return np.nan, np.nan
    q_low, q_high = ttc_series.quantile([lower, upper])
    return float(q_low), float(q_high)


def align_raw_and_processed(df_processed: pd.DataFrame, df_raw: pd.DataFrame, sample_size: int = 5,
                             random_state: int = 0) -> List[Dict[str, float]]:
    """Sample pairs of (trackId, frame) and compare raw coordinates/velocity to processed values.

    Returns a list of dictionaries reporting absolute differences.
    """
    rng = np.random.default_rng(random_state)
    processed_subset = df_processed[df_processed["recordingId"] == 1]
    if processed_subset.empty:
        return []
    sample = processed_subset[["trackId", "frame", "drivingDirection", "x_raw", "y_raw", "v_long_raw"]]
    sample = sample.dropna()
    if sample.empty:
        return []
    sample = sample.sample(n=min(sample_size, len(sample)), random_state=random_state)
    results: List[Dict[str, float]] = []
    for _, row in sample.iterrows():
        tid = int(row["trackId"])
        frame = int(row["frame"])
        raw_matches = df_raw[(df_raw["id"] == tid) & (df_raw["frame"] == frame)]
        if raw_matches.empty:
            continue
        raw_row = raw_matches.iloc[0]
        x_diff = float(abs(raw_row["x"] - row["x_raw"]))
        y_diff = float(abs(raw_row["y"] - row["y_raw"]))
        x_vel = float(raw_row["xVelocity"])
        direction = int(row["drivingDirection"])
        expected_v_long_raw = x_vel if direction == 2 else -x_vel
        v_diff = float(abs(expected_v_long_raw - row["v_long_raw"]))
        results.append({
            "trackId": tid,
            "frame": frame,
            "x_diff": x_diff,
            "y_diff": y_diff,
            "v_long_diff": v_diff,
        })
    return results
