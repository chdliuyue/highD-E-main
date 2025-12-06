"""Input/output helpers for loading raw highD CSV files."""
from pathlib import Path
import pandas as pd


def load_recording_meta(rec_id: int, raw_data_dir: Path) -> pd.DataFrame:
    """Load the recording-level metadata CSV for a given recording id."""
    path = raw_data_dir / f"{rec_id:02d}_recordingMeta.csv"
    return pd.read_csv(path)


def load_tracks_meta(rec_id: int, raw_data_dir: Path) -> pd.DataFrame:
    """Load the vehicle-level metadata CSV for a given recording id."""
    path = raw_data_dir / f"{rec_id:02d}_tracksMeta.csv"
    return pd.read_csv(path)


def load_tracks(rec_id: int, raw_data_dir: Path) -> pd.DataFrame:
    """Load the frame-level tracks CSV for a given recording id."""
    path = raw_data_dir / f"{rec_id:02d}_tracks.csv"
    return pd.read_csv(path)


__all__ = ["load_recording_meta", "load_tracks_meta", "load_tracks"]
