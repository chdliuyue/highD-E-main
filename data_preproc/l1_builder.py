"""Frame-level L1 preprocessing pipeline for highD data."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config
from data_preproc.io_highd import load_recording_meta, load_tracks, load_tracks_meta
from data_preproc.schema import Columns, column_order
from data_preproc.vt_cpfm import apply_vt_cpfm_to_df
from data_preproc.vsp import vsp_emissions
from utils.misc import apply_savgol_by_group, compute_drac, ensure_directory, safe_divide

C = Columns()


class L1Builder:
    """Build per-frame master tables (L1) for highD recordings."""

    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        frame_rate: float = config.FRAME_RATE_DEFAULT,
        num_workers: int = 1,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.frame_rate = float(frame_rate)
        self.num_workers = max(int(num_workers), 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_one(self, rec_id: int) -> None:
        """Build ``L1_master_frame.parquet`` for a single recording."""

        rec_meta = self._load_recording_meta(rec_id)
        tracks_meta = self._load_tracks_meta(rec_id)
        tracks = self._load_tracks(rec_id)

        df = self._merge_meta_to_tracks(rec_meta, tracks_meta, tracks)
        df = self._unify_coordinates(df)
        df = self._smooth_kinematics(df)
        df = self._compute_interactions(df)
        df = self._compute_safety_metrics(df, config.RECOMPUTE_SAFETY_METRICS)
        df = self._compute_emissions_vt_cpfm(df)
        df = self._compute_emissions_vsp(df)
        df = self._add_visual_coordinates(df, rec_meta)

        df = df[column_order()]
        self._save_parquet(df, rec_id)

    def build_many(self, rec_ids: Iterable[int], num_workers: Optional[int] = None) -> None:
        """Build multiple recordings sequentially or in parallel."""

        rec_ids = list(rec_ids)
        workers = self.num_workers if num_workers is None else max(int(num_workers), 1)

        if workers <= 1:
            for idx, rec_id in enumerate(rec_ids, start=1):
                print(f"Processing recording {rec_id:02d} ({idx}/{len(rec_ids)})...")
                self.build_one(int(rec_id))
        else:
            print(f"Parallel processing with {workers} workers. Linux/WSL is recommended for best speed.")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                args = [
                    (
                        self.raw_data_dir,
                        self.processed_data_dir,
                        int(rec_id),
                        workers,
                        self.frame_rate,
                    )
                    for rec_id in rec_ids
                ]
                for _ in ex.map(_process_single_recording_entry, args):
                    pass

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_recording_meta(self, rec_id: int) -> pd.DataFrame:
        return load_recording_meta(rec_id, self.raw_data_dir)

    def _load_tracks_meta(self, rec_id: int) -> pd.DataFrame:
        return load_tracks_meta(rec_id, self.raw_data_dir)

    def _load_tracks(self, rec_id: int) -> pd.DataFrame:
        return load_tracks(rec_id, self.raw_data_dir)

    # ------------------------------------------------------------------
    # Feature construction helpers
    # ------------------------------------------------------------------
    def _merge_meta_to_tracks(
        self, rec_meta: pd.DataFrame, tracks_meta: pd.DataFrame, tracks: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge vehicle-level metadata into frame-level tracks and add index fields."""

        recording_id = int(rec_meta.iloc[0]["id"])
        frame_rate = float(rec_meta.iloc[0].get("frameRate", self.frame_rate))

        df = tracks.copy()
        df.rename(columns={"id": C.track_id}, inplace=True)
        df[C.recording_id] = np.int32(recording_id)
        df[C.time] = df["frame"] / frame_rate
        df[C.x_raw] = df["x"]
        df[C.y_raw] = df["y"]
        df[C.lane_id_raw] = df.get("laneId", 0)
        df[C.global_track_id] = df[C.recording_id] * 10000 + df[C.track_id]
        df["frameRate"] = frame_rate

        meta = tracks_meta.rename(columns={"id": C.track_id, "class": C.veh_class})[
            [C.track_id, C.veh_class, "drivingDirection", "width", "height"]
        ]
        meta[C.veh_class] = meta[C.veh_class].map(config.VEHICLE_CLASS_ENCODING).astype("Int8")
        df = df.merge(meta, on=C.track_id, how="left")
        df[C.driving_direction] = df["drivingDirection"].astype(np.int8)
        df[C.veh_class] = df[C.veh_class].astype("Int8")
        df[C.width] = df["width_y"].fillna(df["width_x"])
        df[C.height] = df["height_y"].fillna(df["height_x"])
        df.drop(columns=[col for col in df.columns if col.endswith("_x") or col.endswith("_y")], inplace=True)
        return df

    def _unify_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unified longitudinal/lateral coordinates and raw kinematics."""

        x_center = df[C.x_raw] + df[C.width] / 2.0
        max_x = (df[C.x_raw] + df[C.width]).max()

        df[C.s_long] = np.where(df[C.driving_direction] == 2, x_center, max_x - x_center)
        df[C.d_lat] = df[C.y_raw] + df[C.height] / 2.0
        df[C.v_long_raw] = np.where(df[C.driving_direction] == 2, df["xVelocity"], -df["xVelocity"])
        df[C.a_long_raw] = np.where(df[C.driving_direction] == 2, df["xAcceleration"], -df["xAcceleration"])
        return df

    def _smooth_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth longitudinal speed and acceleration per trajectory using Savitzky-Golay."""

        df = df.sort_values([C.track_id, C.frame])
        df[C.v_long_smooth] = apply_savgol_by_group(
            df, [C.track_id], C.v_long_raw, config.SAVGOL_WINDOW, config.SAVGOL_POLY
        ).astype(np.float32)
        df[C.a_long_smooth] = apply_savgol_by_group(
            df, [C.track_id], C.v_long_raw, config.SAVGOL_WINDOW, config.SAVGOL_POLY, deriv=1
        ).astype(np.float32)
        return df

    def _compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute leader-related metrics using vectorized merges."""

        leader_info = df[
            [C.track_id, C.frame, C.s_long, C.v_long_smooth, C.a_long_smooth, C.width]
        ].rename(
            columns={
                C.track_id: C.preceding_id,
                C.s_long: C.leader_s_long,
                C.v_long_smooth: C.leader_v_long,
                C.a_long_smooth: C.leader_a_long,
                C.width: "leader_width",
            }
        )
        df = df.merge(leader_info, on=[C.preceding_id, C.frame], how="left")
        df[C.dist_headway_raw] = df.get("dhw", np.nan).astype(np.float32)
        df[C.time_headway_raw] = df.get("thw", np.nan).astype(np.float32)
        df[C.ttc_raw] = df.get("ttc", np.nan).astype(np.float32)

        df[C.dist_headway] = df[C.leader_s_long] - df[C.s_long] - 0.5 * (
            df.get("leader_width", df[C.width]) + df[C.width]
        )
        df[C.rel_velocity] = df[C.v_long_smooth] - df[C.leader_v_long]
        df[C.time_headway] = safe_divide(df[C.dist_headway].to_numpy(), df[C.v_long_smooth].to_numpy())
        return df

    def _compute_safety_metrics(self, df: pd.DataFrame, recompute: bool) -> pd.DataFrame:
        """Compute TTC, DRAC and risk labels."""

        if recompute:
            gap = df[C.dist_headway].to_numpy()
            rel_v = df[C.leader_v_long].to_numpy() - df[C.v_long_smooth].to_numpy()
            ttc = safe_divide(gap, np.maximum(rel_v, 0.0))
            drac = compute_drac(gap, np.maximum(-rel_v, 0.0))
        else:
            ttc = df[C.ttc_raw].replace(0, np.nan).to_numpy()
            drac = compute_drac(df[C.dist_headway_raw].to_numpy(), df[C.rel_velocity].to_numpy())

        df[C.TTC] = np.nan_to_num(ttc, nan=99.0, posinf=99.0, neginf=99.0).astype(np.float32)
        df[C.DRAC] = np.nan_to_num(drac, nan=np.inf, posinf=np.inf).astype(np.float32)

        df[C.risk_level] = np.where(df[C.TTC] < config.TTC_HIGH_RISK, 2, 0)
        df.loc[(df[C.TTC] >= config.TTC_HIGH_RISK) & (df[C.TTC] < config.TTC_LOW_RISK), C.risk_level] = 1
        df[C.risk_level] = df[C.risk_level].astype(np.int8)
        return df

    def _compute_emissions_vt_cpfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VT-CPFM power, fuel, and CO2 rates for each frame."""

        df = apply_vt_cpfm_to_df(df)
        return df

    def _compute_emissions_vsp(self, df: pd.DataFrame) -> pd.DataFrame:
        emissions = vsp_emissions(
            df[C.vehicle_type].to_numpy(), df[C.v_long_smooth].to_numpy(), df[C.a_long_smooth].to_numpy()
        )
        df[C.vsp_value] = emissions["vsp_value"]
        df[C.vsp_co2_rate] = emissions["co2_rate"]
        df[C.vsp_nox_rate] = emissions["nox_rate"]
        df[C.dt] = (1.0 / df["frameRate"]).astype(np.float32)
        return df

    def _add_visual_coordinates(self, df: pd.DataFrame, rec_meta: pd.DataFrame) -> pd.DataFrame:
        """Approximate mapping from metric coordinates to image pixels."""

        img_path = config.HIGHWAY_IMAGE_DIR / f"{int(rec_meta.iloc[0]['id']):02d}_highway.png"
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(img_path)
            img_h, img_w = img.shape[0], img.shape[1]
        except FileNotFoundError:
            img_h, img_w = 720, 1280

        x_min, x_max = df[C.x_raw].min(), (df[C.x_raw] + df[C.width]).max()
        y_min, y_max = df[C.y_raw].min(), (df[C.y_raw] + df[C.height]).max()
        df[C.x_img] = ((df[C.x_raw] - x_min) / max(x_max - x_min, 1e-6) * img_w).astype(np.float32)
        df[C.y_img] = ((df[C.y_raw] - y_min) / max(y_max - y_min, 1e-6) * img_h).astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save_parquet(self, df: pd.DataFrame, rec_id: int) -> None:
        ensure_directory(self.processed_data_dir / f"recording_{rec_id:02d}")
        out_path = self.processed_data_dir / f"recording_{rec_id:02d}" / "L1_master_frame.parquet"
        df.to_parquet(out_path, index=False)


# ----------------------------------------------------------------------
# Multiprocessing entrypoint
# ----------------------------------------------------------------------

def _process_single_recording_entry(args: tuple) -> None:
    raw_dir, out_dir, rec_id, num_workers, frame_rate = args
    builder = L1Builder(Path(raw_dir), Path(out_dir), frame_rate=frame_rate, num_workers=num_workers)
    builder.build_one(rec_id)


__all__ = ["L1Builder", "_process_single_recording_entry"]
