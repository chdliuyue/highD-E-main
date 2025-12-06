"""Column name constants and short descriptions for the master frame table."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Columns:
    """Namespace for commonly used column names to avoid typos."""

    # Index and anchors
    recording_id: str = "recordingId"
    frame: str = "frame"
    track_id: str = "trackId"
    global_track_id: str = "global_track_id"
    time: str = "time"
    x_raw: str = "x_raw"
    y_raw: str = "y_raw"
    width: str = "width"
    height: str = "height"
    lane_id_raw: str = "laneId_raw"
    x_img: str = "x_img"
    y_img: str = "y_img"

    # Physics
    driving_direction: str = "drivingDirection"
    veh_class: str = "veh_class"
    s_long: str = "s_long"
    d_lat: str = "d_lat"
    v_long_raw: str = "v_long_raw"
    a_long_raw: str = "a_long_raw"
    v_long_smooth: str = "v_long_smooth"
    a_long_smooth: str = "a_long_smooth"

    # Interaction
    preceding_id: str = "precedingId"
    following_id: str = "followingId"
    left_preceding_id: str = "leftPrecedingId"
    left_alongside_id: str = "leftAlongsideId"
    left_following_id: str = "leftFollowingId"
    right_preceding_id: str = "rightPrecedingId"
    right_alongside_id: str = "rightAlongsideId"
    right_following_id: str = "rightFollowingId"
    leader_s_long: str = "leader_s_long"
    leader_v_long: str = "leader_v_long"
    leader_a_long: str = "leader_a_long"
    dist_headway_raw: str = "dist_headway_raw"
    time_headway_raw: str = "time_headway_raw"
    ttc_raw: str = "ttc_raw"
    dist_headway: str = "dist_headway"
    rel_velocity: str = "rel_velocity"
    time_headway: str = "time_headway"

    # Safety
    TTC: str = "TTC"
    DRAC: str = "DRAC"
    risk_level: str = "risk_level"

    # Energy
    vehicle_type: str = "vehicle_type"
    v_kmh: str = "v_kmh"
    a_kmhps: str = "a_kmhps"
    vtm_fuel_rate: str = "vtm_fuel_rate"
    vtm_co2_rate: str = "vtm_co2_rate"
    vtm_nox_rate: str = "vtm_nox_rate"
    vsp_value: str = "vsp_value"
    vsp_co2_rate: str = "vsp_co2_rate"
    vsp_nox_rate: str = "vsp_nox_rate"
    dt: str = "dt"


def column_order() -> List[str]:
    """Return the canonical column order for the master frame table."""

    c = Columns()
    return [
        # A. Index & Visual Anchors
        c.recording_id,
        c.frame,
        c.track_id,
        c.global_track_id,
        c.time,
        c.x_raw,
        c.y_raw,
        c.width,
        c.height,
        c.lane_id_raw,
        c.x_img,
        c.y_img,
        # B. Unified Physics Layer
        c.driving_direction,
        c.veh_class,
        c.s_long,
        c.d_lat,
        c.v_long_raw,
        c.a_long_raw,
        c.v_long_smooth,
        c.a_long_smooth,
        # C. Topology & Interaction
        c.preceding_id,
        c.following_id,
        c.left_preceding_id,
        c.left_alongside_id,
        c.left_following_id,
        c.right_preceding_id,
        c.right_alongside_id,
        c.right_following_id,
        c.leader_s_long,
        c.leader_v_long,
        c.leader_a_long,
        c.dist_headway_raw,
        c.time_headway_raw,
        c.ttc_raw,
        c.dist_headway,
        c.rel_velocity,
        c.time_headway,
        # D. Safety
        c.TTC,
        c.DRAC,
        c.risk_level,
        # E. Energy & Environment
        c.vehicle_type,
        c.v_kmh,
        c.a_kmhps,
        c.vtm_fuel_rate,
        c.vtm_co2_rate,
        c.vtm_nox_rate,
        c.vsp_value,
        c.vsp_co2_rate,
        c.vsp_nox_rate,
        c.dt,
    ]


__all__ = ["Columns", "column_order"]

