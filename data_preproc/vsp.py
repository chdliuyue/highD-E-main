"""Vehicle Specific Power (VSP) emission model interface."""
from typing import Dict
import numpy as np

import config


def _compute_vsp(v: np.ndarray, a: np.ndarray, mass: float) -> np.ndarray:
    """Compute VSP using a simplified polynomial model."""
    aerodynamic = config.AIR_RESIST_COEFF * v ** 3
    rolling = config.ROLLING_RESIST_COEFF * v
    drivetrain = config.DRIVETRAIN_LOSS_COEFF * v ** 2
    grade_component = config.GRAVITY * np.sin(config.ROAD_GRADE) * v
    traction = v * a
    vsp = (traction + aerodynamic + rolling + drivetrain + grade_component) / max(mass, 1e-3)
    return vsp.astype(np.float32)


def _piecewise_emission(vsp_value: np.ndarray) -> Dict[str, np.ndarray]:
    """Map VSP to emission rates using a simple piecewise linear model."""
    co2 = np.where(vsp_value < 0, 0.5, 0.7 * vsp_value + 0.5)
    nox = np.where(vsp_value < 0, 0.05, 0.1 * vsp_value + 0.05)
    return {
        "co2": co2.astype(np.float32),
        "nox": nox.astype(np.float32),
    }


def vsp_emissions(
    vehicle_type: np.ndarray,
    v_mps: np.ndarray,
    a_mps2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute VSP and derived emissions using placeholder logic."""
    vehicle_type = np.asarray(vehicle_type)
    v_mps = np.asarray(v_mps, dtype=np.float32)
    a_mps2 = np.asarray(a_mps2, dtype=np.float32)

    vsp = np.zeros_like(v_mps, dtype=np.float32)
    co2 = np.zeros_like(v_mps, dtype=np.float32)
    nox = np.zeros_like(v_mps, dtype=np.float32)

    for veh in np.unique(vehicle_type):
        mask = vehicle_type == veh
        mass = config.VEHICLE_MASS_TRUCK if str(veh) == "Truck" else config.VEHICLE_MASS_CAR
        vsp_segment = _compute_vsp(v_mps[mask], a_mps2[mask], mass)
        emissions = _piecewise_emission(vsp_segment)
        vsp[mask] = vsp_segment
        co2[mask] = emissions["co2"]
        nox[mask] = emissions["nox"]

    return {
        "vsp_value": vsp.astype(np.float32),
        "co2_rate": co2.astype(np.float32),
        "nox_rate": nox.astype(np.float32),
    }


__all__ = ["vsp_emissions"]
