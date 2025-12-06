"""Vehicle Specific Power (VSP) emission model interface."""
from typing import Dict
import numpy as np

from config import (
    AIR_RESIST_COEFF,
    DRIVETRAIN_LOSS_COEFF,
    GRAVITY,
    ROAD_GRADE,
    ROLLING_RESIST_COEFF,
    VEHICLE_MASS_CAR,
    VEHICLE_MASS_TRUCK,
)


def _compute_vsp(v: np.ndarray, a: np.ndarray, mass: float) -> np.ndarray:
    """Compute VSP using a simplified polynomial model.

    The formula follows a generic structure:
    ``VSP = v * (a + g*sin(theta)) + c1*v + c2*v**2 + c3*v**3``.
    Here we assume flat road (sin(theta)=0) and use placeholder coefficients.
    """

    aerodynamic = AIR_RESIST_COEFF * v ** 3
    rolling = ROLLING_RESIST_COEFF * v
    drivetrain = DRIVETRAIN_LOSS_COEFF * v ** 2
    grade_component = GRAVITY * np.sin(ROAD_GRADE) * v
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
    """Compute VSP and derived emissions using placeholder logic.

    Parameters
    ----------
    vehicle_type: np.ndarray
        Vehicle categories ("Car" or "Truck").
    v_mps: np.ndarray
        Longitudinal speed in m/s.
    a_mps2: np.ndarray
        Longitudinal acceleration in m/s^2.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys ``vsp_value``, ``co2_rate`` and ``nox_rate``.
    """
    vehicle_type = np.asarray(vehicle_type)
    v_mps = np.asarray(v_mps, dtype=np.float32)
    a_mps2 = np.asarray(a_mps2, dtype=np.float32)

    vsp = np.zeros_like(v_mps, dtype=np.float32)
    co2 = np.zeros_like(v_mps, dtype=np.float32)
    nox = np.zeros_like(v_mps, dtype=np.float32)

    for veh in np.unique(vehicle_type):
        mask = vehicle_type == veh
        mass = VEHICLE_MASS_TRUCK if str(veh) == "Truck" else VEHICLE_MASS_CAR
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

