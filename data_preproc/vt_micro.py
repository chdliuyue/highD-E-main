"""VT-Micro emission model interface (placeholder implementation)."""
from typing import Dict
import numpy as np

import config


def _poly_eval(coeffs: Dict[tuple, float], v: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Evaluate a sparse 2D polynomial given coefficient dictionary."""
    result = np.zeros_like(v, dtype=np.float32)
    for (i, j), c in coeffs.items():
        result = result + c * (v ** i) * (a ** j)
    return result.astype(np.float32)


def vt_micro_emissions(
    vehicle_type: np.ndarray,
    v_kmh: np.ndarray,
    a_kmhps: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute VT-Micro fuel and emission rates using placeholder coefficients."""
    vehicle_type = np.asarray(vehicle_type)
    v_kmh = np.asarray(v_kmh, dtype=np.float32)
    a_kmhps = np.asarray(a_kmhps, dtype=np.float32)

    fuel = np.zeros_like(v_kmh, dtype=np.float32)
    co2 = np.zeros_like(v_kmh, dtype=np.float32)
    nox = np.zeros_like(v_kmh, dtype=np.float32)

    for veh in np.unique(vehicle_type):
        mask = vehicle_type == veh
        coeff = config.VT_MICRO_COEFFS.get(str(veh), config.VT_MICRO_COEFFS["Car"])
        fuel[mask] = _poly_eval(coeff["fuel"], v_kmh[mask], a_kmhps[mask])
        co2[mask] = _poly_eval(coeff["co2"], v_kmh[mask], a_kmhps[mask])
        nox[mask] = _poly_eval(coeff["nox"], v_kmh[mask], a_kmhps[mask])

    return {
        "fuel_rate": fuel.astype(np.float32),
        "co2_rate": co2.astype(np.float32),
        "nox_rate": nox.astype(np.float32),
    }


__all__ = ["vt_micro_emissions"]
