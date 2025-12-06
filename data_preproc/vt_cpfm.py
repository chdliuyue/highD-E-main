"""VT-CPFM-1 power, fuel, and CO2 estimation utilities.

This module implements a representative-vehicle VT-CPFM-1 (Comprehensive Power-
Based Fuel Consumption Model) pipeline for the highD dataset. Parameter sources
are cited in the constants below:

- LDV_PARAMS (Honda Civic, gasoline): Kamalanathsharma (2014) Table 7.3.
- HDDT_PARAMS (International 9800 SBA, diesel): Fahmin et al. (2022) Table 1 for
  physical parameters; fuel map coefficients currently reuse LDV coefficients as
  a simplifying assumption pending HDDT-specific calibration.
"""
from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd

import config

VehicleCat = Literal["LDV", "HDDT"]

RHO_AIR_KG_M3 = 1.2256  # Sea-level air density [kg/m^3], Fahmin et al. (2022)
GRAVITY = 9.80665  # [m/s^2]

LDV_PARAMS: Dict[str, float] = {
    "mass_kg": 1453.0,
    "cd": 0.30,
    "area_m2": 2.32,
    "eta_d": 0.92,
    "lambda_rot": 0.04,
    "Cr": 1.75,
    "c1": 0.0328,
    "c2": 4.575,
    "alpha0": 4.7738e-4,
    "alpha1": 5.363e-5,
    "alpha2": 1.0e-6,
}

HDDT_PARAMS: Dict[str, float] = {
    "mass_kg": 7239.0,
    "cd": 0.78,
    "area_m2": 8.90,
    "eta_d": 0.95,
    "lambda_rot": 0.10,
    "Cr": 1.75,
    "c1": 0.0328,
    "c2": 4.575,
    "alpha0": 4.7738e-4,  # Simplified assumption: reuse LDV fuel map coefficients
    "alpha1": 5.363e-5,
    "alpha2": 1.0e-6,
}

GASOLINE_CO2_FACTOR = 2310.0  # [g/L]
DIESEL_CO2_FACTOR = 2680.0  # [g/L]

PARAMS_BY_CAT: Dict[VehicleCat, Dict[str, float]] = {
    "LDV": LDV_PARAMS,
    "HDDT": HDDT_PARAMS,
}


def _get_param_array(vehicle_cat: np.ndarray, key: str) -> np.ndarray:
    """Vectorized helper to pull parameter arrays per vehicle category."""

    ldv_val = PARAMS_BY_CAT["LDV"][key]
    hddt_val = PARAMS_BY_CAT["HDDT"][key]
    return np.where(vehicle_cat == "LDV", ldv_val, hddt_val).astype(float)


def compute_power(v_mps: np.ndarray, a_mps2: np.ndarray, vehicle_cat: np.ndarray) -> np.ndarray:
    """Compute traction power P(t) in kW using the VT-CPFM-1 formulation.

    Parameters
    ----------
    v_mps : np.ndarray
        Vehicle speed [m/s].
    a_mps2 : np.ndarray
        Longitudinal acceleration [m/s^2].
    vehicle_cat : np.ndarray
        Array of vehicle category strings ("LDV" or "HDDT").

    Returns
    -------
    np.ndarray
        Traction power in kW for each sample.
    """

    v = np.asarray(v_mps, dtype=float)
    a = np.asarray(a_mps2, dtype=float)
    cat = np.asarray(vehicle_cat)

    mass = _get_param_array(cat, "mass_kg")
    lambda_rot = _get_param_array(cat, "lambda_rot")
    cd = _get_param_array(cat, "cd")
    area = _get_param_array(cat, "area_m2")
    eta_d = _get_param_array(cat, "eta_d")
    cr = _get_param_array(cat, "Cr")
    c1 = _get_param_array(cat, "c1")
    c2 = _get_param_array(cat, "c2")

    v_kmh = v * 3.6
    effective_mass = mass * (1.0 + lambda_rot)
    rolling_resist = mass * GRAVITY * cr / 1000.0 * (c1 * v_kmh + c2)
    aero_resist = 0.5 * RHO_AIR_KG_M3 * area * cd * v**2

    tractive_force = effective_mass * a + rolling_resist + aero_resist
    power_kw = tractive_force * v / (1000.0 * eta_d)
    return power_kw.astype(float)


def fuel_rate_lps(power_kw: np.ndarray, vehicle_cat: np.ndarray) -> np.ndarray:
    """Compute instantaneous fuel consumption rate [L/s] using VT-CPFM-1.

    For P >= 0: FC = alpha0 + alpha1 * P + alpha2 * P^2
    For P < 0:  FC = alpha0 (idling/zero traction).
    """

    power = np.asarray(power_kw, dtype=float)
    cat = np.asarray(vehicle_cat)

    alpha0 = _get_param_array(cat, "alpha0")
    alpha1 = _get_param_array(cat, "alpha1")
    alpha2 = _get_param_array(cat, "alpha2")

    positive = power >= 0
    fc = np.where(positive, alpha0 + alpha1 * power + alpha2 * power**2, alpha0)
    return fc.astype(float)


def co2_rate_gps(fuel_rate_lps: np.ndarray, vehicle_cat: np.ndarray) -> np.ndarray:
    """Convert fuel consumption [L/s] to CO2 emission rate [g/s]."""

    fuel_rate = np.asarray(fuel_rate_lps, dtype=float)
    cat = np.asarray(vehicle_cat)

    factor = np.where(cat == "LDV", GASOLINE_CO2_FACTOR, DIESEL_CO2_FACTOR)
    return (fuel_rate * factor).astype(float)


def apply_vt_cpfm_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply VT-CPFM-1 to an L1 frame-level dataframe.

    Expected columns in ``df`` include:
      - ``v_long_smooth`` [m/s]
      - ``a_long_smooth`` [m/s^2]
      - ``veh_class`` (encoded as numbers or strings; "Car"/"Truck")
      - ``dt`` optional frame duration [s]

    The following columns are appended:
      - ``vehicle_type`` ("LDV"/"HDDT")
      - ``v_kmh`` and ``a_kmhps``
      - ``cpf_power_kw``
      - ``cpf_fuel_rate_lps``
      - ``cpf_co2_rate_gps``
    """

    df_out = df.copy()
    veh_class = df_out["veh_class"]

    mapping = {**config.VEHICLE_CLASS_DECODING, 1: "Car", 2: "Truck", "Car": "Car", "Truck": "Truck"}
    veh_class_str = veh_class.map(mapping).fillna(veh_class).astype(str)
    vehicle_cat = veh_class_str.map({"Car": "LDV", "Truck": "HDDT"}).fillna("LDV")

    v_mps = df_out["v_long_smooth"].to_numpy(dtype=float)
    a_mps2 = df_out["a_long_smooth"].to_numpy(dtype=float)

    df_out["vehicle_type"] = vehicle_cat
    df_out["v_kmh"] = (v_mps * 3.6).astype(np.float32)
    df_out["a_kmhps"] = (a_mps2 * 3.6).astype(np.float32)

    power_kw = compute_power(v_mps, a_mps2, vehicle_cat.to_numpy())
    fuel_rate = fuel_rate_lps(power_kw, vehicle_cat.to_numpy())
    co2_rate = co2_rate_gps(fuel_rate, vehicle_cat.to_numpy())

    df_out["cpf_power_kw"] = power_kw.astype(np.float32)
    df_out["cpf_fuel_rate_lps"] = fuel_rate.astype(np.float32)
    df_out["cpf_co2_rate_gps"] = co2_rate.astype(np.float32)

    return df_out


__all__ = [
    "VehicleCat",
    "LDV_PARAMS",
    "HDDT_PARAMS",
    "compute_power",
    "fuel_rate_lps",
    "co2_rate_gps",
    "apply_vt_cpfm_to_df",
]
