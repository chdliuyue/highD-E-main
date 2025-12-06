"""VT-Micro emission model utilities based on coefficient tables.

This module is retained **only for legacy comparisons**; the primary pipeline now
uses VT-CPFM-derived ``cpf_*`` fields. Coefficients are expected to be supplied
via an external JSON file rather than hard-coded placeholders. The JSON should
follow the structure described in ``load_vtmicro_coeffs``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

CoeffKey = Tuple[str, str, str]  # (vehicle_cat, pollutant, regime)

DEFAULT_VTMICRO_COEFFS_PATH = PROJECT_ROOT / "data" / "vtmicro_coeffs.json"


def load_vtmicro_coeffs(json_path: Path = DEFAULT_VTMICRO_COEFFS_PATH) -> Dict[CoeffKey, np.ndarray]:
    """Load VT-Micro coefficients from a JSON file.

    Parameters
    ----------
    json_path:
        Path to the coefficient JSON. Expected structure::

            {
              "LDV": {
                "CO2": {
                  "pos": [[c00, c01, c02, c03],
                          [c10, c11, c12, c13],
                          [c20, c21, c22, c23],
                          [c30, c31, c32, c33]],
                  "neg": [[...], [...], [...], [...]]
                },
                "NOx": { ... },
                "Fuel": { ... }
              },
              "LDT": { ... }
            }

    Returns
    -------
    Dict[CoeffKey, np.ndarray]
        Dictionary keyed by (vehicle_cat, pollutant, regime) with 4x4 numpy arrays
        holding the polynomial coefficients. If the file does not exist, an empty
        dict is returned so the caller can decide on a fallback strategy.
    """

    coeffs: Dict[CoeffKey, np.ndarray] = {}
    if not json_path.exists():
        return coeffs

    with json_path.open("r") as f:
        raw = json.load(f)

    for veh_cat, pollutant_dict in raw.items():
        for pollutant, regime_dict in pollutant_dict.items():
            for regime, mat in regime_dict.items():
                coeffs[(veh_cat, pollutant, regime)] = np.asarray(mat, dtype=float)
    return coeffs


def vtmicro_eval(v_kmh: np.ndarray, a_kmhps: np.ndarray, coeff_mat: np.ndarray) -> np.ndarray:
    """Evaluate the VT-Micro polynomial for given speed/acceleration.

    Parameters
    ----------
    v_kmh:
        Speed in km/h (scalar or array-like).
    a_kmhps:
        Acceleration in km/h/s (scalar or array-like).
    coeff_mat:
        4x4 coefficient matrix for the desired vehicle/pollutant/regime.

    Returns
    -------
    np.ndarray
        Instantaneous emission rate with the same shape as ``v_kmh``.
    """

    v = np.asarray(v_kmh, dtype=float)
    a = np.asarray(a_kmhps, dtype=float)
    coeff = np.asarray(coeff_mat, dtype=float)

    # Compute \sum_{i=0..3}\sum_{j=0..3} coeff[i,j] * v^i * a^j
    powers_v = np.stack([np.ones_like(v), v, v**2, v**3], axis=0)
    powers_a = np.stack([np.ones_like(a), a, a**2, a**3], axis=0)

    exp_arg = np.zeros_like(v, dtype=float)
    for i in range(4):
        for j in range(4):
            exp_arg += coeff[i, j] * powers_v[i] * powers_a[j]

    return np.exp(exp_arg)


def vt_micro_emissions(
    vehicle_cat: np.ndarray,
    pollutant: str,
    v_kmh: np.ndarray,
    a_kmhps: np.ndarray,
    coeffs: Dict[CoeffKey, np.ndarray],
) -> np.ndarray:
    """Compute VT-Micro emission rate for a specific pollutant.

    Parameters
    ----------
    vehicle_cat:
        Array of VT-Micro vehicle categories (e.g., ``"LDV"`` or ``"LDT"``).
    pollutant:
        Pollutant name such as ``"CO2"``, ``"NOx"`` or ``"Fuel"``.
    v_kmh / a_kmhps:
        Speed and acceleration in km/h and km/h/s.
    coeffs:
        Coefficient dictionary produced by :func:`load_vtmicro_coeffs`.

    Returns
    -------
    np.ndarray
        Emission rate array. If coefficients for a sample are missing, ``NaN`` is
        returned for that position.
    """

    vehicle_cat = np.asarray(vehicle_cat)
    v_kmh = np.asarray(v_kmh, dtype=float)
    a_kmhps = np.asarray(a_kmhps, dtype=float)

    results = np.full_like(v_kmh, np.nan, dtype=float)
    for veh in np.unique(vehicle_cat):
        mask = vehicle_cat == veh
        regime = np.where(a_kmhps[mask] >= 0, "pos", "neg")

        for reg in ["pos", "neg"]:
            reg_mask = mask & (regime == reg)
            if not reg_mask.any():
                continue

            key = (str(veh), pollutant, reg)
            coeff_mat = coeffs.get(key)
            if coeff_mat is None:
                # No coefficient available; leave NaN so downstream code can handle.
                continue
            results[reg_mask] = vtmicro_eval(v_kmh[reg_mask], a_kmhps[reg_mask], coeff_mat)

    return results


def compute_vtmicro_for_df(
    df: pd.DataFrame,
    coeffs: Dict[CoeffKey, np.ndarray],
    vehicle_cat_mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute VT-Micro rates for a frame-level dataframe.

    Parameters
    ----------
    df:
        L1 master frame dataframe containing ``veh_class``, ``v_long_smooth`` and
        ``a_long_smooth`` columns.
    coeffs:
        Coefficient dictionary produced by :func:`load_vtmicro_coeffs`.
    vehicle_cat_mapping:
        Mapping from ``veh_class`` to VT-Micro categories, e.g.,
        ``{"Car": "LDV", "Truck": "LDT"}``. If omitted, the original values are
        used directly.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with additional columns ``vtm_fuel_rate``,
        ``vtm_co2_rate`` and ``vtm_nox_rate`` appended. Missing coefficients
        result in ``NaN`` values.
    """

    df_out = df.copy()
    mapping = vehicle_cat_mapping or {}

    vehicle_cat = df_out["veh_class"].map(mapping).fillna(df_out["veh_class"])
    v_kmh = df_out["v_long_smooth"] * 3.6
    a_kmhps = df_out["a_long_smooth"] * 3.6

    df_out["vehicle_type"] = vehicle_cat
    df_out["v_kmh"] = v_kmh.astype(np.float32)
    df_out["a_kmhps"] = a_kmhps.astype(np.float32)

    df_out["vtm_fuel_rate"] = vt_micro_emissions(vehicle_cat, "Fuel", v_kmh, a_kmhps, coeffs).astype(
        np.float32
    )
    df_out["vtm_co2_rate"] = vt_micro_emissions(vehicle_cat, "CO2", v_kmh, a_kmhps, coeffs).astype(
        np.float32
    )
    df_out["vtm_nox_rate"] = vt_micro_emissions(vehicle_cat, "NOx", v_kmh, a_kmhps, coeffs).astype(
        np.float32
    )

    return df_out


__all__ = [
    "CoeffKey",
    "DEFAULT_VTMICRO_COEFFS_PATH",
    "load_vtmicro_coeffs",
    "vtmicro_eval",
    "vt_micro_emissions",
    "compute_vtmicro_for_df",
]
