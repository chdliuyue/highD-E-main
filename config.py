"""Global configuration for the highD preprocessing project."""
from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "highD" / "data"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed" / "highD" / "data"
HIGHWAY_IMAGE_DIR: Path = RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Processing options
# ---------------------------------------------------------------------------
TEST_MODE: bool = True
TEST_RECORDINGS: List[int] = [1]
NUM_WORKERS: int = 1  # 1 = sequential; >1 = multiprocessing
MAX_RECORDING_ID: int = 60

# Safety metric options
RECOMPUTE_SAFETY_METRICS: bool = True
TTC_HIGH_RISK: float = 2.0
TTC_LOW_RISK: float = 4.0
DRAC_HIGH_RISK: float = 3.0  # [m/s^2], placeholder threshold

# Smoothing parameters
SAVGOL_WINDOW: int = 9
SAVGOL_POLY: int = 3

# Vehicle Specific Power parameters (placeholder coefficients)
AIR_RESIST_COEFF: float = 0.00035
ROLLING_RESIST_COEFF: float = 0.132
DRIVETRAIN_LOSS_COEFF: float = 0.000302
VEHICLE_MASS_CAR: float = 1500.0  # kg, placeholder
VEHICLE_MASS_TRUCK: float = 8000.0  # kg, placeholder
GRAVITY: float = 9.81
ROAD_GRADE: float = 0.0  # assume flat road

# VT-Micro placeholder coefficients
VT_MICRO_COEFFS = {
    "Car": {
        "fuel": {(0, 0): -0.3, (1, 0): 0.02, (0, 1): 0.03, (1, 1): 0.0002},
        "co2": {(0, 0): 0.5, (1, 0): 0.05, (0, 1): 0.06},
        "nox": {(0, 0): 0.05, (1, 0): 0.01, (0, 1): 0.008},
    },
    "Truck": {
        "fuel": {(0, 0): -0.2, (1, 0): 0.03, (0, 1): 0.04, (1, 1): 0.0003},
        "co2": {(0, 0): 0.6, (1, 0): 0.06, (0, 1): 0.07},
        "nox": {(0, 0): 0.06, (1, 0): 0.015, (0, 1): 0.012},
    },
}

# Mapping helpers
VEHICLE_TYPE_MAP = {"Car": "Car", "Truck": "Truck"}

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "HIGHWAY_IMAGE_DIR",
    "TEST_MODE",
    "TEST_RECORDINGS",
    "NUM_WORKERS",
    "MAX_RECORDING_ID",
    "RECOMPUTE_SAFETY_METRICS",
    "TTC_HIGH_RISK",
    "TTC_LOW_RISK",
    "DRAC_HIGH_RISK",
    "SAVGOL_WINDOW",
    "SAVGOL_POLY",
    "AIR_RESIST_COEFF",
    "ROLLING_RESIST_COEFF",
    "DRIVETRAIN_LOSS_COEFF",
    "VEHICLE_MASS_CAR",
    "VEHICLE_MASS_TRUCK",
    "GRAVITY",
    "ROAD_GRADE",
    "VT_MICRO_COEFFS",
    "VEHICLE_TYPE_MAP",
]
