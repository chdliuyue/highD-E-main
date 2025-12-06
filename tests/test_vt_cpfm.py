from __future__ import annotations

import numpy as np

from data_preproc.vt_cpfm import co2_rate_gps, compute_power, fuel_rate_lps


def test_power_monotonic_with_speed_and_accel():
    v = np.array([0.0, 5.0, 10.0])
    a = np.array([0.0, 0.2, 0.5])
    cat = np.array(["LDV", "LDV", "LDV"])

    power = compute_power(v, a, cat)
    assert power[0] <= power[1] <= power[2]


def test_positive_power_increases_fuel_and_co2():
    power = np.array([0.0, 10.0, 50.0])
    cat = np.array(["HDDT", "HDDT", "HDDT"])

    fuel = fuel_rate_lps(power, cat)
    co2 = co2_rate_gps(fuel, cat)

    assert fuel[0] >= 0 and co2[0] >= 0
    assert fuel[1] > fuel[0]
    assert fuel[2] > fuel[1]
    assert co2[2] > co2[1] > co2[0]
