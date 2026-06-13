"""바람응력·마찰 강제 검증."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.forcing import wind


def test_single_gyre_wind_endpoints():
    Ly, tau0 = 1.0, 0.1
    y = np.array([0.0, Ly])
    tau = wind.single_gyre_wind(y, Ly, tau0)
    assert tau[0] == pytest.approx(-tau0)  # -τ0 cos(0)
    assert tau[1] == pytest.approx(tau0)  # -τ0 cos(π)


def test_double_gyre_wind_sign_change():
    Ly, tau0 = 1.0, 0.1
    y = np.array([0.0, Ly / 2, Ly])
    tau = wind.double_gyre_wind(y, Ly, tau0)
    assert tau[0] == pytest.approx(-tau0)  # -τ0 cos(0)
    assert tau[1] == pytest.approx(tau0)  # -τ0 cos(π)
    assert tau[2] == pytest.approx(-tau0)  # -τ0 cos(2π)


def test_wind_stress_curl_kind():
    Ly, tau0 = 1.0, 0.1
    y = np.array([0.25 * Ly])
    cs = wind.wind_stress_curl(y, Ly, tau0, "single")
    cd = wind.wind_stress_curl(y, Ly, tau0, "double")
    assert cs[0] < 0  # -τ0 (π/Ly) sin(π/4) < 0
    assert np.isfinite(cd[0])
    with pytest.raises(ValueError):
        wind.wind_stress_curl(y, Ly, tau0, "triple")


def test_boundary_layer_formulas():
    assert wind.stommel_boundary_layer(r=2.0, beta=4.0) == pytest.approx(0.5)
    assert wind.munk_boundary_layer(A_h=8.0, beta=1.0) == pytest.approx(2.0)
