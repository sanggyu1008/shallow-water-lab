"""진단량(에너지·소용돌이도·지형류·유선함수) 검증."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.dynamics import diagnostics as diag
from shallow_water.grids.cgrid import CGrid


def test_energy_components_nonnegative_and_sum():
    grid = CGrid(nx=10, ny=10, Lx=10.0, Ly=10.0)
    eta = np.ones((10, 10))
    u = np.full((10, 11), 0.5)
    v = np.full((11, 10), -0.3)
    e = diag.energy_components(grid, eta, u, v, g=9.81, H=10.0)
    assert e["KE"] > 0 and e["PE"] > 0
    assert e["total"] == pytest.approx(e["KE"] + e["PE"])


def test_relative_vorticity_solid_body_rotation():
    """강체회전 u=-Ωy, v=Ωx 의 소용돌이도는 2Ω."""
    grid = CGrid(nx=40, ny=40, Lx=40.0, Ly=40.0)
    Omega = 0.1
    u = np.zeros((40, 41))
    v = np.zeros((41, 40))
    # u-점은 y_eta 행, v-점은 x_eta 열에 위치
    u[:] = (-Omega * grid.y_eta).reshape(-1, 1)
    v[:] = (Omega * grid.x_eta).reshape(1, -1)
    zeta = diag.relative_vorticity(grid, u, v)
    np.testing.assert_allclose(zeta[2:-2, 2:-2], 2 * Omega, atol=1e-9)


def test_geostrophic_velocity_linear_eta():
    """η = a·y 이면 u_g = -(g/f) a, v_g = 0."""
    grid = CGrid(nx=20, ny=20, Lx=20.0, Ly=20.0)
    g, f, a = 9.81, 1e-4, 0.01
    _, Y = grid.meshgrid_eta()
    eta = a * Y
    u_g, v_g = diag.geostrophic_velocity(grid, eta, g, f)
    np.testing.assert_allclose(u_g[2:-2, 2:-2], -(g / f) * a, rtol=1e-6)
    np.testing.assert_allclose(v_g[2:-2, 2:-2], 0.0, atol=1e-6)


def test_transport_streamfunction_uniform_u():
    """균일류 u=U 이면 ψ = -U (y - y0)."""
    grid = CGrid(nx=10, ny=12, Lx=10.0, Ly=12.0)
    U = 0.7
    u = np.full((12, 11), U)
    psi = diag.transport_streamfunction(grid, u)
    expected = -U * (grid.y_eta - grid.y_eta[0]).reshape(-1, 1)
    np.testing.assert_allclose(psi, np.broadcast_to(expected, psi.shape), atol=1e-9)
