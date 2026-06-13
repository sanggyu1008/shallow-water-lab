"""Arakawa C-grid 차분/보간 연산자 검증."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.grids.cgrid import CGrid


def test_field_shapes():
    g = CGrid(nx=8, ny=6, Lx=8.0, Ly=6.0)
    eta, u, v = g.zeros()
    assert eta.shape == (6, 8)
    assert u.shape == (6, 9)  # (ny, nx+1)
    assert v.shape == (7, 8)  # (ny+1, nx)


def test_divergence_of_uniform_is_zero():
    g = CGrid(nx=8, ny=6, Lx=8.0, Ly=6.0)
    u = np.full((6, 9), 2.3)
    v = np.full((7, 8), -1.1)
    div = g.divergence(u, v)
    np.testing.assert_allclose(div, 0.0, atol=1e-12)


def test_deta_dx_linear():
    g = CGrid(nx=10, ny=4, Lx=10.0, Ly=4.0)  # dx=1
    X, _ = g.meshgrid_eta()
    eta = 3.0 * X  # ∂η/∂x = 3
    dpdx = g.deta_dx(eta)
    # 내부 face 에서 정확히 3
    np.testing.assert_allclose(dpdx[:, 1:-1], 3.0, atol=1e-12)


def test_v_at_u_of_constant():
    g = CGrid(nx=8, ny=6, Lx=8.0, Ly=6.0)
    v = np.full((7, 8), 4.0)
    vu = g.v_at_u(v)
    assert vu.shape == (6, 9)
    np.testing.assert_allclose(vu, 4.0, atol=1e-12)


def test_laplacian_of_quadratic():
    # ∇²(x² + y²) = 4 (내부)
    g = CGrid(nx=20, ny=20, Lx=20.0, Ly=20.0)  # dx=dy=1
    X, Y = g.meshgrid_eta()
    f = X**2 + Y**2
    lap = g.laplacian_u(f)  # eta 점이지만 연산자는 형태 무관
    np.testing.assert_allclose(lap[2:-2, 2:-2], 4.0, atol=1e-9)


def test_divergence_telescopes_to_boundary_flux():
    # 닫힌 벽에서 u,v 가 경계에서 0이면 전체 발산 적분 = 0
    g = CGrid(nx=12, ny=10, Lx=12.0, Ly=10.0)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((10, 13))
    v = rng.standard_normal((11, 12))
    u[:, 0] = u[:, -1] = 0.0
    v[0, :] = v[-1, :] = 0.0
    div = g.divergence(u, v)
    assert abs(np.sum(div) * g.dx * g.dy) < 1e-10
