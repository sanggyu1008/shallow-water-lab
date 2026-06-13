"""2차원 C-grid 천수방정식 검증 (보존, 회전, 지형류)."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.dynamics import diagnostics as diag
from shallow_water.dynamics.swe2d import ShallowWater2D
from shallow_water.grids.cgrid import CGrid


def test_cfl_and_deformation_radius():
    grid = CGrid(nx=50, ny=50, Lx=5e5, Ly=5e5)
    m = ShallowWater2D(grid=grid, g=9.81, H=500.0, f0=1e-4)
    assert m.gravity_wave_speed() == pytest.approx(np.sqrt(9.81 * 500.0))
    assert m.deformation_radius() == pytest.approx(np.sqrt(9.81 * 500.0) / 1e-4)
    dt = 0.5 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    assert m.cfl(dt) == pytest.approx(0.5, rel=1e-6)


def test_mass_conserved_closed_basin():
    grid = CGrid(nx=40, ny=40, Lx=4e5, Ly=4e5)
    m = ShallowWater2D(grid=grid, g=9.81, H=100.0, f0=0.0)
    X, Y = grid.meshgrid_eta()
    eta0 = np.exp(-(((X - 2e5) ** 2 + (Y - 2e5) ** 2)) / (3e4) ** 2)
    u0 = np.zeros((40, 41))
    v0 = np.zeros((41, 40))
    dt = 0.4 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    out = m.run(eta0, u0, v0, dt, nsteps=600)
    assert np.isclose(np.sum(out["eta"]), np.sum(eta0), rtol=1e-9)


def test_gravity_wave_energy_bounded():
    """회전 없는 중력파: 에너지가 폭주하지 않는다(FB 안정)."""
    grid = CGrid(nx=40, ny=40, Lx=4e5, Ly=4e5)
    m = ShallowWater2D(grid=grid, g=9.81, H=100.0, f0=0.0)
    X, Y = grid.meshgrid_eta()
    eta0 = np.exp(-(((X - 2e5) ** 2 + (Y - 2e5) ** 2)) / (3e4) ** 2)
    u0 = np.zeros((40, 41))
    v0 = np.zeros((41, 40))
    dt = 0.4 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    out = m.run(eta0, u0, v0, dt, nsteps=1500)
    e = out["energy"]
    assert np.max(e) < 1.05 * e[0]
    assert np.all(np.isfinite(out["eta"]))


def test_gravity_wave_radiates_symmetrically():
    """중심 봉우리는 사방으로 퍼져 중심 진폭이 줄어든다(반사 전)."""
    grid = CGrid(nx=80, ny=80, Lx=8e5, Ly=8e5)
    m = ShallowWater2D(grid=grid, g=9.81, H=100.0, f0=0.0)
    X, Y = grid.meshgrid_eta()
    eta0 = np.exp(-(((X - 4e5) ** 2 + (Y - 4e5) ** 2)) / (2e4) ** 2)
    u0 = np.zeros((80, 81))
    v0 = np.zeros((81, 80))
    dt = 0.4 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    out = m.run(eta0, u0, v0, dt, nsteps=200)
    # 중심 진폭이 줄고, 대칭이라 x/y 방향 단면이 거의 같다
    assert out["eta"].max() < eta0.max()
    cx = out["eta"][40, :]
    cy = out["eta"][:, 40]
    assert np.allclose(cx, cy, atol=1e-3 * eta0.max())


def test_inertial_oscillation_amplitude_no_growth():
    """균일류의 관성진동: symplectic 처리로 진폭이 발산하지 않는다."""
    grid = CGrid(nx=4, ny=4, Lx=4.0, Ly=4.0, periodic_x=True, periodic_y=True)
    f0 = 1.0
    m = ShallowWater2D(grid=grid, g=10.0, H=10.0, f0=f0)
    eta0, u0, v0 = grid.zeros()
    u0[:] = 1.0  # 균일 동서류
    dt = 0.02 / f0  # f dt = 0.02
    nsteps = int(round(10 * 2 * np.pi / (f0 * dt)))  # 10 관성주기
    out = m.run(eta0, u0, v0, dt, nsteps=nsteps, save_every=20)
    # 운동에너지 ∝ (진폭)^2 이므로 sqrt(KE/KE0) 로 진폭비를 본다
    ke = np.array(
        [
            diag.energy_components(grid, e, uu, vv, 10.0, 10.0)["KE"]
            for e, uu, vv in zip(out["eta_hist"], out["u_hist"], out["v_hist"])
        ]
    )
    amp = np.sqrt(ke / ke[0])
    # 10 관성주기 동안 진폭이 1 근처에서 발산/감쇠 없이 유지 (symplectic)
    assert np.max(amp) < 1.02
    assert np.min(amp) > 0.98


def test_geostrophic_jet_approximately_steady():
    """지형류 평형 zonal jet 은 거의 정상상태로 유지된다 (주기채널).

    동서로 흐르는 jet 은 동서벽이 있으면 벽으로 흘러들어 모순이므로,
    x-주기 채널(periodic_x)에서 시험한다.
    """
    grid = CGrid(nx=60, ny=60, Lx=3e6, Ly=3e6, periodic_x=True)
    g, H, f0 = 9.81, 500.0, 1e-4
    m = ShallowWater2D(grid=grid, g=g, H=H, f0=f0)
    Ld = m.deformation_radius()
    # η(y): y에만 의존하는 매끈한 능선, u_geo = -(g/f) dη/dy
    yc = grid.Ly / 2
    eta_amp = 0.5
    E = lambda y: eta_amp * np.tanh((y - yc) / Ld)
    Eprime = lambda y: eta_amp / Ld / np.cosh((y - yc) / Ld) ** 2
    eta0 = np.tile(E(grid.y_eta).reshape(-1, 1), (1, grid.nx))
    u0 = np.zeros((grid.ny, grid.nx + 1))
    u0[:] = (-(g / f0) * Eprime(grid.y_eta)).reshape(-1, 1)
    v0 = np.zeros((grid.ny + 1, grid.nx))
    dt = 0.3 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    period = 2 * np.pi / f0
    nsteps = int(round(2 * period / dt))
    out = m.run(eta0, u0, v0, dt, nsteps=nsteps)
    # v 는 작게 유지되고 u 는 거의 변하지 않는다
    assert np.max(np.abs(out["v"])) < 0.01 * np.max(np.abs(u0))
    assert np.max(np.abs(out["u"] - u0)) < 0.01 * np.max(np.abs(u0))


def test_potential_vorticity_conserved_in_adjustment():
    """지형류 조정 동안 위치소용돌이도가 근사적으로 보존된다."""
    grid = CGrid(nx=80, ny=80, Lx=4e6, Ly=4e6)
    g, H, f0 = 9.81, 500.0, 1e-4
    m = ShallowWater2D(grid=grid, g=g, H=H, f0=f0)
    Ld = m.deformation_radius()
    X, Y = grid.meshgrid_eta()
    # 초기: x 방향 단차(rest), 폭 ~ Ld
    eta0 = 0.5 * np.tanh((X - grid.Lx / 2) / (0.3 * Ld))
    u0 = np.zeros((grid.ny, grid.nx + 1))
    v0 = np.zeros((grid.ny + 1, grid.nx))
    dt = 0.3 * grid.dx / (m.gravity_wave_speed() * np.sqrt(2))
    nsteps = int(round(3 * 2 * np.pi / f0 / dt))
    out = m.run(eta0, u0, v0, dt, nsteps=nsteps)
    q0 = diag.potential_vorticity(grid, eta0, u0, v0, f0, H)
    q1 = diag.potential_vorticity(grid, out["eta"], out["u"], out["v"], f0, H)
    scale = np.max(np.abs(q0))
    # 내부에서 PV 변화가 작다
    err = np.max(np.abs(q1 - q0)[5:-5, 5:-5]) / scale
    assert err < 0.2
