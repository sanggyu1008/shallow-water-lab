"""1차원 선형 천수방정식 검증 (파속, 보존, CFL)."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.dynamics import swe1d
from shallow_water.grids.grid1d import StaggeredGrid1D

G, H = 10.0, 10.0  # c = sqrt(gH) = 10


def _gaussian(x, x0, w):
    return np.exp(-(((x - x0) / w) ** 2))


def test_mass_conserved_periodic():
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=True)
    eta0 = _gaussian(g.x_eta, 100.0, 8.0)
    u0 = np.zeros(g.n_u)
    dt = 0.5 * g.dx / g.gravity_wave_speed(G, H)
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=400)
    assert np.isclose(np.sum(out["eta"]), np.sum(eta0), rtol=1e-10)


def test_mass_conserved_wall():
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=False)
    eta0 = _gaussian(g.x_eta, 100.0, 8.0)
    u0 = np.zeros(g.n_u)
    dt = 0.5 * g.dx / g.gravity_wave_speed(G, H)
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=400)
    assert np.isclose(np.sum(out["eta"]), np.sum(eta0), rtol=1e-10)


def test_energy_bounded_when_stable():
    """CFL<1 에서 forward-backward 에너지는 안정적으로 유지(발산 없음)."""
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=True)
    eta0 = _gaussian(g.x_eta, 100.0, 8.0)
    u0 = np.zeros(g.n_u)
    dt = 0.5 * g.dx / g.gravity_wave_speed(G, H)
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=2000)
    e = out["energy"]
    # 에너지가 초기값 근처에 머문다(폭주하지 않음).
    assert np.max(e) < 1.05 * e[0]
    assert np.min(e) > 0.5 * e[0]


def test_cfl_violation_blows_up():
    """CFL>1 이면 불안정하게 폭주한다."""
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=True)
    eta0 = 0.01 * np.sin(2 * np.pi * 5 * g.x_eta / g.Lx)
    u0 = np.zeros(g.n_u)
    dt = 1.3 * g.dx / g.gravity_wave_speed(G, H)  # CFL=1.3
    with np.errstate(over="ignore", invalid="ignore"):
        out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=400)
    assert np.max(np.abs(out["eta"])) > 100.0 * np.max(np.abs(eta0))


def test_gravity_wave_speed_split_pulse():
    """정지 상태의 봉우리는 ±c 로 갈라져 이동한다."""
    g = StaggeredGrid1D(nx=400, Lx=400.0, periodic=True)
    c = g.gravity_wave_speed(G, H)  # 10
    eta0 = _gaussian(g.x_eta, 200.0, 6.0)
    u0 = np.zeros(g.n_u)
    dt = 0.5 * g.dx / c
    T = 8.0
    nsteps = int(round(T / dt))
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=nsteps)
    eta = out["eta"]
    # 오른쪽 절반(x>200)에서 최대 위치가 200 + c*T 근처
    right = g.x_eta > 200.0
    x_peak = g.x_eta[right][np.argmax(eta[right])]
    assert x_peak == pytest.approx(200.0 + c * T, abs=4.0)


def test_leapfrog_runs_and_conserves_mass():
    # 엇갈린 격자 leapfrog 은 grid-scale 모드 때문에 CFL<=0.5 가 한계이고,
    # Robert-Asselin 필터를 쓰면 여유(CFL~0.4)가 더 필요하다.
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=True)
    eta0 = _gaussian(g.x_eta, 100.0, 8.0)
    u0 = np.zeros(g.n_u)
    dt = 0.4 * g.dx / g.gravity_wave_speed(G, H)
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=600, scheme="leapfrog", ra_nu=0.05)
    assert np.isclose(np.sum(out["eta"]), np.sum(eta0), rtol=1e-8)
    assert np.all(np.isfinite(out["eta"]))


def test_leapfrog_unstable_at_cfl_half():
    """엇갈린 leapfrog+RA 필터는 CFL=0.5(격자 한계)에서 불안정해진다."""
    g = StaggeredGrid1D(nx=200, Lx=200.0, periodic=True)
    eta0 = _gaussian(g.x_eta, 100.0, 8.0)
    u0 = np.zeros(g.n_u)
    dt = 0.5 * g.dx / g.gravity_wave_speed(G, H)
    out = swe1d.run(g, eta0, u0, G, H, dt, nsteps=600, scheme="leapfrog", ra_nu=0.05)
    assert np.max(np.abs(out["eta"])) > 100.0
