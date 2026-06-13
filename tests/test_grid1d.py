"""1차원 엇갈린 격자 기하/CFL 검증."""

from __future__ import annotations

import numpy as np
import pytest

from shallow_water.grids.grid1d import StaggeredGrid1D


def test_eta_at_cell_centers():
    g = StaggeredGrid1D(nx=10, Lx=1.0)
    assert g.dx == pytest.approx(0.1)
    # 첫 중심은 dx/2, 마지막은 Lx - dx/2
    assert g.x_eta[0] == pytest.approx(0.05)
    assert g.x_eta[-1] == pytest.approx(0.95)
    assert g.x_eta.size == 10


def test_u_points_wall_vs_periodic():
    gw = StaggeredGrid1D(nx=10, Lx=1.0, periodic=False)
    gp = StaggeredGrid1D(nx=10, Lx=1.0, periodic=True)
    assert gw.n_u == 11  # 벽이면 끝 face 포함
    assert gp.n_u == 10  # 주기면 끝이 처음으로 감김
    # u 는 face(정수 격자점)에 위치
    assert gw.x_u[0] == pytest.approx(0.0)
    assert gw.x_u[-1] == pytest.approx(1.0)


def test_gravity_wave_speed_and_cfl():
    g = StaggeredGrid1D(nx=100, Lx=1000.0)
    c = g.gravity_wave_speed(g=10.0, H=10.0)  # sqrt(100)=10
    assert c == pytest.approx(10.0)
    dt = 0.5 * g.dx / c
    assert g.cfl(g=10.0, H=10.0, dt=dt) == pytest.approx(0.5)
    assert g.max_dt(g=10.0, H=10.0, cfl=0.9) == pytest.approx(0.9 * g.dx / c)


def test_invalid_params():
    with pytest.raises(ValueError):
        StaggeredGrid1D(nx=1, Lx=1.0)
    with pytest.raises(ValueError):
        StaggeredGrid1D(nx=10, Lx=-1.0)
