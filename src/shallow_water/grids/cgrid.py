"""2차원 Arakawa C-grid.

천수방정식의 표준 격자. 변수들을 한 셀 안에서 다음처럼 엇갈리게 배치한다.

    v(j+1) ──┐
             │
    u(i) ─ eta(i,j) ─ u(i+1)
             │
    v(j) ────┘

- ``eta`` (자유표면): 셀 중심,            shape ``(ny, nx)``
- ``u``   (동서 유속): 동서 face(x-face),  shape ``(ny, nx+1)``
- ``v``   (남북 유속): 남북 face(y-face),  shape ``(ny+1, nx)``

좌표:

    x_eta = (i + 0.5) dx,  i = 0..nx-1      y_eta = (j + 0.5) dy,  j = 0..ny-1
    x_u   = i dx,          i = 0..nx        y_u   = (j + 0.5) dy
    x_v   = (i + 0.5) dx                    y_v   = j dy,          j = 0..ny

C-grid 를 쓰는 이유: 압력경도와 발산이 한 번의 인접차분으로 깔끔히 계산되고,
중력파 분산관계가 격자에서 가장 정확하다(특히 변형반지름이 격자로 해상될 때).

이 모듈은 **격자 기하 + 엇갈린 보간/차분 연산자** 만 제공한다.
실제 시간적분(천수방정식 RHS)은 :mod:`shallow_water.dynamics.swe2d` 에 있다.
경계조건(`periodic_x`, `periodic_y`)에 따라 차분/보간이 벽 또는 주기로 처리된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CGrid:
    """2차원 Arakawa C-grid (균일 간격).

    Parameters
    ----------
    nx, ny : int
        x, y 방향 셀 개수.
    Lx, Ly : float
        도메인 크기. ``dx = Lx/nx``, ``dy = Ly/ny``.
    periodic_x, periodic_y : bool
        각 방향 주기경계 여부. False 이면 그 방향 양 끝이 벽.
    """

    nx: int
    ny: int
    Lx: float
    Ly: float
    periodic_x: bool = False
    periodic_y: bool = False

    dx: float = field(init=False)
    dy: float = field(init=False)
    x_eta: np.ndarray = field(init=False)
    y_eta: np.ndarray = field(init=False)
    x_u: np.ndarray = field(init=False)
    y_v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx, ny는 2 이상이어야 한다.")
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx, Ly는 양수여야 한다.")
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x_eta = (np.arange(self.nx) + 0.5) * self.dx
        self.y_eta = (np.arange(self.ny) + 0.5) * self.dy
        self.x_u = np.arange(self.nx + 1) * self.dx
        self.y_v = np.arange(self.ny + 1) * self.dy

    # ------------------------------------------------------------------
    # 빈 상태 / 좌표
    # ------------------------------------------------------------------
    def zeros(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(eta, u, v) 를 0으로 초기화해 올바른 shape으로 반환."""
        eta = np.zeros((self.ny, self.nx))
        u = np.zeros((self.ny, self.nx + 1))
        v = np.zeros((self.ny + 1, self.nx))
        return eta, u, v

    def meshgrid_eta(self) -> tuple[np.ndarray, np.ndarray]:
        """eta 점의 (X, Y) 메쉬 (shape (ny, nx))."""
        return np.meshgrid(self.x_eta, self.y_eta)

    # ------------------------------------------------------------------
    # 차분 연산자
    # ------------------------------------------------------------------
    def deta_dx(self, eta: np.ndarray) -> np.ndarray:
        """eta의 x-경도를 u-face 에서 계산. 반환 shape (ny, nx+1)."""
        out = np.zeros((self.ny, self.nx + 1))
        out[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / self.dx
        if self.periodic_x:
            wrap = (eta[:, 0] - eta[:, -1]) / self.dx
            out[:, 0] = wrap
            out[:, -1] = wrap
        # 벽이면 끝 face 의 경도는 의미 없음(그 face의 u=0). 0으로 둔다.
        return out

    def deta_dy(self, eta: np.ndarray) -> np.ndarray:
        """eta의 y-경도를 v-face 에서 계산. 반환 shape (ny+1, nx)."""
        out = np.zeros((self.ny + 1, self.nx))
        out[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dy
        if self.periodic_y:
            wrap = (eta[0, :] - eta[-1, :]) / self.dy
            out[0, :] = wrap
            out[-1, :] = wrap
        return out

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """발산 ∂u/∂x + ∂v/∂y 을 eta 점에서 계산. 반환 shape (ny, nx)."""
        dudx = (u[:, 1:] - u[:, :-1]) / self.dx
        dvdy = (v[1:, :] - v[:-1, :]) / self.dy
        return dudx + dvdy

    # ------------------------------------------------------------------
    # 엇갈린 보간 (코리올리 항용)
    # ------------------------------------------------------------------
    def v_at_u(self, v: np.ndarray) -> np.ndarray:
        """v 를 u-face 위치로 보간. 반환 shape (ny, nx+1).

        먼저 y로 셀중심에 평균한 뒤(v_c, shape (ny, nx)), x로 face에 평균한다.
        """
        v_c = 0.5 * (v[:-1, :] + v[1:, :])  # (ny, nx) at centers
        out = np.zeros((self.ny, self.nx + 1))
        out[:, 1:-1] = 0.5 * (v_c[:, :-1] + v_c[:, 1:])
        if self.periodic_x:
            wrap = 0.5 * (v_c[:, -1] + v_c[:, 0])
            out[:, 0] = wrap
            out[:, -1] = wrap
        else:
            out[:, 0] = v_c[:, 0]
            out[:, -1] = v_c[:, -1]
        return out

    def u_at_v(self, u: np.ndarray) -> np.ndarray:
        """u 를 v-face 위치로 보간. 반환 shape (ny+1, nx)."""
        u_c = 0.5 * (u[:, :-1] + u[:, 1:])  # (ny, nx) at centers
        out = np.zeros((self.ny + 1, self.nx))
        out[1:-1, :] = 0.5 * (u_c[:-1, :] + u_c[1:, :])
        if self.periodic_y:
            wrap = 0.5 * (u_c[-1, :] + u_c[0, :])
            out[0, :] = wrap
            out[-1, :] = wrap
        else:
            out[0, :] = u_c[0, :]
            out[-1, :] = u_c[-1, :]
        return out

    def u_at_center(self, u: np.ndarray) -> np.ndarray:
        """u 를 셀 중심으로 평균. 반환 shape (ny, nx)."""
        return 0.5 * (u[:, :-1] + u[:, 1:])

    def v_at_center(self, v: np.ndarray) -> np.ndarray:
        """v 를 셀 중심으로 평균. 반환 shape (ny, nx)."""
        return 0.5 * (v[:-1, :] + v[1:, :])

    # ------------------------------------------------------------------
    # 라플라시안 (측면 마찰 = Munk 점성)
    # ------------------------------------------------------------------
    def laplacian_u(self, u: np.ndarray) -> np.ndarray:
        """u 의 라플라시안. 벽에서는 no-slip(u=0), 주기면 wrap."""
        return _laplacian(u, self.dx, self.dy, self.periodic_x, self.periodic_y)

    def laplacian_v(self, v: np.ndarray) -> np.ndarray:
        """v 의 라플라시안. 벽에서는 no-slip(v=0), 주기면 wrap."""
        return _laplacian(v, self.dx, self.dy, self.periodic_x, self.periodic_y)


def _laplacian(
    a: np.ndarray, dx: float, dy: float, periodic_x: bool, periodic_y: bool
) -> np.ndarray:
    """ghost 셀(주기=wrap, 벽=0)로 채운 5점 라플라시안."""
    mode_x = "wrap" if periodic_x else "constant"
    mode_y = "wrap" if periodic_y else "constant"
    ap = np.pad(a, ((1, 1), (0, 0)), mode=mode_y)
    ap = np.pad(ap, ((0, 0), (1, 1)), mode=mode_x)
    d2x = (ap[1:-1, 2:] - 2 * ap[1:-1, 1:-1] + ap[1:-1, :-2]) / dx**2
    d2y = (ap[2:, 1:-1] - 2 * ap[1:-1, 1:-1] + ap[:-2, 1:-1]) / dy**2
    return d2x + d2y
