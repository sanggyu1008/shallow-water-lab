"""1차원 엇갈린 격자 (staggered grid).

천수방정식을 1차원에서 풀 때, 자유표면 변위 $\\eta$ 와 유속 $u$ 를
**반 칸 어긋나게** 배치한다.

    eta:  o     o     o     o        (셀 중심, nx 개)
    u:    |  x  |  x  |  x  |  x  |   (셀 face, 벽이면 nx+1 개)
          0                       Lx

이렇게 엇갈리게 두면

- 압력경도 $\\partial\\eta/\\partial x$ 가 두 $\\eta$ 사이의 face(=$u$ 위치)에서
  중앙차분 한 번으로 자연스럽게 계산되고,
- 발산 $\\partial u/\\partial x$ 가 두 $u$ 사이의 셀 중심(=$\\eta$ 위치)에서 계산된다.

격자점이 겹치지 않으므로 비엇갈린(collocated) 격자에서 나타나는
**격자 잡음(grid-scale 2Δx noise)** 이 억제된다.

경계조건:

- ``periodic=True``  : $u$ 도 $\\eta$ 처럼 nx 개, 끝이 처음으로 감긴다.
- ``periodic=False`` : 양 끝이 벽(wall). $u$ 는 nx+1 개이고 양 끝 face에서 $u=0$.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StaggeredGrid1D:
    """1차원 엇갈린 균일 격자.

    Parameters
    ----------
    nx : int
        셀(=$\\eta$ 격자점) 개수.
    Lx : float
        도메인 길이. 격자간격은 ``dx = Lx / nx``.
    periodic : bool
        주기경계 여부. False 이면 양 끝이 벽.
    """

    nx: int
    Lx: float
    periodic: bool = False

    dx: float = field(init=False)
    x_eta: np.ndarray = field(init=False)
    x_u: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.nx < 2:
            raise ValueError(f"nx는 2 이상이어야 한다. 현재 nx: {self.nx}")
        if self.Lx <= 0:
            raise ValueError(f"Lx는 양수여야 한다. 현재 Lx: {self.Lx}")
        self.dx = self.Lx / self.nx
        # eta 는 셀 중심
        self.x_eta = (np.arange(self.nx) + 0.5) * self.dx
        # u 는 face. 주기면 nx 개(i*dx), 벽이면 nx+1 개(끝 포함)
        n_u = self.nx if self.periodic else self.nx + 1
        self.x_u = np.arange(n_u) * self.dx

    @property
    def n_u(self) -> int:
        """u 격자점 개수."""
        return self.nx if self.periodic else self.nx + 1

    def gravity_wave_speed(self, g: float, H: float) -> float:
        """선형 중력파 속도 c = sqrt(gH)."""
        return float(np.sqrt(g * H))

    def cfl(self, g: float, H: float, dt: float) -> float:
        """CFL 수 c·dt/dx (안정 조건은 <= 1)."""
        return self.gravity_wave_speed(g, H) * dt / self.dx

    def max_dt(self, g: float, H: float, cfl: float = 0.9) -> float:
        """주어진 CFL 한계에 대응하는 최대 dt."""
        return cfl * self.dx / self.gravity_wave_speed(g, H)
