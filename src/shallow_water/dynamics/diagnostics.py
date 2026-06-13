"""천수방정식 진단량.

에너지, 상대/위치 소용돌이도, 지형류 속도, 수송 유선함수 등
모델 상태에서 물리적으로 의미 있는 양을 뽑아낸다. 모두 셀 중심 기준.
"""

from __future__ import annotations

import numpy as np

from shallow_water.grids.cgrid import CGrid


def energy_components(
    grid: CGrid,
    eta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    g: float,
    H: float,
) -> dict[str, float]:
    """운동/위치/총 에너지 (선형 천수계, 전체 도메인 적분).

        KE = (1/2) H ∫ (u² + v²) dA,   PE = (1/2) g ∫ η² dA
    """
    u_c = grid.u_at_center(u)
    v_c = grid.v_at_center(v)
    cell = grid.dx * grid.dy
    ke = 0.5 * H * float(np.sum(u_c**2 + v_c**2)) * cell
    pe = 0.5 * g * float(np.sum(eta**2)) * cell
    return {"KE": ke, "PE": pe, "total": ke + pe}


def total_energy(
    grid: CGrid,
    eta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    g: float,
    H: float,
) -> float:
    """총 에너지(KE+PE)."""
    return energy_components(grid, eta, u, v, g, H)["total"]


def relative_vorticity(grid: CGrid, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """상대 소용돌이도 ζ = ∂v/∂x - ∂u/∂y 를 셀 중심에서 계산. shape (ny, nx).

    유속을 셀 중심으로 평균한 뒤 중앙차분(np.gradient)한다.
    """
    u_c = grid.u_at_center(u)
    v_c = grid.v_at_center(v)
    dvdx = np.gradient(v_c, grid.dx, axis=1)
    dudy = np.gradient(u_c, grid.dy, axis=0)
    return dvdx - dudy


def potential_vorticity(
    grid: CGrid,
    eta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    f: float | np.ndarray,
    H: float,
) -> np.ndarray:
    """선형 위치소용돌이도 변칙량 q = ζ - (f/H) η. shape (ny, nx).

    선형 천수방정식에서 ``∂q/∂t = 0`` 이므로 보존되어야 한다.
    ``f`` 는 스칼라 또는 길이 ny 의 y-프로파일.
    """
    zeta = relative_vorticity(grid, u, v)
    f_arr = np.asarray(f, dtype=float)
    if f_arr.ndim == 1:
        f_arr = f_arr.reshape(grid.ny, 1)
    return zeta - (f_arr / H) * eta


def geostrophic_velocity(
    grid: CGrid, eta: np.ndarray, g: float, f: float
) -> tuple[np.ndarray, np.ndarray]:
    """η 로부터 지형류 속도를 셀 중심에서 계산.

        u_g = -(g/f) ∂η/∂y,   v_g = (g/f) ∂η/∂x

    f-평면(스칼라 f)을 가정한다. (u_g, v_g) 각각 shape (ny, nx).
    """
    detady = np.gradient(eta, grid.dy, axis=0)
    detadx = np.gradient(eta, grid.dx, axis=1)
    u_g = -(g / f) * detady
    v_g = (g / f) * detadx
    return u_g, v_g


def transport_streamfunction(grid: CGrid, u: np.ndarray) -> np.ndarray:
    """수평 유선함수 ψ (u = -∂ψ/∂y) 를 셀 중심에서 근사. shape (ny, nx).

    남쪽 벽에서 ψ=0 으로 두고 ``ψ(y) = -∫_0^y u dy'`` 를 누적합으로 계산한다.
    풍성순환의 gyre 구조를 한눈에 보기 위한 시각화용 진단량.
    """
    u_c = grid.u_at_center(u)
    psi = -np.cumsum(u_c, axis=0) * grid.dy
    # 남쪽 벽(ψ=0)을 기준으로 한 칸 내려 정렬
    psi = psi - psi[0:1, :]
    return psi
