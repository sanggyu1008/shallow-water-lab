"""바람응력 강제와 마찰.

풍성순환(wind-driven circulation)을 위한 바람응력 τ_x(y) 프로파일과,
Stommel(바닥마찰)·Munk(측면점성) 모형의 마찰 파라미터를 다룬다.

바람응력은 운동량방정식에 ``τ_x / (ρ₀ H)`` 로 들어간다. 회전과 마찰이 더해지면
내부는 Sverdrup 균형, 서쪽 경계는 좁은 경계류(서안경계강화)가 나타난다.
"""

from __future__ import annotations

import numpy as np


def single_gyre_wind(y: np.ndarray, Ly: float, tau0: float = 0.1) -> np.ndarray:
    """단일 gyre 바람응력 τ_x(y) = -τ0 cos(π y / Ly).

    응력의 회전(curl)이 도메인 내내 한 부호라 단일 순환을 만든다.
    (북반구 f>0 이면 시계방향 subtropical gyre.)
    """
    return -tau0 * np.cos(np.pi * y / Ly)


def double_gyre_wind(y: np.ndarray, Ly: float, tau0: float = 0.1) -> np.ndarray:
    """이중 gyre 바람응력 τ_x(y) = -τ0 cos(2π y / Ly).

    응력 회전의 부호가 중앙에서 바뀌어 위/아래로 반대 방향의 두 gyre
    (subpolar + subtropical)를 만든다. double-gyre capstone의 강제.
    """
    return -tau0 * np.cos(2.0 * np.pi * y / Ly)


def wind_stress_curl(y: np.ndarray, Ly: float, tau0: float, kind: str) -> np.ndarray:
    """바람응력의 연직성분 회전 ∂τ_x/∂y 의 부호 구조 (해석적).

    Sverdrup 균형 ``β V = (1/ρ₀) curl(τ)`` 의 우변을 보기 위한 보조 함수.
    여기서는 -∂τ_x/∂y 를 반환한다(2차원 curl_z = ∂τ_y/∂x - ∂τ_x/∂y, τ_y=0).
    """
    if kind == "single":
        # τ_x = -τ0 cos(πy/Ly) -> -dτ_x/dy = -τ0 (π/Ly) sin(πy/Ly)
        return -tau0 * (np.pi / Ly) * np.sin(np.pi * y / Ly)
    if kind == "double":
        return -tau0 * (2.0 * np.pi / Ly) * np.sin(2.0 * np.pi * y / Ly)
    raise ValueError(f"kind 는 'single' 또는 'double'. 받은 값: {kind!r}")


def stommel_boundary_layer(r: float, beta: float) -> float:
    """Stommel(바닥마찰) 서안경계층 두께 δ_S = r / β."""
    return r / beta


def munk_boundary_layer(A_h: float, beta: float) -> float:
    """Munk(측면점성) 서안경계층 두께 δ_M = (A_h / β)^(1/3)."""
    return (A_h / beta) ** (1.0 / 3.0)
