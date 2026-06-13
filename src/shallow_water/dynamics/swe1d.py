"""1차원 선형 천수방정식 (회전 없음).

지배방정식:

    ∂u/∂t = -g ∂η/∂x
    ∂η/∂t = -H ∂u/∂x

여기서 $u$ 는 유속, $\\eta$ 는 자유표면 변위, $H$ 는 평균 수심, $g$ 는 중력가속도.
중력파 속도는 $c = \\sqrt{gH}$, CFL 조건은 $c\\,\\Delta t/\\Delta x \\le 1$.

격자는 :class:`~shallow_water.grids.grid1d.StaggeredGrid1D` (엇갈린 격자).
시간적분은 **forward–backward** 를 기본으로 하고, 비교용 **leapfrog** 도 제공한다.
"""

from __future__ import annotations

import numpy as np

from shallow_water.grids.grid1d import StaggeredGrid1D


def _deta_dx(grid: StaggeredGrid1D, eta: np.ndarray) -> np.ndarray:
    """eta 의 x-경도를 u 위치에서 계산."""
    if grid.periodic:
        # u_i 는 eta_{i-1} 와 eta_i 사이 face
        return (eta - np.roll(eta, 1)) / grid.dx
    out = np.zeros(grid.n_u)
    out[1:-1] = (eta[1:] - eta[:-1]) / grid.dx  # 내부 face
    return out  # 끝 face(벽)는 0


def _dudx(grid: StaggeredGrid1D, u: np.ndarray) -> np.ndarray:
    """u 의 x-발산을 eta 위치(셀 중심)에서 계산."""
    if grid.periodic:
        return (np.roll(u, -1) - u) / grid.dx
    return (u[1:] - u[:-1]) / grid.dx


def step_forward_backward(
    grid: StaggeredGrid1D,
    eta: np.ndarray,
    u: np.ndarray,
    g: float,
    H: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward–backward 한 스텝.

    1. 옛 η 로 u 를 전진:  u^{n+1} = u^n - g dt ∂η^n/∂x
    2. 새 u 로 η 를 전진:  η^{n+1} = η^n - H dt ∂u^{n+1}/∂x
    """
    u_new = u - g * dt * _deta_dx(grid, eta)
    if not grid.periodic:
        u_new[0] = 0.0
        u_new[-1] = 0.0
    eta_new = eta - H * dt * _dudx(grid, u_new)
    return eta_new, u_new


def step_leapfrog(
    grid: StaggeredGrid1D,
    eta_prev: np.ndarray,
    u_prev: np.ndarray,
    eta: np.ndarray,
    u: np.ndarray,
    g: float,
    H: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Leapfrog(중앙시간차분) 한 스텝. n-1, n 으로 n+1 을 만든다."""
    u_new = u_prev - 2.0 * g * dt * _deta_dx(grid, eta)
    if not grid.periodic:
        u_new[0] = 0.0
        u_new[-1] = 0.0
    eta_new = eta_prev - 2.0 * H * dt * _dudx(grid, u)
    return eta_new, u_new


def energy(
    grid: StaggeredGrid1D,
    eta: np.ndarray,
    u: np.ndarray,
    g: float,
    H: float,
) -> dict[str, float]:
    """선형 천수계의 운동/위치/총 에너지(단위 폭당).

        KE = (1/2) H ∫ u² dx,   PE = (1/2) g ∫ η² dx
    """
    ke = 0.5 * H * float(np.sum(u**2)) * grid.dx
    pe = 0.5 * g * float(np.sum(eta**2)) * grid.dx
    return {"KE": ke, "PE": pe, "total": ke + pe}


def run(
    grid: StaggeredGrid1D,
    eta0: np.ndarray,
    u0: np.ndarray,
    g: float,
    H: float,
    dt: float,
    nsteps: int,
    scheme: str = "forward_backward",
    save_every: int | None = None,
    ra_nu: float = 0.1,
) -> dict[str, object]:
    """1차원 천수방정식을 nsteps 만큼 적분한다.

    Parameters
    ----------
    scheme : {"forward_backward", "leapfrog"}
        leapfrog 는 첫 스텝을 forward–backward 로 시동하고
        Robert–Asselin 필터(``ra_nu``)로 계산모드를 억제한다.
    save_every : int, optional
        스냅샷 저장 간격. None 이면 마지막만.

    Returns
    -------
    dict with keys: ``eta``, ``u`` (최종), ``times``, ``energy``,
    그리고 save_every 가 있으면 ``eta_hist``, ``u_hist``, ``save_times``.
    """
    eta = np.asarray(eta0, dtype=float).copy()
    u = np.asarray(u0, dtype=float).copy()

    energies = [energy(grid, eta, u, g, H)["total"]]
    eta_hist, u_hist, save_steps = [], [], []

    def maybe_save(n: int) -> None:
        if save_every is not None and n % save_every == 0:
            eta_hist.append(eta.copy())
            u_hist.append(u.copy())
            save_steps.append(n)

    maybe_save(0)

    if scheme == "forward_backward":
        for n in range(1, nsteps + 1):
            eta, u = step_forward_backward(grid, eta, u, g, H, dt)
            energies.append(energy(grid, eta, u, g, H)["total"])
            maybe_save(n)
    elif scheme == "leapfrog":
        from shallow_water.numerics.timestep import robert_asselin

        # 시동: 첫 스텝은 forward–backward
        eta_prev, u_prev = eta.copy(), u.copy()
        eta, u = step_forward_backward(grid, eta, u, g, H, dt)
        energies.append(energy(grid, eta, u, g, H)["total"])
        maybe_save(1)
        for n in range(2, nsteps + 1):
            eta_next, u_next = step_leapfrog(
                grid, eta_prev, u_prev, eta, u, g, H, dt
            )
            # Robert–Asselin 필터
            eta = robert_asselin(eta_prev, eta, eta_next, ra_nu)
            u = robert_asselin(u_prev, u, u_next, ra_nu)
            eta_prev, u_prev = eta, u
            eta, u = eta_next, u_next
            energies.append(energy(grid, eta, u, g, H)["total"])
            maybe_save(n)
    else:
        raise ValueError(f"알 수 없는 scheme: {scheme!r}")

    result: dict[str, object] = {
        "eta": eta,
        "u": u,
        "times": np.arange(nsteps + 1) * dt,
        "energy": np.asarray(energies),
    }
    if save_every is not None:
        result["eta_hist"] = np.asarray(eta_hist)
        result["u_hist"] = np.asarray(u_hist)
        result["save_times"] = np.asarray(save_steps) * dt
    return result
