"""2차원 선형 천수방정식 (Arakawa C-grid).

지배방정식 (회전·바람강제·마찰 포함):

    ∂u/∂t - f v = -g ∂η/∂x + τ_x/(ρ₀H) + (마찰)_u
    ∂v/∂t + f u = -g ∂η/∂y                 + (마찰)_v
    ∂η/∂t + H (∂u/∂x + ∂v/∂y) = 0

- 코리올리 파라미터는 β-평면: ``f(y) = f0 + β (y - y0)``.
- 마찰은 두 종류를 지원한다.
  * 선형 바닥마찰(Stommel):  -r u,  -r v
  * 측면 점성(Munk):         A_h ∇²u,  A_h ∇²v
- 바람응력 τ_x 는 u-점에서 주어진다(보통 y의 함수).

시간적분
--------
중력파 항은 **forward–backward**: 유속을 옛 η 로 갱신한 뒤, 갱신된 유속으로 η 를 갱신.
코리올리 항은 **심플렉틱(symplectic) 순서**로 처리한다 — u 는 옛 v 로, v 는 *갱신된* u 로
갱신한다. 이 순서는 관성진동(inertial oscillation)의 진폭을 보존(사상의 행렬식 = 1)하고
``f Δt < 2`` 에서 안정적이라, 회전계를 오래 적분해도 에너지가 새지 않는다.

안정조건은 2차원 중력파 CFL ``c Δt √(1/Δx² + 1/Δy²) ≤ 1`` (c=√(gH)) 와
``f Δt < 2``, 점성 ``A_h Δt/Δx² < 1/4`` 를 함께 만족해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shallow_water.grids.cgrid import CGrid


@dataclass
class ShallowWater2D:
    """C-grid 위의 2차원 선형 천수방정식 모델.

    Parameters
    ----------
    grid : CGrid
        격자.
    g : float
        중력가속도(축소중력 g' 로 써도 된다).
    H : float
        평균 수심.
    f0 : float
        기준 코리올리 파라미터.
    beta : float
        ``df/dy`` (β-평면). 0 이면 f-평면.
    y0 : float, optional
        f0 를 정의하는 기준 위도(=y). 기본은 도메인 중앙.
    r : float
        선형 바닥마찰 계수(Stommel). 0 이면 끔.
    A_h : float
        측면 점성계수(Munk). 0 이면 끔.
    tau_x : float | np.ndarray | None
        바람응력 τ_x. 스칼라, 길이 ny 의 y-프로파일, 또는 (ny, nx+1) 배열.
    rho0 : float
        기준 밀도(바람응력 정규화용).
    """

    grid: CGrid
    g: float = 9.81
    H: float = 1.0
    f0: float = 0.0
    beta: float = 0.0
    y0: float | None = None
    r: float = 0.0
    A_h: float = 0.0
    tau_x: object = None
    rho0: float = 1025.0

    f_u: np.ndarray = field(init=False)
    f_v: np.ndarray = field(init=False)
    tau_x_u: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        gr = self.grid
        y0 = 0.5 * gr.Ly if self.y0 is None else self.y0
        # f 를 u-점(y=y_eta)과 v-점(y=y_v)에서 평가, 열로 브로드캐스트
        self.f_u = (self.f0 + self.beta * (gr.y_eta - y0)).reshape(gr.ny, 1)
        self.f_v = (self.f0 + self.beta * (gr.y_v - y0)).reshape(gr.ny + 1, 1)
        # 바람응력 → u-점 (ny, nx+1)
        if self.tau_x is None:
            self.tau_x_u = None
        else:
            arr = np.asarray(self.tau_x, dtype=float)
            if arr.ndim == 0:
                self.tau_x_u = np.full((gr.ny, gr.nx + 1), float(arr))
            elif arr.ndim == 1 and arr.size == gr.ny:
                self.tau_x_u = np.broadcast_to(
                    arr.reshape(gr.ny, 1), (gr.ny, gr.nx + 1)
                ).copy()
            elif arr.shape == (gr.ny, gr.nx + 1):
                self.tau_x_u = arr.copy()
            else:
                raise ValueError(
                    "tau_x 는 스칼라, 길이 ny 배열, 또는 (ny, nx+1) 배열이어야 한다. "
                    f"받은 shape: {arr.shape}"
                )

    # ------------------------------------------------------------------
    def gravity_wave_speed(self) -> float:
        return float(np.sqrt(self.g * self.H))

    def cfl(self, dt: float) -> float:
        """2차원 중력파 CFL 수 (<=1 권장)."""
        c = self.gravity_wave_speed()
        return c * dt * np.sqrt(1.0 / self.grid.dx**2 + 1.0 / self.grid.dy**2)

    def deformation_radius(self) -> float:
        """Rossby 변형반지름 L_d = sqrt(gH)/|f0| (f0=0 이면 inf)."""
        if self.f0 == 0:
            return np.inf
        return self.gravity_wave_speed() / abs(self.f0)

    # ------------------------------------------------------------------
    def step(
        self,
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """한 스텝 전진 (eta, u, v) -> (eta, u, v)."""
        gr = self.grid
        eta, u, v = state

        # --- u 갱신 (옛 η, 옛 v 로 코리올리) ---
        du = self.f_u * gr.v_at_u(v) - self.g * gr.deta_dx(eta)
        if self.tau_x_u is not None:
            du = du + self.tau_x_u / (self.rho0 * self.H)
        if self.r:
            du = du - self.r * u
        if self.A_h:
            du = du + self.A_h * gr.laplacian_u(u)
        u_new = u + dt * du
        if not gr.periodic_x:
            u_new[:, 0] = 0.0
            u_new[:, -1] = 0.0

        # --- v 갱신 (옛 η, 갱신된 u 로 코리올리: symplectic) ---
        dv = -self.f_v * gr.u_at_v(u_new) - self.g * gr.deta_dy(eta)
        if self.r:
            dv = dv - self.r * v
        if self.A_h:
            dv = dv + self.A_h * gr.laplacian_v(v)
        v_new = v + dt * dv
        if not gr.periodic_y:
            v_new[0, :] = 0.0
            v_new[-1, :] = 0.0

        # --- η 갱신 (forward–backward: 갱신된 u, v) ---
        eta_new = eta - self.H * dt * gr.divergence(u_new, v_new)
        return eta_new, u_new, v_new

    # ------------------------------------------------------------------
    def run(
        self,
        eta0: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        dt: float,
        nsteps: int,
        save_every: int | None = None,
        track_energy: bool = True,
    ) -> dict[str, object]:
        """모델을 nsteps 만큼 적분한다.

        Returns
        -------
        dict: ``eta``, ``u``, ``v`` (최종), ``times``;
        track_energy 면 ``energy`` (총에너지 시계열);
        save_every 면 ``eta_hist``, ``u_hist``, ``v_hist``, ``save_times``.
        """
        from shallow_water.dynamics.diagnostics import total_energy

        eta = np.asarray(eta0, dtype=float).copy()
        u = np.asarray(u0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()

        energies: list[float] = []
        eta_hist, u_hist, v_hist, save_steps = [], [], [], []

        def record(n: int) -> None:
            if track_energy:
                energies.append(total_energy(self.grid, eta, u, v, self.g, self.H))
            if save_every is not None and n % save_every == 0:
                eta_hist.append(eta.copy())
                u_hist.append(u.copy())
                v_hist.append(v.copy())
                save_steps.append(n)

        record(0)
        for n in range(1, nsteps + 1):
            eta, u, v = self.step((eta, u, v), dt)
            record(n)

        result: dict[str, object] = {
            "eta": eta,
            "u": u,
            "v": v,
            "times": np.arange(nsteps + 1) * dt,
        }
        if track_energy:
            result["energy"] = np.asarray(energies)
        if save_every is not None:
            result["eta_hist"] = np.asarray(eta_hist)
            result["u_hist"] = np.asarray(u_hist)
            result["v_hist"] = np.asarray(v_hist)
            result["save_times"] = np.asarray(save_steps) * dt
        return result
