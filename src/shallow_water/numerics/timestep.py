"""시간적분 보조 도구.

천수방정식의 시간적분에 쓰는 두 스킴을 정리한다.

**Forward–Backward (FB)**
    유속을 먼저 (옛 $\\eta$ 로) 전진시키고, 갱신된 유속으로 $\\eta$ 를 전진시킨다.
    중력파에 대해 단일 시간층만 쓰면서도 leapfrog 수준으로 정확하고,
    계산모드(computational mode)가 없으며 CFL 한계까지 안정적이다.
    이 프로젝트의 기본 스킴.

**Leapfrog (중앙시간차분)**
    $\\phi^{n+1} = \\phi^{n-1} + 2\\Delta t\\, F(\\phi^n)$.
    2차 정확하지만 두 시간층을 분리시키는 **계산모드** 가 생겨,
    이를 억제하려면 Robert–Asselin 시간필터가 필요하다.

여기서는 스킴에 독립적인 **적분 드라이버** 와 **Robert–Asselin 필터** 만 둔다.
실제 RHS(천수방정식)는 :mod:`shallow_water.dynamics` 에 있다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

State = Sequence[np.ndarray]
StepFn = Callable[[Any], Any]


def integrate(
    step: StepFn,
    state0: State,
    nsteps: int,
    save_every: int | None = None,
) -> tuple[Any, list[int], list[Any]]:
    """상태를 ``step`` 으로 ``nsteps`` 만큼 전진시킨다.

    Parameters
    ----------
    step : callable
        ``state -> state`` 한 스텝 전진 함수.
    state0 : 상태(보통 (eta, u, v) 같은 배열들의 튜플)
        초기 상태. 내부에서 복사해 쓰므로 원본은 보존된다.
    nsteps : int
        전진 스텝 수.
    save_every : int, optional
        None 이면 마지막 상태만, 정수 ``k`` 면 ``k`` 스텝마다(그리고 처음)
        스냅샷을 저장한다.

    Returns
    -------
    state : 최종 상태
    saved_steps : list[int]
        저장된 스냅샷의 스텝 번호.
    snapshots : list[state]
        저장된 상태들(각각 복사본).
    """
    state = tuple(np.array(a, dtype=float, copy=True) for a in state0)
    saved_steps: list[int] = []
    snapshots: list[Any] = []

    def maybe_save(n: int) -> None:
        if save_every is not None and n % save_every == 0:
            saved_steps.append(n)
            snapshots.append(tuple(a.copy() for a in state))

    maybe_save(0)
    for n in range(1, nsteps + 1):
        state = step(state)
        maybe_save(n)
    return state, saved_steps, snapshots


def robert_asselin(
    phi_prev: np.ndarray,
    phi_now: np.ndarray,
    phi_next: np.ndarray,
    nu: float = 0.1,
) -> np.ndarray:
    """Robert–Asselin 시간필터로 leapfrog 계산모드를 억제한다.

        phi_now <- phi_now + nu * (phi_prev - 2 phi_now + phi_next)

    필터링된 ``phi_now`` 를 반환한다(다음 스텝의 'prev'로 쓴다).
    """
    return phi_now + nu * (phi_prev - 2.0 * phi_now + phi_next)
