"""Shallow Water Lab.

천수방정식(shallow water equations)을 격자에서 적분하는 해양 전진모델 실습 패키지.

- ``grids``    — 1D 엇갈린 격자, 2D Arakawa C-grid
- ``dynamics`` — 1D/2D 선형 천수방정식, 진단량(에너지·소용돌이도·지형류)
- ``numerics`` — 시간적분 드라이버, Robert–Asselin 필터
- ``forcing``  — 바람응력·마찰(Stommel/Munk)
- ``utils``    — 경로/입출력 보조

학습 노트북(Stage 1~6)은 ``notebooks/`` 에, 학습 계획은 ``docs/roadmap.md`` 에 있다.
"""

__all__ = [
    "grids",
    "dynamics",
    "numerics",
    "forcing",
    "utils",
]
