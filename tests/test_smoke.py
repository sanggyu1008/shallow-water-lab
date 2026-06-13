"""스캐폴딩 smoke 테스트.

Stage 1부터 grids/dynamics/numerics 테스트로 채운다. 지금은 패키지가
정상적으로 import 되고 경로 헬퍼가 동작하는지만 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import shallow_water
from shallow_water.utils.io import find_project_root


def test_package_imports():
    assert hasattr(shallow_water, "__all__")


def test_find_project_root_returns_path():
    root = find_project_root()
    assert isinstance(root, Path)
