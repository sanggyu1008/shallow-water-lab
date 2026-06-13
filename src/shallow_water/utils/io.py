"""프로젝트 경로 탐색과 출력 디렉토리 관리.

노트북에는 ``__file__`` 이 없으므로, 현재 작업 위치에서 위로 올라가며
``pyproject.toml`` 또는 ``.git`` 이 있는 디렉토리를 프로젝트 루트로 본다.
프로젝트 폴더를 다른 경로로 옮겨도 경로가 깨지지 않는다.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(markers: tuple[str, ...] = ("pyproject.toml", ".git")) -> Path:
    """현재 작업 위치에서 위로 올라가며 프로젝트 루트를 찾는다."""
    start = Path.cwd().resolve()
    for path in (start, *start.parents):
        if any((path / marker).exists() for marker in markers):
            return path
    return start.parent if start.name == "notebooks" else start


def get_output_dirs(root: Path | None = None) -> dict[str, Path]:
    """outputs/ 하위 디렉토리들을 만들고 경로 딕셔너리를 반환한다."""
    if root is None:
        root = find_project_root()
    outputs = root / "outputs"
    dirs = {
        "figures": outputs / "figures",
        "fields": outputs / "fields",
        "logs": outputs / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
