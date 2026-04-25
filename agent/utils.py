import fnmatch
from pathlib import Path
from typing import Sequence

from .constants import LOW_VALUE_DIRS, LOW_VALUE_FILENAMES


def _is_ignored(
    path: Path,
    root: Path,
    patterns: Sequence[str],
    *,
    check_low_value: bool = True,
) -> bool:
    rel = str(path.relative_to(root))
    rel_parts = Path(rel).parts
    if check_low_value:
        if path.name.lower() in LOW_VALUE_FILENAMES:
            return True
        if any(part.lower() in LOW_VALUE_DIRS for part in rel_parts):
            return True

    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
        for part in rel_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_repo_relative_path(path: str, label: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if Path(normalized).is_absolute():
        raise ValueError(f"{label} must be a relative path, got {path!r}.")
    if any(part == ".." for part in Path(normalized).parts):
        raise ValueError(
            f"{label} must not contain parent-directory traversal: {path!r}."
        )
    return normalized
