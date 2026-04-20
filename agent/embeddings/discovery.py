from pathlib import Path

from ..config import Config
from ..utils import _is_ignored, _is_within_root


def iter_code_files(root: str | Path, config: Config) -> list[Path]:
    root_path = Path(root).resolve()
    files: list[Path] = []

    for dirpath, dirnames, filenames in root_path.walk():
        current = dirpath
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _is_ignored(current / dirname, root_path, config.ignore_patterns)
            and not (current / dirname).is_symlink()
            and _is_within_root(current / dirname, root_path)
        ]

        for filename in sorted(filenames):
            path = current / filename
            if _is_ignored(path, root_path, config.ignore_patterns):
                continue
            if path.is_symlink() or not _is_within_root(path, root_path):
                continue
            try:
                if path.stat().st_size > config.max_file_size:
                    continue
            except (OSError, PermissionError):
                continue
            files.append(path)

    return files


def _read_file_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (OSError, PermissionError):
        return None
