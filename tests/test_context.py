import pytest
from pathlib import Path

from agent.config import Config
from agent.context import _is_ignored, build_context


@pytest.fixture
def temp_codebase(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    return tmp_path


def test_is_ignored() -> None:
    root = Path("/root")
    patterns = [".git", "node_modules", "*.pyc"]

    assert _is_ignored(root / ".git", root, patterns) is True
    assert _is_ignored(root / "src" / "main.py", root, patterns) is False
    assert _is_ignored(root / "node_modules" / "package", root, patterns) is True
    assert _is_ignored(root / "test.pyc", root, patterns) is True


def test_build_context(temp_codebase: Path) -> None:
    config = Config(ignore_patterns=[".git", "node_modules"])
    prompt = "Update the Python source files"

    result = build_context(str(temp_codebase), prompt, config)

    paths = [entry.path for entry in result.entries]
    assert "README.md" in paths
    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert all(".git" not in path for path in paths)
    assert all("node_modules" not in path for path in paths)

    assert "<codebase>" in result.context
    assert '<file path="src/main.py">' in result.context
    assert "print('hello')" in result.context
    assert result.used_tokens > 0


def test_context_token_limit(temp_codebase: Path) -> None:
    (temp_codebase / "large.txt").write_text("word " * 1000, encoding="utf-8")

    config = Config(
        ignore_patterns=[".git", "node_modules"],
        max_context_tokens=2200,
        max_response_tokens=100,
    )
    prompt = "Update the Python source files"

    result = build_context(str(temp_codebase), prompt, config)

    paths = [entry.path for entry in result.entries]
    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert "large.txt" not in paths
    assert "large.txt" in result.skipped_paths


def test_build_context_skips_symlinks(temp_codebase: Path, tmp_path_factory) -> None:
    outside_dir = tmp_path_factory.mktemp("outside")
    outside = outside_dir / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (temp_codebase / "linked.txt").symlink_to(outside)

    config = Config(ignore_patterns=[".git", "node_modules"])
    result = build_context(str(temp_codebase), "Inspect files", config)

    paths = [entry.path for entry in result.entries]
    assert "linked.txt" not in paths
    assert "secret" not in result.context
