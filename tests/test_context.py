import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from agent.events import EventBus
from agent.context import build_context_async
from agent.utils import _is_ignored


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
    assert _is_ignored(root / "package-lock.json", root, patterns) is True
    assert _is_ignored(root / "__pycache__", root, patterns) is True


def test_is_ignored_uses_relative_path_for_low_value_dirs() -> None:
    root = Path("/home/user/build/myproject")
    patterns: list[str] = []

    assert _is_ignored(root / "src" / "main.py", root, patterns) is False


def test_is_ignored_matches_low_value_dirs_case_insensitively() -> None:
    root = Path("/root")
    patterns: list[str] = []

    assert _is_ignored(root / "Node_Modules" / "pkg", root, patterns) is True


def test_is_ignored_pattern_uses_relative_path() -> None:
    root = Path("/home/user/build/myproject")
    patterns = ["build"]

    assert _is_ignored(root / "src" / "main.py", root, patterns) is False


def test_build_context(temp_codebase: Path) -> None:
    config = Config(ignore_patterns=[".git", "node_modules"])
    prompt = "Update the Python source files"

    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            prompt,
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    paths = [entry.path for entry in result.entries]
    assert "README.md" in paths
    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert all(".git" not in path for path in paths)
    assert all("node_modules" not in path for path in paths)

    assert "<codebase>" in result.context
    assert "<file_tree>" in result.context
    assert "src/" in result.context
    assert '<file path="src/main.py">' in result.context
    assert "print('hello')" in result.context
    assert result.used_tokens > 0


def test_build_context_includes_empty_dirs_in_file_tree(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()

    result = asyncio.run(
        build_context_async(
            str(tmp_path),
            "Build a React ButtonComponent in src",
            Config(),
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    assert "<file_tree>" in result.context
    assert "src/" in result.context
    assert '<file path="src/' not in result.context


def test_context_token_limit(temp_codebase: Path) -> None:
    (temp_codebase / "large.txt").write_text("word " * 1000, encoding="utf-8")

    config = Config(
        ignore_patterns=[".git", "node_modules"],
        max_context_tokens=2200,
        max_response_tokens=100,
    )
    prompt = "Update the Python source files"

    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            prompt,
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    paths = [entry.path for entry in result.entries]
    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert "large.txt" not in paths
    assert "large.txt" in result.skipped_paths
    assert "<file_tree>" in result.context
    assert "large.txt" in result.context


def test_build_context_reports_budget_breakdown_and_zero_budget_warning(
    temp_codebase: Path,
) -> None:
    config = Config(
        ignore_patterns=[".git", "node_modules"],
        max_context_tokens=256,
        max_response_tokens=128,
    )

    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            "Update the Python source files",
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    assert result.budget_breakdown is not None
    assert result.budget_breakdown.context_window_tokens == 256
    assert result.budget_breakdown.response_reserve_tokens == 128
    assert result.token_budget == result.budget_breakdown.file_budget_tokens
    assert result.token_budget == 0
    assert any(
        "No file context budget remains" in warning
        or "leaves no room for file context" in warning
        for warning in result.token_warnings
    )


def test_build_context_skips_symlinks(temp_codebase: Path, tmp_path_factory) -> None:
    outside_dir = tmp_path_factory.mktemp("outside")
    outside = outside_dir / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (temp_codebase / "linked.txt").symlink_to(outside)

    config = Config(ignore_patterns=[".git", "node_modules"])
    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            "Inspect files",
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    paths = [entry.path for entry in result.entries]
    assert "linked.txt" not in paths
    assert "secret" not in result.context


class FakeSemanticEmbeddingClient:
    async def validate(self) -> None:
        return None

    async def probe(self) -> None:
        return None

    async def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "rate limiting" in lower:
                vectors.append((1.0, 0.0))
            elif "authentication" in lower or "middleware" in lower:
                vectors.append((0.99, 0.01))
            elif "rate limiter" in lower or "rate" in lower:
                vectors.append((0.05, 0.95))
            else:
                vectors.append((0.0, 1.0))
        return vectors


def test_build_context_uses_semantic_ranking(temp_codebase: Path) -> None:
    (temp_codebase / "src" / "auth").mkdir(parents=True, exist_ok=True)
    (temp_codebase / "src" / "auth" / "middleware.py").write_text(
        "def apply_authentication(request):\n    return request\n",
        encoding="utf-8",
    )
    (temp_codebase / "src" / "rate_limiter.py").write_text(
        "def helper():\n    return 'unrelated'\n",
        encoding="utf-8",
    )

    config = Config(
        ignore_patterns=[".git", "node_modules"],
        max_context_tokens=6000,
        max_response_tokens=100,
    )
    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            "Implement rate limiting",
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
    )

    assert result.entries[0].path == "src/auth/middleware.py"
    assert result.semantic_matches
    assert result.semantic_matches[0].path == "src/auth/middleware.py"


def test_build_context_async_runs_inside_event_loop(temp_codebase: Path) -> None:
    (temp_codebase / "src" / "auth").mkdir(parents=True, exist_ok=True)
    (temp_codebase / "src" / "auth" / "middleware.py").write_text(
        "def apply_authentication(request):\n    return request\n",
        encoding="utf-8",
    )

    config = Config(ignore_patterns=[".git", "node_modules"])

    async def runner() -> None:
        result = await build_context_async(
            str(temp_codebase),
            "Implement rate limiting",
            config,
            embedding_client=FakeSemanticEmbeddingClient(),
        )
        assert result.entries
        assert result.semantic_matches

    asyncio.run(runner())


def test_build_context_emits_warning_on_semantic_failure(
    temp_codebase: Path,
) -> None:
    (temp_codebase / "src" / "auth").mkdir(parents=True, exist_ok=True)
    (temp_codebase / "src" / "auth" / "middleware.py").write_text(
        "def apply_authentication(request):\n    return request\n",
        encoding="utf-8",
    )

    class FailingEmbeddingClient:
        async def validate(self) -> None:
            return None

        async def probe(self) -> None:
            return None

        async def embed(self, texts):
            raise RuntimeError("boom")

    bus = EventBus()
    events: list[object] = []
    bus.subscribe(events.append)

    result = asyncio.run(
        build_context_async(
            str(temp_codebase),
            "Implement rate limiting",
            Config(ignore_patterns=[".git", "node_modules"]),
            embedding_client=FailingEmbeddingClient(),
            event_bus=bus,
        )
    )

    assert result.entries
    assert any(getattr(event, "name", "") == "context.warning" for event in events)
