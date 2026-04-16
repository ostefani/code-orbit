import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .constants import (
    CONFIG_HINTS,
    LOW_VALUE_DIRS,
    LOW_VALUE_FILENAMES,
    SOURCE_EXTENSIONS,
    STOPWORDS,
    TEST_HINTS,
)
from .llm import SYSTEM_PROMPT
from .token_counter import count_tokens_int


@dataclass
class FileEntry:
    path: str  # relative path
    content: str
    size: int
    tokens: int  # rendered token count as included in final context


@dataclass(frozen=True)
class FileCandidate:
    path: str
    size_bytes: int


def _is_ignored(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = str(path.relative_to(root))
    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
        for part in path.parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_candidate(root: Path, path: Path) -> FileCandidate | None:
    try:
        if path.is_symlink():
            return None
        resolved = path.resolve()
        resolved.relative_to(root)
        stat_result = resolved.stat()
    except (OSError, PermissionError, ValueError):
        return None

    return FileCandidate(
        path=str(path.relative_to(root)),
        size_bytes=stat_result.st_size,
    )


def _extract_prompt_terms(prompt: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_.-]+", prompt.lower())
    return {word for word in words if len(word) > 1 and word not in STOPWORDS}


def _score_file_entry(entry: FileEntry, prompt: str) -> float:
    return _score_path(entry.path, entry.tokens, prompt)


def _score_path(path: str, estimated_tokens: int, prompt: str) -> float:
    path_obj = Path(path)
    path_lower = path.lower()
    name_lower = path_obj.name.lower()
    suffix = path_obj.suffix.lower()
    parts_lower = {part.lower() for part in path_obj.parts}
    prompt_terms = _extract_prompt_terms(prompt)

    score = 0.0

    # 1. Prefer source files over config/docs/etc.
    score += SOURCE_EXTENSIONS.get(suffix, 0)

    # 2. Strong signal: prompt terms appearing in filename/path.
    for term in prompt_terms:
        if term == name_lower:
            score += 80
        elif term in name_lower:
            score += 40
        elif term in path_lower:
            score += 18

    # 3. Mild preference for shallower files.
    depth = len(path_obj.parts)
    score += max(0, 12 - depth * 2)

    # 4. Penalize low-value generated/lock/cache files.
    if name_lower in LOW_VALUE_FILENAMES:
        score -= 80
    if parts_lower & LOW_VALUE_DIRS:
        score -= 80

    # 5. Tests only get a boost if the prompt suggests tests.
    mentions_tests = bool(prompt_terms & TEST_HINTS)
    is_test_file = bool(parts_lower & TEST_HINTS) or any(
        hint in name_lower for hint in TEST_HINTS
    )
    if is_test_file:
        score += 20 if mentions_tests else -20

    # 6. Config files get a boost only for config/tooling prompts.
    if suffix in {".json", ".yaml", ".yml", ".toml"} and prompt_terms & CONFIG_HINTS:
        score += 22

    # 7. Size-aware penalty: large files are more expensive.
    if estimated_tokens > 12000:
        score -= 50
    elif estimated_tokens > 8000:
        score -= 30
    elif estimated_tokens > 4000:
        score -= 15
    elif estimated_tokens > 2000:
        score -= 6

    return score


def _render_file_block(path: str, content: str) -> str:
    return f'<file path="{path}">\n{content}\n</file>\n'


def _compute_context_budget(prompt: str, config: Config) -> int:
    """
    Compute how many tokens may be spent on file blocks only.

    Subtract:
    - system prompt
    - task wrapper and prompt text
    - outer codebase wrapper
    - reserved response tokens
    - small safety margin
    """
    scaffold = (
        f"{SYSTEM_PROMPT}\n" f"<codebase>\n</codebase>\n" f"<task>\n{prompt}\n</task>"
    )
    base_overhead = count_tokens_int(scaffold, config)

    if config.tokenizer_backend == "estimate":
        # Estimate backend is intentionally conservative.
        safety_margin = max(128, int(config.max_context_tokens * 0.10))
    else:
        # Exact tokenizers need only a small buffer for drift.
        safety_margin = 64

    budget = (
        config.max_context_tokens
        - config.max_response_tokens
        - base_overhead
        - safety_margin
    )
    return max(0, budget)


def build_context(
    root: str, prompt: str, config: Config
) -> tuple[list[FileEntry], str]:
    """
    Walk the directory and collect files into context.
    Returns (file_entries, formatted_context_string).

    Files are ranked by relevance heuristics, then packed into the
    remaining file-token budget.
    """
    root_path = Path(root).resolve()
    candidates: list[FileCandidate] = []

    for dirpath, dirnames, filenames in root_path.walk():
        current = dirpath

        dirnames[:] = [
            d
            for d in dirnames
            if not _is_ignored(current / d, root_path, config.ignore_patterns)
            and not (current / d).is_symlink()
            and _is_within_root(current / d, root_path)
        ]

        for fname in sorted(filenames):
            fpath = current / fname
            if _is_ignored(fpath, root_path, config.ignore_patterns):
                continue
            candidate = _safe_candidate(root_path, fpath)
            if candidate is None:
                continue
            if candidate.size_bytes > config.max_file_size:
                continue
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -_score_path(candidate.path, max(1, candidate.size_bytes // 3), prompt),
            candidate.size_bytes,
            candidate.path,
        )
    )

    token_budget = _compute_context_budget(prompt, config)
    used_tokens = 0
    included: list[FileEntry] = []
    skipped: list[str] = []

    for candidate in candidates:
        path = root_path / candidate.path
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue

        rendered = _render_file_block(candidate.path, content)
        tokens = count_tokens_int(rendered, config)

        if used_tokens + tokens > token_budget:
            skipped.append(candidate.path)
            continue

        included.append(
            FileEntry(
                path=candidate.path,
                content=content,
                size=len(content),
                tokens=tokens,
            )
        )
        used_tokens += tokens

    parts = ["<codebase>"]
    for entry in included:
        parts.append(f'<file path="{entry.path}">')
        parts.append(entry.content)
        parts.append("</file>")
    parts.append("</codebase>")

    context_str = "\n".join(parts)

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} file(s) due to context limit:")
        for s in skipped[:5]:
            print(f"   - {s}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped) - 5} more")

    print(f"\n📂 Context: {len(included)} files | ~{used_tokens:,} file tokens")
    return included, context_str


def get_file_tree(root: str, config: Config) -> str:
    """Return a compact file tree string for display."""
    root_path = Path(root).resolve()
    lines = [f"{root_path.name}/"]

    for dirpath, dirnames, filenames in root_path.walk():
        current = dirpath
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not _is_ignored(current / d, root_path, config.ignore_patterns)
            and not (current / d).is_symlink()
            and _is_within_root(current / d, root_path)
        ]
        depth = len(current.relative_to(root_path).parts)
        indent = "  " * depth
        for fname in sorted(filenames):
            fpath = current / fname
            if (
                not _is_ignored(fpath, root_path, config.ignore_patterns)
                and not fpath.is_symlink()
                and _is_within_root(fpath, root_path)
            ):
                lines.append(f"{indent}{fname}")

    return "\n".join(lines)
