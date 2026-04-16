import fnmatch
import re
from pathlib import Path
from dataclasses import dataclass

from .config import Config
from .token_counter import count_tokens_int


@dataclass
class FileEntry:
    path: str  # relative path
    content: str
    size: int
    tokens: int  # exact or estimated token count


from .constants import (
    SOURCE_EXTENSIONS,
    LOW_VALUE_FILENAMES,
    LOW_VALUE_DIRS,
    TEST_HINTS,
    CONFIG_HINTS,
    STOPWORDS,
)

TEST_HINTS: set[str] = {"test", "tests", "spec", "__tests__"}


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


def _extract_prompt_terms(prompt: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_.-]+", prompt.lower())
    return {word for word in words if len(word) > 1 and word not in STOPWORDS}


def _score_file_entry(entry: FileEntry, prompt: str) -> float:
    path_obj = Path(entry.path)
    path_lower = entry.path.lower()
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
    if entry.tokens > 12000:
        score -= 50
    elif entry.tokens > 8000:
        score -= 30
    elif entry.tokens > 4000:
        score -= 15
    elif entry.tokens > 2000:
        score -= 6

    return score


def build_context(
    root: str, prompt: str, config: Config
) -> tuple[list[FileEntry], str]:
    """
    Walk the directory and collect files into context.
    Returns (file_entries, formatted_context_string).

    Files are ranked by lightweight relevance heuristics:
    - prompt-term path/name matches
    - file extension priority
    - shallow directory preference
    - size-aware penalties

    Context is then packed within the token budget.
    """
    root_path = Path(root).resolve()
    entries: list[FileEntry] = []

    for dirpath, dirnames, filenames in root_path.walk():
        current = dirpath

        dirnames[:] = [
            d
            for d in dirnames
            if not _is_ignored(current / d, root_path, config.ignore_patterns)
        ]

        for fname in sorted(filenames):
            fpath = current / fname
            if _is_ignored(fpath, root_path, config.ignore_patterns):
                continue
            if fpath.stat().st_size > config.max_file_size:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                rel = str(fpath.relative_to(root_path))
                tokens = count_tokens_int(content, config)
                entries.append(
                    FileEntry(
                        path=rel,
                        content=content,
                        size=len(content),
                        tokens=tokens,
                    )
                )
            except (OSError, PermissionError):
                continue

    # Rank by relevance first, then by token cost, then path for stable ordering.
    entries.sort(
        key=lambda entry: (
            -_score_file_entry(entry, prompt),
            entry.tokens,
            entry.path,
        )
    )

    token_budget = config.max_context_tokens - config.max_response_tokens - 2048
    used_tokens = 0
    included: list[FileEntry] = []
    skipped: list[str] = []

    for entry in entries:
        if used_tokens + entry.tokens > token_budget:
            skipped.append(entry.path)
            continue
        included.append(entry)
        used_tokens += entry.tokens

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

    print(f"\n📂 Context: {len(included)} files | ~{used_tokens:,} tokens")
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
        ]
        depth = len(current.relative_to(root_path).parts)
        indent = "  " * depth
        for fname in sorted(filenames):
            fpath = current / fname
            if not _is_ignored(fpath, root_path, config.ignore_patterns):
                lines.append(f"{indent}{fname}")

    return "\n".join(lines)
