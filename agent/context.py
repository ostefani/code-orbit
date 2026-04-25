import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config, require_chat_context_window
from .constants import (
    CONFIG_HINTS,
    LOW_VALUE_DIRS,
    LOW_VALUE_FILENAMES,
    SOURCE_EXTENSIONS,
    STOPWORDS,
    TEST_HINTS,
)
from .embeddings import (
    EmbeddingAdapter,
    EmbeddingSyncResult,
    build_embedding_index,
    create_embedding_adapter,
    default_embedding_cache_path,
)
from .events import ContextWarningPayload, EventBus
from .llm import SYSTEM_PROMPT
from .token_counter import count_tokens
from .utils import _is_ignored, _is_within_root


_SEMANTIC_SCORE_WEIGHT = 45.0


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


@dataclass(frozen=True)
class ContextBudgetBreakdown:
    context_window_tokens: int
    response_reserve_tokens: int
    scaffold_tokens: int
    safety_margin_tokens: int
    file_budget_tokens: int


@dataclass(frozen=True)
class SemanticMatch:
    path: str
    semantic_score: float
    lexical_score: float
    blended_score: float


@dataclass(frozen=True)
class ContextBuildResult:
    entries: tuple[FileEntry, ...]
    context: str
    skipped_paths: tuple[str, ...]
    used_tokens: int
    token_budget: int
    token_warnings: tuple[str, ...]
    semantic_matches: tuple[SemanticMatch, ...] = ()
    budget_breakdown: ContextBudgetBreakdown | None = None


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


def _score_path(
    path: str,
    estimated_tokens: int,
    prompt: str,
    semantic_score: float = 0.0,
) -> float:
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

    # 8. Semantic relevance from RAG.
    score += max(0.0, min(1.0, semantic_score)) * _SEMANTIC_SCORE_WEIGHT

    return score


def _render_file_block(path: str, content: str) -> str:
    return f'<file path="{path}">\n{content}\n</file>\n'


def _compute_context_budget(
    prompt: str, config: Config
) -> tuple[ContextBudgetBreakdown, tuple[str, ...]]:
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
    scaffold_result = count_tokens(scaffold, config)
    scaffold_tokens = scaffold_result.count
    context_window = require_chat_context_window(config)

    if config.tokenizer_backend == "estimate":
        # Estimate backend is intentionally conservative.
        safety_margin = max(128, int(context_window * 0.10))
    else:
        # Exact tokenizers need only a small buffer for drift.
        safety_margin = 64

    file_budget = (
        context_window - config.max_response_tokens - scaffold_tokens - safety_margin
    )
    available_file_tokens = max(0, file_budget)

    warnings = list(scaffold_result.warnings)
    if available_file_tokens == 0:
        if config.max_response_tokens >= context_window:
            warnings.append(
                "max_response_tokens leaves no room for file context. "
                "Lower max_response_tokens or raise max_context_tokens."
            )
        else:
            warnings.append(
                "No file context budget remains after reserving "
                f"{config.max_response_tokens} response tokens, "
                f"{scaffold_tokens} scaffold tokens, and {safety_margin} "
                "safety tokens."
            )

    return (
        ContextBudgetBreakdown(
            context_window_tokens=context_window,
            response_reserve_tokens=config.max_response_tokens,
            scaffold_tokens=scaffold_tokens,
            safety_margin_tokens=safety_margin,
            file_budget_tokens=available_file_tokens,
        ),
        tuple(warnings),
    )


async def build_context_async(
    root: str,
    prompt: str,
    config: Config,
    *,
    embedding_client: EmbeddingAdapter | None = None,
    cache_path: Path | None = None,
    event_bus: EventBus | None = None,
) -> ContextBuildResult:
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

    semantic_scores: dict[str, float] = {}
    owns_semantic_client = embedding_client is None
    if embedding_client is None:
        semantic_client = await create_embedding_adapter(config)
    else:
        semantic_client = embedding_client
        await semantic_client.validate()

    try:
        try:
            embedding_result: EmbeddingSyncResult = await build_embedding_index(
                root_path,
                config,
                cache_path=cache_path or default_embedding_cache_path(root_path),
                client=semantic_client,
                batch_size=config.embedding_batch_size,
            )
            prompt_vector = (await semantic_client.embed([prompt]))[0]
            semantic_scores = embedding_result.vector_store.semantic_scores(
                prompt_vector
            )
        except Exception as exc:
            if event_bus is not None:
                event_bus.emit(
                    "context.warning",
                    ContextWarningPayload(
                        warning=f"Semantic ranking unavailable: {exc}"
                    ),
                    level="warning",
                    state="building_context",
                    message="Semantic ranking unavailable.",
                )
            semantic_scores = {}
    finally:
        if owns_semantic_client:
            close = getattr(semantic_client, "aclose", None)
            if close is not None:
                await close()

    scored_candidates: list[tuple[float, FileCandidate, float, float]] = []
    for candidate in candidates:
        semantic_score = semantic_scores.get(candidate.path, 0.0)
        lexical_score = _score_path(
            candidate.path,
            max(1, candidate.size_bytes // 3),
            prompt,
            semantic_score=0.0,
        )
        blended_score = (
            lexical_score + max(0.0, min(1.0, semantic_score)) * _SEMANTIC_SCORE_WEIGHT
        )
        scored_candidates.append(
            (blended_score, candidate, lexical_score, semantic_score)
        )

    scored_candidates.sort(
        key=lambda item: (-item[0], item[1].size_bytes, item[1].path)
    )

    budget_breakdown, initial_warnings = _compute_context_budget(prompt, config)
    token_budget = budget_breakdown.file_budget_tokens
    used_tokens = 0
    included: list[FileEntry] = []
    skipped: list[str] = []
    token_warnings: list[str] = list(initial_warnings)
    included_paths: set[str] = set()

    for _, candidate, _lexical_score, _semantic_score in scored_candidates:
        path = root_path / candidate.path
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue

        rendered = _render_file_block(candidate.path, content)
        token_result = count_tokens(rendered, config)
        tokens = token_result.count
        token_warnings.extend(token_result.warnings)

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
        included_paths.add(candidate.path)
        used_tokens += tokens

    semantic_matches: tuple[SemanticMatch, ...] = ()
    if semantic_scores:
        score_lookup = {
            candidate.path: (lexical_score, blended_score, semantic_score)
            for blended_score, candidate, lexical_score, semantic_score in scored_candidates
        }
        ordered_matches = sorted(
            [
                (
                    path,
                    semantic_scores[path],
                    score_lookup[path][0],
                    score_lookup[path][1],
                )
                for path in included_paths
                if path in semantic_scores and semantic_scores[path] > 0.0
            ],
            key=lambda item: (-item[1], item[0]),
        )
        semantic_matches = tuple(
            SemanticMatch(
                path=path,
                semantic_score=score,
                lexical_score=lexical,
                blended_score=blended,
            )
            for path, score, lexical, blended in ordered_matches[:10]
        )

    parts = ["<codebase>"]
    parts.append(f"<file_tree>\n{get_file_tree(str(root_path), config)}\n</file_tree>")
    for entry in included:
        parts.append(f'<file path="{entry.path}">')
        parts.append(entry.content)
        parts.append("</file>")
    parts.append("</codebase>")

    context_str = "\n".join(parts)
    deduped_warnings = tuple(dict.fromkeys(token_warnings))
    return ContextBuildResult(
        entries=tuple(included),
        context=context_str,
        skipped_paths=tuple(skipped),
        used_tokens=used_tokens,
        token_budget=token_budget,
        token_warnings=deduped_warnings,
        semantic_matches=semantic_matches,
        budget_breakdown=budget_breakdown,
    )


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
        for dirname in sorted(dirnames):
            lines.append(f"{indent}{dirname}/")
        for fname in sorted(filenames):
            fpath = current / fname
            if (
                not _is_ignored(fpath, root_path, config.ignore_patterns)
                and not fpath.is_symlink()
                and _is_within_root(fpath, root_path)
            ):
                lines.append(f"{indent}{fname}")

    return "\n".join(lines)
