from __future__ import annotations

import asyncio
import ast
import inspect
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from .config import Config
from .constants import LOW_VALUE_DIRS, LOW_VALUE_FILENAMES


DEFAULT_CACHE_DIR = ".code-orbit"
DEFAULT_CACHE_FILENAME = "embeddings_cache.json"
DEFAULT_CHUNK_CHAR_LIMIT = 4_000


@dataclass(frozen=True)
class CodeChunk:
    path: str
    index: int
    content: str
    start_line: int
    end_line: int
    content_hash: str


@dataclass(frozen=True)
class ChunkEmbedding:
    index: int
    vector: tuple[float, ...]
    start_line: int
    end_line: int
    content_hash: str


@dataclass(frozen=True)
class FileEmbeddingRecord:
    path: str
    sha256: str
    chunks: tuple[ChunkEmbedding, ...]


@dataclass(frozen=True)
class EmbeddingSearchResult:
    path: str
    score: float
    chunk_index: int
    start_line: int
    end_line: int
    sha256: str


@dataclass(frozen=True)
class EmbeddingSyncResult:
    cache_path: Path
    vector_store: "VectorStore"
    updated_files: tuple[str, ...]
    reused_files: tuple[str, ...]
    chunk_count: int


class EmbeddingClient(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class OpenAICompatibleEmbeddingClient:
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise RuntimeError(
                    "The 'openai' package is required for embedding generation."
                ) from exc

            self._client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=list(texts))
        return [tuple(item.embedding) for item in response.data]

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return

        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        self._client = None


@dataclass
class EmbeddingCache:
    version: int = 1
    model: str = ""
    api_base: str = ""
    files: dict[str, FileEmbeddingRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCache":
        if not path.exists():
            return cls()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()

        files: dict[str, FileEmbeddingRecord] = {}
        for rel_path, payload in raw.get("files", {}).items():
            chunks = tuple(
                ChunkEmbedding(
                    index=int(chunk["index"]),
                    vector=tuple(float(value) for value in chunk["vector"]),
                    start_line=int(chunk["start_line"]),
                    end_line=int(chunk["end_line"]),
                    content_hash=str(chunk["content_hash"]),
                )
                for chunk in payload.get("chunks", [])
            )
            files[rel_path] = FileEmbeddingRecord(
                path=rel_path,
                sha256=str(payload.get("sha256", "")),
                chunks=chunks,
            )

        return cls(
            version=int(raw.get("version", 1)),
            model=str(raw.get("model", "")),
            api_base=str(raw.get("api_base", "")),
            files=files,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "model": self.model,
            "api_base": self.api_base,
            "files": {
                rel_path: {
                    "sha256": record.sha256,
                    "chunks": [
                        {
                            "index": chunk.index,
                            "vector": list(chunk.vector),
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "content_hash": chunk.content_hash,
                        }
                        for chunk in record.chunks
                    ],
                }
                for rel_path, record in sorted(self.files.items())
            },
        }
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, path)

    def is_compatible(self, config: Config) -> bool:
        return self.model == config.embedding_model and self.api_base == config.embedding_api_base


class VectorStore:
    def __init__(self, records: Sequence[FileEmbeddingRecord] | None = None) -> None:
        self._records: dict[str, FileEmbeddingRecord] = {
            record.path: record for record in (records or [])
        }

    @property
    def records(self) -> tuple[FileEmbeddingRecord, ...]:
        return tuple(self._records[path] for path in sorted(self._records))

    def add(self, record: FileEmbeddingRecord) -> None:
        self._records[record.path] = record

    def semantic_scores(self, query_vector: Sequence[float]) -> dict[str, float]:
        try:
            import numpy as np
        except ImportError:
            scores: dict[str, float] = {}
            for record in self._records.values():
                best_score = 0.0
                for chunk in record.chunks:
                    best_score = max(
                        best_score,
                        max(0.0, _cosine_similarity(query_vector, chunk.vector)),
                    )
                scores[record.path] = best_score
            return scores

        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-10:
            return {record.path: 0.0 for record in self._records.values()}

        scores: dict[str, float] = {}
        for record in self._records.values():
            if not record.chunks:
                scores[record.path] = 0.0
                continue

            matrix = np.asarray([chunk.vector for chunk in record.chunks], dtype=np.float64)
            norms = np.linalg.norm(matrix, axis=1)
            valid = norms > 1e-10
            if not np.any(valid):
                scores[record.path] = 0.0
                continue

            dot_products = matrix @ query
            cosines = np.zeros(len(record.chunks), dtype=np.float64)
            cosines[valid] = dot_products[valid] / (norms[valid] * query_norm)
            scores[record.path] = float(np.max(np.maximum(cosines, 0.0)))

        return scores

    def score_path(self, path: str, query_vector: Sequence[float]) -> float:
        record = self._records.get(path)
        if record is None:
            return 0.0
        best_score = 0.0
        for chunk in record.chunks:
            best_score = max(
                best_score,
                max(0.0, _cosine_similarity(query_vector, chunk.vector)),
            )
        return best_score

    def search(self, query_vector: Sequence[float], top_k: int = 10) -> list[EmbeddingSearchResult]:
        results: list[EmbeddingSearchResult] = []
        for record in self._records.values():
            best_result: EmbeddingSearchResult | None = None
            for chunk in record.chunks:
                score = max(0.0, _cosine_similarity(query_vector, chunk.vector))
                candidate = EmbeddingSearchResult(
                    path=record.path,
                    score=score,
                    chunk_index=chunk.index,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    sha256=record.sha256,
                )
                if best_result is None or candidate.score > best_result.score:
                    best_result = candidate
            if best_result is not None:
                results.append(best_result)

        results.sort(key=lambda item: (-item.score, item.path, item.chunk_index))
        return results[:top_k]

    async def search_text(
        self,
        text: str,
        client: EmbeddingClient,
        top_k: int = 10,
    ) -> list[EmbeddingSearchResult]:
        vector = (await client.embed([text]))[0]
        return self.search(vector, top_k=top_k)


def default_embedding_cache_path(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    return root_path / DEFAULT_CACHE_DIR / DEFAULT_CACHE_FILENAME


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_ignored(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    import fnmatch

    rel = str(path.relative_to(root))
    if path.name in LOW_VALUE_FILENAMES:
        return True
    if any(part in LOW_VALUE_DIRS for part in path.parts):
        return True
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


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return None


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


def _chunk_python_source(path: Path, content: str, max_chunk_chars: int) -> list[CodeChunk]:
    lines = content.splitlines()
    try:
        module = ast.parse(content)
    except SyntaxError:
        return _chunk_by_lines(path, content, max_chunk_chars)

    spans: list[tuple[int, int]] = []
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end_line))

    if not spans:
        return _chunk_by_lines(path, content, max_chunk_chars)

    chunks: list[CodeChunk] = []
    current_index = 0
    last_end = 0

    for start_line, end_line in spans:
        if start_line - 1 > last_end:
            gap_text = "\n".join(lines[last_end : start_line - 1]).strip()
            if gap_text:
                chunks.append(
                    _make_chunk(
                        path=path,
                        index=current_index,
                        content=gap_text,
                        start_line=last_end + 1,
                        end_line=start_line - 1,
                    )
                )
                current_index += 1

        block_text = "\n".join(lines[start_line - 1 : end_line]).strip()
        if block_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=current_index,
                    content=block_text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            current_index += 1
        last_end = max(last_end, end_line)

    if last_end < len(lines):
        tail_text = "\n".join(lines[last_end:]).strip()
        if tail_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=current_index,
                    content=tail_text,
                    start_line=last_end + 1,
                    end_line=len(lines),
                )
            )

    return chunks or _chunk_by_lines(path, content, max_chunk_chars)


def _chunk_by_lines(path: Path, content: str, max_chunk_chars: int) -> list[CodeChunk]:
    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    buffer: list[str] = []
    start_line = 1
    current_chars = 0
    index = 0

    for line_number, line in enumerate(lines, start=1):
        line_length = len(line) + 1
        if buffer and current_chars + line_length > max_chunk_chars:
            chunk_text = "\n".join(buffer).strip()
            if chunk_text:
                chunks.append(
                    _make_chunk(
                        path=path,
                        index=index,
                        content=chunk_text,
                        start_line=start_line,
                        end_line=line_number - 1,
                    )
                )
                index += 1
            buffer = []
            current_chars = 0
            start_line = line_number

        buffer.append(line)
        current_chars += line_length

    if buffer:
        chunk_text = "\n".join(buffer).strip()
        if chunk_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=index,
                    content=chunk_text,
                    start_line=start_line,
                    end_line=len(lines),
                )
            )

    return chunks


def _make_chunk(
    path: Path,
    index: int,
    content: str,
    start_line: int,
    end_line: int,
) -> CodeChunk:
    normalized_content = content.strip()
    return CodeChunk(
        path=str(path),
        index=index,
        content=normalized_content,
        start_line=start_line,
        end_line=end_line,
        content_hash=hashlib.sha256(normalized_content.encode("utf-8")).hexdigest(),
    )


def chunk_file(path: Path, content: str, max_chunk_chars: int = DEFAULT_CHUNK_CHAR_LIMIT) -> list[CodeChunk]:
    if not content.strip():
        return []

    if len(content) <= max_chunk_chars:
        return [
            _make_chunk(
                path=path,
                index=0,
                content=content.strip(),
                start_line=1,
                end_line=max(1, content.count("\n") + 1),
            )
        ]

    if path.suffix == ".py":
        return _chunk_python_source(path, content, max_chunk_chars)

    return _chunk_by_lines(path, content, max_chunk_chars)


async def build_embedding_index(
    root: str | Path,
    config: Config,
    *,
    cache_path: Path | None = None,
    client: EmbeddingClient | None = None,
    batch_size: int | None = None,
) -> EmbeddingSyncResult:
    root_path = Path(root).resolve()
    cache_file = cache_path or default_embedding_cache_path(root_path)
    cache = EmbeddingCache.load(cache_file)
    if not cache.is_compatible(config):
        cache = EmbeddingCache(model=config.embedding_model, api_base=config.embedding_api_base)

    owns_client = client is None
    embedding_client = client or OpenAICompatibleEmbeddingClient(
        api_base=config.embedding_api_base,
        api_key=config.api_key,
        model=config.embedding_model,
    )
    effective_batch_size = max(1, batch_size or config.embedding_batch_size)
    max_concurrency = max(1, config.embedding_max_concurrency)

    updated_files: list[str] = []
    reused_files: list[str] = []
    pending_files: dict[str, tuple[str, list[CodeChunk]]] = {}
    pending_chunks: list[tuple[str, str, CodeChunk]] = []
    current_paths: set[str] = set()

    for path in iter_code_files(root_path, config):
        rel_path = str(path.relative_to(root_path))
        current_paths.add(rel_path)
        file_hash = hash_file(path)
        cached_record = cache.files.get(rel_path)
        if cached_record is not None and cached_record.sha256 == file_hash:
            reused_files.append(rel_path)
            continue

        content = _safe_read_text(path)
        if content is None:
            continue

        chunks = chunk_file(path, content)
        if not chunks:
            continue

        pending_files[rel_path] = (file_hash, chunks)
        pending_chunks.extend((rel_path, file_hash, chunk) for chunk in chunks)
        updated_files.append(rel_path)

    refreshed_records: dict[str, list[ChunkEmbedding]] = {path: [] for path in pending_files}

    async def embed_batch(batch: list[tuple[str, str, CodeChunk]]) -> tuple[
        list[tuple[str, str, CodeChunk]],
        Sequence[Sequence[float]],
    ]:
        vectors = await embedding_client.embed([chunk.content for _, _, chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding client returned a mismatched number of vectors for a batch."
            )
        return batch, vectors

    semaphore = asyncio.Semaphore(max_concurrency)

    async def guarded_embed_batch(
        batch: list[tuple[str, str, CodeChunk]]
    ) -> tuple[list[tuple[str, str, CodeChunk]], Sequence[Sequence[float]]]:
        async with semaphore:
            return await embed_batch(batch)

    batch_tasks: list[asyncio.Task[tuple[list[tuple[str, str, CodeChunk]], Sequence[Sequence[float]]]]] = []
    for start in range(0, len(pending_chunks), effective_batch_size):
        batch = pending_chunks[start : start + effective_batch_size]
        batch_tasks.append(asyncio.create_task(guarded_embed_batch(batch)))

    batch_results = await asyncio.gather(*batch_tasks) if batch_tasks else []
    for batch, vectors in batch_results:
        for (rel_path, file_hash, chunk), vector in zip(batch, vectors, strict=True):
            refreshed_records.setdefault(rel_path, []).append(
                ChunkEmbedding(
                    index=chunk.index,
                    vector=tuple(float(value) for value in vector),
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content_hash=chunk.content_hash,
                )
            )

    try:
        for rel_path, (file_hash, _) in pending_files.items():
            cache.files[rel_path] = FileEmbeddingRecord(
                path=rel_path,
                sha256=file_hash,
                chunks=tuple(
                    sorted(refreshed_records.get(rel_path, []), key=lambda item: item.index)
                ),
            )

        for stale_path in set(cache.files) - current_paths:
            cache.files.pop(stale_path, None)

        cache.save(cache_file)
        vector_store = VectorStore(cache.files.values())
        return EmbeddingSyncResult(
            cache_path=cache_file,
            vector_store=vector_store,
            updated_files=tuple(updated_files),
            reused_files=tuple(reused_files),
            chunk_count=sum(len(record.chunks) for record in cache.files.values()),
        )
    finally:
        if owns_client and hasattr(embedding_client, "aclose"):
            await embedding_client.aclose()  # type: ignore[union-attr]


def build_embedding_sync(
    root: str | Path,
    config: Config,
    *,
    cache_path: Path | None = None,
    client: EmbeddingClient | None = None,
    batch_size: int | None = None,
) -> EmbeddingSyncResult:
    return asyncio.run(
        build_embedding_index(
            root,
            config,
            cache_path=cache_path,
            client=client,
            batch_size=batch_size,
        )
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lhs, rhs in zip(left, right, strict=True):
        dot += lhs * rhs
        left_norm += lhs * lhs
        right_norm += rhs * rhs

    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denom < 1e-10:
        return 0.0
    return dot / denom
