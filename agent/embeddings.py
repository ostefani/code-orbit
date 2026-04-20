import asyncio
import ast
import hashlib
import inspect
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import Config
from .utils import _is_ignored, _is_within_root


DEFAULT_CACHE_DIR = ".code-orbit"
DEFAULT_CACHE_FILENAME = "embeddings_cache.npz"
LEGACY_CACHE_FILENAME = "embeddings_cache.json"
_CACHE_FORMAT_VERSION = 2
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


class EmbeddingClient(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class _EmbeddingVectorsAPI(Protocol):
    async def create(self, *, model: str, input: list[str]) -> Any: ...


class _EmbeddingClientAPI(Protocol):
    @property
    def embeddings(self) -> _EmbeddingVectorsAPI: ...


class OpenAICompatibleEmbeddingClient:
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: _EmbeddingClientAPI | None = None

    def _get_client(self) -> _EmbeddingClientAPI:
        client = self._client
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise RuntimeError(
                    "The 'openai' package is required for embedding generation."
                ) from exc

            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            self._client = client
        return client

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


@dataclass(frozen=True)
class _VectorStoreNumpyIndex:
    record_paths: tuple[str, ...]
    chunk_matrix: np.ndarray
    chunk_path_ids: np.ndarray
    chunk_norms: np.ndarray


class VectorStore:
    def __init__(self, records: Iterable[FileEmbeddingRecord] = ()) -> None:
        self._records: dict[str, FileEmbeddingRecord] = {
            record.path: record for record in records
        }
        self._numpy_index: _VectorStoreNumpyIndex | None = None

    @property
    def records(self) -> tuple[FileEmbeddingRecord, ...]:
        return tuple(self._records[path] for path in sorted(self._records))

    def add(self, record: FileEmbeddingRecord) -> None:
        self._records[record.path] = record
        self._numpy_index = None

    def _ensure_numpy_index(self) -> _VectorStoreNumpyIndex:
        if self._numpy_index is None:
            self._numpy_index = self._build_numpy_index()
        return self._numpy_index

    def _build_numpy_index(self) -> _VectorStoreNumpyIndex:
        record_paths: list[str] = []
        chunk_vectors: list[Sequence[float]] = []
        chunk_path_ids: list[int] = []

        for record_index, record in enumerate(self.records):
            record_paths.append(record.path)
            for chunk in sorted(record.chunks, key=lambda item: item.index):
                chunk_vectors.append(chunk.vector)
                chunk_path_ids.append(record_index)

        if not chunk_vectors:
            chunk_matrix = np.empty((0, 0), dtype=np.float64)
            chunk_norms = np.empty((0,), dtype=np.float64)
            chunk_path_ids_array = np.empty((0,), dtype=np.int64)
        else:
            chunk_matrix = np.asarray(chunk_vectors, dtype=np.float64)
            chunk_norms = np.linalg.norm(chunk_matrix, axis=1)
            chunk_path_ids_array = np.asarray(chunk_path_ids, dtype=np.int64)

        return _VectorStoreNumpyIndex(
            record_paths=tuple(record_paths),
            chunk_matrix=chunk_matrix,
            chunk_path_ids=chunk_path_ids_array,
            chunk_norms=chunk_norms,
        )

    def semantic_scores(self, query_vector: Sequence[float]) -> dict[str, float]:
        index = self._ensure_numpy_index()
        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-10:
            return {path: 0.0 for path in index.record_paths}

        if index.chunk_matrix.size == 0:
            return {path: 0.0 for path in index.record_paths}

        dot_products = index.chunk_matrix @ query
        valid = index.chunk_norms > 1e-10
        cosines = np.zeros(len(index.chunk_path_ids), dtype=np.float64)
        cosines[valid] = dot_products[valid] / (index.chunk_norms[valid] * query_norm)
        cosines = np.maximum(cosines, 0.0)

        scores = np.zeros(len(index.record_paths), dtype=np.float64)
        np.maximum.at(scores, index.chunk_path_ids, cosines)
        return {path: float(score) for path, score in zip(index.record_paths, scores)}

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


@dataclass(frozen=True)
class EmbeddingSyncResult:
    cache_path: Path
    vector_store: VectorStore
    updated_files: tuple[str, ...]
    reused_files: tuple[str, ...]
    chunk_count: int


@dataclass
class EmbeddingCache:
    version: int = _CACHE_FORMAT_VERSION
    model: str = ""
    api_base: str = ""
    files: dict[str, FileEmbeddingRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> EmbeddingCache:
        if not path.exists():
            legacy_path = _legacy_cache_sibling_path(path)
            if legacy_path.exists():
                cache = _load_legacy_json_cache(legacy_path)
                if cache is not None:
                    return cache
            return cls()

        if _looks_like_npz(path):
            cache = _load_npz_cache(path)
            if cache is not None:
                return cache

        cache = _load_legacy_json_cache(path)
        if cache is not None:
            return cache

        return cls()

    def save(self, path: Path) -> None:
        metadata, vectors = _serialize_cache_arrays(self.files, np)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("wb") as file_handle:
            np.savez_compressed(
                file_handle,
                version=np.array(_CACHE_FORMAT_VERSION, dtype=np.int64),
                model=np.array(self.model, dtype=f"<U{max(1, len(self.model))}"),
                api_base=np.array(
                    self.api_base, dtype=f"<U{max(1, len(self.api_base))}"
                ),
                metadata=metadata,
                vectors=vectors,
            )
        os.replace(temp_path, path)

    def is_compatible(self, config: Config) -> bool:
        return self.model == config.embedding_model and self.api_base == config.embedding_api_base


def default_embedding_cache_path(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    return root_path / DEFAULT_CACHE_DIR / DEFAULT_CACHE_FILENAME


def _legacy_embedding_cache_path(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    return root_path / DEFAULT_CACHE_DIR / LEGACY_CACHE_FILENAME


def _legacy_cache_sibling_path(path: Path) -> Path:
    return path.parent / LEGACY_CACHE_FILENAME


def _looks_like_npz(path: Path) -> bool:
    try:
        with path.open("rb") as file_handle:
            return file_handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _serialize_cache_arrays(
    files: dict[str, FileEmbeddingRecord],
    np,
):
    rows: list[tuple[str, str, int, int, int, str]] = []
    vectors: list[tuple[float, ...]] = []
    vector_width: int | None = None
    max_path_len = 1
    max_sha256_len = 64
    max_content_hash_len = 64

    for rel_path, record in sorted(files.items()):
        max_path_len = max(max_path_len, len(rel_path))
        max_sha256_len = max(max_sha256_len, len(record.sha256))
        for chunk in sorted(record.chunks, key=lambda item: item.index):
            if vector_width is None:
                vector_width = len(chunk.vector)
            elif len(chunk.vector) != vector_width:
                raise ValueError(
                    f"Embedding cache has inconsistent vector dimensions for {rel_path!r}."
                )
            max_content_hash_len = max(max_content_hash_len, len(chunk.content_hash))
            rows.append(
                (
                    rel_path,
                    record.sha256,
                    chunk.index,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content_hash,
                )
            )
            vectors.append(tuple(float(value) for value in chunk.vector))

    metadata_dtype = np.dtype(
        [
            ("path", f"<U{max_path_len}"),
            ("sha256", f"<U{max_sha256_len}"),
            ("chunk_index", np.int64),
            ("start_line", np.int64),
            ("end_line", np.int64),
            ("content_hash", f"<U{max_content_hash_len}"),
        ]
    )

    if rows:
        metadata = np.array(rows, dtype=metadata_dtype)
        vectors_array = np.asarray(vectors, dtype=np.float64)
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
    else:
        metadata = np.empty((0,), dtype=metadata_dtype)
        vectors_array = np.empty((0, 0), dtype=np.float64)

    return metadata, vectors_array


def _load_legacy_json_cache(path: Path) -> EmbeddingCache | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

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

    return EmbeddingCache(
        version=int(raw.get("version", _CACHE_FORMAT_VERSION)),
        model=str(raw.get("model", "")),
        api_base=str(raw.get("api_base", "")),
        files=files,
    )


def _load_npz_cache(path: Path) -> EmbeddingCache | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive or "vectors" not in archive:
                return None

            metadata = archive["metadata"]
            vectors = archive["vectors"]
            version = int(archive["version"].item()) if "version" in archive else _CACHE_FORMAT_VERSION
            model = str(archive["model"].item()) if "model" in archive else ""
            api_base = str(archive["api_base"].item()) if "api_base" in archive else ""
    except (OSError, ValueError, KeyError, TypeError):
        return None

    if metadata.size == 0:
        return EmbeddingCache(version=version, model=model, api_base=api_base, files={})

    if metadata.shape[0] != vectors.shape[0]:
        return None

    if vectors.ndim == 1:
        if metadata.shape[0] != 1:
            return None
        vectors = vectors.reshape(1, -1)

    files: dict[str, list[ChunkEmbedding]] = {}
    sha256_by_path: dict[str, str] = {}
    for row, vector in zip(metadata, vectors, strict=True):
        rel_path = str(row["path"])
        sha256 = str(row["sha256"])
        existing_sha256 = sha256_by_path.setdefault(rel_path, sha256)
        if existing_sha256 != sha256:
            return None
        files.setdefault(rel_path, []).append(
            ChunkEmbedding(
                index=int(row["chunk_index"]),
                vector=tuple(float(value) for value in vector),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                content_hash=str(row["content_hash"]),
            )
        )

    return EmbeddingCache(
        version=version,
        model=model,
        api_base=api_base,
        files={
            rel_path: FileEmbeddingRecord(
                path=rel_path,
                sha256=sha256_by_path[rel_path],
                chunks=tuple(
                    sorted(chunk_list, key=lambda chunk: chunk.index)
                ),
            )
            for rel_path, chunk_list in sorted(files.items())
        },
    )


def _read_file_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
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
    cache_load_path = cache_file
    if cache_path is None and not cache_file.exists():
        legacy_cache_file = _legacy_embedding_cache_path(root_path)
        if legacy_cache_file.exists():
            cache_load_path = legacy_cache_file

    cache = EmbeddingCache.load(cache_load_path)
    if not cache.is_compatible(config):
        cache = EmbeddingCache(
            model=config.embedding_model,
            api_base=config.embedding_api_base,
        )

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
        data = _read_file_bytes(path)
        if data is None:
            continue

        file_hash = hashlib.sha256(data).hexdigest()
        cached_record = cache.files.get(rel_path)
        if cached_record is not None and cached_record.sha256 == file_hash:
            reused_files.append(rel_path)
            continue

        content = data.decode("utf-8", errors="ignore")

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
