import asyncio
import hashlib
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Config
from .cache import (
    EmbeddingCache,
    _legacy_embedding_cache_path,
    default_embedding_cache_path,
)
from .adapters import EmbeddingAdapter
from .chunking import chunk_file
from .factory import create_embedding_adapter
from .discovery import _read_file_bytes, iter_code_files
from .store import VectorStore
from .types import (
    CodeChunk,
    ChunkEmbedding,
    FileEmbeddingRecord,
)


@runtime_checkable
class _ClosableEmbeddingAdapter(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class EmbeddingSyncResult:
    cache_path: Path
    vector_store: VectorStore
    updated_files: tuple[str, ...]
    reused_files: tuple[str, ...]
    chunk_count: int
    failed_files: tuple[str, ...] = ()
    timed_out_files: tuple[str, ...] = ()


async def build_embedding_index(
    root: str | Path,
    config: Config,
    *,
    cache_path: Path | None = None,
    client: EmbeddingAdapter | None = None,
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
    if client is None:
        embedding_client = await create_embedding_adapter(config)
    else:
        embedding_client = client
        await embedding_client.validate()
    effective_batch_size = max(1, batch_size or config.embedding_batch_size)
    max_concurrency = max(1, config.embedding_max_concurrency)

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

    refreshed_records: dict[str, list[ChunkEmbedding]] = {
        path: [] for path in pending_files
    }

    async def embed_batch(
        batch: list[tuple[str, str, CodeChunk]]
    ) -> tuple[list[tuple[str, str, CodeChunk]], list[Sequence[float]]]:
        vectors = await embedding_client.embed([chunk.content for _, _, chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding client returned a mismatched number of vectors for a batch."
            )
        return batch, list(vectors)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def guarded_embed_batch(
        batch: list[tuple[str, str, CodeChunk]]
    ) -> tuple[list[tuple[str, str, CodeChunk]], list[Sequence[float]]]:
        async with semaphore:
            return await asyncio.wait_for(
                embed_batch(batch),
                timeout=config.embedding_timeout_seconds,
            )

    batch_tasks: list[
        tuple[
            list[tuple[str, str, CodeChunk]],
            asyncio.Task[tuple[list[tuple[str, str, CodeChunk]], list[Sequence[float]]]],
        ]
    ] = []
    for start in range(0, len(pending_chunks), effective_batch_size):
        batch = pending_chunks[start : start + effective_batch_size]
        batch_tasks.append((batch, asyncio.create_task(guarded_embed_batch(batch))))

    failed_files: set[str] = set()
    timed_out_files: set[str] = set()
    if batch_tasks:
        batch_results = await asyncio.gather(
            *(task for _, task in batch_tasks),
            return_exceptions=True,
        )
    else:
        batch_results = []

    for (requested_batch, _task), result in zip(
        batch_tasks,
        batch_results,
        strict=True,
    ):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, TimeoutError):
            timed_out_files.update(rel_path for rel_path, _, _ in requested_batch)
            continue
        if isinstance(result, Exception):
            failed_files.update(rel_path for rel_path, _, _ in requested_batch)
            continue
        if isinstance(result, BaseException):
            raise result

        _batch, vectors = result
        for (rel_path, _file_hash, chunk), vector in zip(
            requested_batch,
            vectors,
            strict=True,
        ):
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
        updated_files: list[str] = []
        for rel_path, (file_hash, _) in pending_files.items():
            if rel_path in failed_files or rel_path in timed_out_files:
                continue

            chunks = refreshed_records.get(rel_path, [])
            expected_chunk_count = len(pending_files[rel_path][1])
            if len(chunks) != expected_chunk_count:
                failed_files.add(rel_path)
                continue

            cache.files[rel_path] = FileEmbeddingRecord(
                path=rel_path,
                sha256=file_hash,
                chunks=tuple(sorted(chunks, key=lambda item: item.index)),
            )
            updated_files.append(rel_path)

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
            failed_files=tuple(sorted(failed_files)),
            timed_out_files=tuple(sorted(timed_out_files)),
        )
    finally:
        if owns_client and isinstance(embedding_client, _ClosableEmbeddingAdapter):
            await embedding_client.aclose()


def build_embedding_sync(
    root: str | Path,
    config: Config,
    *,
    cache_path: Path | None = None,
    client: EmbeddingAdapter | None = None,
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
