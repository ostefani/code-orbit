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
from .chunking import chunk_file
from .client import EmbeddingClient, OpenAICompatibleEmbeddingClient
from .discovery import _read_file_bytes, iter_code_files
from .store import VectorStore
from .types import (
    CodeChunk,
    ChunkEmbedding,
    FileEmbeddingRecord,
)


@runtime_checkable
class _ClosableEmbeddingClient(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class EmbeddingSyncResult:
    cache_path: Path
    vector_store: VectorStore
    updated_files: tuple[str, ...]
    reused_files: tuple[str, ...]
    chunk_count: int


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
            return await embed_batch(batch)

    batch_tasks: list[
        asyncio.Task[tuple[list[tuple[str, str, CodeChunk]], list[Sequence[float]]]]
    ] = []
    for start in range(0, len(pending_chunks), effective_batch_size):
        batch = pending_chunks[start : start + effective_batch_size]
        batch_tasks.append(asyncio.create_task(guarded_embed_batch(batch)))

    batch_results = await asyncio.gather(*batch_tasks) if batch_tasks else []
    for batch, vectors in batch_results:
        for (rel_path, _file_hash, chunk), vector in zip(batch, vectors, strict=True):
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
        if owns_client and isinstance(embedding_client, _ClosableEmbeddingClient):
            await embedding_client.aclose()


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
