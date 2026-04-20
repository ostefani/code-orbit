import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from .types import ChunkEmbedding, FileEmbeddingRecord


DEFAULT_CACHE_DIR = ".code-orbit"
DEFAULT_CACHE_FILENAME = "embeddings_cache.npz"
LEGACY_CACHE_FILENAME = "embeddings_cache.json"
_CACHE_FORMAT_VERSION = 2


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
        return (
            self.model == config.embedding_model
            and self.api_base == config.embedding_api_base
        )


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
    np_module,
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
            max_content_hash_len = max(
                max_content_hash_len, len(chunk.content_hash)
            )
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

    metadata_dtype = np_module.dtype(
        [
            ("path", f"<U{max_path_len}"),
            ("sha256", f"<U{max_sha256_len}"),
            ("chunk_index", np_module.int64),
            ("start_line", np_module.int64),
            ("end_line", np_module.int64),
            ("content_hash", f"<U{max_content_hash_len}"),
        ]
    )

    if rows:
        metadata = np_module.array(rows, dtype=metadata_dtype)
        vectors_array = np_module.asarray(vectors, dtype=np_module.float64)
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
    else:
        metadata = np_module.empty((0,), dtype=metadata_dtype)
        vectors_array = np_module.empty((0, 0), dtype=np_module.float64)

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
            version = (
                int(archive["version"].item())
                if "version" in archive
                else _CACHE_FORMAT_VERSION
            )
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
                chunks=tuple(sorted(chunk_list, key=lambda chunk: chunk.index)),
            )
            for rel_path, chunk_list in sorted(files.items())
        },
    )
