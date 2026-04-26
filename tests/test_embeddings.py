import json
import hashlib
from pathlib import Path

import pytest

from agent.config import Config
from agent.embeddings import (
    EmbeddingCache,
    FileEmbeddingRecord,
    ChunkEmbedding,
    VectorStore,
    build_embedding_sync,
    default_embedding_cache_path,
    iter_code_files,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    async def embed(self, texts):
        batch = list(texts)
        self.requests.append(batch)
        vectors: list[tuple[float, float]] = []
        for text in batch:
            lower = text.lower()
            if "auth" in lower or "middleware" in lower:
                vectors.append((0.9, 0.1))
            elif "rate" in lower or "limit" in lower:
                vectors.append((0.2, 0.9))
            else:
                vectors.append((0.5, 0.5))
        return vectors

    async def validate(self) -> None:
        return None

    async def probe(self) -> None:
        return None


class PartiallyFailingEmbeddingClient(FakeEmbeddingClient):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on

    async def embed(self, texts):
        batch = list(texts)
        if any(self.fail_on in text for text in batch):
            self.requests.append(batch)
            raise RuntimeError("embedding batch failed")
        return await super().embed(batch)


def _write_codebase(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "auth").mkdir()
    (root / "src" / "auth" / "middleware.py").write_text(
        "def rate_limit(request):\n    return request\n",
        encoding="utf-8",
    )
    (root / "src" / "tests.py").write_text(
        "def test_rate_limit():\n    assert True\n",
        encoding="utf-8",
    )


def test_build_embedding_sync_creates_cache_and_reuses_files(tmp_path: Path) -> None:
    _write_codebase(tmp_path)
    config = Config(ignore_patterns=[".git", "node_modules"])
    client = FakeEmbeddingClient()

    result = build_embedding_sync(
        tmp_path,
        config,
        client=client,
        cache_path=default_embedding_cache_path(tmp_path),
        batch_size=2,
    )

    assert result.cache_path.exists()
    assert result.cache_path.suffix == ".npz"
    assert result.cache_path.read_bytes().startswith(b"PK")
    assert set(result.updated_files) == {
        "src/auth/middleware.py",
        "src/tests.py",
    }
    assert len(client.requests) == 1
    assert result.chunk_count == 2

    cache = EmbeddingCache.load(result.cache_path)
    expected_hash = hashlib.sha256(
        (tmp_path / "src" / "auth" / "middleware.py").read_bytes()
    ).hexdigest()
    assert cache.files["src/auth/middleware.py"].sha256 == expected_hash

    second_client = FakeEmbeddingClient()
    second = build_embedding_sync(
        tmp_path,
        config,
        client=second_client,
        cache_path=result.cache_path,
        batch_size=2,
    )

    assert second.updated_files == ()
    assert set(second.reused_files) == {
        "src/auth/middleware.py",
        "src/tests.py",
    }
    assert second_client.requests == []

    (tmp_path / "src" / "tests.py").unlink()
    third_client = FakeEmbeddingClient()
    third = build_embedding_sync(
        tmp_path,
        config,
        client=third_client,
        cache_path=result.cache_path,
        batch_size=2,
    )

    assert third.updated_files == ()
    assert third.reused_files == ("src/auth/middleware.py",)
    assert "src/tests.py" not in EmbeddingCache.load(third.cache_path).files


def test_embedding_cache_round_trips_npz(tmp_path: Path) -> None:
    cache = EmbeddingCache(
        version=2,
        model="embedding-model",
        api_base="http://example.invalid/v1",
        files={
            "src/app.py": FileEmbeddingRecord(
                path="src/app.py",
                sha256="abc123",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.1, 0.2, 0.3),
                        start_line=1,
                        end_line=3,
                        content_hash="chunk-hash",
                    ),
                ),
            ),
        },
    )

    cache_path = tmp_path / "embeddings_cache.npz"
    cache.save(cache_path)

    assert cache_path.read_bytes().startswith(b"PK")

    loaded = EmbeddingCache.load(cache_path)
    assert loaded.version == 2
    assert loaded.model == "embedding-model"
    assert loaded.api_base == "http://example.invalid/v1"
    assert loaded.files == cache.files


def test_embedding_cache_loads_legacy_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "legacy-model",
                "api_base": "http://legacy.invalid/v1",
                "files": {
                    "src/app.py": {
                        "sha256": "abc123",
                        "chunks": [
                            {
                                "index": 0,
                                "vector": [0.4, 0.5],
                                "start_line": 1,
                                "end_line": 2,
                                "content_hash": "legacy-chunk",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cache = EmbeddingCache.load(cache_path)
    assert cache.version == 1
    assert cache.model == "legacy-model"
    assert cache.api_base == "http://legacy.invalid/v1"
    assert cache.files["src/app.py"].sha256 == "abc123"
    assert cache.files["src/app.py"].chunks[0].vector == (0.4, 0.5)


def test_embedding_cache_loads_legacy_json_sibling_for_missing_npz(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / ".code-orbit" / "embeddings_cache.npz"
    cache_path.parent.mkdir()
    legacy_path = cache_path.parent / "embeddings_cache.json"
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "legacy-model",
                "api_base": "http://legacy.invalid/v1",
                "files": {
                    "src/app.py": {
                        "sha256": "abc123",
                        "chunks": [
                            {
                                "index": 0,
                                "vector": [0.4, 0.5],
                                "start_line": 1,
                                "end_line": 2,
                                "content_hash": "legacy-chunk",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cache = EmbeddingCache.load(cache_path)
    assert cache.model == "legacy-model"
    assert cache.files["src/app.py"].chunks[0].vector == (0.4, 0.5)


def test_build_embedding_sync_migrates_legacy_cache(tmp_path: Path) -> None:
    _write_codebase(tmp_path)
    config = Config(ignore_patterns=[".git", "node_modules"])
    legacy_cache_dir = tmp_path / ".code-orbit"
    legacy_cache_dir.mkdir()
    legacy_cache_path = legacy_cache_dir / "embeddings_cache.json"

    middleware_hash = hashlib.sha256(
        (tmp_path / "src" / "auth" / "middleware.py").read_bytes()
    ).hexdigest()
    tests_hash = hashlib.sha256(
        (tmp_path / "src" / "tests.py").read_bytes()
    ).hexdigest()
    legacy_cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": config.embedding_model,
                "api_base": config.embedding_api_base,
                "files": {
                    "src/auth/middleware.py": {
                        "sha256": middleware_hash,
                        "chunks": [
                            {
                                "index": 0,
                                "vector": [0.9, 0.1],
                                "start_line": 1,
                                "end_line": 2,
                                "content_hash": "middleware-chunk",
                            }
                        ],
                    },
                    "src/tests.py": {
                        "sha256": tests_hash,
                        "chunks": [
                            {
                                "index": 0,
                                "vector": [0.1, 0.9],
                                "start_line": 1,
                                "end_line": 2,
                                "content_hash": "tests-chunk",
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    client = FakeEmbeddingClient()
    result = build_embedding_sync(
        tmp_path,
        config,
        client=client,
        cache_path=default_embedding_cache_path(tmp_path),
        batch_size=2,
    )

    assert result.cache_path.suffix == ".npz"
    assert result.updated_files == ()
    assert set(result.reused_files) == {
        "src/auth/middleware.py",
        "src/tests.py",
    }
    assert client.requests == []
    assert set(EmbeddingCache.load(result.cache_path).files) == {
        "src/auth/middleware.py",
        "src/tests.py",
    }


def test_build_embedding_sync_reads_each_file_once(tmp_path: Path, monkeypatch) -> None:
    _write_codebase(tmp_path)
    config = Config(ignore_patterns=[".git", "node_modules"])
    client = FakeEmbeddingClient()
    expected_files = len(iter_code_files(tmp_path, config))

    original_read_bytes = Path.read_bytes
    calls = {"count": 0}

    def counting_read_bytes(self: Path):
        calls["count"] += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    build_embedding_sync(
        tmp_path,
        config,
        client=client,
        cache_path=default_embedding_cache_path(tmp_path),
        batch_size=2,
    )

    assert calls["count"] == expected_files


def test_build_embedding_sync_wraps_batches_in_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_codebase(tmp_path)
    config = Config(
        ignore_patterns=[".git", "node_modules"],
        embedding_timeout_seconds=12.5,
    )
    client = FakeEmbeddingClient()
    observed_timeouts: list[float | None] = []
    call_count = {"count": 0}

    async def fake_wait_for(coro, timeout=None):
        observed_timeouts.append(timeout)
        call_count["count"] += 1
        if call_count["count"] == 1:
            coro.close()
            raise TimeoutError("embedding request timed out")
        return await coro

    monkeypatch.setattr("agent.embeddings.index.asyncio.wait_for", fake_wait_for)

    result = build_embedding_sync(
        tmp_path,
        config,
        client=client,
        cache_path=default_embedding_cache_path(tmp_path),
        batch_size=1,
    )

    assert len(observed_timeouts) == 2
    assert observed_timeouts == [12.5, 12.5]
    assert set(result.updated_files) == {"src/auth/middleware.py"}
    assert len(result.timed_out_files) == 1
    assert result.failed_files == ()
    assert set(result.timed_out_files) == {"src/tests.py"}
    assert result.all_failed_files == ("src/tests.py",)


def test_build_embedding_sync_keeps_successful_batches_when_one_batch_fails(
    tmp_path: Path,
) -> None:
    _write_codebase(tmp_path)
    config = Config(ignore_patterns=[".git", "node_modules"])
    client = PartiallyFailingEmbeddingClient("assert True")

    result = build_embedding_sync(
        tmp_path,
        config,
        client=client,
        cache_path=default_embedding_cache_path(tmp_path),
        batch_size=1,
    )

    assert result.updated_files == ("src/auth/middleware.py",)
    assert result.failed_files == ("src/tests.py",)
    assert result.chunk_count == 1
    assert set(EmbeddingCache.load(result.cache_path).files) == {
        "src/auth/middleware.py",
    }


def test_build_embedding_sync_does_not_overwrite_cache_for_failed_batch(
    tmp_path: Path,
) -> None:
    _write_codebase(tmp_path)
    config = Config(ignore_patterns=[".git", "node_modules"])
    cache_path = default_embedding_cache_path(tmp_path)

    initial = build_embedding_sync(
        tmp_path,
        config,
        client=FakeEmbeddingClient(),
        cache_path=cache_path,
        batch_size=1,
    )
    cached_before = EmbeddingCache.load(initial.cache_path)
    old_record = cached_before.files["src/tests.py"]

    (tmp_path / "src" / "tests.py").write_text(
        "def test_rate_limit():\n    assert False\n",
        encoding="utf-8",
    )
    result = build_embedding_sync(
        tmp_path,
        config,
        client=PartiallyFailingEmbeddingClient("assert False"),
        cache_path=cache_path,
        batch_size=1,
    )

    cached_after = EmbeddingCache.load(result.cache_path)
    assert result.updated_files == ()
    assert result.failed_files == ("src/tests.py",)
    assert cached_after.files["src/tests.py"] == old_record


def test_vector_store_ranks_closest_chunk() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.9, 0.1),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                ),
            ),
            FileEmbeddingRecord(
                path="tests/test_rate.py",
                sha256="b",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.1, 0.9),
                        start_line=1,
                        end_line=2,
                        content_hash="y",
                    ),
                ),
            ),
        ]
    )

    results = store.search((0.95, 0.05))

    assert results[0].path == "auth/middleware.py"
    assert results[0].score > results[1].score


def test_vector_store_search_returns_best_chunk_metadata() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.0, 1.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                    ChunkEmbedding(
                        index=1,
                        vector=(1.0, 0.0),
                        start_line=3,
                        end_line=4,
                        content_hash="y",
                    ),
                ),
            ),
            FileEmbeddingRecord(
                path="tests/test_rate.py",
                sha256="b",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.0, 1.0),
                        start_line=1,
                        end_line=2,
                        content_hash="z",
                    ),
                ),
            ),
        ]
    )

    results = store.search((1.0, 0.0))
    scores = store.semantic_scores((1.0, 0.0))

    assert results[0].path == "auth/middleware.py"
    assert results[0].score == scores["auth/middleware.py"]
    assert results[0].chunk_index == 1
    assert results[0].start_line == 3


def test_vector_store_search_ties_use_lowest_chunk_index() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=2,
                        vector=(1.0, 0.0),
                        start_line=5,
                        end_line=6,
                        content_hash="later",
                    ),
                    ChunkEmbedding(
                        index=1,
                        vector=(1.0, 0.0),
                        start_line=3,
                        end_line=4,
                        content_hash="earlier",
                    ),
                ),
            ),
        ]
    )

    results = store.search((1.0, 0.0))

    assert results[0].chunk_index == 1
    assert results[0].start_line == 3


def test_vector_store_rejects_wrong_query_dimension() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(1.0, 0.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                ),
            ),
        ]
    )

    with pytest.raises(ValueError, match="dimension"):
        store.search((1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="dimension"):
        store.semantic_scores((1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="dimension"):
        store.score_path("auth/middleware.py", (1.0, 0.0, 0.0))


def test_vector_store_semantic_scores_use_best_chunk() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.0, 1.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                    ChunkEmbedding(
                        index=1,
                        vector=(1.0, 0.0),
                        start_line=3,
                        end_line=4,
                        content_hash="y",
                    ),
                ),
            ),
            FileEmbeddingRecord(
                path="tests/test_rate.py",
                sha256="b",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(0.0, 1.0),
                        start_line=1,
                        end_line=2,
                        content_hash="z",
                    ),
                ),
            ),
        ]
    )

    scores = store.semantic_scores((1.0, 0.0))

    assert scores["auth/middleware.py"] == 1.0
    assert scores["tests/test_rate.py"] == 0.0


def test_vector_store_builds_numpy_index_lazily() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(1.0, 0.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                ),
            ),
        ]
    )

    build_calls = 0
    original = store._build_numpy_index

    def counting_build():
        nonlocal build_calls
        build_calls += 1
        return original()

    store._build_numpy_index = counting_build  # type: ignore[method-assign]

    store.semantic_scores((1.0, 0.0))
    store.semantic_scores((1.0, 0.0))
    assert build_calls == 1

    store.add(
        FileEmbeddingRecord(
            path="auth/middleware.py",
            sha256="b",
            chunks=(
                ChunkEmbedding(
                    index=0,
                    vector=(0.0, 1.0),
                    start_line=1,
                    end_line=2,
                    content_hash="y",
                ),
            ),
        )
    )
    scores = store.semantic_scores((0.0, 1.0))

    assert build_calls == 2
    assert scores["auth/middleware.py"] == 1.0


def test_vector_store_add_batches_pending_records_for_warm_numpy_index() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(1.0, 0.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                ),
            ),
        ]
    )

    assert store.semantic_scores((1.0, 0.0))["auth/middleware.py"] == 1.0

    store.add(
        FileEmbeddingRecord(
            path="tests/test_rate.py",
            sha256="b",
            chunks=(
                ChunkEmbedding(
                    index=0,
                    vector=(0.0, 1.0),
                    start_line=1,
                    end_line=2,
                    content_hash="y",
                ),
            ),
        )
    )
    store.add(
        FileEmbeddingRecord(
            path="docs/empty.md",
            sha256="c",
            chunks=(),
        )
    )

    scores = store.semantic_scores((0.0, 1.0))

    assert scores["tests/test_rate.py"] == 1.0
    assert scores["auth/middleware.py"] == 0.0
    assert scores["docs/empty.md"] == 0.0


def test_vector_store_add_rejects_mismatched_dimensions_for_warm_index() -> None:
    store = VectorStore(
        records=[
            FileEmbeddingRecord(
                path="auth/middleware.py",
                sha256="a",
                chunks=(
                    ChunkEmbedding(
                        index=0,
                        vector=(1.0, 0.0),
                        start_line=1,
                        end_line=2,
                        content_hash="x",
                    ),
                ),
            ),
        ]
    )
    store.semantic_scores((1.0, 0.0))

    store.add(
        FileEmbeddingRecord(
            path="tests/test_rate.py",
            sha256="b",
            chunks=(
                ChunkEmbedding(
                    index=0,
                    vector=(1.0, 0.0, 0.0),
                    start_line=1,
                    end_line=2,
                    content_hash="y",
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="dimension"):
        store.semantic_scores((1.0, 0.0))
