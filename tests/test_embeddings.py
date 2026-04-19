from pathlib import Path

from agent.config import Config
from agent.embeddings import (
    EmbeddingCache,
    FileEmbeddingRecord,
    ChunkEmbedding,
    VectorStore,
    build_embedding_sync,
    default_embedding_cache_path,
    hash_file,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    def embed(self, texts):
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
    assert set(result.updated_files) == {
        "src/auth/middleware.py",
        "src/tests.py",
    }
    assert len(client.requests) == 1
    assert result.chunk_count == 2

    cache = EmbeddingCache.load(result.cache_path)
    assert cache.files["src/auth/middleware.py"].sha256 == hash_file(
        tmp_path / "src" / "auth" / "middleware.py"
    )

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
