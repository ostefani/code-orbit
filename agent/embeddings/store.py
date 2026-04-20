import math
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np

from .client import EmbeddingClient
from .types import EmbeddingSearchResult, FileEmbeddingRecord


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

    def search(
        self, query_vector: Sequence[float], top_k: int = 10
    ) -> list[EmbeddingSearchResult]:
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
