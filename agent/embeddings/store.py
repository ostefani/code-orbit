from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np

from .adapters import EmbeddingAdapter
from .types import EmbeddingSearchResult, FileEmbeddingRecord


@dataclass(frozen=True)
class _VectorStoreNumpyIndex:
    record_paths: tuple[str, ...]
    record_sha256s: tuple[str, ...]
    chunk_matrix: np.ndarray
    chunk_path_ids: np.ndarray
    chunk_norms: np.ndarray
    chunk_indexes: np.ndarray
    chunk_start_lines: np.ndarray
    chunk_end_lines: np.ndarray


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
        record_sha256s: list[str] = []
        chunk_vectors: list[Sequence[float]] = []
        chunk_path_ids: list[int] = []
        chunk_indexes: list[int] = []
        chunk_start_lines: list[int] = []
        chunk_end_lines: list[int] = []

        for record_index, record in enumerate(self.records):
            record_paths.append(record.path)
            record_sha256s.append(record.sha256)
            for chunk in sorted(record.chunks, key=lambda item: item.index):
                chunk_vectors.append(chunk.vector)
                chunk_path_ids.append(record_index)
                chunk_indexes.append(chunk.index)
                chunk_start_lines.append(chunk.start_line)
                chunk_end_lines.append(chunk.end_line)

        if not chunk_vectors:
            chunk_matrix = np.empty((0, 0), dtype=np.float64)
            chunk_norms = np.empty((0,), dtype=np.float64)
            chunk_path_ids_array = np.empty((0,), dtype=np.int64)
            chunk_indexes_array = np.empty((0,), dtype=np.int64)
            chunk_start_lines_array = np.empty((0,), dtype=np.int64)
            chunk_end_lines_array = np.empty((0,), dtype=np.int64)
        else:
            chunk_matrix = np.asarray(chunk_vectors, dtype=np.float64)
            chunk_norms = np.linalg.norm(chunk_matrix, axis=1)
            chunk_path_ids_array = np.asarray(chunk_path_ids, dtype=np.int64)
            chunk_indexes_array = np.asarray(chunk_indexes, dtype=np.int64)
            chunk_start_lines_array = np.asarray(chunk_start_lines, dtype=np.int64)
            chunk_end_lines_array = np.asarray(chunk_end_lines, dtype=np.int64)

        return _VectorStoreNumpyIndex(
            record_paths=tuple(record_paths),
            record_sha256s=tuple(record_sha256s),
            chunk_matrix=chunk_matrix,
            chunk_path_ids=chunk_path_ids_array,
            chunk_norms=chunk_norms,
            chunk_indexes=chunk_indexes_array,
            chunk_start_lines=chunk_start_lines_array,
            chunk_end_lines=chunk_end_lines_array,
        )

    def _chunk_cosines(self, query_vector: Sequence[float]) -> tuple[
        _VectorStoreNumpyIndex,
        np.ndarray,
    ]:
        index = self._ensure_numpy_index()
        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-10:
            return index, np.zeros(len(index.chunk_path_ids), dtype=np.float64)

        if index.chunk_matrix.size == 0:
            return index, np.zeros(len(index.chunk_path_ids), dtype=np.float64)

        if index.chunk_matrix.shape[1] != len(query):
            raise ValueError(
                "Query vector dimension does not match indexed embeddings: "
                f"got {len(query)}, expected {index.chunk_matrix.shape[1]}."
            )

        dot_products = index.chunk_matrix @ query
        valid = index.chunk_norms > 1e-10
        cosines = np.zeros(len(index.chunk_path_ids), dtype=np.float64)
        cosines[valid] = dot_products[valid] / (index.chunk_norms[valid] * query_norm)
        return index, np.maximum(cosines, 0.0)

    def semantic_scores(self, query_vector: Sequence[float]) -> dict[str, float]:
        index, cosines = self._chunk_cosines(query_vector)

        scores = np.zeros(len(index.record_paths), dtype=np.float64)
        if len(cosines):
            np.maximum.at(scores, index.chunk_path_ids, cosines)
        return {path: float(score) for path, score in zip(index.record_paths, scores)}

    def score_path(self, path: str, query_vector: Sequence[float]) -> float:
        record = self._records.get(path)
        if record is None or not record.chunks:
            return 0.0

        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-10:
            return 0.0

        chunk_matrix = np.asarray(
            [chunk.vector for chunk in record.chunks],
            dtype=np.float64,
        )
        if chunk_matrix.shape[1] != len(query):
            raise ValueError(
                "Query vector dimension does not match indexed embeddings: "
                f"got {len(query)}, expected {chunk_matrix.shape[1]}."
            )

        chunk_norms = np.linalg.norm(chunk_matrix, axis=1)
        dot_products = chunk_matrix @ query
        valid = chunk_norms > 1e-10
        cosines = np.zeros(len(record.chunks), dtype=np.float64)
        cosines[valid] = dot_products[valid] / (chunk_norms[valid] * query_norm)
        return float(np.max(np.maximum(cosines, 0.0)))

    def search(
        self, query_vector: Sequence[float], top_k: int = 10
    ) -> list[EmbeddingSearchResult]:
        index, cosines = self._chunk_cosines(query_vector)
        if len(cosines) == 0:
            return []

        scores = np.full(len(index.record_paths), -np.inf, dtype=np.float64)
        np.maximum.at(scores, index.chunk_path_ids, cosines)

        best_chunk_positions = np.full(len(index.record_paths), -1, dtype=np.int64)
        order = np.lexsort(
            (
                index.chunk_indexes,
                -cosines,
                index.chunk_path_ids,
            )
        )
        record_ids, first_positions = np.unique(
            index.chunk_path_ids[order],
            return_index=True,
        )
        best_chunk_positions[record_ids] = order[first_positions]

        results: list[EmbeddingSearchResult] = []
        for record_index, chunk_position in enumerate(best_chunk_positions):
            if chunk_position == -1:
                continue
            results.append(
                EmbeddingSearchResult(
                    path=index.record_paths[record_index],
                    score=float(scores[record_index]),
                    chunk_index=int(index.chunk_indexes[chunk_position]),
                    start_line=int(index.chunk_start_lines[chunk_position]),
                    end_line=int(index.chunk_end_lines[chunk_position]),
                    sha256=index.record_sha256s[record_index],
                )
            )

        results.sort(key=lambda item: (-item.score, item.path, item.chunk_index))
        return results[:top_k]

    async def search_text(
        self,
        text: str,
        client: EmbeddingAdapter,
        top_k: int = 10,
    ) -> list[EmbeddingSearchResult]:
        vector = (await client.embed([text]))[0]
        return self.search(vector, top_k=top_k)
