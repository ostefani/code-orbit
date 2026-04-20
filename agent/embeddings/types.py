from dataclasses import dataclass


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
