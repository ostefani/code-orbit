from .cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_FILENAME,
    LEGACY_CACHE_FILENAME,
    EmbeddingCache,
    default_embedding_cache_path,
)
from .chunking import DEFAULT_CHUNK_CHAR_LIMIT, chunk_file
from .client import EmbeddingClient, OpenAICompatibleEmbeddingClient
from .discovery import iter_code_files
from .index import EmbeddingSyncResult, build_embedding_index, build_embedding_sync
from .store import VectorStore
from .types import CodeChunk, ChunkEmbedding, EmbeddingSearchResult, FileEmbeddingRecord

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_FILENAME",
    "LEGACY_CACHE_FILENAME",
    "DEFAULT_CHUNK_CHAR_LIMIT",
    "EmbeddingCache",
    "EmbeddingClient",
    "EmbeddingSearchResult",
    "EmbeddingSyncResult",
    "FileEmbeddingRecord",
    "CodeChunk",
    "ChunkEmbedding",
    "OpenAICompatibleEmbeddingClient",
    "VectorStore",
    "build_embedding_index",
    "build_embedding_sync",
    "chunk_file",
    "default_embedding_cache_path",
    "iter_code_files",
]
