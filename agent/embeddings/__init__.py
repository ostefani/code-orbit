from .cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_FILENAME,
    LEGACY_CACHE_FILENAME,
    EmbeddingCache,
    default_embedding_cache_path,
)
from .adapters import (
    EmbeddingAdapter,
    EmbeddingProviderAuthenticationError,
    EmbeddingProviderConfig,
    EmbeddingProviderConfigurationError,
    EmbeddingProviderError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderRequestError,
    EmbeddingProviderUnavailableError,
    UnsupportedEmbeddingProviderError,
)
from .chunking import DEFAULT_CHUNK_CHAR_LIMIT, chunk_file
from .client import EmbeddingClient, OpenAICompatibleEmbeddingClient
from .factory import build_embedding_provider_config, create_embedding_adapter
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
    "EmbeddingAdapter",
    "EmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "EmbeddingSearchResult",
    "EmbeddingSyncResult",
    "EmbeddingProviderAuthenticationError",
    "EmbeddingProviderConfig",
    "EmbeddingProviderConfigurationError",
    "EmbeddingProviderError",
    "EmbeddingProviderRateLimitError",
    "EmbeddingProviderRequestError",
    "EmbeddingProviderUnavailableError",
    "UnsupportedEmbeddingProviderError",
    "FileEmbeddingRecord",
    "CodeChunk",
    "ChunkEmbedding",
    "VectorStore",
    "build_embedding_provider_config",
    "build_embedding_index",
    "build_embedding_sync",
    "chunk_file",
    "default_embedding_cache_path",
    "create_embedding_adapter",
    "iter_code_files",
]
