from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable


RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS = {"api_base", "base_url", "api_key", "model"}


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Structural adapter contract for embedding providers.

    `@runtime_checkable` only verifies callable members at runtime, so the
    protocol deliberately constrains behavior and leaves provider metadata to
    concrete adapter classes.
    """

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def validate(self) -> None: ...

    async def probe(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    provider: str
    api_base: str
    api_key: str
    model: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


class EmbeddingProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(message)
        self.provider = provider


class UnsupportedEmbeddingProviderError(EmbeddingProviderError):
    pass


class EmbeddingProviderConfigurationError(EmbeddingProviderError):
    pass


class EmbeddingProviderAuthenticationError(EmbeddingProviderError):
    pass


class EmbeddingProviderRateLimitError(EmbeddingProviderError):
    pass


class EmbeddingProviderUnavailableError(EmbeddingProviderError):
    pass


class EmbeddingProviderRequestError(EmbeddingProviderError):
    pass
