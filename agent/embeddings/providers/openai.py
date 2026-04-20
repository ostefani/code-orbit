from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, TYPE_CHECKING

from ..adapters import (
    EmbeddingProviderAuthenticationError,
    EmbeddingProviderError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderRequestError,
    EmbeddingProviderUnavailableError,
    RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS,
)
from ..adapters import EmbeddingProviderConfig


if TYPE_CHECKING:
    from openai import APIConnectionError as _APIConnectionError
    from openai import APIError as _APIError
    from openai import APIStatusError as _APIStatusError
    from openai import APITimeoutError as _APITimeoutError
    from openai import AuthenticationError as _AuthenticationError
    from openai import BadRequestError as _BadRequestError
    from openai import ConflictError as _ConflictError
    from openai import InternalServerError as _InternalServerError
    from openai import NotFoundError as _NotFoundError
    from openai import PermissionDeniedError as _PermissionDeniedError
    from openai import RateLimitError as _RateLimitError


try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    from openai import APIConnectionError as _APIConnectionError
    from openai import APIError as _APIError
    from openai import APIStatusError as _APIStatusError
    from openai import APITimeoutError as _APITimeoutError
    from openai import AuthenticationError as _AuthenticationError
    from openai import BadRequestError as _BadRequestError
    from openai import ConflictError as _ConflictError
    from openai import InternalServerError as _InternalServerError
    from openai import NotFoundError as _NotFoundError
    from openai import PermissionDeniedError as _PermissionDeniedError
    from openai import RateLimitError as _RateLimitError
except ImportError:  # pragma: no cover - optional dependency path
    _OPENAI_AVAILABLE = False
    _AsyncOpenAI = None
    _APIConnectionError = None
    _APIError = None
    _APIStatusError = None
    _APITimeoutError = None
    _AuthenticationError = None
    _BadRequestError = None
    _ConflictError = None
    _InternalServerError = None
    _NotFoundError = None
    _PermissionDeniedError = None
    _RateLimitError = None
else:
    _OPENAI_AVAILABLE = True


class _EmbeddingVectorsAPI(Protocol):
    async def create(self, *, model: str, input: list[str]) -> Any: ...


class _EmbeddingClientAPI(Protocol):
    @property
    def embeddings(self) -> _EmbeddingVectorsAPI: ...

    async def close(self) -> None: ...


class OpenAIEmbeddingAdapter:
    provider_name: ClassVar[str] = "openai"

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        self._config = config
        self._client: _EmbeddingClientAPI | None = None

    def _get_client(self) -> _EmbeddingClientAPI:
        client = self._client
        if client is None:
            if not _OPENAI_AVAILABLE or _AsyncOpenAI is None:
                raise EmbeddingProviderUnavailableError(
                    self.provider_name,
                    "The 'openai' package is required for the OpenAI embedding provider.",
                )

            client_kwargs = {
                "base_url": self._config.api_base,
                "api_key": self._config.api_key,
            }
            client_kwargs.update(
                {
                    key: value
                    for key, value in self._config.options.items()
                    if key not in RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS
                }
            )

            try:
                client = _AsyncOpenAI(**client_kwargs)
            except Exception as exc:  # pragma: no cover - constructor validation
                if isinstance(exc, EmbeddingProviderError):
                    raise
                raise _map_openai_exception(self.provider_name, exc) from exc
            self._client = client
        return client

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self._config.model,
                input=list(texts),
            )
        except Exception as exc:
            raise _map_openai_exception(self.provider_name, exc) from exc
        return [tuple(item.embedding) for item in response.data]

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return

        await client.close()
        self._client = None

    async def validate(self) -> None:
        self._get_client()

    async def probe(self) -> None:
        client = self._get_client()
        try:
            await client.embeddings.create(
                model=self._config.model,
                input=["probe"],
            )
        except Exception as exc:
            raise _map_openai_exception(self.provider_name, exc) from exc


def _map_openai_exception(provider: str, exc: Exception) -> EmbeddingProviderError:
    if not _OPENAI_AVAILABLE:
        return EmbeddingProviderRequestError(provider, str(exc))

    if isinstance(exc, _AuthenticationError):
        return EmbeddingProviderAuthenticationError(provider, str(exc))
    if isinstance(exc, _RateLimitError):
        return EmbeddingProviderRateLimitError(provider, str(exc))
    if isinstance(exc, (_APIConnectionError, _APITimeoutError, _InternalServerError)):
        return EmbeddingProviderUnavailableError(provider, str(exc))
    if isinstance(
        exc,
        (
            _BadRequestError,
            _ConflictError,
            _NotFoundError,
            _PermissionDeniedError,
            _APIError,
        ),
    ):
        return EmbeddingProviderRequestError(provider, str(exc))
    if isinstance(exc, _APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code is not None and int(status_code) >= 500:
            return EmbeddingProviderUnavailableError(provider, str(exc))
        return EmbeddingProviderRequestError(provider, str(exc))
    return EmbeddingProviderRequestError(provider, str(exc))
