from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypedDict, cast

from httpx import AsyncClient, Timeout, URL
from openai import NotGiven

from ...openai_shared import (
    BASE_OPENAI_OPTION_KEYS,
    OpenAIErrorKind,
    OpenAIExceptionTypes,
    base_openai_client_kwargs,
    classify_openai_exception,
    coerce_openai_client_options,
)
from ..adapters import (
    EmbeddingProviderAuthenticationError,
    EmbeddingProviderError,
    EmbeddingProviderConfigurationError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderRequestError,
    EmbeddingProviderUnavailableError,
    RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS,
)
from ..adapters import EmbeddingProviderConfig


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


class OpenAIClientOptions(TypedDict, total=False):
    timeout: float | Timeout | NotGiven | None
    max_retries: int
    default_headers: Mapping[str, str] | None
    default_query: Mapping[str, object] | None
    http_client: AsyncClient | None
    _strict_response_validation: bool
    organization: str | None
    project: str | None
    webhook_secret: str | None
    websocket_base_url: str | URL | None


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

            openai_options = _coerce_openai_options(
                self.provider_name, self._config.options
            )

            client_kwargs = base_openai_client_kwargs(
                openai_options,
                api_key=self._config.api_key,
                base_url=self._config.api_base,
            )
            client_kwargs["http_client"] = openai_options.get("http_client")
            client_kwargs["_strict_response_validation"] = openai_options.get(
                "_strict_response_validation", False
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


def _exception_types() -> OpenAIExceptionTypes:
    # Sourced from this module's globals (not the shared module) so tests can
    # keep patching these names with fakes.
    return OpenAIExceptionTypes(
        available=_OPENAI_AVAILABLE,
        authentication_error=_AuthenticationError,
        rate_limit_error=_RateLimitError,
        connection_error=_APIConnectionError,
        timeout_error=_APITimeoutError,
        internal_server_error=_InternalServerError,
        bad_request_error=_BadRequestError,
        conflict_error=_ConflictError,
        not_found_error=_NotFoundError,
        permission_denied_error=_PermissionDeniedError,
        api_error=_APIError,
        status_error=_APIStatusError,
    )


def _map_openai_exception(provider: str, exc: Exception) -> EmbeddingProviderError:
    kind = classify_openai_exception(exc, _exception_types())
    message = str(exc)
    if kind is OpenAIErrorKind.RATE_LIMIT:
        return EmbeddingProviderRateLimitError(provider, message)
    if kind is OpenAIErrorKind.AUTHENTICATION:
        return EmbeddingProviderAuthenticationError(provider, message)
    if kind is OpenAIErrorKind.UNAVAILABLE:
        return EmbeddingProviderUnavailableError(provider, message)
    return EmbeddingProviderRequestError(provider, message)


_EMBEDDING_OPTION_KEYS = (
    RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS
    | BASE_OPENAI_OPTION_KEYS
    | {"http_client", "_strict_response_validation"}
)


def _coerce_openai_options(
    provider: str, options: Mapping[str, object]
) -> OpenAIClientOptions:
    result = cast(
        OpenAIClientOptions,
        coerce_openai_client_options(
            provider,
            options,
            label="embedding",
            error_cls=EmbeddingProviderConfigurationError,
            allowed_keys=_EMBEDDING_OPTION_KEYS,
        ),
    )

    if "http_client" in options:
        http_client = options["http_client"]
        if http_client is None or isinstance(http_client, AsyncClient):
            result["http_client"] = http_client
        else:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'http_client' must be an httpx.AsyncClient or null.",
            )

    if "_strict_response_validation" in options:
        strict_response_validation = options["_strict_response_validation"]
        if isinstance(strict_response_validation, bool):
            result["_strict_response_validation"] = strict_response_validation
        else:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option '_strict_response_validation' must be a boolean.",
            )

    return result
