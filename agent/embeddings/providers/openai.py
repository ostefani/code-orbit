from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypedDict

from httpx import AsyncClient, Timeout, URL
from openai import NOT_GIVEN, NotGiven

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

            timeout: float | Timeout | NotGiven | None = NOT_GIVEN
            if "timeout" in openai_options:
                timeout = openai_options["timeout"]

            max_retries = 2
            if "max_retries" in openai_options:
                max_retries = openai_options["max_retries"]

            default_headers: Mapping[str, str] | None = None
            if "default_headers" in openai_options:
                default_headers = openai_options["default_headers"]

            default_query: Mapping[str, object] | None = None
            if "default_query" in openai_options:
                default_query = openai_options["default_query"]

            http_client: AsyncClient | None = None
            if "http_client" in openai_options:
                http_client = openai_options["http_client"]

            strict_response_validation = False
            if "_strict_response_validation" in openai_options:
                strict_response_validation = openai_options[
                    "_strict_response_validation"
                ]

            organization: str | None = None
            if "organization" in openai_options:
                organization = openai_options["organization"]

            project: str | None = None
            if "project" in openai_options:
                project = openai_options["project"]

            webhook_secret: str | None = None
            if "webhook_secret" in openai_options:
                webhook_secret = openai_options["webhook_secret"]

            websocket_base_url: str | URL | None = None
            if "websocket_base_url" in openai_options:
                websocket_base_url = openai_options["websocket_base_url"]

            try:
                client = _AsyncOpenAI(
                    api_key=self._config.api_key,
                    organization=organization,
                    project=project,
                    webhook_secret=webhook_secret,
                    base_url=self._config.api_base,
                    websocket_base_url=websocket_base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                    default_headers=default_headers,
                    default_query=default_query,
                    http_client=http_client,
                    _strict_response_validation=strict_response_validation,
                )
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

    if _AuthenticationError is not None and isinstance(exc, _AuthenticationError):
        return EmbeddingProviderAuthenticationError(provider, str(exc))
    if _RateLimitError is not None and isinstance(exc, _RateLimitError):
        return EmbeddingProviderRateLimitError(provider, str(exc))
    if (
        _APIConnectionError is not None
        and _APITimeoutError is not None
        and _InternalServerError is not None
        and isinstance(exc, (_APIConnectionError, _APITimeoutError, _InternalServerError))
    ):
        return EmbeddingProviderUnavailableError(provider, str(exc))
    if (
        _BadRequestError is not None
        and _ConflictError is not None
        and _NotFoundError is not None
        and _PermissionDeniedError is not None
        and _APIError is not None
        and isinstance(
            exc,
            (
                _BadRequestError,
                _ConflictError,
                _NotFoundError,
                _PermissionDeniedError,
                _APIError,
            ),
        )
    ):
        return EmbeddingProviderRequestError(provider, str(exc))
    if _APIStatusError is not None and isinstance(exc, _APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code is not None and int(status_code) >= 500:
            return EmbeddingProviderUnavailableError(provider, str(exc))
        return EmbeddingProviderRequestError(provider, str(exc))
    return EmbeddingProviderRequestError(provider, str(exc))


def _coerce_openai_options(
    provider: str, options: Mapping[str, object]
) -> OpenAIClientOptions:
    unknown_keys = set(options) - (
        RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS
        | {
            "timeout",
            "max_retries",
            "default_headers",
            "default_query",
            "http_client",
            "_strict_response_validation",
            "organization",
            "project",
            "webhook_secret",
            "websocket_base_url",
        }
    )
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise EmbeddingProviderConfigurationError(
            provider,
            f"Unsupported OpenAI embedding option(s): {keys}.",
        )

    result: OpenAIClientOptions = {}

    if "timeout" in options:
        timeout = options["timeout"]
        if timeout is None or isinstance(timeout, (int, float, Timeout)):
            result["timeout"] = timeout
        else:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'timeout' must be a number, httpx.Timeout, or null.",
            )

    if "max_retries" in options:
        max_retries = options["max_retries"]
        if isinstance(max_retries, int) and not isinstance(max_retries, bool):
            result["max_retries"] = max_retries
        else:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'max_retries' must be an integer.",
            )

    if "default_headers" in options:
        default_headers = options["default_headers"]
        if isinstance(default_headers, Mapping):
            headers: dict[str, str] = {}
            for header_key, header_value in default_headers.items():
                if not isinstance(header_key, str) or not isinstance(
                    header_value, str
                ):
                    raise EmbeddingProviderConfigurationError(
                        provider,
                        "OpenAI embedding option 'default_headers' must map strings to strings.",
                    )
                headers[header_key] = header_value
            result["default_headers"] = headers
        elif default_headers is not None:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'default_headers' must be a mapping or null.",
            )

    if "default_query" in options:
        default_query = options["default_query"]
        if isinstance(default_query, Mapping):
            query: dict[str, object] = {}
            for query_key, query_value in default_query.items():
                if not isinstance(query_key, str):
                    raise EmbeddingProviderConfigurationError(
                        provider,
                        "OpenAI embedding option 'default_query' must use string keys.",
                    )
                query[query_key] = query_value
            result["default_query"] = query
        elif default_query is not None:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'default_query' must be a mapping or null.",
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

    for key in ("organization", "project", "webhook_secret"):
        if key in options:
            value = options[key]
            if isinstance(value, str):
                result[key] = value
            elif value is not None:
                raise EmbeddingProviderConfigurationError(
                    provider,
                    f"OpenAI embedding option '{key}' must be a string or null.",
                )

    if "websocket_base_url" in options:
        websocket_base_url = options["websocket_base_url"]
        if isinstance(websocket_base_url, (str, URL)):
            result["websocket_base_url"] = websocket_base_url
        elif websocket_base_url is not None:
            raise EmbeddingProviderConfigurationError(
                provider,
                "OpenAI embedding option 'websocket_base_url' must be a string, httpx.URL, or null.",
            )

    return result
