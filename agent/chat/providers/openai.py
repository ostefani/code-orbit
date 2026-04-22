from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypedDict, cast

from httpx import Timeout, URL
from openai import NOT_GIVEN, NotGiven
from collections.abc import AsyncGenerator
from ..adapters import ChatProviderConfig
from ..errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from ..types import (
    AdapterCapabilities,
    ChatDelta,
    ChatGenerationSettings,
    ChatMessage,
    ChatResponse,
    ChatUsage,
)

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


class _ChatCompletionMessageAPI(Protocol):
    content: str | None


class _ChatCompletionDeltaAPI(Protocol):
    content: str | None


class _ChatCompletionChoiceAPI(Protocol):
    message: _ChatCompletionMessageAPI
    delta: _ChatCompletionDeltaAPI
    finish_reason: str | None


class _ChatUsageAPI(Protocol):
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class _ChatResponseAPI(Protocol):
    choices: Sequence[_ChatCompletionChoiceAPI]
    usage: _ChatUsageAPI | None


class _ChatStreamEventAPI(Protocol):
    choices: Sequence[_ChatCompletionChoiceAPI]


class OpenAIClientOptions(TypedDict, total=False):
    timeout: float | Timeout | NotGiven | None
    max_retries: int
    default_headers: Mapping[str, str] | None
    default_query: Mapping[str, object] | None
    organization: str | None
    project: str | None
    webhook_secret: str | None
    websocket_base_url: str | URL | None


class _BaseCreateParams(TypedDict):
    model: str
    messages: list[dict[str, str]]


class _NonStreamCreateParams(_BaseCreateParams, total=False):
    max_tokens: int
    temperature: float
    response_format: dict[str, str]


class _StreamCreateParams(_BaseCreateParams, total=False):
    max_tokens: int
    temperature: float
    response_format: dict[str, str]
    stream: bool


class OpenAIChatAdapter:
    provider_name: ClassVar[str] = "openai"
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        chat=True,
        streaming=True,
        embeddings=False,
        reranking=False,
    )

    def __init__(self, config: ChatProviderConfig) -> None:
        self._config = config
        self.context_window = config.context_window
        self._client: Any | None = None

    def _get_client(self) -> Any:
        client = self._client
        if client is None:
            if not _OPENAI_AVAILABLE or _AsyncOpenAI is None:
                raise ProviderUnavailableError(
                    self.provider_name,
                    "The 'openai' package is required for the OpenAI chat provider.",
                )

            openai_options = _coerce_openai_options(
                self.provider_name,
                self._config.options,
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
                    api_key=self._config.api_key.get_secret_value(),
                    organization=organization,
                    project=project,
                    webhook_secret=webhook_secret,
                    base_url=self._config.api_base,
                    websocket_base_url=websocket_base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                    default_headers=default_headers,
                    default_query=default_query,
                )
            except Exception as exc:  # pragma: no cover - constructor validation
                if isinstance(exc, ProviderError):
                    raise
                raise _map_openai_exception(self.provider_name, exc) from exc

            self._client = client
        return client

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> ChatResponse:
        client = self._get_client()
        request = _build_non_stream_request(self._config.model, messages, generation)

        try:
            response = cast(
                _ChatResponseAPI,
                await client.chat.completions.create(**request),
            )
        except Exception as exc:
            raise _map_openai_exception(self.provider_name, exc) from exc

        return _coerce_chat_response(response)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> AsyncGenerator[ChatDelta, None]:
        client = self._get_client()
        request = _build_stream_request(self._config.model, messages, generation)

        try:
            stream = await client.chat.completions.create(**request)
            async for event in cast(AsyncIterator[_ChatStreamEventAPI], stream):
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content
                if isinstance(delta, str) and delta:
                    yield ChatDelta(content=delta)
        except Exception as exc:
            raise _map_openai_exception(self.provider_name, exc) from exc

    async def validate(self) -> None:
        _validate_local_configuration(self._config)
        if not _OPENAI_AVAILABLE or _AsyncOpenAI is None:
            raise ProviderUnavailableError(
                self.provider_name,
                "The 'openai' package is required for the OpenAI chat provider.",
            )

    async def probe(self) -> None:
        client = self._get_client()
        try:
            await client.models.list()
        except Exception as exc:
            raise _map_openai_exception(self.provider_name, exc) from exc

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return
        await client.close()
        self._client = None


def _build_non_stream_request(
    model: str,
    messages: Sequence[ChatMessage],
    generation: ChatGenerationSettings | None,
) -> _NonStreamCreateParams:
    request: _NonStreamCreateParams = {
        "model": model,
        "messages": _serialize_messages(messages),
    }
    if generation is not None:
        if generation.max_tokens is not None:
            request["max_tokens"] = generation.max_tokens
        if generation.temperature is not None:
            request["temperature"] = generation.temperature
        if generation.response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
    return request


def _build_stream_request(
    model: str,
    messages: Sequence[ChatMessage],
    generation: ChatGenerationSettings | None,
) -> _StreamCreateParams:
    request: _StreamCreateParams = {
        "model": model,
        "messages": _serialize_messages(messages),
        "stream": True,
    }
    if generation is not None:
        if generation.max_tokens is not None:
            request["max_tokens"] = generation.max_tokens
        if generation.temperature is not None:
            request["temperature"] = generation.temperature
        if generation.response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
    return request


def _serialize_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _coerce_chat_response(response: _ChatResponseAPI) -> ChatResponse:
    if not response.choices:
        raise ProviderRequestError("openai", "The chat provider returned no choices.")

    choice = response.choices[0]
    message_content = choice.message.content or ""
    usage = _coerce_chat_usage(response.usage)
    return ChatResponse(
        content=message_content,
        finish_reason=choice.finish_reason,
        usage=usage,
    )


def _coerce_chat_usage(usage: _ChatUsageAPI | None) -> ChatUsage | None:
    if usage is None:
        return None
    return ChatUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _map_openai_exception(provider: str, exc: Exception) -> ProviderError:
    if not _OPENAI_AVAILABLE:
        return ProviderRequestError(provider, str(exc))

    if _AuthenticationError is not None and isinstance(exc, _AuthenticationError):
        return ProviderAuthenticationError(provider, str(exc))
    if _RateLimitError is not None and isinstance(exc, _RateLimitError):
        return ProviderRateLimitError(provider, str(exc))
    if (
        _APIConnectionError is not None
        and _APITimeoutError is not None
        and _InternalServerError is not None
        and isinstance(
            exc, (_APIConnectionError, _APITimeoutError, _InternalServerError)
        )
    ):
        return ProviderUnavailableError(provider, str(exc))
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
        return ProviderRequestError(provider, str(exc))
    if _APIStatusError is not None and isinstance(exc, _APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return ProviderRateLimitError(provider, str(exc))
        if status_code in {401, 403}:
            return ProviderAuthenticationError(provider, str(exc))
        if isinstance(status_code, int) and status_code >= 500:
            return ProviderUnavailableError(provider, str(exc))
        return ProviderRequestError(provider, str(exc))
    return ProviderRequestError(provider, str(exc))


def _coerce_openai_options(
    provider: str,
    options: Mapping[str, object],
) -> OpenAIClientOptions:
    result: OpenAIClientOptions = {}

    allowed_keys = {
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "organization",
        "project",
        "webhook_secret",
        "websocket_base_url",
    }
    unexpected = sorted(key for key in options if key not in allowed_keys)
    if unexpected:
        raise ProviderConfigurationError(
            provider,
            "Unsupported OpenAI chat options: "
            + ", ".join(repr(key) for key in unexpected),
        )

    if "timeout" in options:
        timeout = options["timeout"]
        if timeout is None or isinstance(timeout, (int, float, Timeout)):
            result["timeout"] = timeout
        else:
            raise ProviderConfigurationError(
                provider,
                "OpenAI chat option 'timeout' must be a number, httpx.Timeout, or null.",
            )

    if "max_retries" in options:
        max_retries = options["max_retries"]
        if isinstance(max_retries, int) and not isinstance(max_retries, bool):
            result["max_retries"] = max_retries
        else:
            raise ProviderConfigurationError(
                provider,
                "OpenAI chat option 'max_retries' must be an integer.",
            )

    if "default_headers" in options:
        default_headers = options["default_headers"]
        if isinstance(default_headers, Mapping):
            headers: dict[str, str] = {}
            for header_key, header_value in default_headers.items():
                if not isinstance(header_key, str) or not isinstance(header_value, str):
                    raise ProviderConfigurationError(
                        provider,
                        "OpenAI chat option 'default_headers' must map strings to strings.",
                    )
                headers[header_key] = header_value
            result["default_headers"] = headers
        elif default_headers is not None:
            raise ProviderConfigurationError(
                provider,
                "OpenAI chat option 'default_headers' must be a mapping or null.",
            )

    if "default_query" in options:
        default_query = options["default_query"]
        if isinstance(default_query, Mapping):
            query: dict[str, object] = {}
            for query_key, query_value in default_query.items():
                if not isinstance(query_key, str):
                    raise ProviderConfigurationError(
                        provider,
                        "OpenAI chat option 'default_query' must use string keys.",
                    )
                query[query_key] = query_value
            result["default_query"] = query
        elif default_query is not None:
            raise ProviderConfigurationError(
                provider,
                "OpenAI chat option 'default_query' must be a mapping or null.",
            )

    for key in ("organization", "project", "webhook_secret"):
        if key in options:
            value = options[key]
            if isinstance(value, str):
                result[key] = value
            elif value is not None:
                raise ProviderConfigurationError(
                    provider,
                    f"OpenAI chat option '{key}' must be a string or null.",
                )

    if "websocket_base_url" in options:
        websocket_base_url = options["websocket_base_url"]
        if isinstance(websocket_base_url, (str, URL)):
            result["websocket_base_url"] = websocket_base_url
        elif websocket_base_url is not None:
            raise ProviderConfigurationError(
                provider,
                "OpenAI chat option 'websocket_base_url' must be a string, httpx.URL, or null.",
            )

    return result


def _validate_local_configuration(config: ChatProviderConfig) -> None:
    if not config.api_base.strip():
        raise ProviderConfigurationError(
            config.provider,
            "chat_api_base must not be empty.",
        )

    api_key = config.api_key.get_secret_value()
    if not api_key.strip():
        raise ProviderConfigurationError(
            config.provider,
            "chat_api_key must not be empty.",
        )

    if not config.model.strip():
        raise ProviderConfigurationError(
            config.provider,
            "chat_model must not be empty.",
        )

    if config.context_window <= 0:
        raise ProviderConfigurationError(
            config.provider,
            "chat_context_window must be greater than zero.",
        )

    _coerce_openai_options(config.provider, config.options)
