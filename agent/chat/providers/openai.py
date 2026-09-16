from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypedDict, cast

from httpx import Timeout, URL
from openai import NotGiven
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
from ...openai_shared import (
    BASE_OPENAI_OPTION_KEYS,
    OpenAIErrorKind,
    OpenAIExceptionTypes,
    base_openai_client_kwargs,
    classify_openai_exception,
    coerce_openai_client_options,
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

            try:
                client = _AsyncOpenAI(
                    **base_openai_client_kwargs(
                        openai_options,
                        api_key=self._config.api_key.get_secret_value(),
                        base_url=self._config.api_base,
                    )
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


def _map_openai_exception(provider: str, exc: Exception) -> ProviderError:
    kind = classify_openai_exception(exc, _exception_types())
    message = str(exc)
    if kind is OpenAIErrorKind.RATE_LIMIT:
        return ProviderRateLimitError(provider, message)
    if kind is OpenAIErrorKind.AUTHENTICATION:
        return ProviderAuthenticationError(provider, message)
    if kind is OpenAIErrorKind.UNAVAILABLE:
        return ProviderUnavailableError(provider, message)
    return ProviderRequestError(provider, message)


def _coerce_openai_options(
    provider: str,
    options: Mapping[str, object],
) -> OpenAIClientOptions:
    return cast(
        OpenAIClientOptions,
        coerce_openai_client_options(
            provider,
            options,
            label="chat",
            error_cls=ProviderConfigurationError,
            allowed_keys=BASE_OPENAI_OPTION_KEYS,
        ),
    )


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
