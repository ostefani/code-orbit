import asyncio
import types

import pytest

from agent.chat import (
    AdapterCapabilities,
    ChatAdapter,
    ChatDelta,
    ChatGenerationSettings,
    ChatMessage,
    ChatProviderConfig,
    ChatResponse,
    ChatUsage,
    CapabilityNotSupportedError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    UnsupportedChatProviderError,
    build_chat_adapter,
    validate_chat_adapter,
)
from agent.chat import factory as chat_factory
from agent.chat import orchestrator as chat_orchestrator
from agent.chat.providers import openai as openai_provider
from agent.chat.providers.openai import OpenAIChatAdapter
from agent.config import Config


class FakeChatAdapter:
    capabilities = AdapterCapabilities(
        chat=True,
        streaming=True,
        embeddings=False,
        reranking=False,
    )
    provider_name = "fake"

    def __init__(self) -> None:
        self.completed: list[dict[str, object]] = []
        self.streamed: list[dict[str, object]] = []
        self.validates = 0
        self.closed = False
        self.context_window = 8192

    async def complete(
        self,
        messages,
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> ChatResponse:
        self.completed.append(
            {
                "messages": list(messages),
                "generation": generation,
            }
        )
        return ChatResponse(
            content="complete",
            finish_reason="stop",
            usage=ChatUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        )

    async def stream(
        self,
        messages,
        *,
        generation: ChatGenerationSettings | None = None,
    ):
        self.streamed.append(
            {
                "messages": list(messages),
                "generation": generation,
            }
        )
        yield ChatDelta(content="pro")
        yield ChatDelta(content="be")

    async def validate(self) -> None:
        self.validates += 1

    async def aclose(self) -> None:
        self.closed = True


class FakeNonStreamingChatAdapter(FakeChatAdapter):
    capabilities = AdapterCapabilities(
        chat=True,
        streaming=False,
        embeddings=False,
        reranking=False,
    )


async def _assert_chat_adapter_contract(adapter: ChatAdapter) -> None:
    await adapter.validate()
    response = await adapter.complete(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ],
        generation=ChatGenerationSettings(
            max_tokens=32,
            temperature=0.2,
            response_format="json_object",
        ),
    )
    assert response.content
    assert response.finish_reason is not None
    assert response.usage is not None

    chunks: list[str] = []
    async for delta in adapter.stream(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ],
        generation=ChatGenerationSettings(
            max_tokens=32,
            temperature=0.2,
            response_format="json_object",
        ),
    ):
        chunks.append(delta.content)
    assert "".join(chunks)

    await adapter.aclose()
    await adapter.aclose()


def _install_fake_openai(monkeypatch, *, error: Exception | None = None):
    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        def __init__(self, message: str = "api status error", status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(APIError):
        pass

    class AuthenticationError(APIError):
        pass

    class BadRequestError(APIError):
        pass

    class ConflictError(APIError):
        pass

    class InternalServerError(APIError):
        pass

    class NotFoundError(APIError):
        pass

    class PermissionDeniedError(APIError):
        pass

    class RateLimitError(APIError):
        pass

    class AsyncOpenAI:
        instances: list["AsyncOpenAI"] = []
        error: Exception | None = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[dict[str, object]] = []
            self.closed_via: str | None = None
            type(self).instances.append(self)

        @property
        def chat(self):
            owner = self

            class _ChatCompletionsAPI:
                async def create(
                    self,
                    *,
                    model: str,
                    messages: list[dict[str, str]],
                    max_tokens: int | None = None,
                    temperature: float | None = None,
                    response_format: dict[str, str] | None = None,
                    stream: bool = False,
                ):
                    owner.calls.append(
                        {
                            "model": model,
                            "messages": list(messages),
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "response_format": response_format,
                            "stream": stream,
                        }
                    )
                    if owner.error is not None:
                        raise owner.error
                    if stream:
                        return _StreamResponse()
                    return types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(content="complete"),
                                delta=types.SimpleNamespace(content=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=types.SimpleNamespace(
                            prompt_tokens=7,
                            completion_tokens=3,
                            total_tokens=10,
                        ),
                    )

            class _ChatAPI:
                completions = _ChatCompletionsAPI()

            return _ChatAPI()

        async def close(self) -> None:
            self.closed_via = "close"

    class _StreamResponse:
        def __aiter__(self):
            async def _gen():
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="pro"),
                            message=types.SimpleNamespace(content=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="be"),
                            message=types.SimpleNamespace(content=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )

            return _gen()

    AsyncOpenAI.error = error
    monkeypatch.setattr(openai_provider, "_OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider, "_AsyncOpenAI", AsyncOpenAI, raising=False)
    monkeypatch.setattr(openai_provider, "_APIConnectionError", APIConnectionError, raising=False)
    monkeypatch.setattr(openai_provider, "_APIError", APIError, raising=False)
    monkeypatch.setattr(openai_provider, "_APIStatusError", APIStatusError, raising=False)
    monkeypatch.setattr(openai_provider, "_APITimeoutError", APITimeoutError, raising=False)
    monkeypatch.setattr(openai_provider, "_AuthenticationError", AuthenticationError, raising=False)
    monkeypatch.setattr(openai_provider, "_BadRequestError", BadRequestError, raising=False)
    monkeypatch.setattr(openai_provider, "_ConflictError", ConflictError, raising=False)
    monkeypatch.setattr(openai_provider, "_InternalServerError", InternalServerError, raising=False)
    monkeypatch.setattr(openai_provider, "_NotFoundError", NotFoundError, raising=False)
    monkeypatch.setattr(openai_provider, "_PermissionDeniedError", PermissionDeniedError, raising=False)
    monkeypatch.setattr(openai_provider, "_RateLimitError", RateLimitError, raising=False)
    return AsyncOpenAI, AuthenticationError, RateLimitError


def test_chat_adapter_contract_with_fake_adapter() -> None:
    adapter = FakeChatAdapter()

    asyncio.run(_assert_chat_adapter_contract(adapter))

    assert adapter.validates == 1
    assert adapter.completed[0]["generation"].max_tokens == 32
    assert adapter.completed[0]["generation"].response_format == "json_object"
    assert adapter.streamed[0]["generation"].temperature == 0.2
    assert adapter.closed is True


def test_openai_chat_adapter_contract(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    adapter = OpenAIChatAdapter(
        ChatProviderConfig(
            provider="openai",
            api_base="http://chat.example/v1",
            api_key="chat-secret",
            model="chat-model",
            context_window=4096,
            streaming=True,
            options={"timeout": 9},
        )
    )

    asyncio.run(adapter.validate())
    assert len(AsyncOpenAI.instances) == 0

    asyncio.run(_assert_chat_adapter_contract(adapter))

    assert len(AsyncOpenAI.instances) == 1
    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["base_url"] == "http://chat.example/v1"
    assert instance.kwargs["api_key"] == "chat-secret"
    assert instance.kwargs["timeout"] == 9
    assert instance.calls[0]["messages"][0]["role"] == "system"
    assert instance.calls[0]["model"] == "chat-model"
    assert instance.calls[0]["response_format"] == {"type": "json_object"}
    assert instance.calls[1]["stream"] is True
    assert instance.closed_via == "close"


def test_openai_chat_validate_makes_no_network_call(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    adapter = OpenAIChatAdapter(
        ChatProviderConfig(
            provider="openai",
            api_base="http://chat.example/v1",
            api_key="chat-secret",
            model="chat-model",
            context_window=4096,
            streaming=True,
            options={"timeout": 9},
        )
    )

    asyncio.run(adapter.validate())
    assert len(AsyncOpenAI.instances) == 0

    asyncio.run(adapter.probe())
    assert len(AsyncOpenAI.instances) == 1
    assert AsyncOpenAI.instances[0].calls[0]["messages"][0]["content"] == "probe"

    asyncio.run(adapter.aclose())


def test_openai_chat_adapter_reports_unavailable_package(monkeypatch) -> None:
    monkeypatch.setattr(openai_provider, "_OPENAI_AVAILABLE", False, raising=False)
    monkeypatch.setattr(openai_provider, "_AsyncOpenAI", None, raising=False)

    adapter = OpenAIChatAdapter(
        ChatProviderConfig(
            provider="openai",
            api_base="http://chat.example/v1",
            api_key="chat-secret",
            model="chat-model",
            context_window=4096,
            streaming=True,
            options={},
        )
    )

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(adapter.validate())


def test_chat_factory_uses_provider_specific_config(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    config = Config(
        chat_provider="openai",
        chat_api_base="http://chat.example/v1",
        chat_api_key="chat-secret",
        chat_model="chat-model",
        chat_context_window=4096,
        chat_streaming=True,
        chat_provider_options={
            "timeout": 9,
            "api_key": "should-not-clobber",
            "base_url": "should-not-clobber",
        },
    )

    adapter = build_chat_adapter(config)
    assert len(AsyncOpenAI.instances) == 0

    asyncio.run(validate_chat_adapter(adapter))
    assert len(AsyncOpenAI.instances) == 0

    asyncio.run(adapter.probe())
    assert len(AsyncOpenAI.instances) == 1
    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["base_url"] == "http://chat.example/v1"
    assert instance.kwargs["api_key"] == "chat-secret"
    assert instance.kwargs["timeout"] == 9
    assert instance.calls[0]["messages"][0]["content"] == "probe"
    asyncio.run(adapter.aclose())


def test_chat_factory_rejects_missing_api_key() -> None:
    adapter = OpenAIChatAdapter(
        ChatProviderConfig(
            provider="openai",
            api_base="http://chat.example/v1",
            api_key="",
            model="chat-model",
            context_window=4096,
            streaming=True,
            options={},
        )
    )

    with pytest.raises(ProviderConfigurationError):
        asyncio.run(validate_chat_adapter(adapter))


def test_chat_factory_rejects_streaming_mismatch(monkeypatch) -> None:
    monkeypatch.setitem(chat_factory._CHAT_PROVIDER_BUILDERS, "fake", "unused")
    monkeypatch.setattr(
        chat_factory,
        "_load_chat_builder",
        lambda _dotted_path: lambda _provider_config: FakeNonStreamingChatAdapter(),
    )

    config = Config(
        chat_provider="fake",
        chat_api_base="http://chat.example/v1",
        chat_api_key="chat-secret",
        chat_model="chat-model",
        chat_context_window=4096,
        chat_streaming=True,
    )

    try:
        build_chat_adapter(config)
    except CapabilityNotSupportedError as exc:
        assert "streaming" in str(exc)
    else:
        raise AssertionError("Expected streaming capability validation to fail")


def test_chat_factory_rejects_unknown_provider() -> None:
    config = Config(
        chat_provider="anthropic",
        chat_api_base="http://chat.example/v1",
        chat_api_key="chat-secret",
        chat_model="chat-model",
        chat_context_window=4096,
        chat_streaming=True,
    )

    try:
        build_chat_adapter(config)
    except UnsupportedChatProviderError as exc:
        assert "Unsupported chat provider" in str(exc)
    else:
        raise AssertionError("Expected an unsupported provider error")


def test_config_rejects_invalid_chat_context_window() -> None:
    with pytest.raises(ValueError):
        Config(
            chat_provider="openai",
            chat_api_base="http://chat.example/v1",
            chat_api_key="chat-secret",
            chat_model="chat-model",
            chat_context_window=0,
            chat_streaming=True,
        )


def test_chat_factory_rejects_openai_errors(monkeypatch) -> None:
    AsyncOpenAI, AuthenticationError, RateLimitError = _install_fake_openai(monkeypatch)
    AsyncOpenAI.error = AuthenticationError("auth failed")

    config = Config(
        chat_provider="openai",
        chat_api_base="http://chat.example/v1",
        chat_api_key="chat-secret",
        chat_model="chat-model",
        chat_context_window=4096,
        chat_streaming=True,
    )

    adapter = build_chat_adapter(config)

    with pytest.raises(ProviderAuthenticationError):
        asyncio.run(
            adapter.complete(
                [ChatMessage(role="user", content="hello")],
                generation=ChatGenerationSettings(
                    max_tokens=8,
                    temperature=0.2,
                ),
            )
        )

    AsyncOpenAI.error = RateLimitError("rate limited")
    with pytest.raises(ProviderRateLimitError):
        asyncio.run(
            adapter.stream(
                [ChatMessage(role="user", content="hello")],
                generation=ChatGenerationSettings(
                    max_tokens=8,
                    temperature=0.2,
                ),
            ).__anext__()
        )

    AsyncOpenAI.error = None
    asyncio.run(adapter.aclose())


def test_run_chat_closes_owned_adapter(monkeypatch) -> None:
    adapter = FakeChatAdapter()

    monkeypatch.setattr(chat_orchestrator, "build_chat_adapter", lambda config: adapter)

    response = asyncio.run(
        chat_orchestrator.run_chat(
            [ChatMessage(role="user", content="hello")],
            Config(
                chat_provider="openai",
                chat_api_base="http://chat.example/v1",
                chat_api_key="chat-secret",
                chat_model="chat-model",
                chat_context_window=4096,
                chat_streaming=False,
            ),
        )
    )

    assert response.content == "complete"
    assert adapter.closed is True
