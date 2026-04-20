from collections.abc import AsyncIterator, Sequence

from ..config import Config
from .adapters import ChatAdapter
from .errors import CapabilityNotSupportedError
from .factory import create_chat_adapter
from .types import ChatDelta, ChatMessage, ChatResponse


def _provider_name(adapter: ChatAdapter) -> str:
    return getattr(adapter, "provider_name", type(adapter).__name__)


async def _acquire_chat_adapter(
    config: Config,
    adapter: ChatAdapter | None,
) -> tuple[ChatAdapter, bool]:
    if adapter is None:
        return await create_chat_adapter(config), True
    await adapter.validate()
    return adapter, False


async def run_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatResponse:
    chat_adapter, owns_adapter = await _acquire_chat_adapter(config, adapter)
    try:
        if not chat_adapter.capabilities.chat:
            raise CapabilityNotSupportedError(
                _provider_name(chat_adapter),
                "The selected provider does not support chat completion.",
            )
        return await chat_adapter.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    finally:
        if owns_adapter:
            await chat_adapter.aclose()


async def stream_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[ChatDelta]:
    chat_adapter, owns_adapter = await _acquire_chat_adapter(config, adapter)
    try:
        if not chat_adapter.capabilities.chat:
            raise CapabilityNotSupportedError(
                _provider_name(chat_adapter),
                "The selected provider does not support chat completion.",
            )
        if not chat_adapter.capabilities.streaming:
            raise CapabilityNotSupportedError(
                _provider_name(chat_adapter),
                "The selected provider does not support streaming chat completion.",
            )
        async for delta in chat_adapter.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield delta
    finally:
        if owns_adapter:
            await chat_adapter.aclose()
