from collections.abc import AsyncIterator, Sequence

from ..config import Config
from .adapters import ChatAdapter
from .factory import build_chat_adapter
from .types import ChatDelta, ChatGenerationSettings, ChatMessage, ChatResponse


async def run_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    generation: ChatGenerationSettings | None = None,
) -> ChatResponse:
    chat_adapter = adapter or build_chat_adapter(config)
    owns_adapter = adapter is None
    try:
        return await chat_adapter.complete(messages, generation=generation)
    finally:
        if owns_adapter:
            await chat_adapter.aclose()


async def stream_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    generation: ChatGenerationSettings | None = None,
) -> AsyncIterator[ChatDelta]:
    chat_adapter = adapter or build_chat_adapter(config)
    owns_adapter = adapter is None
    try:
        async for delta in chat_adapter.stream(messages, generation=generation):
            yield delta
    finally:
        if owns_adapter:
            await chat_adapter.aclose()
