from collections.abc import AsyncIterator, Sequence, AsyncGenerator
from contextlib import asynccontextmanager, aclosing

from ..config import Config
from .adapters import ChatAdapter
from .factory import build_chat_adapter
from .types import ChatDelta, ChatGenerationSettings, ChatMessage, ChatResponse


@asynccontextmanager
async def _chat_adapter_context(
    config: Config,
    adapter: ChatAdapter | None,
) -> AsyncGenerator[ChatAdapter, None]:
    if adapter is not None:
        yield adapter
        return
    chat_adapter = build_chat_adapter(config)
    try:
        yield chat_adapter
    finally:
        await chat_adapter.aclose()


async def run_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    generation: ChatGenerationSettings | None = None,
) -> ChatResponse:
    async with _chat_adapter_context(config, adapter) as chat_adapter:
        return await chat_adapter.complete(messages, generation=generation)


async def stream_chat(
    messages: Sequence[ChatMessage],
    config: Config,
    *,
    adapter: ChatAdapter | None = None,
    generation: ChatGenerationSettings | None = None,
) -> AsyncIterator[ChatDelta]:
    async with _chat_adapter_context(config, adapter) as chat_adapter:
        async with aclosing(
            chat_adapter.stream(messages, generation=generation)
        ) as stream:
            async for delta in stream:
                yield delta
