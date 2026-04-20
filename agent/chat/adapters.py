from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from .types import (
    AdapterCapabilities,
    ChatDelta,
    ChatGenerationSettings,
    ChatMessage,
    ChatResponse,
)


RESERVED_CHAT_PROVIDER_OPTION_KEYS = {
    "api_base",
    "base_url",
    "api_key",
    "model",
    "context_window",
    "streaming",
}


class ChatAdapter(Protocol):
    capabilities: AdapterCapabilities
    context_window: int

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> ChatResponse: ...

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> AsyncIterator[ChatDelta]: ...

    async def validate(self) -> None: ...

    async def probe(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class ChatProviderConfig:
    provider: str
    api_base: str
    api_key: str
    model: str
    context_window: int
    streaming: bool
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
