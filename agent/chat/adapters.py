from collections.abc import AsyncIterator, Mapping, Sequence, AsyncGenerator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import SecretStr

from .types import (
    AdapterCapabilities,
    ChatDelta,
    ChatGenerationSettings,
    ChatMessage,
    ChatResponse,
)


RESERVED_CHAT_PROVIDER_OPTION_KEYS = frozenset(
    (
        "api_base",
        "base_url",
        "api_key",
        "model",
        "context_window",
        "streaming",
    )
)


class ChatAdapter(Protocol):
    capabilities: ClassVar[AdapterCapabilities]
    context_window: int

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> ChatResponse: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: ChatGenerationSettings | None = None,
    ) -> AsyncGenerator[ChatDelta, None]: ...

    async def validate(self) -> None: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class ProbingChatAdapter(Protocol):
    async def probe(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatProviderConfig:
    provider: str
    api_base: str
    api_key: SecretStr
    model: str
    context_window: int
    streaming: bool
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
