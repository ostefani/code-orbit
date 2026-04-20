from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatDelta:
    content: str


@dataclass(frozen=True)
class ChatUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatResponse:
    content: str
    finish_reason: str | None = None
    usage: ChatUsage | None = None


@dataclass(frozen=True)
class AdapterCapabilities:
    chat: bool
    streaming: bool
    embeddings: bool
    reranking: bool
