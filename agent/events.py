from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Generic, TypeVar, Callable


PayloadT = TypeVar("PayloadT", covariant=True)


@dataclass(frozen=True)
class EmptyPayload:
    pass


@dataclass(frozen=True)
class ConfigMessagePayload:
    text: str


@dataclass(frozen=True)
class RunStartedPayload:
    target_dir: str
    model: str


@dataclass(frozen=True)
class StateChangedPayload:
    pass


@dataclass(frozen=True)
class ContextWarningPayload:
    warning: str


@dataclass(frozen=True)
class ContextSkippedPayload:
    skipped_count: int
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ContextSummaryPayload:
    file_count: int
    used_tokens: int
    token_budget: int


@dataclass(frozen=True)
class RunSummaryPayload:
    summary: str
    change_count: int


@dataclass(frozen=True)
class RunProposalReadyPayload:
    summary: str
    change_count: int


@dataclass(frozen=True)
class RunCompletedPayload:
    affected_count: int


@dataclass(frozen=True)
class PreviewChangePayload:
    path: str
    action: str
    content: str | None = None
    diff_text: str | None = None
    exists: bool = False
    missing: bool = False
    unchanged: bool = False


@dataclass(frozen=True)
class ApplyFilePayload:
    path: str
    action: str
    performed: bool


@dataclass(frozen=True)
class GitCommitSucceededPayload:
    files: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class GitCommitFailedPayload:
    stderr: str


@dataclass(frozen=True)
class AgentEvent(Generic[PayloadT]):
    name: str
    payload: PayloadT
    level: str = "info"
    state: str | None = None
    message: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


EventSubscriber = Callable[[AgentEvent[object]], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

    def subscribe(self, subscriber: EventSubscriber) -> None:
        self._subscribers.append(subscriber)

    def publish(self, event: AgentEvent[PayloadT]) -> AgentEvent[PayloadT]:
        safe_event = AgentEvent(
            name=event.name,
            level=event.level,
            state=event.state,
            message=event.message,
            payload=copy.deepcopy(event.payload),
            timestamp=event.timestamp,
        )
        for subscriber in self._subscribers:
            try:
                subscriber(safe_event)
            except Exception as exc:
                print(f"Subscriber error: {exc}", file=sys.stderr)
        return safe_event

    def emit(self, name: str, payload: PayloadT, **kwargs) -> AgentEvent[PayloadT]:
        event = AgentEvent(name=name, payload=payload, **kwargs)
        return self.publish(event)


class JsonEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", None)
        if isinstance(event, AgentEvent):
            payload_obj = event.payload
            if is_dataclass(payload_obj) and not isinstance(payload_obj, type):
                payload = asdict(payload_obj)
            else:
                payload = payload_obj
            body = {
                "timestamp": event.timestamp,
                "level": event.level,
                "event": event.name,
                "state": event.state,
                "message": event.message,
                "payload": payload,
            }
            return json.dumps(body, ensure_ascii=False)
        return super().format(record)


class LoggingEventSubscriber:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("code_orbit.events")

    def __call__(self, event: AgentEvent[object]) -> None:
        level = getattr(logging, event.level.upper(), logging.INFO)
        self._logger.log(level, event.message or event.name, extra={"event": event})


def build_event_logger() -> logging.Logger:
    logger = logging.getLogger("code_orbit.events")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(
        getattr(handler, "_code_orbit_events", False) for handler in logger.handlers
    ):
        log_dir = Path(".code-orbit")
        log_dir.mkdir(exist_ok=True)

        handler = RotatingFileHandler(
            log_dir / "trace.jsonl",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler._code_orbit_events = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonEventFormatter())
        logger.addHandler(handler)

    return logger
