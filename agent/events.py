import copy
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Generic, TypeVar, IO


PayloadT = TypeVar("PayloadT")


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
    context_window_tokens: int = 0
    response_reserve_tokens: int = 0
    scaffold_tokens: int = 0
    safety_margin_tokens: int = 0


@dataclass(frozen=True)
class ContextSemanticMatchItem:
    path: str
    semantic_score: float
    lexical_score: float
    blended_score: float


@dataclass(frozen=True)
class ContextSemanticMatchPayload:
    prompt: str
    selected_count: int
    matches: tuple[ContextSemanticMatchItem, ...]


@dataclass(frozen=True)
class RunSummaryPayload:
    summary: str
    change_count: int


@dataclass(frozen=True)
class PlanReadyPayload:
    summary: str
    task_count: int
    draft_path: str


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
    src: str | None = None


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
        payload = event.payload
        if is_dataclass(payload) and not isinstance(payload, type):
            dataclass_params = getattr(type(payload), "__dataclass_params__", None)
            if dataclass_params is not None and dataclass_params.frozen:
                safe_payload = payload
            else:
                safe_payload = copy.deepcopy(payload)
        else:
            safe_payload = copy.deepcopy(payload)

        safe_event = AgentEvent(
            name=event.name,
            level=event.level,
            state=event.state,
            message=event.message,
            payload=safe_payload,
            timestamp=event.timestamp,
        )
        for subscriber in self._subscribers:
            try:
                subscriber(safe_event)
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"Subscriber error: {exc}\n{tb}", file=sys.stderr)
        return safe_event

    def emit(
        self,
        name: str,
        payload: PayloadT,
        *,
        level: str = "info",
        state: str | None = None,
        message: str | None = None,
    ) -> AgentEvent[PayloadT]:
        event = AgentEvent(
            name=name,
            payload=payload,
            level=level,
            state=state,
            message=message,
        )
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
        else:
            # Non-AgentEvent records: wrap in a uniform JSON envelope so the
            # .jsonl file remains machine-parseable regardless of log source.
            body = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname.lower(),
                "event": "log",
                "state": None,
                "message": record.getMessage(),
                "payload": {},
            }
        return json.dumps(body, ensure_ascii=False)


class LoggingEventSubscriber:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __call__(self, event: AgentEvent[object]) -> None:
        level = getattr(logging, event.level.upper(), logging.INFO)
        self._logger.log(level, event.message or event.name, extra={"event": event})


class DeferredRotatingFileHandler(RotatingFileHandler):
    def _open(self) -> IO[bytes]:
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def _resolve_log_dir(log_dir: Path | str | None) -> Path:
    if log_dir is None:
        return (Path.cwd() / ".code-orbit").resolve()
    return Path(log_dir).expanduser().resolve()


def _build_event_logger_name(log_dir: Path) -> str:
    digest = hashlib.sha1(
        str(log_dir).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return f"code_orbit.events.{digest}"


def build_event_log_handler(log_dir: Path) -> DeferredRotatingFileHandler:
    handler = DeferredRotatingFileHandler(
        log_dir / "trace.jsonl",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(JsonEventFormatter())
    return handler


def _configure_event_logger(
    logger: logging.Logger,
    resolved_log_dir: Path,
) -> logging.Logger:
    existing_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, DeferredRotatingFileHandler)
    ]
    if existing_handlers:
        existing_paths = {
            Path(Path(handler.baseFilename).resolve())
            for handler in existing_handlers
            if handler.baseFilename
        }
        desired_path = (resolved_log_dir / "trace.jsonl").resolve()
        if existing_paths != {desired_path}:
            raise ValueError(
                "Event logger is already configured for a different log directory."
            )
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(build_event_log_handler(resolved_log_dir))

    return logger


def configure_event_logger(
    logger: logging.Logger,
    log_dir: Path | str | None = None,
) -> logging.Logger:
    resolved_log_dir = _resolve_log_dir(log_dir)
    return _configure_event_logger(logger, resolved_log_dir)


def build_event_logger(log_dir: Path | str | None = None) -> logging.Logger:
    resolved_log_dir = _resolve_log_dir(log_dir)
    logger = logging.getLogger(_build_event_logger_name(resolved_log_dir))
    return _configure_event_logger(logger, resolved_log_dir)
