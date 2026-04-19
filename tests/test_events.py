from dataclasses import dataclass
import json
import logging
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from logging.handlers import RotatingFileHandler

from agent.events import (
    AgentEvent,
    ApplyFilePayload,
    EmptyPayload,
    EventBus,
    JsonEventFormatter,
    LoggingEventSubscriber,
    RunStartedPayload,
    build_event_logger,
)


def test_event_bus_notifies_subscribers() -> None:
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    bus.publish(
        AgentEvent(
            name="state.changed",
            state="planning",
            message="Building context.",
            payload=EmptyPayload(),
        )
    )

    assert len(events) == 1
    assert events[0].name == "state.changed"
    assert events[0].state == "planning"


def test_logging_subscriber_emits_json() -> None:
    logger = logging.getLogger("test.code_orbit.events")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    records: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)

    subscriber = LoggingEventSubscriber(logger)
    subscriber(
        EventBus().publish(
            AgentEvent(
            name="run.started",
            state="starting",
            message="Agent run started.",
            payload=RunStartedPayload(model="local", target_dir="/tmp/project"),
            )
        )
    )

    payload = json.loads(records[0])
    assert payload["event"] == "run.started"
    assert payload["state"] == "starting"
    assert payload["payload"]["model"] == "local"


def test_event_bus_isolates_subscriber_failures() -> None:
    bus = EventBus()
    events = []

    def failing_subscriber(_event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(failing_subscriber)
    bus.subscribe(events.append)

    stderr = StringIO()
    with redirect_stderr(stderr):
        bus.publish(
            AgentEvent(
                name="state.changed",
                state="planning",
                message="Building context.",
                payload=EmptyPayload(),
            )
        )

    assert len(events) == 1
    assert "Subscriber error: boom" in stderr.getvalue()


def test_logging_formatter_preserves_utf8() -> None:
    logger = logging.getLogger("test.code_orbit.utf8")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    records: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)

    subscriber = LoggingEventSubscriber(logger)
    subscriber(
        EventBus().publish(
            AgentEvent(
            name="apply.file",
            state="applying",
            message="✅ Archivo actualizado",
            payload=ApplyFilePayload(path="demo.py", action="update", performed=True),
            )
        )
    )

    assert "\\u2705" not in records[0]
    assert "✅ Archivo actualizado" in records[0]


def test_event_bus_keeps_frozen_payload_identity() -> None:
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    payload = RunStartedPayload(target_dir="/tmp/demo", model="local")
    event = AgentEvent(name="run.started", payload=payload)

    published = bus.publish(event)

    assert published.payload == payload
    assert published.payload is payload


def test_event_bus_deep_copies_mutable_payload() -> None:
    @dataclass
    class MutablePayload:
        values: list[int]

    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    payload = MutablePayload(values=[1, 2, 3])
    event = AgentEvent(name="demo.mutable", payload=payload)

    published = bus.publish(event)

    assert published.payload == payload
    assert published.payload is not payload
    payload.values.append(4)
    assert published.payload.values == [1, 2, 3]


def test_build_event_logger_uses_rotating_file_handler(tmp_path: Path) -> None:
    logger = logging.getLogger("code_orbit.events")
    prior_handlers = logger.handlers[:]
    logger.handlers.clear()

    try:
        current_dir = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)
            configured = build_event_logger()
        finally:
            os.chdir(current_dir)

        assert configured is logger
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3
        assert (tmp_path / ".code-orbit" / "trace.jsonl").parent.exists()
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.handlers.extend(prior_handlers)
