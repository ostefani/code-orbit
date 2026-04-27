import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.events import EventBus
from api import AgentRunRequest, AgentRunStatus
from workflow.core import run_workflow_core
from workflow.editing import run_editing_plan_stage
from workflow._state import WorkflowRuntime, WorkflowState


class _DummyAdapter:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        return None


def test_run_editing_plan_stage_auto_approves_when_non_interactive(
    monkeypatch, tmp_path
) -> None:
    plan = SimpleNamespace(summary="Ship it", tasks=[])
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Make it so",
        config=Config(interactive=False),
        plan_path=tmp_path / "plan.json",
        architect_plan=plan,
    )
    events = []
    bus = EventBus()
    bus.subscribe(events.append)

    def fail_open_editor(_plan_path):
        raise AssertionError("non-interactive runs must not open the editor")

    monkeypatch.setattr("workflow.editing.open_plan_in_editor", fail_open_editor)

    state = run_editing_plan_stage(runtime, bus)

    assert state is WorkflowState.EXECUTING
    assert runtime.approved_plan is plan
    assert events == []


def test_run_workflow_core_returns_completed_result(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    config = Config(interactive=False, auto_commit=False, allow_delete=False)

    adapter = _DummyAdapter()
    seen_config = None
    planning_callback = None
    execution_callback = None

    async def fake_create_chat_adapter(_config):
        nonlocal seen_config
        seen_config = _config
        return adapter

    async def fake_build_context(runtime, _event_bus):
        runtime.context_result = SimpleNamespace(context="context")
        assert runtime.target_dir == "/tmp/project"
        return WorkflowState.PLANNING

    async def fake_plan(runtime, _event_bus, on_chunk=None):
        nonlocal planning_callback
        planning_callback = on_chunk
        if on_chunk is not None:
            on_chunk("plan chunk")
        runtime.approved_plan = SimpleNamespace(
            summary="Ship it",
            tasks=[SimpleNamespace(goal="Task", files=["src/app.py"], reasoning="why")],
        )
        runtime.plan_path = Path("/tmp/plan.json")
        return WorkflowState.EDITING_PLAN

    def fake_editing(_runtime, _event_bus):
        return WorkflowState.EXECUTING

    async def fake_execution(runtime, _event_bus, on_chunk=None):
        nonlocal execution_callback
        execution_callback = on_chunk
        if on_chunk is not None:
            on_chunk(1, 1, "task chunk")
        runtime.final_summary = "Ship it"
        runtime.affected_files = ["src/app.py"]
        return WorkflowState.REVIEWING_DIFF

    def fake_review(_runtime, _event_bus):
        return WorkflowState.APPLYING

    def fake_applying(runtime, _event_bus):
        runtime.final_summary = runtime.final_summary or "Ship it"
        runtime.affected_files = ["src/app.py"]
        return WorkflowState.COMPLETED

    def fake_committing(_runtime, _event_bus):
        return WorkflowState.COMPLETED

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)
    monkeypatch.setattr("workflow.core.run_planning_stage", fake_plan)
    monkeypatch.setattr("workflow.core.run_editing_plan_stage", fake_editing)
    monkeypatch.setattr("workflow.core.run_execution_stage", fake_execution)
    monkeypatch.setattr("workflow.core.run_review_diff_stage", fake_review)
    monkeypatch.setattr("workflow.core.run_applying_stage", fake_applying)
    monkeypatch.setattr("workflow.core.run_committing_stage", fake_committing)

    plan_chunks = []
    task_chunks = []
    plan_callback = plan_chunks.append

    result = asyncio.run(
        run_workflow_core(
            request,
            config=config,
            event_bus=bus,
            on_plan_chunk=plan_callback,
            on_task_chunk=lambda task_index, task_total, chunk: task_chunks.append(
                (task_index, task_total, chunk)
            ),
        )
    )

    assert seen_config is not None
    assert seen_config.auto_commit is False
    assert seen_config.allow_delete is False
    assert planning_callback is plan_callback
    assert execution_callback is not None
    assert adapter.closed is True
    assert plan_chunks == ["plan chunk"]
    assert task_chunks == [(1, 1, "task chunk")]
    assert result.status is AgentRunStatus.COMPLETED
    assert result.run_id == request.run_id
    assert result.summary == "Ship it"
    assert result.affected_files == ["src/app.py"]
    assert [event.name for event in events] == ["run.started", "run.completed"]


def test_run_workflow_core_returns_failed_result(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    bus = EventBus()

    async def fake_create_chat_adapter(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=bus,
            on_plan_chunk=None,
            on_task_chunk=None,
        )
    )

    assert result.status is AgentRunStatus.FAILED
    assert result.error == "boom"


def test_run_workflow_core_merges_request_flags(monkeypatch) -> None:
    request = AgentRunRequest(
        target_dir="/tmp/project",
        prompt="Make it so",
        auto_commit=True,
        allow_delete=True,
    )
    seen_config = None

    async def fake_create_chat_adapter(_config):
        nonlocal seen_config
        seen_config = _config
        raise RuntimeError("stop")

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False, auto_commit=False, allow_delete=False),
            event_bus=EventBus(),
        )
    )

    assert seen_config is not None
    assert seen_config.auto_commit is True
    assert seen_config.allow_delete is True
    assert result.status is AgentRunStatus.FAILED


def test_run_workflow_core_returns_failed_result_for_failed_state(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    adapter = _DummyAdapter()

    async def fake_create_chat_adapter(_config):
        return adapter

    async def fake_build_context(_runtime, _event_bus):
        return WorkflowState.FAILED

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=EventBus(),
        )
    )

    assert adapter.closed is True
    assert result.status is AgentRunStatus.FAILED
    assert result.error == "Workflow failed."


def test_run_workflow_core_returns_failed_result_for_stage_exception(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    adapter = _DummyAdapter()
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    async def fake_create_chat_adapter(_config):
        return adapter

    async def fake_build_context(_runtime, _event_bus):
        raise RuntimeError("context exploded")

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=bus,
        )
    )

    assert adapter.closed is True
    assert result.status is AgentRunStatus.FAILED
    assert result.error == "context exploded"
    assert [event.name for event in events] == ["run.started", "run.failed"]


def test_run_workflow_core_cleans_up_plan_path(monkeypatch, tmp_path) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    adapter = _DummyAdapter()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("temporary plan", encoding="utf-8")

    async def fake_create_chat_adapter(_config):
        return adapter

    async def fake_build_context(runtime, _event_bus):
        runtime.plan_path = plan_path
        raise RuntimeError("boom")

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=EventBus(),
        )
    )

    assert result.status is AgentRunStatus.FAILED
    assert adapter.closed is True
    assert not plan_path.exists()
