import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.events import EventBus
from api import AgentRunRequest, AgentRunStatus
from workflow.core import run_workflow_core
from workflow._state import WorkflowState


class _DummyAdapter:
    async def aclose(self) -> None:
        return None


def test_run_workflow_core_returns_completed_result(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    config = Config(interactive=False, auto_commit=False, allow_delete=False)

    async def fake_create_chat_adapter(_config):
        return _DummyAdapter()

    async def fake_build_context(runtime, _event_bus):
        runtime.context_result = SimpleNamespace(context="context")
        return WorkflowState.PLANNING

    async def fake_plan(runtime, _event_bus, on_chunk=None):
        runtime.approved_plan = SimpleNamespace(
            summary="Ship it",
            tasks=[SimpleNamespace(goal="Task", files=["src/app.py"], reasoning="why")],
        )
        runtime.plan_path = Path("/tmp/plan.json")
        return WorkflowState.EDITING_PLAN

    def fake_editing(_runtime, _event_bus):
        return WorkflowState.EXECUTING

    async def fake_execution(runtime, _event_bus, on_chunk=None):
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

    result = asyncio.run(
        run_workflow_core(
            request,
            config=config,
            event_bus=bus,
            on_plan_chunk=lambda _chunk: None,
            on_task_chunk=lambda _task_index, _task_total, _chunk: None,
        )
    )

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
