import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.events import EventBus
from agent.schemas import CodeChangeSchema
from api import AgentRunRequest, AgentRunStatus
from workflow.core import run_workflow_core
from workflow.context import run_build_context_stage
from workflow.editing import run_editing_plan_stage
from workflow.execution import run_execution_stage
from workflow.output import run_review_diff_stage
from workflow.planning import run_planning_stage
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


def test_run_planning_stage_approves_direct_answer_plan(monkeypatch, tmp_path) -> None:
    plan = SimpleNamespace(
        summary="Explanation",
        answer="This code does the thing.",
        tasks=[],
        model_dump_json=lambda indent=None: "{}",
    )
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Explain this",
        config=Config(interactive=False),
        context_result=SimpleNamespace(context="context"),
    )

    async def fake_call_architect(*_args, **_kwargs):
        return plan

    monkeypatch.setattr("workflow.planning.call_architect", fake_call_architect)

    state = asyncio.run(run_planning_stage(runtime, EventBus()))

    assert state is WorkflowState.COMPLETED
    assert runtime.architect_plan is plan
    assert runtime.approved_plan is plan


def test_run_build_context_stage_emits_building_context_state(monkeypatch, tmp_path) -> None:
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Make it so",
        config=Config(interactive=False),
    )
    events = []
    bus = EventBus()
    bus.subscribe(events.append)

    fake_context_result = SimpleNamespace(
        context="context",
        token_warnings=[],
        skipped_paths=[],
        budget_breakdown=None,
        entries=[SimpleNamespace()],
        used_tokens=12,
        token_budget=34,
        semantic_matches=[],
    )

    async def fake_build_context_async(*_args, **_kwargs):
        return fake_context_result

    monkeypatch.setattr(
        "workflow.context.build_context_async",
        fake_build_context_async,
    )

    state = asyncio.run(run_build_context_stage(runtime, bus))

    assert state is WorkflowState.PLANNING
    assert runtime.context_result is fake_context_result
    assert runtime.working_context == "context"
    assert [event.state for event in events if event.name == "state.changed"] == [
        "building_context"
    ]


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
    assert [event.name for event in events] == ["run.completed"]


def test_run_execution_stage_emits_one_state_change_per_task(monkeypatch, tmp_path) -> None:
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Make it so",
        config=Config(interactive=False),
        context_result=SimpleNamespace(context="context"),
        approved_plan=SimpleNamespace(
            summary="Ship it",
            tasks=[
                SimpleNamespace(goal="Task 1", files=["a.py"], reasoning="why"),
                SimpleNamespace(goal="Task 2", files=["b.py"], reasoning="why"),
            ],
        ),
    )
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    calls = []

    async def fake_call_coder_for_task(_plan, task, *_args, **_kwargs):
        calls.append(task.goal)
        return SimpleNamespace(
            summary=f"Done {task.goal}",
            changes=[
                CodeChangeSchema(
                    path=f"{task.goal.lower().replace(' ', '_')}.py",
                    action="create",
                    content="print('ok')",
                )
            ],
        )

    monkeypatch.setattr(
        "workflow.execution.call_coder_for_task",
        fake_call_coder_for_task,
    )

    state = asyncio.run(run_execution_stage(runtime, bus))

    executing_events = [
        event for event in events
        if event.name == "state.changed" and event.state == "executing"
    ]
    assert state is WorkflowState.REVIEWING_DIFF
    assert calls == ["Task 1", "Task 2"]
    assert [event.message for event in executing_events] == [
        "Generating file replacements for task 1/2.",
        "Generating file replacements for task 2/2.",
    ]
    assert [
        (event.payload.task_index, event.payload.task_total)
        for event in executing_events
    ] == [(1, 2), (2, 2)]
    task_completed_events = [
        event for event in events
        if event.name == "task.completed" and event.state == "executing"
    ]
    assert [event.payload.summary for event in task_completed_events] == [
        "Done Task 1",
        "Done Task 2",
    ]
    assert [event.payload.change_count for event in task_completed_events] == [1, 1]


def test_run_review_diff_stage_uses_rich_confirm_prompt(monkeypatch, tmp_path) -> None:
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Make it so",
        config=Config(interactive=True),
        approved_plan=SimpleNamespace(summary="Ship it", tasks=[]),
    )
    runtime.all_changes = [CodeChangeSchema(path="src/app.py", action="update", content="x")]
    events = []
    bus = EventBus()
    bus.subscribe(events.append)

    monkeypatch.setattr("workflow.output.preview_changes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("workflow.output.Confirm.ask", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("input() must not be used")),
    )

    state = run_review_diff_stage(runtime, bus)

    assert state is WorkflowState.EDITING_PLAN
    assert runtime.all_changes == []
    assert runtime.task_summaries == []
    assert runtime.final_summary == ""
    assert [event.state for event in events if event.name == "state.changed"] == [
        "reviewing_diff",
        "waiting_for_user",
    ]


def test_run_review_diff_stage_treats_eof_as_decline(monkeypatch, tmp_path) -> None:
    runtime = WorkflowRuntime(
        target_dir=str(tmp_path),
        prompt="Make it so",
        config=Config(interactive=True),
        approved_plan=SimpleNamespace(summary="Ship it", tasks=[]),
    )
    runtime.all_changes = [CodeChangeSchema(path="src/app.py", action="update", content="x")]

    monkeypatch.setattr("workflow.output.preview_changes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("workflow.output.Confirm.ask", lambda *_args, **_kwargs: (_ for _ in ()).throw(EOFError))

    state = run_review_diff_stage(runtime, EventBus())

    assert state is WorkflowState.EDITING_PLAN


def test_run_workflow_core_uses_plan_summary_when_execution_has_no_changes(
    monkeypatch,
) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Make it so")
    adapter = _DummyAdapter()

    async def fake_create_chat_adapter(_config):
        return adapter

    async def fake_build_context(runtime, _event_bus):
        runtime.context_result = SimpleNamespace(context="context")
        return WorkflowState.PLANNING

    async def fake_plan(runtime, _event_bus, on_chunk=None):
        runtime.approved_plan = SimpleNamespace(
            summary="Keep it as-is",
            tasks=[SimpleNamespace(goal="Task", files=["src/app.py"], reasoning="why")],
        )
        return WorkflowState.EXECUTING

    async def fake_execution(runtime, _event_bus, on_chunk=None):
        runtime.affected_files = []
        runtime.final_summary = None
        return WorkflowState.COMPLETED

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)
    monkeypatch.setattr("workflow.core.run_planning_stage", fake_plan)
    monkeypatch.setattr("workflow.core.run_execution_stage", fake_execution)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=EventBus(),
        )
    )

    assert adapter.closed is True
    assert result.status is AgentRunStatus.COMPLETED
    assert result.summary == "Keep it as-is"
    assert result.affected_files == []


def test_run_workflow_core_returns_answered_result(monkeypatch) -> None:
    request = AgentRunRequest(target_dir="/tmp/project", prompt="Explain this")
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    adapter = _DummyAdapter()

    async def fake_create_chat_adapter(_config):
        return adapter

    async def fake_build_context(runtime, _event_bus):
        runtime.context_result = SimpleNamespace(context="context")
        return WorkflowState.PLANNING

    async def fake_plan(runtime, _event_bus, on_chunk=None):
        runtime.architect_plan = SimpleNamespace(
            summary="Explanation",
            answer="This code does the thing.",
            tasks=[],
        )
        runtime.approved_plan = runtime.architect_plan
        return WorkflowState.COMPLETED

    monkeypatch.setattr("workflow.core.create_chat_adapter", fake_create_chat_adapter)
    monkeypatch.setattr("workflow.core.run_build_context_stage", fake_build_context)
    monkeypatch.setattr("workflow.core.run_planning_stage", fake_plan)

    result = asyncio.run(
        run_workflow_core(
            request,
            config=Config(interactive=False),
            event_bus=bus,
        )
    )

    assert adapter.closed is True
    assert result.status is AgentRunStatus.ANSWERED
    assert result.summary == "Explanation"
    assert result.answer == "This code does the thing."
    assert result.affected_files == []
    assert [event.name for event in events] == ["run.completed"]


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
    assert [event.name for event in events] == ["run.failed"]


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
