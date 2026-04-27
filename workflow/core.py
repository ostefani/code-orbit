from collections.abc import Callable
from datetime import datetime, timezone

from agent.config import Config
from agent.chat import create_chat_adapter
from agent.events import (
    AgentEvent,
    EmptyPayload,
    EventBus,
    RunCompletedPayload,
    RunStartedPayload,
)
from api import AgentRunRequest, AgentRunResult, AgentRunStatus

from .context import run_build_context_stage
from .editing import run_editing_plan_stage
from .execution import run_execution_stage
from .output import run_applying_stage, run_committing_stage, run_review_diff_stage
from .planning import run_planning_stage
from ._state import WorkflowRuntime, WorkflowState


def _terminal_time() -> datetime:
    return datetime.now(timezone.utc)


def _build_failed_result(
    request: AgentRunRequest,
    error: str,
    *,
    completed_at: datetime | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        request=request,
        status=AgentRunStatus.FAILED,
        error=error,
        completed_at=completed_at or _terminal_time(),
    )


async def run_workflow_core(
    request: AgentRunRequest,
    *,
    config: Config,
    event_bus: EventBus,
    on_plan_chunk: Callable[[str], None] | None = None,
    on_task_chunk: Callable[[int, int, str], None] | None = None,
) -> AgentRunResult:
    target_path = str(request.target_dir)
    config = config.model_copy(
        update={
            "auto_commit": config.auto_commit or request.auto_commit,
            "allow_delete": config.allow_delete or request.allow_delete,
        }
    )

    event_bus.publish(AgentEvent(
        name="run.started",
        state="starting",
        message="Agent run started.",
        payload=RunStartedPayload(target_dir=target_path, model=config.chat_model),
    ))

    runtime = WorkflowRuntime(
        target_dir=target_path,
        prompt=request.prompt,
        config=config,
    )

    try:
        runtime.chat_adapter = await create_chat_adapter(config)
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="loading_chat",
            message=str(exc),
            payload=EmptyPayload(),
        ))
        return _build_failed_result(request, str(exc))

    state = WorkflowState.BUILDING_CONTEXT
    try:
        while state not in {WorkflowState.COMPLETED, WorkflowState.FAILED}:
            match state:
                case WorkflowState.BUILDING_CONTEXT:
                    state = await run_build_context_stage(runtime, event_bus)
                case WorkflowState.PLANNING:
                    state = await run_planning_stage(
                        runtime,
                        event_bus,
                        on_chunk=on_plan_chunk,
                    )
                case WorkflowState.EDITING_PLAN:
                    state = run_editing_plan_stage(runtime, event_bus)
                case WorkflowState.EXECUTING:
                    state = await run_execution_stage(
                        runtime,
                        event_bus,
                        on_chunk=on_task_chunk,
                    )
                case WorkflowState.REVIEWING_DIFF:
                    state = run_review_diff_stage(runtime, event_bus)
                case WorkflowState.APPLYING:
                    state = run_applying_stage(runtime, event_bus)
                case WorkflowState.COMMITTING:
                    state = run_committing_stage(runtime, event_bus)
                case _:
                    state = WorkflowState.FAILED

        if state == WorkflowState.COMPLETED:
            approved_plan = runtime.approved_plan or runtime.architect_plan
            if (
                approved_plan is not None
                and not approved_plan.tasks
                and approved_plan.answer
            ):
                result = AgentRunResult(
                    request=request,
                    status=AgentRunStatus.ANSWERED,
                    summary=approved_plan.summary,
                    answer=approved_plan.answer,
                    completed_at=_terminal_time(),
                )
                event_bus.publish(AgentEvent(
                    name="run.completed",
                    state=WorkflowState.COMPLETED.value,
                    message="Agent run completed.",
                    payload=RunCompletedPayload(
                        affected_count=0,
                    ),
                ))
                return result

            result = AgentRunResult(
                request=request,
                status=AgentRunStatus.COMPLETED,
                summary=runtime.final_summary or "",
                affected_files=runtime.affected_files,
                completed_at=_terminal_time(),
            )
            event_bus.publish(AgentEvent(
                name="run.completed",
                state=WorkflowState.COMPLETED.value,
                message="Agent run completed.",
                payload=RunCompletedPayload(
                    affected_count=len(runtime.affected_files),
                ),
            ))
            return result

        return _build_failed_result(
            request,
            "Workflow failed.",
            completed_at=_terminal_time(),
        )
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state=WorkflowState.FAILED.value,
            message=str(exc),
            payload=EmptyPayload(),
        ))
        return _build_failed_result(request, str(exc))
    finally:
        if runtime.plan_path is not None:
            runtime.plan_path.unlink(missing_ok=True)
        if runtime.chat_adapter is not None:
            await runtime.chat_adapter.aclose()
