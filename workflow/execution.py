from collections.abc import Callable

from agent.config import Config
from agent.events import (
    AgentEvent,
    EmptyPayload,
    EventBus,
    RunProposalReadyPayload,
    StateChangedPayload,
)
from agent.llm import CodeResponseSchema, call_coder_for_task
from agent.schemas import CodeChangeSchema
from agent.utils import validate_repo_relative_path

from ._state import (
    WorkflowRuntime,
    WorkflowState,
    require_approved_plan,
    reset_execution_state,
)

ALLOWED_ACTIONS = {"create", "update", "delete", "mkdir", "copy", "move"}
MAX_REPLAN_ATTEMPTS = 3


def render_applied_changes_context(changes: list[CodeChangeSchema]) -> str:
    if not changes:
        return ""

    blocks = ["<applied_changes>"]
    for change in changes:
        path = change.path
        action = change.action
        content = change.content
        src = change.src
        attrs = f'path="{path}" action="{action}"'
        if src:
            attrs += f' src="{src}"'
        blocks.append(f"<change {attrs}>")
        if content is not None:
            blocks.append(content)
        blocks.append("</change>")
    blocks.append("</applied_changes>")
    return "\n".join(blocks)


def build_working_context(base_context: str, changes: list[CodeChangeSchema]) -> str:
    applied_changes_context = render_applied_changes_context(changes)
    if not applied_changes_context:
        return base_context
    return f"{base_context}\n\n{applied_changes_context}"


def format_execution_feedback(
    error: str,
    partial_changes: list[CodeChangeSchema],
) -> str:
    parts = [f"Execution failed with error: {error}"]
    if partial_changes:
        parts.append("Partial task outputs so far:")
        parts.append(render_applied_changes_context(partial_changes))
    return "\n".join(parts)


def _handle_task_failure(
    runtime: WorkflowRuntime,
    event_bus: EventBus,
    exc: Exception,
) -> WorkflowState:
    runtime.execution_feedback = format_execution_feedback(
        str(exc),
        runtime.all_changes,
    )
    runtime.replan_attempts += 1
    reset_execution_state(runtime)
    event_bus.publish(AgentEvent(
        name="run.failed",
        level="error",
        state=WorkflowState.EXECUTING.value,
        message=str(exc),
        payload=EmptyPayload(),
    ))
    return (
        WorkflowState.PLANNING
        if runtime.replan_attempts <= MAX_REPLAN_ATTEMPTS
        else WorkflowState.FAILED
    )


def validate_llm_result(
    result: CodeResponseSchema,
    config: Config,
) -> tuple[str, list[CodeChangeSchema]]:
    validated_changes: list[CodeChangeSchema] = []
    seen_paths: set[str] = set()

    for index, change in enumerate(result.changes, 1):
        path = change.path
        action = change.action

        # Defense-in-depth: Pydantic rejects empty paths during normal parsing,
        # but validate_llm_result can still see model_construct'd inputs in tests
        # or partially validated callers.
        if not path.strip():
            raise ValueError(f"Change #{index} is missing a valid 'path' string.")

        normalized_path = validate_repo_relative_path(path, f"Change #{index} path")
        if normalized_path in seen_paths:
            raise ValueError(f"Duplicate change path detected: {normalized_path!r}")
        seen_paths.add(normalized_path)

        if action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"Change #{index} has unsupported action {action!r}. "
                f"Allowed actions: {sorted(ALLOWED_ACTIONS)}"
            )

        if action == "delete":
            if not config.allow_delete:
                raise ValueError(
                    "Model proposed a delete action, but deletes are disabled. "
                    "Re-run with --allow-delete or set allow_delete: true in config."
                )
            validated_changes.append(
                CodeChangeSchema(path=normalized_path, action=action)
            )
            continue

        if action == "mkdir":
            validated_changes.append(
                CodeChangeSchema(path=normalized_path, action=action)
            )
            continue

        if action in {"copy", "move"}:
            src = change.src
            if not src or not src.strip():
                raise ValueError(
                    f"Change #{index} action={action!r} requires a non-empty 'src'."
                )
            normalized_src = validate_repo_relative_path(
                src, f"Change #{index} src"
            )
            validated_changes.append(
                CodeChangeSchema(
                    path=normalized_path,
                    action=action,
                    src=normalized_src,
                )
            )
            continue

        # Defense-in-depth: the schema validator should already enforce this for
        # create/update actions, but keep the guard for bypassed validation.
        content = change.content
        if content is None:
            raise ValueError(f"Change #{index} action={action!r} requires content.")
        validated_changes.append(
            CodeChangeSchema(
                path=normalized_path,
                action=action,
                content=content,
            )
        )

    return result.summary, validated_changes


async def run_execution_stage(
    runtime: WorkflowRuntime,
    event_bus: EventBus,
    on_chunk: Callable[[int, int, str], None] | None = None,
) -> WorkflowState:
    assert runtime.context_result is not None
    approved_plan = require_approved_plan(runtime)

    runtime.all_changes = []
    runtime.task_summaries = []
    runtime.working_context = runtime.context_result.context

    task_total = len(approved_plan.tasks)
    for index, task in enumerate(approved_plan.tasks, 1):
        event_bus.publish(AgentEvent(
            name="state.changed",
            state=WorkflowState.EXECUTING.value,
            message=(
                f"Generating file replacements for task {index}/"
                f"{task_total}."
            ),
            payload=StateChangedPayload(),
        ))
        try:
            task_chunk_callback = None
            if on_chunk is not None:
                def _task_chunk(
                    chunk: str,
                    _i: int = index,
                    _t: int = task_total,
                    _callback: Callable[[int, int, str], None] = on_chunk,
                ) -> None:
                    _callback(_i, _t, chunk)

                task_chunk_callback = _task_chunk

            result = await call_coder_for_task(
                approved_plan,
                task,
                runtime.working_context,
                runtime.config,
                chat_adapter=runtime.chat_adapter,
                on_chunk=task_chunk_callback,
            )
        except Exception as exc:
            return _handle_task_failure(runtime, event_bus, exc)

        try:
            summary, changes = validate_llm_result(result, runtime.config)
        except ValueError as exc:
            return _handle_task_failure(runtime, event_bus, exc)

        if summary:
            runtime.task_summaries.append(summary)
        runtime.all_changes.extend(changes)
        runtime.working_context = build_working_context(
            runtime.context_result.context,
            runtime.all_changes,
        )

    if not runtime.all_changes:
        event_bus.publish(AgentEvent(
            name="run.no_changes",
            state=WorkflowState.COMPLETED.value,
            message="No changes needed.",
            payload=EmptyPayload(),
        ))
        return WorkflowState.COMPLETED

    runtime.final_summary = approved_plan.summary
    if len(runtime.task_summaries) == 1:
        runtime.final_summary = runtime.task_summaries[0]

    event_bus.emit(
        "run.proposal_ready",
        RunProposalReadyPayload(
            summary=runtime.final_summary,
            change_count=len(runtime.all_changes),
        ),
        state="validated",
        message="Model response validated.",
    )
    return WorkflowState.REVIEWING_DIFF
