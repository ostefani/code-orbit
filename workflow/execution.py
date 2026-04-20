from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent.config import Config
from agent.events import (
    AgentEvent,
    EmptyPayload,
    EventBus,
    RunProposalReadyPayload,
    StateChangedPayload,
)
from agent.llm import CodeResponseSchema, call_coder_for_task

from ._state import (
    WorkflowRuntime,
    WorkflowState,
    require_approved_plan,
    reset_execution_state,
)

ALLOWED_ACTIONS = {"create", "update", "delete"}
MAX_REPLAN_ATTEMPTS = 3


def render_applied_changes_context(changes: list[dict[str, str]]) -> str:
    if not changes:
        return ""

    blocks = ["<applied_changes>"]
    for change in changes:
        path = change["path"]
        action = change["action"]
        content = change.get("content")
        blocks.append(f'<change path="{path}" action="{action}">')
        if content is not None:
            blocks.append(content)
        blocks.append("</change>")
    blocks.append("</applied_changes>")
    return "\n".join(blocks)


def build_working_context(base_context: str, changes: list[dict[str, str]]) -> str:
    applied_changes_context = render_applied_changes_context(changes)
    if not applied_changes_context:
        return base_context
    return f"{base_context}\n\n{applied_changes_context}"


def format_execution_feedback(
    error: str,
    partial_changes: list[dict[str, str]],
) -> str:
    parts = [f"Execution failed with error: {error}"]
    if partial_changes:
        parts.append("Partial task outputs so far:")
        parts.append(render_applied_changes_context(partial_changes))
    return "\n".join(parts)


def validate_llm_result(
    result: CodeResponseSchema,
    config: Config,
) -> tuple[str, list[dict[str, str]]]:
    validated_changes: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for index, change in enumerate(result.changes, 1):
        path = change.path
        action = change.action

        # Defense-in-depth: Pydantic rejects empty paths during normal parsing,
        # but validate_llm_result can still see model_construct'd inputs in tests
        # or partially validated callers.
        if not path.strip():
            raise ValueError(f"Change #{index} is missing a valid 'path' string.")

        normalized_path = path.strip()
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
            validated_changes.append({"path": normalized_path, "action": action})
            continue

        # Defense-in-depth: the schema validator should already enforce this for
        # create/update actions, but keep the guard for bypassed validation.
        content = change.content
        if content is None:
            raise ValueError(f"Change #{index} action={action!r} requires content.")
        validated_changes.append(
            {
                "path": normalized_path,
                "action": action,
                "content": content,
            }
        )

    return result.summary, validated_changes


async def run_execution_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.context_result is not None
    approved_plan = require_approved_plan(runtime)

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.EXECUTING.value,
        message="Generating file replacements.",
        payload=StateChangedPayload(),
    ))

    runtime.all_changes = []
    runtime.task_summaries = []
    runtime.working_context = runtime.context_result.context

    for index, task in enumerate(approved_plan.tasks, 1):
        event_bus.publish(AgentEvent(
            name="state.changed",
            state=WorkflowState.EXECUTING.value,
            message=(
                f"Generating file replacements for task {index}/"
                f"{len(approved_plan.tasks)}."
            ),
            payload=StateChangedPayload(),
        ))
        try:
            progress = Progress(
                SpinnerColumn(style="bold magenta"),
                TextColumn("{task.description}"),
                transient=True,
                console=runtime.console,
            )
            task_id = progress.add_task(
                f"Coder is streaming task {index}/{len(approved_plan.tasks)}...",
                total=None,
            )
            chunk_count = 0

            # _i binds index by value; if tasks are ever parallelised,
            # task_id and progress would also need default-argument binding.
            def on_chunk(_chunk: str, _i: int = index) -> None:
                nonlocal chunk_count
                chunk_count += 1
                progress.update(
                    task_id,
                    description=(
                        f"Coder is streaming task {_i}/"
                        f"{len(approved_plan.tasks)}... ({chunk_count} chunks)"
                    ),
                )

            with Live(progress, console=runtime.console, refresh_per_second=12):
                result = await call_coder_for_task(
                    approved_plan,
                    task,
                    runtime.working_context,
                    runtime.config,
                    on_chunk=on_chunk,
                )
        except Exception as exc:
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

        try:
            summary, changes = validate_llm_result(result, runtime.config)
        except ValueError as exc:
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

        if summary:
            runtime.task_summaries.append(summary)
        runtime.all_changes.extend(changes)
        runtime.working_context = build_working_context(
            runtime.context_result.context,
            runtime.all_changes,
        )

        if changes:
            runtime.console.print(
                f"[green]✓[/green] Task {index}/{len(approved_plan.tasks)}: "
                f"{summary} ({len(changes)} file(s))"
            )
        else:
            runtime.console.print(
                f"[yellow]⚠[/yellow] Task {index}/{len(approved_plan.tasks)}: "
                f"No changes required for '{task.goal}'"
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
