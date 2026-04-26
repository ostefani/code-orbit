from collections.abc import Callable
import os
import tempfile
from pathlib import Path

from agent.events import AgentEvent, EmptyPayload, EventBus, PlanReadyPayload, StateChangedPayload
from agent.llm import PlanSchema, call_architect

from .errors import WorkflowError
from ._state import WorkflowRuntime, WorkflowState, reset_execution_state


def build_architect_prompt(prompt: str, execution_feedback: str | None = None) -> str:
    if not execution_feedback:
        return prompt
    return (
        f"{prompt}\n\n<execution_feedback>\n"
        f"{execution_feedback}\n</execution_feedback>"
    )


def create_plan_draft_path() -> Path:
    fd, temp_name = tempfile.mkstemp(prefix="code-orbit-plan-", suffix=".json")
    os.close(fd)
    return Path(temp_name)


def write_plan_draft(plan_path: Path, plan: PlanSchema) -> None:
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan_draft(plan_path: Path) -> PlanSchema:
    return PlanSchema.model_validate_json(plan_path.read_text(encoding="utf-8"))


def format_plan_for_display(plan: PlanSchema) -> str:
    lines = [f"[bold]{plan.summary}[/bold]"]
    if not plan.tasks:
        if plan.answer:
            lines.append("")
            lines.append(plan.answer)
        else:
            lines.append("[dim]No implementation tasks were proposed.[/dim]")
        return "\n".join(lines)

    lines.append("")
    for index, task in enumerate(plan.tasks, 1):
        files = ", ".join(task.files)
        lines.append(f"[bold cyan]{index}. {task.goal}[/bold cyan]")
        lines.append(f"[dim]Files:[/dim] {files}")
        lines.append(f"[dim]Reason:[/dim] {task.reasoning}")
        if index < len(plan.tasks):
            lines.append("")
    return "\n".join(lines)


async def run_planning_stage(
    runtime: WorkflowRuntime,
    event_bus: EventBus,
    on_chunk: Callable[[str], None] | None = None,
) -> WorkflowState:
    if runtime.context_result is None:
        raise RuntimeError(
            "run_planning_stage called before context was built. "
            "run_build_context_stage must complete first."
        )

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.PLANNING.value,
        message="Drafting implementation plan.",
        payload=StateChangedPayload(),
    ))
    prompt = build_architect_prompt(runtime.prompt, runtime.execution_feedback)
    try:
        runtime.architect_plan = await call_architect(
            prompt,
            runtime.context_result.context,
            runtime.config,
            chat_adapter=runtime.chat_adapter,
            on_chunk=on_chunk,
        )
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state=WorkflowState.PLANNING.value,
            message=str(exc),
            payload=EmptyPayload(),
        ))
        raise WorkflowError(str(exc)) from exc

    if runtime.plan_path is None:
        runtime.plan_path = create_plan_draft_path()
    write_plan_draft(runtime.plan_path, runtime.architect_plan)
    event_bus.publish(AgentEvent(
        name="plan.ready",
        state="reviewing_plan",
        message="Implementation plan ready.",
        payload=PlanReadyPayload(
            summary=runtime.architect_plan.summary,
            task_count=len(runtime.architect_plan.tasks),
            draft_path=str(runtime.plan_path),
        ),
    ))
    reset_execution_state(runtime)
    runtime.execution_feedback = None
    if not runtime.architect_plan.tasks:
        return WorkflowState.COMPLETED
    return WorkflowState.EDITING_PLAN
