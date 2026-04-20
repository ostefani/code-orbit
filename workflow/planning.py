import os
import tempfile
from pathlib import Path

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.context_result is not None

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.PLANNING.value,
        message="Drafting implementation plan.",
        payload=StateChangedPayload(),
    ))
    prompt = build_architect_prompt(runtime.prompt, runtime.execution_feedback)
    try:
        progress = Progress(
            SpinnerColumn(style="bold magenta"),
            TextColumn("{task.description}"),
            transient=True,
            console=runtime.console,
        )
        task_id = progress.add_task("Architect is streaming response...", total=None)
        chunk_count = 0

        def on_plan_chunk(_chunk: str) -> None:
            nonlocal chunk_count
            chunk_count += 1
            progress.update(
                task_id,
                description=f"Architect is streaming response... ({chunk_count} chunks)",
            )

        with Live(progress, console=runtime.console, refresh_per_second=12):
            runtime.architect_plan = await call_architect(
                prompt,
                runtime.context_result.context,
                runtime.config,
                on_chunk=on_plan_chunk,
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
    runtime.console.print(
        Panel.fit(
            format_plan_for_display(runtime.architect_plan),
            title="🧭 plan",
            border_style="cyan",
        )
    )
    reset_execution_state(runtime)
    runtime.execution_feedback = None
    if not runtime.architect_plan.tasks:
        return WorkflowState.COMPLETED
    return WorkflowState.EDITING_PLAN
