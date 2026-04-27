from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.config import Config
from agent.events import (
    AgentEvent,
    ConfigMessagePayload,
    EmptyPayload,
    EventBus,
    LoggingEventSubscriber,
    RunStartedPayload,
    StateChangedPayload,
    build_event_logger,
)
from api import AgentRunRequest, AgentRunStatus

from .core import run_workflow_core
from .errors import WorkflowError

if TYPE_CHECKING:
    from agent.llm import PlanSchema
    from rich.console import Console

__all__ = ["run_workflow", "WorkflowError"]


def format_plan_for_display(plan: "PlanSchema") -> str:
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


async def run_workflow(
    *,
    target_dir: str,
    prompt: str,
    config_path: str = "config.yaml",
    profile_name: str | None = None,
    no_interactive: bool = False,
    auto_commit: bool = False,
    allow_delete: bool = False,
    tree: bool = False,
    console: Console | None = None,
) -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

    from agent.context import get_file_tree
    from agent.rendering import CliEventRenderer

    console_obj = console or Console()
    event_bus = EventBus()
    target_path = str(Path(target_dir).resolve())

    progress = Progress(
        SpinnerColumn(style="bold magenta"),
        TextColumn("{task.description}"),
        transient=True,
        console=console_obj,
    )

    @dataclass
    class _ProgressState:
        live: Live | None = None
        context_task_id: TaskID | None = None
        plan_task_id: TaskID | None = None
        plan_chunk_count: int = 0
        task_task_ids: dict[int, TaskID] = field(default_factory=dict)
        task_chunk_counts: dict[int, int] = field(default_factory=dict)

    progress_state = _ProgressState()

    def start_live() -> None:
        if progress_state.live is None:
            progress_state.live = Live(
                progress,
                console=console_obj,
                refresh_per_second=12,
            )
            progress_state.live.start()

    def stop_live() -> None:
        if progress_state.live is not None:
            progress_state.live.stop()
            progress_state.live = None

    def _remove_plan_task() -> None:
        if progress_state.plan_task_id is not None:
            progress.remove_task(progress_state.plan_task_id)
            progress_state.plan_task_id = None
        progress_state.plan_chunk_count = 0

    def _remove_context_task() -> None:
        if progress_state.context_task_id is not None:
            progress.remove_task(progress_state.context_task_id)
            progress_state.context_task_id = None

    def _remove_task_tasks() -> None:
        for task_id in progress_state.task_task_ids.values():
            progress.remove_task(task_id)
        progress_state.task_task_ids.clear()
        progress_state.task_chunk_counts.clear()

    def progress_subscriber(event: AgentEvent[object]) -> None:
        if event.name in {"context.summary", "run.completed", "run.failed"}:
            stop_live()
            _remove_context_task()
            if event.name != "context.summary":
                _remove_plan_task()
                _remove_task_tasks()
            return

        if event.name == "plan.ready":
            stop_live()
            _remove_plan_task()
            return

        if event.name == "task.completed":
            stop_live()
            _remove_task_tasks()
            return

        if event.name == "run.proposal_ready":
            stop_live()
            _remove_task_tasks()
            return

        if event.name != "state.changed":
            return

        if event.state == "building_context":
            _remove_context_task()
            progress_state.context_task_id = progress.add_task(
                "Analyzing codebase...",
                total=None,
            )
            start_live()
        elif event.state == "planning":
            _remove_plan_task()
            progress_state.plan_task_id = progress.add_task(
                "Architect is streaming response...",
                total=None,
            )
            start_live()
        elif event.state == "executing":
            _remove_plan_task()
            _remove_task_tasks()
            if isinstance(event.payload, StateChangedPayload):
                task_index = event.payload.task_index
                task_total = event.payload.task_total
            else:
                task_index = None
                task_total = None
            if task_index is not None and task_total is not None:
                task_id = progress.add_task(
                    f"Coder is streaming task {task_index}/{task_total}...",
                    total=None,
                )
                progress_state.task_task_ids[task_index] = task_id
                progress_state.task_chunk_counts[task_index] = 0
                start_live()
        elif event.state in {
            "editing_plan",
            "reviewing_diff",
            "waiting_for_user",
            "applying",
            "committing",
        }:
            stop_live()
            _remove_context_task()
            _remove_plan_task()
            _remove_task_tasks()

    def on_plan_chunk(_chunk: str) -> None:
        if progress_state.plan_task_id is None:
            return
        progress_state.plan_chunk_count += 1
        progress.update(
            progress_state.plan_task_id,
            description=(
                "Architect is streaming response... "
                f"({progress_state.plan_chunk_count})"
            ),
        )
        start_live()

    def on_task_chunk(task_index: int, task_total: int, _chunk: str) -> None:
        task_id = progress_state.task_task_ids.get(task_index)
        if task_id is None:
            return

        progress_state.task_chunk_counts[task_index] += 1
        progress.update(
            task_id,
            description=(
                f"Coder is streaming task {task_index}/{task_total}... "
                f"({progress_state.task_chunk_counts[task_index]} chunks)"
            ),
        )
        start_live()

    event_bus.subscribe(progress_subscriber)
    event_bus.subscribe(CliEventRenderer(console_obj))
    event_bus.subscribe(
        LoggingEventSubscriber(
            build_event_logger(log_dir=Path(target_path) / ".code-orbit")
        )
    )

    try:
        config_result = Config.load_with_diagnostics(
            config_path, profile_name=profile_name
        )
        config = config_result.config
    except Exception as exc:
        event_bus.publish(
            AgentEvent(
                name="run.failed",
                level="error",
                state="loading_config",
                message=str(exc),
                payload=EmptyPayload(),
            )
        )
        console_obj.print(f"[bold red]Error loading config:[/bold red] {exc}")
        raise WorkflowError(str(exc)) from exc

    for message in config_result.messages:
        event_bus.publish(
            AgentEvent(
                name="config.message",
                level=message.level,
                state="loading_config",
                message=message.text,
                payload=ConfigMessagePayload(text=message.text),
            )
        )

    config = config.model_copy(
        update={
            "interactive": config.interactive and not no_interactive,
        }
    )

    event_bus.publish(
        AgentEvent(
            name="run.started",
            state="starting",
            message="Agent run started.",
            payload=RunStartedPayload(target_dir=target_path, model=config.chat_model),
        )
    )

    console_obj.print(
        Panel.fit(
            f"[bold blue]Code Orbit[/bold blue]\n"
            f"[dim]Target :[/dim] [green]{target_path}[/green]\n"
            f"[dim]Model  :[/dim] [magenta]{config.chat_api_base} ({config.chat_model})[/magenta]\n"
            f"[dim]Prompt :[/dim] [yellow]{prompt}[/yellow]",
            title="🔧 settings",
            border_style="blue",
        )
    )

    if tree:
        console_obj.print(
            Panel(
                get_file_tree(target_path, config),
                title="📂 File Tree",
                border_style="dim",
            )
        )
        return

    request = AgentRunRequest(
        target_dir=Path(target_path),
        prompt=prompt,
        auto_commit=auto_commit,
        allow_delete=allow_delete,
    )

    try:
        result = await run_workflow_core(
            request,
            config=config,
            event_bus=event_bus,
            on_plan_chunk=on_plan_chunk,
            on_task_chunk=on_task_chunk,
        )
    finally:
        stop_live()

    if result.status is AgentRunStatus.ANSWERED:
        console_obj.print(
            Panel(
                result.answer or "",
                title="Answer",
                border_style="green",
            )
        )
        return

    if result.status is AgentRunStatus.FAILED:
        raise WorkflowError(result.error or "Workflow failed.")
