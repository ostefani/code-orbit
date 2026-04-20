from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent.config import Config
from agent.context import get_file_tree
from agent.events import (
    AgentEvent,
    ConfigMessagePayload,
    EmptyPayload,
    EventBus,
    LoggingEventSubscriber,
    RunCompletedPayload,
    RunStartedPayload,
    build_event_logger,
)
from agent.rendering import CliEventRenderer

from .errors import WorkflowError
from ._state import WorkflowRuntime, WorkflowState
from .context import run_build_context_stage
from .editing import run_editing_plan_stage
from .execution import run_execution_stage
from .output import run_applying_stage, run_committing_stage, run_review_diff_stage
from .planning import run_planning_stage

__all__ = ["run_workflow", "WorkflowError"]


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
    console_obj = console or Console()
    event_bus = EventBus()
    event_bus.subscribe(LoggingEventSubscriber(build_event_logger()))
    event_bus.subscribe(CliEventRenderer(console_obj))

    runtime: WorkflowRuntime | None = None

    try:
        config_result = Config.load_with_diagnostics(
            config_path, profile_name=profile_name
        )
        config = config_result.config
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="loading_config",
            message=str(exc),
            payload=EmptyPayload(),
        ))
        console_obj.print(f"[bold red]Error loading config:[/bold red] {exc}")
        raise WorkflowError(str(exc)) from exc

    for message in config_result.messages:
        event_bus.publish(AgentEvent(
            name="config.message",
            level=message.level,
            state="loading_config",
            message=message.text,
            payload=ConfigMessagePayload(text=message.text),
        ))

    config = replace(
        config,
        interactive=config.interactive and not no_interactive,
        auto_commit=config.auto_commit or auto_commit,
        allow_delete=config.allow_delete or allow_delete,
    )

    target_path = str(Path(target_dir).resolve())
    event_bus.publish(AgentEvent(
        name="run.started",
        state="starting",
        message="Agent run started.",
        payload=RunStartedPayload(target_dir=target_path, model=config.model),
    ))

    console_obj.print(
        Panel.fit(
            f"[bold blue]Code Orbit[/bold blue]\n"
            f"[dim]Target :[/dim] [green]{target_path}[/green]\n"
            f"[dim]Model  :[/dim] [magenta]{config.api_base} ({config.model})[/magenta]\n"
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

    runtime = WorkflowRuntime(
        target_dir=target_path,
        prompt=prompt,
        config=config,
        console=console_obj,
    )
    state = WorkflowState.BUILDING_CONTEXT
    try:
        while state not in {WorkflowState.COMPLETED, WorkflowState.FAILED}:
            match state:
                case WorkflowState.BUILDING_CONTEXT:
                    state = await run_build_context_stage(runtime, event_bus)
                case WorkflowState.PLANNING:
                    state = await run_planning_stage(runtime, event_bus)
                case WorkflowState.EDITING_PLAN:
                    state = run_editing_plan_stage(runtime, event_bus)
                case WorkflowState.EXECUTING:
                    state = await run_execution_stage(runtime, event_bus)
                case WorkflowState.REVIEWING_DIFF:
                    state = run_review_diff_stage(runtime, event_bus)
                case WorkflowState.APPLYING:
                    state = run_applying_stage(runtime, event_bus)
                case WorkflowState.COMMITTING:
                    state = run_committing_stage(runtime, event_bus)
                case _:
                    state = WorkflowState.FAILED

        if state == WorkflowState.COMPLETED:
            event_bus.publish(AgentEvent(
                name="run.completed",
                state=WorkflowState.COMPLETED.value,
                message="Agent run completed.",
                payload=RunCompletedPayload(
                    affected_count=len(runtime.affected_files),
                ),
            ))
        elif state == WorkflowState.FAILED:
            raise WorkflowError("Workflow failed.")
    finally:
        if runtime is not None and runtime.plan_path is not None:
            runtime.plan_path.unlink(missing_ok=True)
