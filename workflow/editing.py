import os
import shlex
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from agent.events import AgentEvent, EmptyPayload, EventBus, StateChangedPayload
from agent.llm import PlanSchema

from ._state import WorkflowRuntime, WorkflowState
from .planning import load_plan_draft


def open_plan_in_editor(plan_path: Path, console: Console | None = None) -> PlanSchema:
    console = console or Console()
    while True:
        # Treat EDITOR as a command line, not a shell string.
        # This supports editors with flags (for example, "code --wait")
        # without invoking a shell.
        editor_cmd = shlex.split(os.environ.get("EDITOR", "vim"))
        if not editor_cmd:
            editor_cmd = ["vim"]

        result = subprocess.run([*editor_cmd, str(plan_path)], check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Editor exited with status {result.returncode}. "
                "Plan editing was aborted."
            )

        try:
            return load_plan_draft(plan_path)
        except Exception as exc:
            console.print(f"\n[bold red]Invalid plan file:[/bold red] {exc}")
            if not Confirm.ask(
                "[bold yellow]Open the editor again to fix the plan?[/bold yellow]",
                default=True,
                console=console,
            ):
                raise


def run_editing_plan_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.plan_path is not None
    assert runtime.architect_plan is not None

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.EDITING_PLAN.value,
        message="Opening plan editor.",
        payload=StateChangedPayload(),
    ))
    try:
        runtime.approved_plan = open_plan_in_editor(runtime.plan_path, runtime.console)
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state=WorkflowState.EDITING_PLAN.value,
            message=str(exc),
            payload=EmptyPayload(),
        ))
        raise
    return WorkflowState.EXECUTING
