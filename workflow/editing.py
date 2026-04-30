import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agent.events import AgentEvent, EmptyPayload, EventBus, StateChangedPayload

from ._state import WorkflowRuntime, WorkflowState
from .planning import load_plan_draft

if TYPE_CHECKING:
    from agent.llm import PlanSchema


def open_plan_in_editor(plan_path: Path) -> "PlanSchema":
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
            print(f"\nInvalid plan file: {exc}", file=sys.stderr)
            try:
                choice = input("Open the editor again to fix the plan? [Y/n] ")
            except EOFError as input_exc:
                raise RuntimeError("Plan editing was aborted.") from input_exc
            if choice.strip().lower() not in {"", "y", "yes"}:
                raise


_default_open_plan_in_editor = open_plan_in_editor


def run_editing_plan_stage(
    runtime: WorkflowRuntime,
    event_bus: EventBus,
    *,
    open_plan_in_editor: Callable[[Path], "PlanSchema"] = _default_open_plan_in_editor,
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
        runtime.approved_plan = open_plan_in_editor(runtime.plan_path)
    except Exception as exc:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state=WorkflowState.EDITING_PLAN.value,
            message=str(exc),
            payload=EmptyPayload(),
        ))
        return WorkflowState.FAILED
    return WorkflowState.EXECUTING
