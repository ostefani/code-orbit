from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Required, TypedDict

from rich.console import Console

from agent.chat import ChatAdapter
from agent.config import Config
from agent.context import ContextBuildResult
from agent.llm import PlanSchema


class WorkflowState(str, Enum):
    BUILDING_CONTEXT = "building_context"
    PLANNING = "planning"
    EDITING_PLAN = "editing_plan"
    EXECUTING = "executing"
    REVIEWING_DIFF = "reviewing_diff"
    APPLYING = "applying"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileChange(TypedDict, total=False):
    """Represents a single file change produced by the coder model.

    ``path`` and ``action`` are always present. ``content`` is present for
    ``create`` and ``update`` actions and absent for ``delete`` actions.
    """
    path: Required[str]
    action: Required[str]
    content: str


@dataclass
class WorkflowRuntime:
    target_dir: str
    prompt: str
    config: Config
    console: Console = field(default_factory=Console)
    chat_adapter: ChatAdapter | None = None
    context_result: ContextBuildResult | None = None
    plan_path: Path | None = None
    architect_plan: PlanSchema | None = None
    approved_plan: PlanSchema | None = None
    all_changes: list[FileChange] = field(default_factory=list)
    task_summaries: list[str] = field(default_factory=list)
    working_context: str = ""
    execution_feedback: str | None = None
    replan_attempts: int = 0
    final_summary: str = ""
    affected_files: list[str] = field(default_factory=list)


def reset_execution_state(runtime: WorkflowRuntime) -> None:
    runtime.all_changes.clear()
    runtime.task_summaries.clear()
    runtime.final_summary = ""
    if runtime.context_result is not None:
        runtime.working_context = runtime.context_result.context


def require_approved_plan(runtime: WorkflowRuntime) -> PlanSchema:
    approved_plan = runtime.approved_plan
    if approved_plan is None:
        raise ValueError("Approved plan is not available.")
    return approved_plan
