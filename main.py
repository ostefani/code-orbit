"""
llm-agent: Apply AI-powered code changes to a local directory using local LLMs.

Usage:
    python main.py --dir ./my-project --prompt "Add type hints to all functions"
    python main.py --dir . --prompt "Refactor the auth module to use JWT" --no-interactive
    python main.py --dir . --prompt "..." --config config.yaml
"""

import argparse
import os
import json
import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from agent.config import Config
from agent.context import ContextBuildResult, build_context_async, get_file_tree
from agent.llm import (
    LLMResponseSchema,
    PlanSchema,
    call_architect,
    call_coder_for_task,
    format_plan_roadmap,
)
from agent.events import (
    AgentEvent,
    ConfigMessagePayload,
    ContextSkippedPayload,
    ContextSummaryPayload,
    ContextSemanticMatchItem,
    ContextSemanticMatchPayload,
    ContextWarningPayload,
    EmptyPayload,
    EventBus,
    LoggingEventSubscriber,
    PlanReadyPayload,
    RunCompletedPayload,
    RunStartedPayload,
    RunProposalReadyPayload,
    StateChangedPayload,
    build_event_logger,
)
from agent.patcher import apply_changes, git_commit, preview_changes
from agent.rendering import CliEventRenderer

console = Console()
HISTORY_FILE = ".code-orbit-history"
ALLOWED_ACTIONS = {"create", "update", "delete"}
MAX_REPLAN_ATTEMPTS = 3


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


@dataclass
class WorkflowRuntime:
    target_dir: str
    prompt: str
    config: Config
    context_result: ContextBuildResult | None = None
    plan_path: Path | None = None
    architect_plan: PlanSchema | None = None
    approved_plan: PlanSchema | None = None
    all_changes: list[dict[str, str]] = field(default_factory=list)
    task_summaries: list[str] = field(default_factory=list)
    working_context: str = ""
    execution_feedback: str | None = None
    replan_attempts: int = 0
    final_summary: str = ""
    affected_files: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-powered codebase editor using local llama.cpp models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=".",
        help="Target directory to edit (default: current directory)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        help="The change you want to make (e.g. 'Add logging to all endpoints')",
    )
    parser.add_argument(
        "--profile",
        "-P",
        help="Configuration profile to use (defined in config.yaml)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Apply changes without confirmation prompt",
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Auto git commit after applying changes",
    )
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="Allow LLM-proposed delete actions",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print file tree and exit (useful for debugging context)",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="code-orbit 0.1.0",
    )
    return parser.parse_args()


def load_history() -> list[str]:
    path = Path(HISTORY_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_history(prompt: str) -> None:
    history = load_history()
    if prompt in history:
        history.remove(prompt)
    history.insert(0, prompt)
    try:
        Path(HISTORY_FILE).write_text(json.dumps(history[:20], indent=2))
    except OSError:
        pass


def get_prompt_interactively(history: list[str]) -> str:
    if history:
        rprint("\n[bold cyan]Recent Prompts:[/bold cyan]")
        for i, h in enumerate(history[:5], 1):
            rprint(f"  [dim]{i}.[/dim] {h}")
        rprint("  [dim]0. Enter new prompt[/dim]")

        choice = Prompt.ask("\nChoose a prompt or enter new one", default="0")
        if choice.isdigit() and 0 < int(choice) <= len(history):
            return history[int(choice) - 1]

    return Prompt.ask(
        "\n[bold yellow]What change would you like to make?[/bold yellow]"
    )


def validate_llm_result(
    result: LLMResponseSchema,
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


def create_plan_draft_path() -> Path:
    fd, temp_name = tempfile.mkstemp(prefix="code-orbit-plan-", suffix=".json")
    os.close(fd)
    return Path(temp_name)


def write_plan_draft(plan_path: Path, plan: PlanSchema) -> None:
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan_draft(plan_path: Path) -> PlanSchema:
    return PlanSchema.model_validate_json(plan_path.read_text(encoding="utf-8"))


def open_plan_in_editor(plan_path: Path) -> PlanSchema:
    while True:
        editor = os.environ.get("EDITOR", "vim")
        result = subprocess.run([editor, str(plan_path)], check=False)
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
            ):
                raise


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


def build_architect_prompt(prompt: str, execution_feedback: str | None = None) -> str:
    if not execution_feedback:
        return prompt
    return (
        f"{prompt}\n\n<execution_feedback>\n"
        f"{execution_feedback}\n</execution_feedback>"
    )


def format_execution_feedback(
    error: str,
    partial_changes: list[dict[str, str]],
) -> str:
    parts = [f"Execution failed with error: {error}"]
    if partial_changes:
        parts.append("Partial task outputs so far:")
        parts.append(render_applied_changes_context(partial_changes))
    return "\n".join(parts)


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


def reset_execution_state(runtime: WorkflowRuntime) -> None:
    runtime.all_changes.clear()
    runtime.task_summaries.clear()
    runtime.final_summary = ""
    if runtime.context_result is not None:
        runtime.working_context = runtime.context_result.context


async def run_build_context_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.BUILDING_CONTEXT.value,
        message="Building context.",
        payload=StateChangedPayload(),
    ))
    with console.status("[bold green]Analyzing codebase..."):
        runtime.context_result = await build_context_async(
            runtime.target_dir,
            runtime.prompt,
            runtime.config,
            event_bus=event_bus,
        )
    runtime.working_context = runtime.context_result.context

    for warning in runtime.context_result.token_warnings:
        event_bus.publish(AgentEvent(
            name="context.warning",
            level="warning",
            state=WorkflowState.BUILDING_CONTEXT.value,
            message=warning,
            payload=ContextWarningPayload(warning=warning),
        ))

    if runtime.context_result.skipped_paths:
        event_bus.publish(AgentEvent(
            name="context.skipped",
            level="warning",
            state=WorkflowState.BUILDING_CONTEXT.value,
            payload=ContextSkippedPayload(
                skipped_count=len(runtime.context_result.skipped_paths),
                paths=tuple(runtime.context_result.skipped_paths),
            ),
        ))

    event_bus.publish(AgentEvent(
        name="context.summary",
        state=WorkflowState.BUILDING_CONTEXT.value,
        payload=ContextSummaryPayload(
            file_count=len(runtime.context_result.entries),
            used_tokens=runtime.context_result.used_tokens,
            token_budget=runtime.context_result.token_budget,
            context_window_tokens=(
                runtime.context_result.budget_breakdown.context_window_tokens
                if runtime.context_result.budget_breakdown is not None
                else runtime.config.max_context_tokens
            ),
            response_reserve_tokens=(
                runtime.context_result.budget_breakdown.response_reserve_tokens
                if runtime.context_result.budget_breakdown is not None
                else runtime.config.max_response_tokens
            ),
            scaffold_tokens=(
                runtime.context_result.budget_breakdown.scaffold_tokens
                if runtime.context_result.budget_breakdown is not None
                else 0
            ),
            safety_margin_tokens=(
                runtime.context_result.budget_breakdown.safety_margin_tokens
                if runtime.context_result.budget_breakdown is not None
                else 0
            ),
        ),
    ))

    if runtime.context_result.semantic_matches:
        event_bus.publish(AgentEvent(
            name="context.semantic_matches",
            level="debug",
            state=WorkflowState.BUILDING_CONTEXT.value,
            message="Semantic matches selected for context.",
            payload=ContextSemanticMatchPayload(
                prompt=runtime.prompt,
                selected_count=len(runtime.context_result.semantic_matches),
                matches=tuple(
                    ContextSemanticMatchItem(
                        path=match.path,
                        semantic_score=match.semantic_score,
                        lexical_score=match.lexical_score,
                        blended_score=match.blended_score,
                    )
                    for match in runtime.context_result.semantic_matches
                ),
            ),
        ))
    return WorkflowState.PLANNING


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
            console=console,
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

        with Live(progress, console=console, refresh_per_second=12):
            runtime.architect_plan = await call_architect(
                prompt,
                runtime.context_result.context,
                runtime.config,
                on_chunk=on_plan_chunk,
            )
    except Exception as e:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state=WorkflowState.PLANNING.value,
            message=str(e),
            payload=EmptyPayload(),
        ))
        raise

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
    rprint(
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
        runtime.approved_plan = open_plan_in_editor(runtime.plan_path)
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


async def run_execution_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.context_result is not None
    assert runtime.approved_plan is not None

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.EXECUTING.value,
        message="Generating file replacements.",
        payload=StateChangedPayload(),
    ))

    runtime.all_changes = []
    runtime.task_summaries = []
    runtime.working_context = runtime.context_result.context

    for index, task in enumerate(runtime.approved_plan.tasks, 1):
        event_bus.publish(AgentEvent(
            name="state.changed",
            state=WorkflowState.EXECUTING.value,
            message=(
                f"Generating file replacements for task {index}/"
                f"{len(runtime.approved_plan.tasks)}."
            ),
            payload=StateChangedPayload(),
        ))
        try:
            progress = Progress(
                SpinnerColumn(style="bold magenta"),
                TextColumn("{task.description}"),
                transient=True,
                console=console,
            )
            task_id = progress.add_task(
                f"Coder is streaming task {index}/{len(runtime.approved_plan.tasks)}...",
                total=None,
            )
            chunk_count = 0

            def on_chunk(_chunk: str) -> None:
                nonlocal chunk_count
                chunk_count += 1
                progress.update(
                    task_id,
                    description=(
                        f"Coder is streaming task {index}/"
                        f"{len(runtime.approved_plan.tasks)}... ({chunk_count} chunks)"
                    ),
                )

            with Live(progress, console=console, refresh_per_second=12):
                result = await call_coder_for_task(
                    runtime.approved_plan,
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
            rprint(
                f"[green]✓[/green] Task {index}/{len(runtime.approved_plan.tasks)}: "
                f"{summary} ({len(changes)} file(s))"
            )
        else:
            rprint(
                f"[yellow]⚠[/yellow] Task {index}/{len(runtime.approved_plan.tasks)}: "
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

    runtime.final_summary = runtime.approved_plan.summary
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


def run_review_diff_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    assert runtime.approved_plan is not None

    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.REVIEWING_DIFF.value,
        message="Previewing proposed changes.",
        payload=StateChangedPayload(),
    ))
    preview_changes(runtime.target_dir, runtime.all_changes, event_bus=event_bus)

    if runtime.config.interactive:
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="waiting_for_user",
            message="Waiting for user confirmation.",
            payload=StateChangedPayload(),
        ))
        print()
        if not Confirm.ask("[bold yellow]Apply these changes?[/bold yellow]"):
            reset_execution_state(runtime)
            return WorkflowState.EDITING_PLAN

    return WorkflowState.APPLYING


def run_applying_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    event_bus.publish(AgentEvent(
        name="state.changed",
        state=WorkflowState.APPLYING.value,
        message="Applying changes.",
        payload=StateChangedPayload(),
    ))
    rprint("\n[bold green]📝 Applying changes...[/bold green]")
    affected = apply_changes(runtime.target_dir, runtime.all_changes, event_bus=event_bus)
    runtime.final_summary = runtime.final_summary or runtime.approved_plan.summary
    runtime.affected_files = affected
    return WorkflowState.COMMITTING if runtime.config.auto_commit and affected else WorkflowState.COMPLETED


def run_committing_stage(
    runtime: WorkflowRuntime, event_bus: EventBus
) -> WorkflowState:
    affected = runtime.affected_files
    if affected:
        event_bus.publish(AgentEvent(
            name="state.changed",
            state=WorkflowState.COMMITTING.value,
            message="Creating git commit.",
            payload=StateChangedPayload(),
        ))
        git_commit(
            runtime.target_dir,
            f"{runtime.final_summary}\n\n{format_plan_roadmap(runtime.approved_plan)}",
            affected,
            event_bus=event_bus,
        )
    return WorkflowState.COMPLETED


async def run_workflow() -> None:
    event_bus = EventBus()
    event_bus.subscribe(LoggingEventSubscriber(build_event_logger()))
    event_bus.subscribe(CliEventRenderer(console))

    args = parse_args()
    try:
        config_result = Config.load_with_diagnostics(
            args.config, profile_name=args.profile
        )
        config = config_result.config
    except Exception as e:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="loading_config",
            message=str(e),
            payload=EmptyPayload(),
        ))
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    for message in config_result.messages:
        event_bus.publish(AgentEvent(
            name="config.message",
            level=message.level,
            state="loading_config",
            message=message.text,
            payload=ConfigMessagePayload(text=message.text),
        ))

    if args.no_interactive:
        config.interactive = False
    if args.auto_commit:
        config.auto_commit = True
    if args.allow_delete:
        config.allow_delete = True

    history = load_history()
    prompt = args.prompt or get_prompt_interactively(history)
    if not prompt:
        rprint("[bold red]❌ No prompt provided.[/bold red]")
        sys.exit(1)

    save_history(prompt)

    target_dir = str(Path(args.dir).resolve())
    event_bus.publish(AgentEvent(
        name="run.started",
        state="starting",
        message="Agent run started.",
        payload=RunStartedPayload(target_dir=target_dir, model=config.model),
    ))

    rprint(
        Panel.fit(
            f"[bold blue]Code Orbit[/bold blue]\n"
            f"[dim]Target :[/dim] [green]{target_dir}[/green]\n"
            f"[dim]Model  :[/dim] [magenta]{config.api_base} ({config.model})[/magenta]\n"
            f"[dim]Prompt :[/dim] [yellow]{prompt}[/yellow]",
            title="🔧 settings",
            border_style="blue",
        )
    )

    if args.tree:
        rprint(
            Panel(
                get_file_tree(target_dir, config),
                title="📂 File Tree",
                border_style="dim",
            )
        )
        return

    runtime = WorkflowRuntime(target_dir=target_dir, prompt=prompt, config=config)
    state = WorkflowState.BUILDING_CONTEXT
    try:
        while state not in {WorkflowState.COMPLETED, WorkflowState.FAILED}:
            match state:
                case WorkflowState.BUILDING_CONTEXT:
                    state = await run_build_context_stage(runtime, event_bus)
                case WorkflowState.PLANNING:
                    state = await run_planning_stage(runtime, event_bus)
                case WorkflowState.EDITING_PLAN:
                    try:
                        state = run_editing_plan_stage(runtime, event_bus)
                    except Exception:
                        state = WorkflowState.FAILED
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
    finally:
        if runtime.plan_path is not None:
            runtime.plan_path.unlink(missing_ok=True)




if __name__ == "__main__":
    try:
        asyncio.run(run_workflow())
    except KeyboardInterrupt:
        rprint("\n[bold red]Stopping...[/bold red]")
        sys.exit(0)
