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
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from agent.config import Config
from agent.context import build_context, get_file_tree
from agent.llm import (
    LLMResponseSchema,
    PlanSchema,
    call_architect,
    call_coder_for_task,
)
from agent.events import (
    AgentEvent,
    ConfigMessagePayload,
    ContextSkippedPayload,
    ContextSummaryPayload,
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

        if not path.strip():
            raise ValueError(f"Change #{index} is missing a valid 'path' string.")
        if Path(path).is_absolute():
            raise ValueError(f"Change #{index} uses an absolute path: {path!r}")

        normalized_path = path.strip()
        normalized_parts = Path(normalized_path).parts
        if any(part == ".." for part in normalized_parts):
            raise ValueError(
                f"Change #{index} uses parent-directory traversal: {normalized_path!r}"
            )
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

        content = change.content
        assert content is not None
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


async def main() -> None:
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

    # CLI overrides
    if args.no_interactive:
        config.interactive = False
    if args.auto_commit:
        config.auto_commit = True
    if args.allow_delete:
        config.allow_delete = True

    history = load_history()
    prompt = args.prompt
    if not prompt:
        prompt = get_prompt_interactively(history)

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

    # 1. Build context
    event_bus.publish(AgentEvent(
        name="state.changed",
        state="planning",
        message="Building context.",
        payload=StateChangedPayload(),
    ))
    with console.status("[bold green]Analyzing codebase..."):
        context_result = build_context(target_dir, prompt, config)

    for warning in context_result.token_warnings:
        event_bus.publish(AgentEvent(
            name="context.warning",
            level="warning",
            state="planning",
            message=warning,
            payload=ContextWarningPayload(warning=warning),
        ))

    if context_result.skipped_paths:
        event_bus.publish(AgentEvent(
            name="context.skipped",
            level="warning",
            state="planning",
            payload=ContextSkippedPayload(
                skipped_count=len(context_result.skipped_paths),
                paths=tuple(context_result.skipped_paths),
            ),
        ))

    event_bus.publish(AgentEvent(
        name="context.summary",
        state="planning",
        payload=ContextSummaryPayload(
            file_count=len(context_result.entries),
            used_tokens=context_result.used_tokens,
            token_budget=context_result.token_budget,
        ),
    ))

    # 2. Call architect
    event_bus.publish(AgentEvent(
        name="state.changed",
        state="planning",
        message="Drafting implementation plan.",
        payload=StateChangedPayload(),
    ))
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
            plan = await call_architect(
                prompt,
                context_result.context,
                config,
                on_chunk=on_plan_chunk,
            )
    except Exception as e:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="planning",
            message=str(e),
            payload=EmptyPayload(),
        ))
        console.print(f"\n[bold red]Error calling architect:[/bold red] {e}")
        sys.exit(1)

    plan_path = create_plan_draft_path()
    try:
        write_plan_draft(plan_path, plan)
    except OSError as e:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="planning",
            message=str(e),
            payload=EmptyPayload(),
        ))
        console.print(f"\n[bold red]Error saving plan draft:[/bold red] {e}")
        sys.exit(1)

    event_bus.publish(AgentEvent(
        name="plan.ready",
        state="reviewing_plan",
        message="Implementation plan ready.",
        payload=PlanReadyPayload(
            summary=plan.summary,
            task_count=len(plan.tasks),
            draft_path=str(plan_path),
        ),
    ))
    rprint(
        Panel.fit(
            format_plan_for_display(plan),
            title="🧭 plan",
            border_style="cyan",
        )
    )

    try:
        if not plan.tasks:
            event_bus.publish(AgentEvent(
                name="run.no_changes",
                state="completed",
                message="No changes needed.",
                payload=EmptyPayload(),
            ))
            return

        approved_plan = plan
        if config.interactive:
            event_bus.publish(AgentEvent(
                name="state.changed",
                state="editing_plan",
                message="Opening plan editor.",
                payload=StateChangedPayload(),
            ))
            try:
                approved_plan = open_plan_in_editor(plan_path)
            except Exception as e:
                event_bus.publish(AgentEvent(
                    name="run.failed",
                    level="error",
                    state="editing_plan",
                    message=str(e),
                    payload=EmptyPayload(),
                ))
                console.print(f"\n[bold red]Plan editing aborted:[/bold red] {e}")
                sys.exit(1)

        # 3. Call coder once per approved task
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="coding",
            message="Generating file replacements.",
            payload=StateChangedPayload(),
        ))
        all_changes: list[dict[str, str]] = []
        task_summaries: list[str] = []
        working_context = context_result.context

        for index, task in enumerate(approved_plan.tasks, 1):
            event_bus.publish(AgentEvent(
                name="state.changed",
                state="coding_task",
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
                    console=console,
                )
                task_id = progress.add_task(
                    f"Coder is streaming task {index}/{len(approved_plan.tasks)}...",
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
                            f"{len(approved_plan.tasks)}... ({chunk_count} chunks)"
                        ),
                    )

                with Live(progress, console=console, refresh_per_second=12):
                    result = await call_coder_for_task(
                        approved_plan,
                        task,
                        working_context,
                        config,
                        on_chunk=on_chunk,
                    )
            except Exception as e:
                event_bus.publish(AgentEvent(
                    name="run.failed",
                    level="error",
                    state="coding",
                    message=str(e),
                    payload=EmptyPayload(),
                ))
                console.print(f"\n[bold red]Error calling coder:[/bold red] {e}")
                sys.exit(1)

            try:
                summary, changes = validate_llm_result(
                    result,
                    config,
                )
            except ValueError as e:
                event_bus.publish(AgentEvent(
                    name="run.failed",
                    level="error",
                    state="validating",
                    message=str(e),
                    payload=EmptyPayload(),
                ))
                console.print(f"\n[bold red]Invalid model response:[/bold red] {e}")
                sys.exit(1)

            if summary:
                task_summaries.append(summary)
            all_changes.extend(changes)
            working_context = build_working_context(
                context_result.context,
                all_changes,
            )

            if changes:
                rprint(
                    f"[green]✓[/green] Task {index}/{len(approved_plan.tasks)}: "
                    f"{summary} ({len(changes)} file(s))"
                )
            else:
                rprint(
                    f"[yellow]⚠[/yellow] Task {index}/{len(approved_plan.tasks)}: "
                    f"No changes required for '{task.goal}'"
                )

        if not all_changes:
            event_bus.publish(AgentEvent(
                name="run.no_changes",
                state="completed",
                message="No changes needed.",
                payload=EmptyPayload(),
            ))
            return

        final_summary = approved_plan.summary
        if len(task_summaries) == 1:
            final_summary = task_summaries[0]

        event_bus.emit(
            "run.proposal_ready",
            RunProposalReadyPayload(
                summary=final_summary,
                change_count=len(all_changes),
            ),
            state="validated",
            message="Model response validated.",
        )

        # 4. Preview
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="previewing",
            message="Previewing proposed changes.",
            payload=StateChangedPayload(),
        ))
        preview_changes(target_dir, all_changes, event_bus=event_bus)

        # 5. Confirm (if interactive)
        if config.interactive:
            event_bus.publish(AgentEvent(
                name="state.changed",
                state="waiting_for_user",
                message="Waiting for user confirmation.",
                payload=StateChangedPayload(),
            ))
            print()
            if not Confirm.ask("[bold yellow]Apply these changes?[/bold yellow]"):
                event_bus.publish(AgentEvent(
                    name="run.aborted",
                    level="warning",
                    state="waiting_for_user",
                    message="User aborted run.",
                    payload=EmptyPayload(),
                ))
                sys.exit(0)

        # 6. Apply
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="applying",
            message="Applying changes.",
            payload=StateChangedPayload(),
        ))
        rprint("\n[bold green]📝 Applying changes...[/bold green]")
        affected = apply_changes(target_dir, all_changes, event_bus=event_bus)

        # 7. Commit (optional)
        if config.auto_commit and affected:
            event_bus.publish(AgentEvent(
                name="state.changed",
                state="committing",
                message="Creating git commit.",
                payload=StateChangedPayload(),
            ))
            git_commit(target_dir, final_summary, affected, event_bus=event_bus)

        event_bus.publish(AgentEvent(
            name="run.completed",
            state="completed",
            message="Agent run completed.",
            payload=RunCompletedPayload(affected_count=len(affected)),
        ))
    finally:
        plan_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rprint("\n[bold red]Stopping...[/bold red]")
        sys.exit(0)
