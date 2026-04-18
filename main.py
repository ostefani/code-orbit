"""
llm-agent: Apply AI-powered code changes to a local directory using local LLMs.

Usage:
    python main.py --dir ./my-project --prompt "Add type hints to all functions"
    python main.py --dir . --prompt "Refactor the auth module to use JWT" --no-interactive
    python main.py --dir . --prompt "..." --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich import print as rprint

from agent.config import Config
from agent.context import build_context, get_file_tree
from agent.llm import LLMResponseSchema, call_llm
from agent.events import (
    AgentEvent,
    ConfigMessagePayload,
    ContextSkippedPayload,
    ContextSummaryPayload,
    ContextWarningPayload,
    EmptyPayload,
    EventBus,
    LoggingEventSubscriber,
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
    result: LLMResponseSchema, config: Config
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


def main() -> None:
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

    # 2. Call LLM
    event_bus.publish(AgentEvent(
        name="state.changed",
        state="coding",
        message="Calling language model.",
        payload=StateChangedPayload(),
    ))
    try:
        with console.status("[bold magenta]Model is thinking..."):
            result = call_llm(prompt, context_result.context, config)
    except Exception as e:
        event_bus.publish(AgentEvent(
            name="run.failed",
            level="error",
            state="coding",
            message=str(e),
            payload=EmptyPayload(),
        ))
        console.print(f"\n[bold red]Error calling LLM:[/bold red] {e}")
        sys.exit(1)

    try:
        summary, changes = validate_llm_result(result, config)
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

    event_bus.emit(
        "run.proposal_ready",
        RunProposalReadyPayload(summary=summary, change_count=len(changes)),
        state="validated",
        message="Model response validated.",
    )

    if not changes:
        event_bus.publish(AgentEvent(
            name="run.no_changes",
            state="completed",
            message="No changes needed.",
            payload=EmptyPayload(),
        ))
        return

    # 3. Preview
    event_bus.publish(AgentEvent(
        name="state.changed",
        state="previewing",
        message="Previewing proposed changes.",
        payload=StateChangedPayload(),
    ))
    preview_changes(target_dir, changes, event_bus=event_bus)

    # 4. Confirm (if interactive)
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

    # 5. Apply
    event_bus.publish(AgentEvent(
        name="state.changed",
        state="applying",
        message="Applying changes.",
        payload=StateChangedPayload(),
    ))
    rprint("\n[bold green]📝 Applying changes...[/bold green]")
    affected = apply_changes(target_dir, changes, event_bus=event_bus)

    # 6. Commit (optional)
    if config.auto_commit and affected:
        event_bus.publish(AgentEvent(
            name="state.changed",
            state="committing",
            message="Creating git commit.",
            payload=StateChangedPayload(),
        ))
        git_commit(target_dir, summary, affected, event_bus=event_bus)

    event_bus.publish(AgentEvent(
        name="run.completed",
        state="completed",
        message="Agent run completed.",
        payload=RunCompletedPayload(affected_count=len(affected)),
    ))


if __name__ == "__main__":
    main()
