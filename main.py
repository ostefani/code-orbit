"""
llm-agent: Apply AI-powered code changes to a local directory using local LLMs.

Usage:
    python main.py --dir ./my-project --prompt "Add type hints to all functions"
    python main.py --dir . --prompt "Refactor the auth module to use JWT" --no-interactive
    python main.py --dir . --prompt "..." --config config.yaml
"""

import argparse
import sys
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.prompt import Confirm, Prompt
from rich import print as rprint

from agent.config import Config
from agent.context import build_context, get_file_tree
from agent.llm import call_llm
from agent.patcher import apply_changes, git_commit, preview_changes

console = Console()
HISTORY_FILE = ".code-orbit-history"


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


def main() -> None:
    args = parse_args()
    try:
        config = Config.load(args.config, profile_name=args.profile)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    # CLI overrides
    if args.no_interactive:
        config.interactive = False
    if args.auto_commit:
        config.auto_commit = True

    history = load_history()
    prompt = args.prompt
    if not prompt:
        prompt = get_prompt_interactively(history)

    if not prompt:
        rprint("[bold red]❌ No prompt provided.[/bold red]")
        sys.exit(1)

    save_history(prompt)

    target_dir = str(Path(args.dir).resolve())

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
    with console.status("[bold green]Analyzing codebase..."):
        _, context_str = build_context(target_dir, prompt, config)

    # 2. Call LLM
    try:
        with console.status("[bold magenta]Model is thinking..."):
            result = call_llm(prompt, context_str, config)
    except Exception as e:
        console.print(f"\n[bold red]Error calling LLM:[/bold red] {e}")
        sys.exit(1)

    summary = result.get("summary", "No summary provided.")
    changes = result.get("changes", [])

    rprint(f"\n[bold cyan]💡 Summary:[/bold cyan] {summary}")
    rprint(f"[dim]Files to change:[/dim] {len(changes)}")

    if not changes:
        rprint("\n[bold green]✅ No changes needed.[/bold green]")
        return

    # 3. Preview
    preview_changes(target_dir, changes)

    # 4. Confirm (if interactive)
    if config.interactive:
        print()
        if not Confirm.ask("[bold yellow]Apply these changes?[/bold yellow]"):
            rprint("[bold red]❌ Aborted.[/bold red]")
            sys.exit(0)

    # 5. Apply
    rprint("\n[bold green]📝 Applying changes...[/bold green]")
    affected = apply_changes(target_dir, changes)

    # 6. Commit (optional)
    if config.auto_commit and affected:
        git_commit(target_dir, summary, affected)

    rprint(
        f"\n[bold green]✅ Done![/bold green] [white]{len(affected)} file(s) updated.[/white]"
    )


if __name__ == "__main__":
    main()
