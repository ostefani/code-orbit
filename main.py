"""
llm-agent: Apply AI-powered code changes to a local directory using local LLMs.

Usage:
    python main.py --dir ./my-project --prompt "Add type hints to all functions"
    python main.py --dir . --prompt "Refactor the auth module to use JWT" --no-interactive
    python main.py --dir . --prompt "..." --config config.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.prompt import Prompt
from rich import print as rprint

from workflow import run_workflow


HISTORY_DIR = Path(".code-orbit")
HISTORY_FILE = HISTORY_DIR / "history.json"
LEGACY_HISTORY_FILE = Path(".code-orbit-history")


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
    path = HISTORY_FILE
    if not path.exists() and LEGACY_HISTORY_FILE.exists():
        path = LEGACY_HISTORY_FILE

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
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history[:20], indent=2))
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
    try:
        args = parse_args()
        history = load_history()
        prompt = args.prompt or get_prompt_interactively(history)
        if not prompt:
            rprint("[bold red]❌ No prompt provided.[/bold red]")
            sys.exit(1)

        save_history(prompt)
        asyncio.run(
            run_workflow(
                target_dir=str(Path(args.dir).resolve()),
                prompt=prompt,
                config_path=args.config,
                profile_name=args.profile,
                no_interactive=args.no_interactive,
                auto_commit=args.auto_commit,
                allow_delete=args.allow_delete,
                tree=args.tree,
            )
        )
    except KeyboardInterrupt:
        rprint("\n[bold red]Stopping...[/bold red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
