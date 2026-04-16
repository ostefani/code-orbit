import subprocess
import difflib

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.columns import Columns
from rich import print as rprint

console = Console()


def _safe_path(root: Path, rel_path: str) -> Path:
    """Resolve rel_path under root, raising PermissionError if it escapes."""
    resolved = (root / rel_path).resolve()
    if not resolved.is_relative_to(root):  # Python 3.9+
        raise PermissionError(f"Path traversal detected: {rel_path!r}")
    return resolved


def preview_changes(root: str, changes: list[dict]) -> None:
    """Print a colored diff or syntax-highlighted code for each change."""
    root_path = Path(root).resolve()

    for change in changes:
        path = _safe_path(root_path, change["path"])
        action = change["action"]
        new_content = change.get("content", "")

        action_label = {
            "update": "[bold yellow]✏️  UPDATE[/bold yellow]",
            "create": "[bold green]✨ CREATE[/bold green]",
            "delete": "[bold red]🗑️  DELETE[/bold red]",
        }.get(action, f"[bold]{action.upper()}[/bold]")

        rprint(
            f"\n[bold blue]─[/bold blue][bold]{action_label}: {change['path']} [/bold][bold blue]{'─' * (console.width - len(change['path']) - 20)}[/bold blue]"
        )

        if action == "delete":
            if path.exists():
                rprint(f"  [dim]File will be deleted.[/dim]")
            continue

        if action == "create":
            syntax = Syntax(
                new_content,
                Path(change["path"]).suffix[1:] or "text",
                theme="monokai",
                line_numbers=True,
            )
            rprint(
                Panel(
                    syntax,
                    title="[green]New File Content[/green]",
                    border_style="green",
                )
            )
            continue

        # update
        if path.exists():
            old_content = path.read_text(encoding="utf-8")
            if old_content == new_content:
                rprint("  [dim](no changes detected)[/dim]")
                continue

            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{change['path']}",
                    tofile=f"b/{change['path']}",
                    lineterm="",
                )
            )

            diff_text = ""
            for line in diff:
                if line.startswith("+") and not line.startswith("+++"):
                    diff_text += f"[green]{line}[/green]\n"
                elif line.startswith("-") and not line.startswith("---"):
                    diff_text += f"[red]{line}[/red]\n"
                elif line.startswith("@@"):
                    diff_text += f"[cyan]{line}[/cyan]\n"
                else:
                    diff_text += f"{line}\n"

            rprint(
                Panel(
                    diff_text.strip(),
                    title="[yellow]Diff Preview[/yellow]",
                    border_style="yellow",
                )
            )
        else:
            rprint(
                f"  [bold yellow]⚠️  File not found locally, will create:[/bold yellow] {change['path']}"
            )
            syntax = Syntax(
                new_content,
                Path(change["path"]).suffix[1:] or "text",
                theme="monokai",
                line_numbers=True,
            )
            rprint(
                Panel(
                    syntax,
                    title="[green]New File Content[/green]",
                    border_style="green",
                )
            )


def apply_changes(root: str, changes: list[dict]) -> list[str]:
    """
    Apply all changes to disk. Returns list of affected file paths.
    """
    root_path = Path(root).resolve()
    affected = []

    for change in changes:
        path = _safe_path(root_path, change["path"])
        action = change["action"]

        if action == "delete":
            if path.exists():
                path.unlink()
                rprint(f"  [red]🗑️  Deleted:[/red] {change['path']}")
            else:
                rprint(
                    f"  [yellow]⚠️  Delete skipped (not found):[/yellow] {change['path']}"
                )

        elif action in ("create", "update"):
            content = change.get("content", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            verb = "Created" if action == "create" else "Updated"
            rprint(f"  [green]✅ {verb}:[/green] {change['path']}")

        affected.append(str(path))

    return affected


def git_commit(root: str, message: str, files: list[str]) -> None:
    """Stage affected files and commit."""
    try:
        subprocess.run(
            ["git", "add"] + files, cwd=root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"[llm-agent] {message}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        rprint(f"\n[bold blue]📦 Committed:[/bold blue] {message}")
    except subprocess.CalledProcessError as e:
        rprint(
            f"\n[bold yellow]⚠️  Git commit failed:[/bold yellow] {e.stderr.decode()}"
        )
