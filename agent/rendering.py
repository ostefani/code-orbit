from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from .events import (
    AgentEvent,
    ApplyFilePayload,
    ConfigMessagePayload,
    ContextSkippedPayload,
    ContextSummaryPayload,
    ContextWarningPayload,
    GitCommitFailedPayload,
    GitCommitSucceededPayload,
    PreviewChangePayload,
    RunCompletedPayload,
    RunProposalReadyPayload,
)


class CliEventRenderer:
    def __init__(self, console: Console) -> None:
        self.console = console

    def __call__(self, event: AgentEvent[object]) -> None:
        if event.name == "config.message" and isinstance(
            event.payload, ConfigMessagePayload
        ):
            style = "yellow" if event.level == "warning" else "dim"
            self.console.print(f"[{style}]{event.payload.text}[/{style}]")
            return

        if event.name == "context.warning" and isinstance(
            event.payload, ContextWarningPayload
        ):
            self.console.print(f"[yellow]Warning:[/yellow] {event.payload.warning}")
            return

        if event.name == "context.summary" and isinstance(
            event.payload, ContextSummaryPayload
        ):
            self.console.print(
                f"\n[dim]Context:[/dim] {event.payload.file_count} files | "
                f"~{event.payload.used_tokens:,} file tokens"
            )
            return

        if event.name == "context.skipped" and isinstance(
            event.payload, ContextSkippedPayload
        ):
            skipped_count = event.payload.skipped_count
            paths = event.payload.paths
            self.console.print(
                f"\n[yellow]Skipped {skipped_count} file(s) due to context limit:[/yellow]"
            )
            for path in paths[:5]:
                self.console.print(f"  - {path}")
            if skipped_count > 5:
                self.console.print(f"  ... and {skipped_count - 5} more")
            return

        if event.name == "run.proposal_ready" and isinstance(
            event.payload, RunProposalReadyPayload
        ):
            rprint(f"\n[bold cyan]💡 Summary:[/bold cyan] {event.payload.summary}")
            rprint(f"[dim]Files to change:[/dim] {event.payload.change_count}")
            return

        if event.name == "run.no_changes":
            rprint("\n[bold green]✅ No changes needed.[/bold green]")
            return

        if event.name == "preview.change":
            self._render_preview_change(event)
            return

        if event.name == "apply.file" and isinstance(event.payload, ApplyFilePayload):
            path = event.payload.path
            action = event.payload.action
            if action == "delete":
                if event.payload.performed:
                    rprint(f"  [red]🗑️  Deleted:[/red] {path}")
                else:
                    rprint(f"  [yellow]⚠️  Delete skipped (not found):[/yellow] {path}")
            else:
                verb = "Created" if action == "create" else "Updated"
                rprint(f"  [green]✅ {verb}:[/green] {path}")
            return

        if event.name == "git.commit_succeeded" and isinstance(
            event.payload, GitCommitSucceededPayload
        ):
            rprint(f"\n[bold blue]📦 Committed:[/bold blue] {event.payload.summary}")
            return

        if event.name == "git.commit_failed" and isinstance(
            event.payload, GitCommitFailedPayload
        ):
            rprint(
                f"\n[bold yellow]⚠️  Git commit failed:[/bold yellow] {event.payload.stderr}"
            )
            return

        if event.name == "run.aborted":
            rprint("[bold red]❌ Aborted.[/bold red]")
            return

        if event.name == "run.completed" and isinstance(
            event.payload, RunCompletedPayload
        ):
            rprint(
                f"\n[bold green]✅ Done![/bold green] "
                f"[white]{event.payload.affected_count} file(s) updated.[/white]"
            )

    def _render_preview_change(self, event: AgentEvent[object]) -> None:
        if not isinstance(event.payload, PreviewChangePayload):
            return

        path = event.payload.path
        action = event.payload.action
        header = {
            "update": "[bold yellow]✏️  UPDATE[/bold yellow]",
            "create": "[bold green]✨ CREATE[/bold green]",
            "delete": "[bold red]🗑️  DELETE[/bold red]",
        }.get(action, f"[bold]{action.upper()}[/bold]")
        width = max(0, self.console.width - len(path) - 20)
        rprint(
            f"\n[bold blue]─[/bold blue][bold]{header}: {path} [/bold]"
            f"[bold blue]{'─' * width}[/bold blue]"
        )

        if action == "delete":
            if event.payload.exists:
                rprint("  [dim]File will be deleted.[/dim]")
            return

        extension = Path(path).suffix[1:] or "text"
        if action == "create":
            syntax = Syntax(
                event.payload.content or "",
                extension,
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
            return

        if event.payload.missing:
            rprint(
                f"  [bold yellow]⚠️  File not found locally, will create:[/bold yellow] {path}"
            )
            syntax = Syntax(
                event.payload.content or "",
                extension,
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
            return

        if event.payload.unchanged:
            rprint("  [dim](no changes detected)[/dim]")
            return

        rprint(
            Panel(
                event.payload.diff_text or "",
                title="[yellow]Diff Preview[/yellow]",
                border_style="yellow",
            )
        )
