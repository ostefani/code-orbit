import subprocess
import difflib
import shutil

from pathlib import Path

from .events import (
    AgentEvent,
    ApplyFilePayload,
    EventBus,
    GitCommitFailedPayload,
    GitCommitSucceededPayload,
    PreviewChangePayload,
)


def _safe_path(root: Path, rel_path: str) -> Path:
    """Resolve rel_path under root, raising PermissionError if it escapes."""
    resolved = (root / rel_path).resolve()
    if not resolved.is_relative_to(root):  # Python 3.9+
        raise PermissionError(f"Path traversal detected: {rel_path!r}")
    return resolved


def preview_changes(root: str, changes: list[dict], event_bus: EventBus | None = None) -> None:
    """Emit preview events for each change."""
    if event_bus is None:
        return

    root_path = Path(root).resolve()

    for change in changes:
        path = _safe_path(root_path, change["path"])
        action = change["action"]
        new_content = change.get("content", "")

        if action == "delete":
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        exists=path.exists(),
                    ),
                ))
            continue

        if action == "create":
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        content=new_content,
                    ),
                ))
            continue

        if action == "mkdir":
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        exists=path.exists(),
                    ),
                ))
            continue

        if action in ("copy", "move"):
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        src=change.get("src"),
                        exists=path.exists(),
                    ),
                ))
            continue

        # update
        if path.exists():
            old_content = path.read_text(encoding="utf-8")
            if old_content == new_content:
                if event_bus:
                    event_bus.publish(AgentEvent(
                        name="preview.change",
                        state="previewing",
                        payload=PreviewChangePayload(
                            path=change["path"],
                            action=action,
                            unchanged=True,
                        ),
                    ))
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

            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        diff_text=diff_text.strip(),
                    ),
                ))
        else:
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change["path"],
                        action=action,
                        content=new_content,
                        missing=True,
                    ),
                ))


def apply_changes(
    root: str, changes: list[dict], event_bus: EventBus | None = None
) -> list[str]:
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
                performed = True
            else:
                performed = False
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change["path"],
                        action=action,
                        performed=performed,
                    ),
                ))

        elif action in ("create", "update"):
            content = change.get("content", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change["path"],
                        action=action,
                        performed=True,
                    ),
                ))

        elif action == "mkdir":
            path.mkdir(parents=True, exist_ok=True)
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change["path"],
                        action=action,
                        performed=True,
                    ),
                ))

        elif action == "copy":
            src_path = _safe_path(root_path, change["src"])
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, path)
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change["path"],
                        action=action,
                        performed=True,
                    ),
                ))

        elif action == "move":
            src_path = _safe_path(root_path, change["src"])
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(path))
            affected.append(str(src_path))
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change["path"],
                        action=action,
                        performed=True,
                    ),
                ))

        affected.append(str(path))

    return affected


def git_commit(
    root: str, message: str, files: list[str], event_bus: EventBus | None = None
) -> None:
    """Stage affected files and commit."""
    root_path = Path(root).resolve()
    relative_files: list[str] = []
    for file_path in files:
        candidate = Path(file_path)
        if candidate.is_absolute():
            relative_files.append(str(candidate.resolve().relative_to(root_path)))
        else:
            relative_files.append(str(candidate))

    try:
        subprocess.run(
            ["git", "add"] + relative_files, cwd=root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"[llm-agent] {message}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        if event_bus:
            event_bus.publish(AgentEvent(
                name="git.commit_succeeded",
                state="committing",
                message="Git commit created.",
                payload=GitCommitSucceededPayload(
                    files=tuple(relative_files),
                    summary=message,
                ),
            ))
    except subprocess.CalledProcessError as e:
        if event_bus:
            event_bus.publish(AgentEvent(
                name="git.commit_failed",
                level="warning",
                state="committing",
                message="Git commit failed.",
                payload=GitCommitFailedPayload(stderr=e.stderr.decode()),
            ))
