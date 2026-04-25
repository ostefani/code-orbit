import subprocess
import difflib
import shutil

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .events import (
    AgentEvent,
    ApplyFilePayload,
    EventBus,
    GitCommitFailedPayload,
    GitCommitSucceededPayload,
    PreviewChangePayload,
)
from .schemas import CodeChangeSchema

CodeChangeInput = CodeChangeSchema | dict[str, Any]


def _safe_path(root: Path, rel_path: str) -> Path:
    """Resolve rel_path under root, raising PermissionError if it escapes."""
    resolved = (root / rel_path).resolve()
    if not resolved.is_relative_to(root):  # Python 3.9+
        raise PermissionError(f"Path traversal detected: {rel_path!r}")
    return resolved


def _relativize_path(resolved_root: Path, file_path: str) -> str:
    assert resolved_root.is_absolute(), "resolved_root must be an absolute Path."

    candidate = Path(file_path)
    resolved_candidate = (resolved_root / candidate).resolve()
    try:
        return str(resolved_candidate.relative_to(resolved_root))
    except ValueError as exc:
        raise ValueError(
            f"Path {file_path!r} is outside repository root {str(resolved_root)!r}"
        ) from exc


def _validate_changes(changes: list[CodeChangeInput]) -> list[CodeChangeSchema]:
    validated: list[CodeChangeSchema] = []
    for index, change in enumerate(changes, 1):
        if isinstance(change, CodeChangeSchema):
            validated.append(change)
            continue

        try:
            validated.append(CodeChangeSchema.model_validate(change))
        except ValidationError as exc:
            message = exc.errors()[0]["msg"]
            raise ValueError(f"Invalid change #{index}: {message}") from exc
    return validated


def preview_changes(
    root: str,
    changes: list[CodeChangeInput],
    event_bus: EventBus | None = None,
) -> None:
    """Emit preview events for each change."""
    if event_bus is None:
        return

    root_path = Path(root).resolve()

    for change in _validate_changes(changes):
        path = _safe_path(root_path, change.path)
        action = change.action

        if action == "delete":
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change.path,
                        action=action,
                        exists=path.exists(),
                    ),
                ))
            continue

        if action == "create":
            content = change.content
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change.path,
                        action=action,
                        content=content,
                    ),
                ))
            continue

        if action == "mkdir":
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="preview.change",
                    state="previewing",
                    payload=PreviewChangePayload(
                        path=change.path,
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
                        path=change.path,
                        action=action,
                        src=change.src,
                        exists=path.exists(),
                    ),
                ))
            continue

        # update
        new_content = change.content
        if path.exists():
            old_content = path.read_text(encoding="utf-8")
            if old_content == new_content:
                if event_bus:
                    event_bus.publish(AgentEvent(
                        name="preview.change",
                        state="previewing",
                        payload=PreviewChangePayload(
                            path=change.path,
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
                    fromfile=f"a/{change.path}",
                    tofile=f"b/{change.path}",
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
                        path=change.path,
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
                        path=change.path,
                        action=action,
                        content=new_content,
                        missing=True,
                    ),
                ))


def apply_changes(
    root: str,
    changes: list[CodeChangeInput],
    event_bus: EventBus | None = None,
) -> list[str]:
    """
    Apply all changes to disk. Returns list of affected file paths.
    """
    root_path = Path(root).resolve()
    affected = []

    for change in _validate_changes(changes):
        path = _safe_path(root_path, change.path)
        action = change.action

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
                        path=change.path,
                        action=action,
                        performed=performed,
                    ),
                ))

        elif action in ("create", "update"):
            content = change.content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change.path,
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
                        path=change.path,
                        action=action,
                        performed=True,
                    ),
                ))

        elif action == "copy":
            src = change.src
            if src is None:
                raise ValueError(f"copy action for {change.path!r} is missing 'src'.")
            src_path = _safe_path(root_path, src)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, path)
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change.path,
                        action=action,
                        performed=True,
                    ),
                ))

        elif action == "move":
            src = change.src
            if src is None:
                raise ValueError(f"move action for {change.path!r} is missing 'src'.")
            src_path = _safe_path(root_path, src)
            path.parent.mkdir(parents=True, exist_ok=True)
            # shutil.move falls back to copy+delete on cross-device moves, so
            # this is not atomic across filesystems.
            shutil.move(str(src_path), str(path))
            affected.append(_relativize_path(root_path, str(src_path)))
            if event_bus:
                event_bus.publish(AgentEvent(
                    name="apply.file",
                    state="applying",
                    payload=ApplyFilePayload(
                        path=change.path,
                        action=action,
                        performed=True,
                    ),
                ))

        affected.append(_relativize_path(root_path, str(path)))

    return affected


def git_commit(
    root: str, message: str, files: list[str], event_bus: EventBus | None = None
) -> None:
    """Stage affected files and commit."""
    root_path = Path(root).resolve()
    relative_files: list[str] = []
    for file_path in files:
        relative_files.append(_relativize_path(root_path, file_path))

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
