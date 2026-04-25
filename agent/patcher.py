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


def _resolve_path_under_root(resolved_root: Path, file_path: str) -> Path:
    """Resolve a path under an already-resolved repository root."""
    assert resolved_root.is_absolute(), "resolved_root must be an absolute Path."

    resolved_candidate = (resolved_root / Path(file_path)).resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise PermissionError(
            f"Path traversal detected: {file_path!r}"
    )
    return resolved_candidate


def _relativize_resolved_path(resolved_root: Path, resolved_path: Path) -> str:
    """Convert an already-resolved path under a resolved root to a relative path."""
    assert resolved_root.is_absolute(), "resolved_root must be an absolute Path."
    assert resolved_path.is_absolute(), "resolved_path must be an absolute Path."
    return str(resolved_path.relative_to(resolved_root))


def _relativize_path(resolved_root: Path, file_path: str) -> str:
    """Convert a file path to a repo-relative path under a resolved root."""
    return _relativize_resolved_path(
        resolved_root,
        _resolve_path_under_root(resolved_root, file_path),
    )


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
        path = _resolve_path_under_root(root_path, change.path)
        action = change.action

        if action == "delete":
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
                diff_text += f"{line}\n"

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
        path = _resolve_path_under_root(root_path, change.path)
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
            if performed:
                affected.append(_relativize_resolved_path(root_path, path))
            continue

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
            src_path = _resolve_path_under_root(root_path, src)
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
            src_path = _resolve_path_under_root(root_path, src)
            path.parent.mkdir(parents=True, exist_ok=True)
            # shutil.move falls back to copy+delete on cross-device moves, so
            # this is not atomic across filesystems.
            shutil.move(str(src_path), str(path))
            affected.append(_relativize_resolved_path(root_path, src_path))
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

        affected.append(_relativize_resolved_path(root_path, path))

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
        raise
