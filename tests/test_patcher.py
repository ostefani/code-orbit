import subprocess

import pytest
from pathlib import Path
from agent.config import Config
from agent.patcher import apply_changes, git_commit, preview_changes
from agent.events import ApplyFilePayload, EventBus, PreviewChangePayload
from agent.schemas import CodeChangeSchema


@pytest.fixture
def temp_project(tmp_path):
    (tmp_path / "existing.py").write_text("print('old')")
    return tmp_path


def test_apply_create(temp_project):
    changes = [
        CodeChangeSchema(
            path="new.py",
            action="create",
            content="print('new')",
        )
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "new.py" in affected
    assert (temp_project / "new.py").read_text() == "print('new')"


def test_apply_accepts_code_change_schema(temp_project):
    changes = [
        CodeChangeSchema(
            path="typed.py",
            action="create",
            content="print('typed')",
        )
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "typed.py" in affected
    assert (temp_project / "typed.py").read_text() == "print('typed')"


def test_apply_create_respects_max_content_bytes_boundary(temp_project):
    config = Config(max_content_bytes=5)

    under_limit = "a" * 4
    exact_limit = "a" * 5

    under_affected = apply_changes(
        str(temp_project),
        [
            CodeChangeSchema(
                path="under.txt",
                action="create",
                content=under_limit,
            )
        ],
        config=config,
    )
    exact_affected = apply_changes(
        str(temp_project),
        [
            CodeChangeSchema(
                path="exact.txt",
                action="create",
                content=exact_limit,
            )
        ],
        config=config,
    )

    assert "under.txt" in under_affected
    assert "exact.txt" in exact_affected
    assert (temp_project / "under.txt").read_text(encoding="utf-8") == under_limit
    assert (temp_project / "exact.txt").read_text(encoding="utf-8") == exact_limit


def test_preview_unchanged_update_accepts_code_change_schema(temp_project):
    changes = [
        CodeChangeSchema(
            path="existing.py",
            action="update",
            content="print('old')",
        )
    ]
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    preview_changes(str(temp_project), changes, event_bus=bus)

    assert len(events) == 1
    assert events[0].name == "preview.change"
    assert isinstance(events[0].payload, PreviewChangePayload)
    assert events[0].payload.path == "existing.py"
    assert events[0].payload.unchanged is True


def test_preview_update_diff_text_is_plain(temp_project):
    changes = [
        CodeChangeSchema(
            path="existing.py",
            action="update",
            content="print('new')",
        )
    ]
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    preview_changes(str(temp_project), changes, event_bus=bus)

    assert len(events) == 1
    payload = events[0].payload
    assert isinstance(payload, PreviewChangePayload)
    assert payload.diff_text is not None
    assert "[green]" not in payload.diff_text
    assert "[red]" not in payload.diff_text
    assert "--- a/existing.py" in payload.diff_text
    assert "+++ b/existing.py" in payload.diff_text


def test_apply_update(temp_project):
    changes = [
        CodeChangeSchema(
            path="existing.py",
            action="update",
            content="print('updated')",
        )
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert (temp_project / "existing.py").read_text() == "print('updated')"


def test_apply_update_rejects_content_over_max_content_bytes(temp_project):
    content = "a" * 6
    changes = [
        CodeChangeSchema(
            path="existing.py",
            action="update",
            content=content,
        )
    ]
    config = Config(max_content_bytes=5)

    with pytest.raises(
        ValueError,
        match=r"exceeds the 5-byte limit \(6 bytes\)",
    ):
        apply_changes(
            str(temp_project),
            changes,
            config=config,
        )

    assert (temp_project / "existing.py").read_text() == "print('old')"


def test_apply_delete(temp_project):
    changes = [CodeChangeSchema(path="existing.py", action="delete")]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert not (temp_project / "existing.py").exists()


def test_apply_delete_missing_file_does_not_append_affected(temp_project):
    changes = [CodeChangeSchema(path="missing.py", action="delete")]
    affected = apply_changes(str(temp_project), changes)

    assert affected == []


def test_apply_mkdir(temp_project):
    changes = [CodeChangeSchema(path="subdir/deep", action="mkdir")]
    affected = apply_changes(str(temp_project), changes)

    assert str(Path("subdir") / "deep") in affected
    assert (temp_project / "subdir" / "deep").is_dir()


def test_apply_copy(temp_project):
    changes = [
        CodeChangeSchema(
            path="copied.py",
            action="copy",
            src="existing.py",
        )
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "copied.py" in affected
    assert (temp_project / "copied.py").read_text() == "print('old')"


def test_apply_move_includes_source_and_destination(temp_project):
    changes = [
        CodeChangeSchema(
            path="moved.py",
            action="move",
            src="existing.py",
        )
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert "moved.py" in affected
    assert not (temp_project / "existing.py").exists()
    assert (temp_project / "moved.py").read_text() == "print('old')"


def test_apply_changes_emits_events(temp_project):
    changes = [
        CodeChangeSchema(
            path="new.py",
            action="create",
            content="print('new')",
        )
    ]
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    apply_changes(str(temp_project), changes, event_bus=bus)

    assert len(events) == 1
    assert events[0].name == "apply.file"
    assert isinstance(events[0].payload, ApplyFilePayload)
    assert events[0].payload.path == "new.py"
    assert events[0].payload.action == "create"


def test_git_commit_stages_relative_paths(temp_project, monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, check=None, capture_output=None):
        calls.append((cmd, cwd, check, capture_output))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agent.patcher.subprocess.run", fake_run)

    git_commit(
        str(temp_project),
        "Add feature",
        ["existing.py", str(temp_project / "new.py")],
    )

    assert calls[0][0] == ["git", "add", "existing.py", "new.py"]
    assert calls[0][1] == str(temp_project)
    assert calls[1][0] == ["git", "commit", "-m", "[llm-agent] Add feature"]


def test_git_commit_rejects_path_traversal(temp_project):
    with pytest.raises(PermissionError, match="Path traversal detected"):
        git_commit(
            str(temp_project),
            "Add feature",
            ["../outside.py"],
        )


def test_git_commit_propagates_git_add_failure(temp_project, monkeypatch):
    def fake_run(cmd, cwd=None, check=None, capture_output=None):
        if cmd[:2] == ["git", "add"]:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr=b"boom",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agent.patcher.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        git_commit(
            str(temp_project),
            "Add feature",
            ["existing.py"],
        )
