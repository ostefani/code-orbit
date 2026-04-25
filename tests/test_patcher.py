import pytest
from pathlib import Path
from agent.patcher import apply_changes, git_commit, preview_changes
from agent.events import ApplyFilePayload, EventBus, PreviewChangePayload
from agent.schemas import CodeChangeSchema


@pytest.fixture
def temp_project(tmp_path):
    (tmp_path / "existing.py").write_text("print('old')")
    return tmp_path


def test_apply_create(temp_project):
    changes = [{"path": "new.py", "action": "create", "content": "print('new')"}]
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


def test_apply_update(temp_project):
    changes = [
        {"path": "existing.py", "action": "update", "content": "print('updated')"}
    ]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert (temp_project / "existing.py").read_text() == "print('updated')"


def test_apply_delete(temp_project):
    changes = [{"path": "existing.py", "action": "delete"}]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert not (temp_project / "existing.py").exists()


def test_apply_mkdir(temp_project):
    changes = [{"path": "subdir/deep", "action": "mkdir"}]
    affected = apply_changes(str(temp_project), changes)

    assert str(Path("subdir") / "deep") in affected
    assert (temp_project / "subdir" / "deep").is_dir()


def test_apply_copy(temp_project):
    changes = [{"path": "copied.py", "action": "copy", "src": "existing.py"}]
    affected = apply_changes(str(temp_project), changes)

    assert "copied.py" in affected
    assert (temp_project / "copied.py").read_text() == "print('old')"


def test_apply_copy_requires_src(temp_project):
    changes = [{"path": "copied.py", "action": "copy"}]

    with pytest.raises(ValueError, match="Invalid change #1.*requires field 'src'"):
        apply_changes(str(temp_project), changes)


def test_apply_move_includes_source_and_destination(temp_project):
    changes = [{"path": "moved.py", "action": "move", "src": "existing.py"}]
    affected = apply_changes(str(temp_project), changes)

    assert "existing.py" in affected
    assert "moved.py" in affected
    assert not (temp_project / "existing.py").exists()
    assert (temp_project / "moved.py").read_text() == "print('old')"


def test_apply_move_requires_src(temp_project):
    changes = [{"path": "moved.py", "action": "move"}]

    with pytest.raises(ValueError, match="Invalid change #1.*requires field 'src'"):
        apply_changes(str(temp_project), changes)


def test_apply_rejects_malformed_change_without_key_error(temp_project):
    with pytest.raises(ValueError, match="Invalid change #1"):
        apply_changes(str(temp_project), [{"action": "create", "content": "x"}])

    assert not (temp_project / "x").exists()


def test_apply_changes_emits_events(temp_project):
    changes = [{"path": "new.py", "action": "create", "content": "print('new')"}]
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
        return __import__("subprocess").CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agent.patcher.subprocess.run", fake_run)

    git_commit(
        str(temp_project),
        "Add feature",
        ["existing.py", str(temp_project / "new.py")],
    )

    assert calls[0][0] == ["git", "add", "existing.py", "new.py"]
    assert calls[0][1] == str(temp_project)
    assert calls[1][0] == ["git", "commit", "-m", "[llm-agent] Add feature"]


def test_git_commit_rejects_path_traversal(temp_project, monkeypatch):
    monkeypatch.setattr("agent.patcher.subprocess.run", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="outside repository root"):
        git_commit(
            str(temp_project),
            "Add feature",
            ["../outside.py"],
        )
