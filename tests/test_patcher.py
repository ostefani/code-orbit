import pytest
from pathlib import Path
from agent.patcher import apply_changes
from agent.events import ApplyFilePayload, EventBus


@pytest.fixture
def temp_project(tmp_path):
    (tmp_path / "existing.py").write_text("print('old')")
    return tmp_path


def test_apply_create(temp_project):
    changes = [{"path": "new.py", "action": "create", "content": "print('new')"}]
    affected = apply_changes(str(temp_project), changes)

    assert str(temp_project / "new.py") in affected
    assert (temp_project / "new.py").read_text() == "print('new')"


def test_apply_update(temp_project):
    changes = [
        {"path": "existing.py", "action": "update", "content": "print('updated')"}
    ]
    affected = apply_changes(str(temp_project), changes)

    assert str(temp_project / "existing.py") in affected
    assert (temp_project / "existing.py").read_text() == "print('updated')"


def test_apply_delete(temp_project):
    changes = [{"path": "existing.py", "action": "delete"}]
    affected = apply_changes(str(temp_project), changes)

    assert str(temp_project / "existing.py") in affected
    assert not (temp_project / "existing.py").exists()


def test_apply_mkdir(temp_project):
    changes = [{"path": "subdir/deep/file.py", "action": "create", "content": "hello"}]
    apply_changes(str(temp_project), changes)

    assert (temp_project / "subdir" / "deep" / "file.py").exists()
    assert (temp_project / "subdir" / "deep" / "file.py").read_text() == "hello"


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
