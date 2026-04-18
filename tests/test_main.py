from agent.config import Config
from agent.events import AgentEvent, EventBus, RunProposalReadyPayload
from agent.llm import ChangeSchema, LLMResponseSchema
from main import validate_llm_result


def test_validate_llm_result_rejects_delete_by_default() -> None:
    result = {
        "summary": "Remove file",
        "changes": [ChangeSchema(path="danger.txt", action="delete")],
    }

    config = Config()

    try:
        validate_llm_result(result, config)
    except ValueError as exc:
        assert "delete action" in str(exc)
    else:
        raise AssertionError("Expected delete action to be rejected")


def test_validate_llm_result_rejects_duplicate_paths() -> None:
    result = {
        "summary": "Conflicting changes",
        "changes": [
            ChangeSchema(path="same.py", action="update", content="a"),
            ChangeSchema(path="same.py", action="create", content="b"),
        ],
    }

    config = Config(allow_delete=True)

    try:
        validate_llm_result(result, config)
    except ValueError as exc:
        assert "Duplicate change path" in str(exc)
    else:
        raise AssertionError("Expected duplicate paths to be rejected")


def test_validate_llm_result_accepts_valid_changes() -> None:
    result = {
        "summary": "Update files",
        "changes": [
            ChangeSchema(path="src/app.py", action="update", content="print('ok')"),
            ChangeSchema(path="src/new.py", action="create", content="print('new')"),
        ],
    }

    summary, changes = validate_llm_result(LLMResponseSchema(**result), Config())

    assert summary == "Update files"
    assert changes == [
        {"path": "src/app.py", "action": "update", "content": "print('ok')"},
        {"path": "src/new.py", "action": "create", "content": "print('new')"},
    ]


def test_run_proposal_ready_event_can_be_published() -> None:
    bus = EventBus()
    events = []
    bus.subscribe(events.append)

    published = bus.emit(
        "run.proposal_ready",
        RunProposalReadyPayload(summary="Update files", change_count=2),
        state="validated",
        message="Model response validated.",
    )

    assert events[0] == published
    assert published.name == "run.proposal_ready"
    assert published.state == "validated"
    assert published.payload.change_count == 2


def test_change_schema_requires_content_for_create_and_update() -> None:
    try:
        ChangeSchema(path="src/app.py", action="update")
    except Exception as exc:
        assert "requires field 'content'" in str(exc)
    else:
        raise AssertionError("Expected content to be required for update actions")


def test_change_schema_rejects_content_for_delete() -> None:
    try:
        ChangeSchema(path="danger.txt", action="delete", content="nope")
    except Exception as exc:
        assert "must not include 'content'" in str(exc)
    else:
        raise AssertionError("Expected content to be rejected for delete actions")
