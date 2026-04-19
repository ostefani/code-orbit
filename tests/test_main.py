from agent.config import Config
from agent.events import AgentEvent, EventBus, RunProposalReadyPayload
from agent.llm import ChangeSchema, LLMResponseSchema, PlanSchema, PlanTaskSchema
from main import build_working_context, open_plan_in_editor, validate_llm_result


def test_config_rejects_impossible_context_budget() -> None:
    try:
        Config(max_context_tokens=1024, max_response_tokens=1024)
    except ValueError as exc:
        assert "room for file context" in str(exc)
    else:
        raise AssertionError("Expected impossible budgets to be rejected")


def test_validate_llm_result_rejects_delete_by_default() -> None:
    result = LLMResponseSchema(
        summary="Remove file",
        changes=[ChangeSchema(path="danger.txt", action="delete")],
    )

    config = Config()

    try:
        validate_llm_result(result, config)
    except ValueError as exc:
        assert "delete action" in str(exc)
    else:
        raise AssertionError("Expected delete action to be rejected")


def test_validate_llm_result_rejects_duplicate_paths() -> None:
    result = LLMResponseSchema(
        summary="Conflicting changes",
        changes=[
            ChangeSchema(path="same.py", action="update", content="a"),
            ChangeSchema(path="same.py", action="create", content="b"),
        ],
    )

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


def test_validate_llm_result_rejects_missing_content() -> None:
    result = LLMResponseSchema.model_construct(
        summary="Update files",
        changes=[
            ChangeSchema.model_construct(
                path="src/app.py",
                action="update",
                content=None,
            )
        ],
    )

    try:
        validate_llm_result(result, Config())
    except ValueError as exc:
        assert "requires content" in str(exc)
    else:
        raise AssertionError("Expected missing content to be rejected")


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


def test_change_schema_rejects_path_traversal() -> None:
    try:
        ChangeSchema(
            path="../escape.txt",
            action="update",
            content="print('escape')",
        )
    except Exception as exc:
        assert "parent-directory traversal" in str(exc)
    else:
        raise AssertionError("Expected traversal paths to be rejected")


def test_change_schema_rejects_absolute_path() -> None:
    try:
        ChangeSchema(
            path="/etc/passwd",
            action="update",
            content="print('owned')",
        )
    except Exception as exc:
        assert "relative path" in str(exc)
    else:
        raise AssertionError("Expected absolute paths to be rejected")


def test_plan_task_schema_requires_relative_files() -> None:
    try:
        PlanTaskSchema(
            files=["/tmp/escape.py"],
            goal="Update the application wiring",
            reasoning="Absolute paths should never be allowed in plans.",
        )
    except Exception as exc:
        assert "relative path" in str(exc)
    else:
        raise AssertionError("Expected absolute paths to be rejected")


def test_plan_schema_accepts_multiple_tasks() -> None:
    plan = PlanSchema(
        summary="Split the feature into a plan and execution step",
        tasks=[
            PlanTaskSchema(
                files=["src/app.py"],
                goal="Update the CLI entry point",
                reasoning="The state machine starts in the CLI layer.",
            ),
            PlanTaskSchema(
                files=["agent/llm.py"],
                goal="Add the architect and coder schemas",
                reasoning="The LLM contract needs to be split before execution.",
            ),
        ],
    )

    assert plan.tasks[0].files == ["src/app.py"]


def test_open_plan_in_editor_parses_modified_plan(monkeypatch, tmp_path) -> None:
    original = PlanSchema(
        summary="Draft plan",
        tasks=[
            PlanTaskSchema(
                files=["src/app.py"],
                goal="Initial goal",
                reasoning="Start with the entry point.",
            )
        ],
    )
    edited = PlanSchema(
        summary="Edited plan",
        tasks=[
            PlanTaskSchema(
                files=["src/app.py", "agent/llm.py"],
                goal="Update both layers",
                reasoning="The user edited the draft before approval.",
            )
        ],
    )

    temp_path = tmp_path / "code-orbit-plan.json"

    def fake_run(cmd, check=False):
        assert cmd[0] == "vim"
        temp_path.write_text(edited.model_dump_json(indent=2), encoding="utf-8")
        return __import__("subprocess").CompletedProcess(cmd, 0)

    monkeypatch.setattr("main.subprocess.run", fake_run)
    monkeypatch.setenv("EDITOR", "vim")
    temp_path.write_text(original.model_dump_json(indent=2), encoding="utf-8")

    approved = open_plan_in_editor(temp_path)

    assert approved.summary == "Edited plan"
    assert approved.tasks[0].files == ["src/app.py", "agent/llm.py"]


def test_build_working_context_appends_applied_changes() -> None:
    base_context = "<codebase>\n<file path=\"src/app.py\">\nprint('old')\n</file>\n</codebase>"
    changes = [
        {
            "path": "src/new.py",
            "action": "create",
            "content": "print('new')",
        }
    ]

    working_context = build_working_context(base_context, changes)

    assert base_context in working_context
    assert "<applied_changes>" in working_context
    assert 'path="src/new.py"' in working_context
    assert "print('new')" in working_context
