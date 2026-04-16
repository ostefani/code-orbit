from agent.config import Config
from main import validate_llm_result


def test_validate_llm_result_rejects_delete_by_default() -> None:
    result = {
        "summary": "Remove file",
        "changes": [{"path": "danger.txt", "action": "delete"}],
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
            {"path": "same.py", "action": "update", "content": "a"},
            {"path": "same.py", "action": "create", "content": "b"},
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
            {"path": "src/app.py", "action": "update", "content": "print('ok')"},
            {"path": "src/new.py", "action": "create", "content": "print('new')"},
        ],
    }

    summary, changes = validate_llm_result(result, Config())

    assert summary == "Update files"
    assert changes == result["changes"]
