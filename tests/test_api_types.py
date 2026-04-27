from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

import api
from api import AgentRunRequest, AgentRunResult, AgentRunStatus


def test_api_exports_public_models() -> None:
    assert api.__all__ == ["AgentRunRequest", "AgentRunResult", "AgentRunStatus"]
    assert not hasattr(api, "AgentRunMode")


def test_agent_run_request_defaults_and_forbids_cli_fields() -> None:
    request = AgentRunRequest(target_dir=Path("."), prompt="implement it")

    UUID(request.run_id)
    assert request.auto_commit is False
    assert request.allow_delete is False

    with pytest.raises(ValidationError):
        AgentRunRequest(
            target_dir=Path("."),
            prompt="implement it",
            config_path="config.yaml",
        )


def test_agent_run_result_exposes_run_id_property() -> None:
    request = AgentRunRequest(target_dir=Path("."), prompt="implement it")
    result = AgentRunResult(
        request=request,
        status=AgentRunStatus.COMPLETED,
        summary="done",
    )

    assert result.run_id == request.run_id
    assert "run_id" not in result.model_dump()


def test_agent_run_result_status_invariants() -> None:
    request = AgentRunRequest(target_dir=Path("."), prompt="implement it")

    with pytest.raises(ValidationError, match="FAILED runs require an error"):
        AgentRunResult(request=request, status=AgentRunStatus.FAILED)

    with pytest.raises(ValidationError, match="COMPLETED runs require a summary"):
        AgentRunResult(request=request, status=AgentRunStatus.COMPLETED)

    with pytest.raises(ValidationError, match="COMPLETED runs must not include"):
        AgentRunResult(
            request=request,
            status=AgentRunStatus.COMPLETED,
            summary="done",
            error="unexpected",
        )

    with pytest.raises(ValidationError, match="ANSWERED runs require an answer"):
        AgentRunResult(request=request, status=AgentRunStatus.ANSWERED)

    with pytest.raises(ValidationError, match="ANSWERED runs must not include an error"):
        AgentRunResult(
            request=request,
            status=AgentRunStatus.ANSWERED,
            answer="Here is why.",
            error="unexpected",
        )

    with pytest.raises(
        ValidationError,
        match="ANSWERED runs must not include affected files",
    ):
        AgentRunResult(
            request=request,
            status=AgentRunStatus.ANSWERED,
            answer="Here is why.",
            affected_files=["src/app.py"],
        )

    with pytest.raises(ValidationError, match="CANCELLED runs require completed_at"):
        AgentRunResult(request=request, status=AgentRunStatus.CANCELLED)

    answered = AgentRunResult(
        request=request,
        status=AgentRunStatus.ANSWERED,
        summary="Explanation",
        answer="Here is why.",
    )
    assert answered.answer == "Here is why."
    assert answered.affected_files == []


def test_agent_run_result_coerces_affected_files_to_strings() -> None:
    request = AgentRunRequest(target_dir=Path("."), prompt="implement it")
    result = AgentRunResult(
        request=request,
        status=AgentRunStatus.RUNNING,
        affected_files=(Path("agent/config.py"), 1),
    )

    assert result.affected_files == ["agent/config.py", "1"]
