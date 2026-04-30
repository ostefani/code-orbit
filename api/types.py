from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ANSWERED = "answered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    target_dir: Path
    prompt: str
    auto_commit: bool = False
    allow_delete: bool = False
    conversation_context: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_target_dir(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        normalized = dict(data)
        target_dir = normalized.get("target_dir")
        if isinstance(target_dir, PathLike):
            normalized["target_dir"] = Path(target_dir)
        return normalized


class AgentRunResult(BaseModel):
    request: AgentRunRequest
    status: AgentRunStatus
    summary: str | None = None
    answer: str | None = None
    error: str | None = None
    affected_files: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def run_id(self) -> str:
        return self.request.run_id

    @field_validator("affected_files", mode="before")
    @classmethod
    def _coerce_affected_files(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError("affected_files must be a list or tuple.")

    @model_validator(mode="after")
    def _validate_status_fields(self) -> "AgentRunResult":
        if self.status is AgentRunStatus.FAILED and not self.error:
            raise ValueError("FAILED runs require an error.")

        if self.status is AgentRunStatus.COMPLETED:
            if not self.summary:
                raise ValueError("COMPLETED runs require a summary.")
            if self.error is not None:
                raise ValueError("COMPLETED runs must not include an error.")

        if self.status is AgentRunStatus.ANSWERED:
            if not self.answer:
                raise ValueError("ANSWERED runs require an answer.")
            if self.error is not None:
                raise ValueError("ANSWERED runs must not include an error.")
            if self.affected_files:
                raise ValueError("ANSWERED runs must not include affected files.")

        if self.status is AgentRunStatus.CANCELLED and self.completed_at is None:
            raise ValueError("CANCELLED runs require completed_at.")

        return self
