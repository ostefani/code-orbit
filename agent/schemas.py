from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils import validate_repo_relative_path


class CodeChangeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    action: Literal["create", "update", "delete", "mkdir", "copy", "move"]
    content: str | None = None
    src: str | None = None

    @model_validator(mode="after")
    def validate_content_for_action(self) -> "CodeChangeSchema":
        self.path = validate_repo_relative_path(self.path, "Change path")
        if self.action in {"create", "update"} and self.content is None:
            raise ValueError(
                f"Action {self.action!r} requires field 'content' to be provided."
            )
        if self.action == "delete" and self.content is not None:
            raise ValueError("Delete actions must not include 'content'.")
        if self.action in {"copy", "move"} and not self.src:
            raise ValueError(f"Action {self.action!r} requires field 'src'.")
        if self.action not in {"copy", "move"} and self.src is not None:
            raise ValueError(f"Action {self.action!r} must not include 'src'.")
        if self.action == "mkdir" and self.content is not None:
            raise ValueError("mkdir must not include 'content'.")
        if self.src:
            self.src = validate_repo_relative_path(self.src, "Change src")
        return self
