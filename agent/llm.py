from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .chat import ChatMessage, run_chat, stream_chat
from .config import Config

ARCHITECT_SYSTEM_PROMPT = """\
You are The Architect, a senior software planner.
You will be given a codebase and a user request. Your job is to produce a
high-level implementation plan only.

RULES:
1. Return ONLY a valid JSON object - no markdown, no explanation, no backticks.
2. The JSON must follow this exact schema:
   {
     "summary": "One sentence describing the overall plan.",
     "tasks": [
       {
         "files": ["relative/path/to/file.py"],
         "goal": "What should happen in these files.",
         "reasoning": "Why this task is needed."
       }
     ]
   }
3. Do not write raw code, diffs, or full file contents.
4. Keep tasks high-level and implementation-oriented.
5. Only reference repository-relative file paths.
6. Include only tasks that materially contribute to the requested change.
7. If no code change is needed, return an empty tasks list with a clear summary.
"""

CODER_SYSTEM_PROMPT = """\
You are The Coder, a precise code editor.
You will be given a codebase, an approved implementation plan, and one approved
task from that plan. Your job is to produce exact file replacements only for
that task.

RULES:
1. Return ONLY a valid JSON object - no markdown, no explanation, no backticks.
2. The JSON must follow this exact schema:
   {
     "summary": "One sentence describing what you did.",
     "changes": [
       {
         "path": "relative/path/to/file.py",
         "action": "update" | "create" | "delete",
         "content": "full new file content as a string (omit for delete)"
       }
     ]
   }
3. For "update" and "create", always provide the COMPLETE file content - not a diff.
4. Only include files that actually need to change.
5. Preserve existing code style, indentation, and conventions.
6. Do not hallucinate file paths - only reference files that exist in the codebase
   (except for "create").
7. Follow the approved plan and the current task exactly. If the task says no
   changes are needed, return an empty changes list.
"""

# Backward compatibility for the context builder and any older callers.
SYSTEM_PROMPT = ARCHITECT_SYSTEM_PROMPT


def _validate_repo_relative_path(path: str, label: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if Path(normalized).is_absolute():
        raise ValueError(f"{label} must be a relative path, got {path!r}.")
    if any(part == ".." for part in Path(normalized).parts):
        raise ValueError(
            f"{label} must not contain parent-directory traversal: {path!r}."
        )
    return normalized


class PlanTaskSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(min_length=1)
    goal: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def validate_files(cls, files: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for index, file_path in enumerate(files, 1):
            cleaned = _validate_repo_relative_path(
                file_path, f"Plan task file #{index}"
            )
            if cleaned in seen:
                raise ValueError(f"Duplicate file path in plan task: {cleaned!r}")
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


class PlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    tasks: list[PlanTaskSchema] = Field(default_factory=list)


class CodeChangeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    action: Literal["create", "update", "delete"]
    content: str | None = None

    @model_validator(mode="after")
    def validate_content_for_action(self) -> "CodeChangeSchema":
        self.path = _validate_repo_relative_path(self.path, "Change path")
        if self.action in {"create", "update"} and self.content is None:
            raise ValueError(
                f"Action {self.action!r} requires field 'content' to be provided."
            )
        if self.action == "delete" and self.content is not None:
            raise ValueError("Delete actions must not include 'content'.")
        return self


class CodeResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    changes: list[CodeChangeSchema] = Field(default_factory=list)


ModelT = TypeVar("ModelT", bound=BaseModel)


def format_plan_roadmap(plan: PlanSchema) -> str:
    lines = [f"Plan summary: {plan.summary}"]
    if not plan.tasks:
        lines.append("Roadmap: no implementation tasks were approved.")
        return "\n".join(lines)

    lines.append("Roadmap:")
    for index, task in enumerate(plan.tasks, 1):
        files = ", ".join(task.files)
        lines.append(f"{index}. {task.goal} | files: {files}")
    return "\n".join(lines)


async def _call_structured_llm(
    *,
    system_prompt: str,
    user_message: str,
    config: Config,
    parser: Callable[[str], ModelT],
    on_chunk: Callable[[str], None] | None = None,
) -> ModelT:
    messages = (
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    )

    if on_chunk is not None and config.chat_streaming:
        chunks: list[str] = []
        async for delta in stream_chat(
            messages,
            config,
            max_tokens=config.max_response_tokens,
            temperature=0.2,
        ):
            if delta.content:
                chunks.append(delta.content)
                on_chunk(delta.content)
        content = "".join(chunks).strip()
    else:
        response = await run_chat(
            messages,
            config,
            max_tokens=config.max_response_tokens,
            temperature=0.2,
        )
        content = response.content.strip()

    if not content:
        raise ValueError("Model returned empty content.")

    try:
        return parser(content)
    except ValidationError as exc:
        raise ValueError(f"Invalid structured response from model: {exc}") from exc


async def call_architect(
    prompt: str,
    context: str,
    config: Config,
    on_chunk: Callable[[str], None] | None = None,
) -> PlanSchema:
    """Ask the architect model for a high-level implementation plan."""
    user_message = f"{context}\n\n<task>\n{prompt}\n</task>"
    result = await _call_structured_llm(
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        user_message=user_message,
        config=config,
        parser=PlanSchema.model_validate_json,
        on_chunk=on_chunk,
    )
    return result


async def call_coder_for_task(
    plan: PlanSchema,
    task: PlanTaskSchema,
    context: str,
    config: Config,
    on_chunk: Callable[[str], None] | None = None,
) -> CodeResponseSchema:
    """Ask the coder model for exact file replacements for one approved task."""
    user_message = (
        f"{context}\n\n<approved_plan_summary>\n{format_plan_roadmap(plan)}\n"
        f"</approved_plan_summary>\n\n<approved_task>\n"
        f"{task.model_dump_json(indent=2)}\n</approved_task>"
    )
    result = await _call_structured_llm(
        system_prompt=CODER_SYSTEM_PROMPT,
        user_message=user_message,
        config=config,
        parser=CodeResponseSchema.model_validate_json,
        on_chunk=on_chunk,
    )
    return result


async def call_coder(
    plan: PlanSchema,
    context: str,
    config: Config,
    on_chunk: Callable[[str], None] | None = None,
) -> CodeResponseSchema:
    """Backward-compatible wrapper that executes the first approved task."""
    if not plan.tasks:
        raise ValueError("Approved plan must contain at least one task.")
    return await call_coder_for_task(
        plan=plan,
        task=plan.tasks[0],
        context=context,
        config=config,
        on_chunk=on_chunk,
    )
