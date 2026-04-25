import asyncio
from collections.abc import Callable
from typing import TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from .chat import (
    ChatAdapter,
    ChatGenerationSettings,
    ChatMessage,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    run_chat,
    stream_chat,
)
from .config import Config
from .schemas import CodeChangeSchema
from .utils import validate_repo_relative_path

ARCHITECT_SYSTEM_PROMPT = """\
You are The Architect, a senior software planner.
You will be given a codebase and a user request. Your job is to produce a
high-level implementation plan only.

RULES:
1. Return ONLY a valid JSON object - no markdown, no explanation, no backticks.
2. The JSON must follow this exact schema:
   {
     "summary": "One sentence describing the overall plan.",
     "answer": "Direct, complete answer to the user when no code changes are needed (e.g. the request is a question, a shell command, or an explanation). Omit this field (or set to null) when implementation tasks are present.",
     "tasks": [
       {
         "files": ["relative/path/to/file.py"],
         "goal": "What should happen in these files.",
         "reasoning": "Why this task is needed."
       }
     ]
   }
3. Do not write raw code, diffs, or full file contents inside tasks.
4. Keep tasks high-level and implementation-oriented.
5. Only reference repository-relative file paths.
6. Include only tasks that materially contribute to the requested change.
7. If no code change is needed, return an empty tasks list and put the full,
   helpful response to the user in the "answer" field. The answer must directly
   address the user's request — do not just state that no changes are needed.
8. Filesystem-only tasks (copy, move, delete, mkdir) are valid tasks even when
   no file content changes. Prefer creating files directly over mkdir whenever
   a file will be placed inside the directory — create actions handle parent
   directory creation automatically. Only use mkdir for explicitly empty
   directories.
9. The <file_tree> block shows the directory structure of the codebase.
   Directories listed there exist and are valid targets even if they contain
   no <file> entries. You may propose creating files inside them.
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
         "action": "update" | "create" | "delete" | "mkdir" | "copy" | "move",
         "content": "full new file content as a string (create/update only)",
         "src": "source path (copy/move only)"
       }
     ]
   }
3. Action rules:
   - create/update: always provide COMPLETE file content in "content". No "src".
     The "create" action automatically creates any missing parent directories;
     do not add a separate "mkdir" step before creating a file.
   - delete: no "content", no "src".
   - mkdir: only use this to create an empty directory with no files in it.
     If you are creating any file inside a directory, use "create" instead.
     It handles directory creation automatically. No "content", no "src".
   - copy: copies file from "src" to "path". No "content".
   - move: moves/renames file from "src" to "path". No "content".
4. Only include files that actually need to change.
5. Preserve existing code style, indentation, and conventions.
6. You may create files in directories shown in <file_tree> even if no files
   from that directory appear in <file> blocks. An empty directory is a valid
   target. You may also create entirely new directories and files using the
   "create" action; parent directories are created automatically.
7. Follow the approved plan and the current task exactly. If the task says no
   changes are needed, return an empty changes list.
"""

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
            cleaned = validate_repo_relative_path(
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
    answer: str | None = Field(default=None)
    tasks: list[PlanTaskSchema] = Field(default_factory=list)


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
    chat_adapter: ChatAdapter | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> ModelT:
    base_messages = (
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    )
    generation = ChatGenerationSettings(
        max_tokens=config.max_response_tokens,
        temperature=config.structured_llm_temperature,
        response_format="json_object",
    )
    retryable_attempts = config.structured_llm_retries
    messages = base_messages

    for attempt in range(retryable_attempts + 1):
        try:
            content = await _get_structured_llm_content(
                messages,
                config,
                generation=generation,
                chat_adapter=chat_adapter,
                on_chunk=on_chunk,
            )
            if not content:
                raise ValueError("Model returned empty content.")

            return parser(content)
        except ValidationError as exc:
            if attempt >= retryable_attempts:
                raise ValueError(
                    f"Invalid structured response from model: {exc}"
                ) from exc
            messages = (
                base_messages[0],
                ChatMessage(
                    role="user",
                    content=_build_structured_llm_retry_message(user_message, exc),
                ),
            )
        except ValueError as exc:
            if str(exc) != "Model returned empty content.":
                raise
            if attempt >= retryable_attempts:
                raise
        except ProviderError as exc:
            if not _is_retryable_structured_llm_error(exc):
                raise
            if attempt >= retryable_attempts:
                raise
            await _sleep_before_structured_llm_retry(config, exc, attempt)

    raise RuntimeError("Structured LLM retry loop exhausted unexpectedly.")


async def _get_structured_llm_content(
    messages: tuple[ChatMessage, ChatMessage],
    config: Config,
    *,
    generation: ChatGenerationSettings,
    chat_adapter: ChatAdapter | None,
    on_chunk: Callable[[str], None] | None,
) -> str:
    if on_chunk is not None and config.chat_streaming:
        chunks: list[str] = []
        async for delta in stream_chat(
            messages,
            config,
            adapter=chat_adapter,
            generation=generation,
        ):
            if delta.content:
                chunks.append(delta.content)
                on_chunk(delta.content)
        return "".join(chunks).strip()

    response = await run_chat(
        messages,
        config,
        adapter=chat_adapter,
        generation=generation,
    )
    return response.content.strip()


def _build_structured_llm_retry_message(
    user_message: str,
    exc: ValidationError,
) -> str:
    return (
        f"{user_message}\n\n"
        "Your previous response was not valid JSON.\n"
        f"Parser error: {exc}\n"
        "Return ONLY a valid JSON object that matches the schema."
    )


def _is_retryable_structured_llm_error(exc: ProviderError) -> bool:
    return isinstance(exc, (ProviderRateLimitError, ProviderUnavailableError))


async def _sleep_before_structured_llm_retry(
    config: Config,
    exc: ProviderError,
    attempt: int,
) -> None:
    if isinstance(exc, ProviderRateLimitError):
        delay = config.structured_llm_retry_delay_seconds * (2**attempt)
        await asyncio.sleep(delay)


async def call_architect(
    prompt: str,
    context: str,
    config: Config,
    chat_adapter: ChatAdapter | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> PlanSchema:
    """Ask the architect model for a high-level implementation plan."""
    user_message = f"{context}\n\n<task>\n{prompt}\n</task>"
    result = await _call_structured_llm(
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        user_message=user_message,
        config=config,
        parser=PlanSchema.model_validate_json,
        chat_adapter=chat_adapter,
        on_chunk=on_chunk,
    )
    return result


async def call_coder_for_task(
    plan: PlanSchema,
    task: PlanTaskSchema,
    context: str,
    config: Config,
    chat_adapter: ChatAdapter | None = None,
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
        chat_adapter=chat_adapter,
        on_chunk=on_chunk,
    )
    return result


async def call_coder(
    plan: PlanSchema,
    context: str,
    config: Config,
    chat_adapter: ChatAdapter | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> CodeResponseSchema:
    """Backward-compatible wrapper for callers that submit exactly one task."""
    if not plan.tasks:
        raise ValueError("Approved plan must contain at least one task.")
    if len(plan.tasks) > 1:
        raise ValueError(
            "call_coder only supports single-task plans. "
            "Use call_coder_for_task for each approved task."
        )
    return await call_coder_for_task(
        plan=plan,
        task=plan.tasks[0],
        context=context,
        config=config,
        chat_adapter=chat_adapter,
        on_chunk=on_chunk,
    )
