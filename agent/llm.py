from collections.abc import Callable
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator
from .config import Config

SYSTEM_PROMPT = """\
You are an expert software engineer and coding assistant.
You will be given a codebase and a task. Your job is to make precise, minimal changes to fulfill the task.

RULES:
1. Return ONLY a valid JSON object — no markdown, no explanation, no backticks.
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
3. For "update" and "create", always provide the COMPLETE file content — not a diff.
4. Only include files that actually need to change.
5. Preserve existing code style, indentation, and conventions.
6. Do not hallucinate file paths — only reference files that exist in the codebase (except for "create").
"""


class ChangeSchema(BaseModel):
    path: str = Field(min_length=1)
    action: Literal["create", "update", "delete"]
    content: str | None = None

    @model_validator(mode="after")
    def validate_content_for_action(self) -> "ChangeSchema":
        if self.action in {"create", "update"} and self.content is None:
            raise ValueError(
                f"Action {self.action!r} requires field 'content' to be provided."
            )
        if self.action == "delete" and self.content is not None:
            raise ValueError("Delete actions must not include 'content'.")
        return self


class LLMResponseSchema(BaseModel):
    summary: str = Field(min_length=1)
    changes: list[ChangeSchema] = Field(default_factory=list)


async def call_llm(
    prompt: str,
    context: str,
    config: Config,
    on_chunk: Callable[[str], None] | None = None,
) -> LLMResponseSchema:
    """
    Send prompt + codebase context to llama.cpp and return parsed JSON response.
    """
    client = AsyncOpenAI(
        base_url=config.api_base,
        api_key=config.api_key,
        timeout=60.0,
    )

    user_message = f"{context}\n\n<task>\n{prompt}\n</task>"

    stream = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.max_response_tokens,
        temperature=0.2,  # Low temp for deterministic code edits
        response_format={"type": "json_object"},
        stream=True,
    )

    chunks: list[str] = []
    async for event in stream:
        delta = event.choices[0].delta.content
        if isinstance(delta, str) and delta:
            chunks.append(delta)
            if on_chunk is not None:
                on_chunk(delta)

    content = "".join(chunks).strip()
    if not content:
        raise ValueError("Model returned empty content.")

    try:
        return LLMResponseSchema.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"Invalid structured response from model: {exc}") from exc
