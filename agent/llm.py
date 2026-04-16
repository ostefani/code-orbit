import json
import re
from openai import OpenAI
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


def extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output:\n{raw}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e}\nRaw output:\n{raw}") from e


def call_llm(prompt: str, context: str, config: Config) -> dict:
    """
    Send prompt + codebase context to llama.cpp and return parsed JSON response.
    """
    client = OpenAI(
        base_url=config.api_base,
        api_key=config.api_key,
    )

    user_message = f"{context}\n\n<task>\n{prompt}\n</task>"

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.max_response_tokens,
        temperature=0.2,  # Low temp for deterministic code edits
    )

    # raw = response.choices[0].message.content.strip()

    message = response.choices[0].message
    content = message.content

    if content is None:
        raise ValueError("Model returned empty content.")

    if not isinstance(content, str):
        raise ValueError(
            f"Expected string content, got {type(content).__name__}: {content!r}"
        )

    raw = content.strip()

    # Strip markdown fences if model ignores instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return extract_json(raw)
