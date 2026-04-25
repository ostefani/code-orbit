from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class ConfigLoadMessage:
    level: str
    text: str


@dataclass(frozen=True)
class ConfigLoadResult:
    config: "Config"
    messages: tuple[ConfigLoadMessage, ...] = ()


DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".git",
    "__pycache__",
    "*.pyc",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".code-orbit",
    "dist",
    "build",
    "*.egg-info",
    ".DS_Store",
    "*.lock",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.min.js",
    "*.min.css",
)


def _resolve_tokenizer_model_path(
    raw_path: Path | str, base_dir: Path | None = None
) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve()
    else:
        path = path.resolve()
    return path


class Config(BaseModel):
    """Immutable runtime configuration for the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # llama.cpp server
    api_base: str = "http://localhost:8081/v1"
    api_key: str = ""
    model: str = "local"

    # Chat provider for planning / code generation
    chat_provider: str = "openai"
    chat_api_base: str = "http://localhost:8081/v1"
    chat_api_key: str = ""
    chat_model: str = "local"
    chat_context_window: int = 16384
    chat_streaming: bool = True
    chat_probe_on_startup: bool = False
    chat_provider_options: dict[str, object] = Field(default_factory=dict)

    # Embeddings for semantic retrieval / RAG
    embedding_provider: str = "openai"
    embedding_api_base: str = "http://localhost:8081/v1"
    embedding_api_key: str = ""
    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 16
    embedding_max_concurrency: int = 4
    embedding_probe_on_startup: bool = False
    embedding_provider_options: dict[str, object] = Field(default_factory=dict)

    max_context_tokens: int = 16384
    max_response_tokens: int = 4096
    structured_llm_temperature: float = 0.2
    structured_llm_retries: int = 1
    structured_llm_retry_delay_seconds: float = 1.0
    tokenizer_backend: str = "estimate"

    tokenizer_model_path: Path | None = None
    tokenizer_model: str | None = None

    # Files to skip when building context
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS

    # Max file size to include in context (bytes)
    max_file_size: int = 100_000

    interactive: bool = True

    auto_commit: bool = False

    allow_delete: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        if isinstance(data, cls):
            data = data.model_dump()
        if not isinstance(data, Mapping):
            return data

        normalized = dict(data)

        for key in ("embedding_provider", "chat_provider"):
            value = normalized.get(key)
            if value is not None:
                normalized[key] = str(value).strip().lower()

        raw_path = normalized.get("tokenizer_model_path")
        if raw_path is not None:
            normalized["tokenizer_model_path"] = _resolve_tokenizer_model_path(
                raw_path
            )

        patterns = normalized.get("ignore_patterns")
        if patterns is not None:
            normalized_patterns = tuple(patterns)
            if ".code-orbit" not in normalized_patterns:
                normalized_patterns = normalized_patterns + (".code-orbit",)
            normalized["ignore_patterns"] = normalized_patterns

        if normalized.get("chat_context_window") is None:
            normalized["chat_context_window"] = normalized.get(
                "max_context_tokens",
                cls.model_fields["max_context_tokens"].default,
            )

        return normalized

    @model_validator(mode="after")
    def _validate_ranges(self) -> "Config":
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")
        if self.max_response_tokens < 0:
            raise ValueError("max_response_tokens must not be negative.")
        if self.max_response_tokens >= self.max_context_tokens:
            raise ValueError(
                "max_response_tokens must be smaller than max_context_tokens "
                "so there is room for file context."
            )
        if self.structured_llm_temperature < 0:
            raise ValueError("structured_llm_temperature must not be negative.")
        if self.structured_llm_retries < 0:
            raise ValueError("structured_llm_retries must not be negative.")
        if self.structured_llm_retry_delay_seconds < 0:
            raise ValueError(
                "structured_llm_retry_delay_seconds must not be negative."
            )
        if self.chat_context_window <= 0:
            raise ValueError("chat_context_window must be greater than zero.")
        return self

    @classmethod
    def load(
        cls, path: str | Path = "config.yaml", profile_name: str | None = None
    ) -> "Config":
        return cls.load_with_diagnostics(path, profile_name).config

    @classmethod
    def load_with_diagnostics(
        cls, path: str | Path = "config.yaml", profile_name: str | None = None
    ) -> ConfigLoadResult:
        config_path = Path(path).expanduser().resolve()
        messages: list[ConfigLoadMessage] = []

        data: dict[str, Any] = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        valid_keys = set(cls.model_fields)
        base_params = {
            k: v for k, v in data.items() if k in valid_keys and k != "profiles"
        }

        profiles_raw = data.get("profiles", {})
        profiles: dict[str, dict[str, Any]] = (
            profiles_raw if isinstance(profiles_raw, dict) else {}
        )
        selected_profile_name: str | None = profile_name or data.get("default_profile")

        profile_data: dict[str, Any] = {}
        if selected_profile_name:
            if selected_profile_name in profiles:
                profile_data = profiles[selected_profile_name]
                messages.append(
                    ConfigLoadMessage(
                        level="info",
                        text=f"Using profile '{selected_profile_name}'",
                    )
                )
                unknown_keys = {k for k in profile_data if k not in valid_keys}
                if unknown_keys:
                    messages.append(
                        ConfigLoadMessage(
                            level="warning",
                            text=(
                                f"Profile '{selected_profile_name}' contains unknown "
                                f"key(s) that will be ignored: "
                                f"{', '.join(sorted(unknown_keys))}"
                            ),
                        )
                    )
            else:
                available = ", ".join(profiles.keys()) or "none"
                messages.append(
                    ConfigLoadMessage(
                        level="warning",
                        text=(
                            f"Profile '{selected_profile_name}' not found "
                            f"(available: {available}). Using base config."
                        ),
                    )
                )

        merged = {**base_params, **profile_data}

        raw_path = merged.get("tokenizer_model_path")
        if raw_path is not None:
            merged["tokenizer_model_path"] = _resolve_tokenizer_model_path(
                raw_path, config_path.parent
            )

        valid_params = {k: v for k, v in merged.items() if k in valid_keys}
        return ConfigLoadResult(config=cls.model_validate(valid_params), messages=tuple(messages))
