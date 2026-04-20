from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Optional
import yaml


@dataclass(frozen=True)
class ConfigLoadMessage:
    level: str
    text: str


@dataclass(frozen=True)
class ConfigLoadResult:
    config: "Config"
    messages: tuple[ConfigLoadMessage, ...] = ()


@dataclass(frozen=True)
class Config:
    # llama.cpp server
    api_base: str = "http://localhost:8081/v1"
    api_key: str = "dummy"
    model: str = "local"

    # Chat provider for planning / code generation
    chat_provider: str = "openai"
    chat_api_base: str = "http://localhost:8081/v1"
    chat_api_key: str = "dummy"
    chat_model: str = "local"
    chat_context_window: int = 16384
    chat_streaming: bool = True
    chat_provider_options: dict[str, object] = field(default_factory=dict)

    # Embeddings for semantic retrieval / RAG
    embedding_provider: str = "openai"
    embedding_api_base: str = "http://localhost:8081/v1"
    embedding_api_key: str = ""
    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 16
    embedding_max_concurrency: int = 4
    embedding_probe_on_startup: bool = False
    embedding_provider_options: dict[str, object] = field(default_factory=dict)

    max_context_tokens: int = 16384
    max_response_tokens: int = 4096
    tokenizer_backend: str = "estimate"

    tokenizer_model_path: Optional[Path] = None
    tokenizer_model: Optional[str] = None

    # Files to skip when building context
    ignore_patterns: tuple[str, ...] = field(
        default_factory=lambda: (
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
    )

    # Max file size to include in context (bytes)
    max_file_size: int = 100_000

    interactive: bool = True

    auto_commit: bool = False

    allow_delete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "embedding_provider",
            (self.embedding_provider or "openai").strip().lower(),
        )
        object.__setattr__(
            self,
            "chat_provider",
            (self.chat_provider or "openai").strip().lower(),
        )

        patterns = tuple(self.ignore_patterns)
        if ".code-orbit" not in patterns:
            patterns = patterns + (".code-orbit",)
        object.__setattr__(self, "ignore_patterns", patterns)

        tokenizer_model_path = self.tokenizer_model_path
        if tokenizer_model_path is not None and not isinstance(tokenizer_model_path, Path):
            object.__setattr__(self, "tokenizer_model_path", Path(tokenizer_model_path))

        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")
        if self.max_response_tokens < 0:
            raise ValueError("max_response_tokens must not be negative.")
        if self.max_response_tokens >= self.max_context_tokens:
            raise ValueError(
                "max_response_tokens must be smaller than max_context_tokens "
                "so there is room for file context."
            )
        if self.chat_context_window == 16384 and self.max_context_tokens != 16384:
            object.__setattr__(self, "chat_context_window", self.max_context_tokens)
        if self.chat_context_window <= 0:
            raise ValueError("chat_context_window must be greater than zero.")

    @classmethod
    def load(
        cls, path: str | Path = "config.yaml", profile_name: Optional[str] = None
    ) -> "Config":
        return cls.load_with_diagnostics(path, profile_name).config

    @classmethod
    def load_with_diagnostics(
        cls, path: str | Path = "config.yaml", profile_name: Optional[str] = None
    ) -> ConfigLoadResult:
        config_path = Path(path).expanduser().resolve()
        messages: list[ConfigLoadMessage] = []

        data: dict = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        valid_keys = {f.name for f in dataclass_fields(cls)}
        base_params = {
            k: v for k, v in data.items() if k in valid_keys and k != "profiles"
        }

        profiles: dict = data.get("profiles", {})
        selected_profile_name: Optional[str] = profile_name or data.get(
            "default_profile"
        )

        profile_data: dict = {}
        if selected_profile_name:
            if selected_profile_name in profiles:
                profile_data = profiles[selected_profile_name]
                messages.append(
                    ConfigLoadMessage(
                        level="info",
                        text=f"Using profile '{selected_profile_name}'",
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
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = (config_path.parent / p).resolve()
            else:
                p = p.resolve()
            merged["tokenizer_model_path"] = p

        valid_params = {k: v for k, v in merged.items() if k in valid_keys}
        return ConfigLoadResult(
            config=cls(**valid_params),
            messages=tuple(messages),
        )
