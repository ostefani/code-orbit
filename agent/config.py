from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    # llama.cpp server
    api_base: str = "http://localhost:8081/v1"
    api_key: str = "dummy"
    model: str = "local"

    max_context_tokens: int = 16384
    max_response_tokens: int = 4096
    tokenizer_backend: str = "estimate"

    tokenizer_model_path: Optional[Path] = None
    tokenizer_model: Optional[str] = None

    # Files to skip when building context
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".venv",
            "venv",
            "env",
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
        ]
    )

    # Max file size to include in context (bytes)
    max_file_size: int = 100_000

    interactive: bool = True

    auto_commit: bool = False

    allow_delete: bool = False

    @classmethod
    def load(
        cls, path: str | Path = "config.yaml", profile_name: Optional[str] = None
    ) -> "Config":
        config_path = Path(path).expanduser().resolve()

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
                print(f"✅ Using profile '{selected_profile_name}'")
            else:
                available = ", ".join(profiles.keys()) or "none"
                print(
                    f"⚠️  Profile '{selected_profile_name}' not found "
                    f"(available: {available}). Using base config."
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
        return cls(**valid_params)
