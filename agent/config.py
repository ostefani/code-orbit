import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # llama.cpp server
    api_base: str = "http://localhost:8081/v1"
    api_key: str = "dummy"
    model: str = "local"

    # Context window (tokens). Set based on your model.
    max_context_tokens: int = 16384
    max_response_tokens: int = 4096

    # ── Token counting ────────────────────────────────────────────────────────
    # Which backend to use for counting tokens when building context.
    #   "tokenizers_json"  exact counts via a local tokenizer.json  (recommended)
    #   "tiktoken"         exact counts for OpenAI-family models
    #   "estimate"         chars // 3 heuristic, no dependencies (default)
    tokenizer_backend: str = "estimate"

    # Absolute or ~ path to your model's tokenizer.json.
    # Required when tokenizer_backend = "tokenizers_json".
    # Keep this pointing at the file next to your GGUF — do not copy it here.
    tokenizer_model_path: Optional[str] = None

    # Model name passed to tiktoken (e.g. "gpt-4o").
    # Required when tokenizer_backend = "tiktoken".
    # Falls back to config.model if not set.
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

    # Show diffs and ask for confirmation before applying
    interactive: bool = True

    # Auto-commit changes with git
    auto_commit: bool = False

    @classmethod
    def load(
        cls, path: str = "config.yaml", profile_name: Optional[str] = None
    ) -> "Config":
        config_path = Path(path)
        data = {}
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

        # Start with base data
        base_params = {
            k: v for k, v in data.items() if hasattr(cls, k) and k != "profiles"
        }

        # Identify profile
        profiles = data.get("profiles", {})
        selected_profile_name = profile_name or data.get("default_profile")

        # Apply profile overrides
        profile_data = {}
        if selected_profile_name and selected_profile_name in profiles:
            profile_data = profiles[selected_profile_name]
        elif selected_profile_name:
            print(f"⚠️  Profile '{selected_profile_name}' not found, using base config.")

        # Merge: base < profile_overrides
        params = {**base_params, **profile_data}

        # Filter to only valid dataclass fields
        from dataclasses import fields

        valid_keys = {f.name for f in fields(cls)}
        valid_params = {k: v for k, v in params.items() if k in valid_keys}

        return cls(**valid_params)
