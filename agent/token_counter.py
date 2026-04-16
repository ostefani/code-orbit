"""
token_counter.py — local-first token counting, no API calls.

Backends
────────
  tokenizers_json  Exact counts using a local tokenizer.json file.
                   Works with Gemma 4 and any model that ships a HuggingFace
                   fast-tokenizer file.  Requires: pip install tokenizers

  tiktoken         Exact counts for OpenAI-family models (GPT-4, GPT-4o …).
                   Requires: pip install tiktoken

  estimate         chars // 3 heuristic.  Always available, no dependencies.
                   Safe default — slightly over-estimates, which is the
                   correct direction for context-budget checks.

Config fields
─────────────
  tokenizer_backend     "tokenizers_json" | "tiktoken" | "estimate"
  tokenizer_model_path  Absolute or ~ path to tokenizer.json
                        (required for tokenizers_json backend)
  tokenizer_model       tiktoken model name, e.g. "gpt-4o"
                        (required for tiktoken backend; falls back to config.model)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenCountResult:
    count: int
    tokenizer_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Warn-once helper
# ─────────────────────────────────────────────────────────────────────────────

_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        print(f"⚠️  {msg}")
        _warned.add(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Estimate fallback  (always available, no dependencies)
# ─────────────────────────────────────────────────────────────────────────────


def _estimate(text: str) -> TokenCountResult:
    # chars // 3 is safer than // 4 for mixed code + prose.
    # Over-estimating is intentional — it keeps us safely inside the budget.
    return TokenCountResult(
        count=max(1, len(text) // 3),
        tokenizer_name="estimate:chars_div_3",
    )


# ─────────────────────────────────────────────────────────────────────────────
# tokenizers_json backend  (Gemma 4 and any HF fast-tokenizer)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _load_tokenizers_json(path: str):
    try:
        from tokenizers import Tokenizer  # pip install tokenizers
    except ImportError as exc:
        raise ImportError(
            "The 'tokenizers' package is not installed. Run: pip install tokenizers"
        ) from exc
    return Tokenizer.from_file(path)


def _count_tokenizers_json(text: str, path: str) -> TokenCountResult:
    tok = _load_tokenizers_json(path)
    encoded = tok.encode(text, add_special_tokens=False)
    return TokenCountResult(
        count=len(encoded.ids),
        tokenizer_name=f"tokenizers:{Path(path).parent.name}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# tiktoken backend  (OpenAI-family models)
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _load_tiktoken(model_name: str):
    import tiktoken  # pip install tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tiktoken(text: str, model_name: str) -> TokenCountResult:
    enc = _load_tiktoken(model_name)
    return TokenCountResult(
        count=len(enc.encode(text)),
        tokenizer_name=f"tiktoken:{model_name}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens(text: str, config: Config) -> TokenCountResult:
    """
    Count tokens in *text* using the backend specified in *config*.
    Never raises — always falls back to the estimate backend.
    """
    backend: str = config.tokenizer_backend

    # ── tokenizers_json (Gemma 4 / any HF fast-tokenizer) ────────────────────
    if backend == "tokenizers_json":
        model_path: str | None = config.tokenizer_model_path

        if not model_path:
            _warn_once(
                "tokenizer_backend='tokenizers_json' requires tokenizer_model_path "
                "in config.  Falling back to estimate."
            )
            return _estimate(text)

        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            _warn_once(
                f"tokenizer.json not found at '{resolved}'. "
                "Falling back to estimate."
            )
            return _estimate(text)

        try:
            return _count_tokenizers_json(text, str(resolved))
        except Exception as exc:
            _warn_once(f"tokenizers_json counting failed: {exc}. Falling back to estimate.")
            return _estimate(text)

    # ── tiktoken (OpenAI-family) ──────────────────────────────────────────────
    if backend == "tiktoken":
        model_name: str = config.tokenizer_model or config.model
        try:
            return _count_tiktoken(text, model_name)
        except Exception as exc:
            _warn_once(
                f"tiktoken counting failed for {model_name!r}: {exc}. "
                "Falling back to estimate."
            )
            return _estimate(text)

    # ── estimate (default / explicit) ─────────────────────────────────────────
    return _estimate(text)


def count_tokens_int(text: str, config: Config) -> int:
    """Convenience wrapper — returns just the integer count."""
    return count_tokens(text, config).count
