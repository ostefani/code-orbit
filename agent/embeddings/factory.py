import importlib
from collections.abc import Callable
from urllib.parse import urlparse

from ..config import Config
from .adapters import (
    EmbeddingAdapter,
    EmbeddingProviderConfig,
    EmbeddingProviderConfigurationError,
    EmbeddingProviderError,
    RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS,
    UnsupportedEmbeddingProviderError,
)


_EMBEDDING_PROVIDER_BUILDERS: dict[str, str] = {
    "openai": "agent.embeddings.providers.openai.OpenAIEmbeddingAdapter",
}

# Placeholder key for local OpenAI-compatible servers (llama.cpp, Ollama,
# LM Studio) that ignore auth but sit behind an SDK requiring a non-empty key.
LOCAL_DUMMY_API_KEY = "dummy"


def _is_local_api_base(api_base: str) -> bool:
    """Return True when ``api_base`` targets a loopback host."""
    try:
        host = urlparse(api_base).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    if host in ("localhost", "::1"):
        return True
    if host.startswith("127."):
        return True
    return False


def build_embedding_provider_config(config: Config) -> EmbeddingProviderConfig:
    options = {
        key: value
        for key, value in dict(config.embedding_provider_options).items()
        if key not in RESERVED_EMBEDDING_PROVIDER_OPTION_KEYS
    }

    # Only substitute the placeholder for loopback targets so a missing key
    # against a remote endpoint still fails fast.
    api_key = config.embedding_api_key or config.api_key
    if not api_key and _is_local_api_base(config.embedding_api_base):
        api_key = LOCAL_DUMMY_API_KEY
    return EmbeddingProviderConfig(
        provider=config.embedding_provider,
        api_base=config.embedding_api_base,
        api_key=api_key,
        model=config.embedding_model,
        options=options,
    )


def _load_embedding_builder(dotted_path: str) -> Callable[[EmbeddingProviderConfig], EmbeddingAdapter]:
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    builder = getattr(module, class_name)
    return builder


async def create_embedding_adapter(config: Config) -> EmbeddingAdapter:
    provider_config = build_embedding_provider_config(config)
    dotted_path = _EMBEDDING_PROVIDER_BUILDERS.get(provider_config.provider)
    if dotted_path is None:
        raise UnsupportedEmbeddingProviderError(
            provider_config.provider,
            f"Unsupported embedding provider: {provider_config.provider!r}.",
        )

    builder = _load_embedding_builder(dotted_path)
    try:
        adapter = builder(provider_config)
    except EmbeddingProviderConfigurationError:
        raise
    except Exception as exc:
        raise EmbeddingProviderConfigurationError(
            provider_config.provider,
            f"Failed to create embedding provider {provider_config.provider!r}: {exc}",
        ) from exc

    try:
        await adapter.validate()
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        raise EmbeddingProviderConfigurationError(
            provider_config.provider,
            f"Failed to validate embedding provider {provider_config.provider!r}: {exc}",
        ) from exc

    if config.embedding_probe_on_startup:
        try:
            await adapter.probe()
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderConfigurationError(
                provider_config.provider,
                f"Failed to probe embedding provider {provider_config.provider!r}: {exc}",
            ) from exc

    return adapter
