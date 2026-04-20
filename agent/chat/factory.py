import importlib
from collections.abc import Callable

from ..config import Config
from .adapters import (
    RESERVED_CHAT_PROVIDER_OPTION_KEYS,
    ChatAdapter,
    ChatProviderConfig,
    ProbingChatAdapter,
)
from .errors import (
    CapabilityNotSupportedError,
    UnsupportedChatProviderError,
)


_CHAT_PROVIDER_BUILDERS: dict[str, str] = {
    "openai": "agent.chat.providers.openai.OpenAIChatAdapter",
}


def build_chat_provider_config(config: Config) -> ChatProviderConfig:
    context_window = (
        config.chat_context_window
        if config.chat_context_window is not None
        else config.max_context_tokens
    )
    options = {
        key: value
        for key, value in dict(config.chat_provider_options).items()
        if key not in RESERVED_CHAT_PROVIDER_OPTION_KEYS
    }
    return ChatProviderConfig(
        provider=config.chat_provider,
        api_base=config.chat_api_base or config.api_base,
        api_key=config.chat_api_key or config.api_key,
        model=config.chat_model or config.model,
        context_window=context_window,
        streaming=config.chat_streaming,
        options=options,
    )


def _load_chat_builder(
    dotted_path: str,
) -> Callable[[ChatProviderConfig], ChatAdapter]:
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    builder = getattr(module, class_name)
    return builder


def build_chat_adapter(config: Config) -> ChatAdapter:
    """Construct an unvalidated chat adapter from config.

    This does not perform local validation, network probing, or any other
    readiness checks. Callers must either validate explicitly or use
    `create_chat_adapter()` when they need a ready-to-use adapter.
    """
    provider_config = build_chat_provider_config(config)
    dotted_path = _CHAT_PROVIDER_BUILDERS.get(provider_config.provider)
    if dotted_path is None:
        raise UnsupportedChatProviderError(
            provider_config.provider,
            f"Unsupported chat provider: {provider_config.provider!r}.",
        )

    builder = _load_chat_builder(dotted_path)
    adapter = builder(provider_config)
    if provider_config.streaming and not adapter.capabilities.streaming:
        raise CapabilityNotSupportedError(
            provider_config.provider,
            "chat_streaming is enabled in config, but the selected provider does not support streaming.",
        )
    if not adapter.capabilities.chat:
        raise CapabilityNotSupportedError(
            provider_config.provider,
            "The selected provider does not support chat completion.",
        )

    return adapter


async def validate_chat_adapter(adapter: ChatAdapter) -> None:
    """Run local, non-network validation on an adapter."""
    await adapter.validate()


async def probe_chat_adapter(adapter: ChatAdapter) -> None:
    """Run an optional live readiness probe for adapters that support it."""
    if not isinstance(adapter, ProbingChatAdapter):
        return
    await adapter.probe()


async def create_chat_adapter(config: Config) -> ChatAdapter:
    """Convenience initializer that returns a ready-to-use adapter."""
    adapter = build_chat_adapter(config)
    await validate_chat_adapter(adapter)
    if config.chat_probe_on_startup:
        await probe_chat_adapter(adapter)
    return adapter
