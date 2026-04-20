import importlib
from collections.abc import Callable

from ..config import Config
from .adapters import (
    RESERVED_CHAT_PROVIDER_OPTION_KEYS,
    ChatAdapter,
    ChatProviderConfig,
)
from .errors import (
    CapabilityNotSupportedError,
    ProviderConfigurationError,
    ProviderError,
    ProviderValidationError,
    UnsupportedChatProviderError,
)


_CHAT_PROVIDER_BUILDERS: dict[str, str] = {
    "openai": "agent.chat.providers.openai.OpenAIChatAdapter",
}


def build_chat_provider_config(config: Config) -> ChatProviderConfig:
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
        context_window=config.chat_context_window,
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


async def create_chat_adapter(config: Config) -> ChatAdapter:
    if config.chat_context_window <= 0:
        raise ProviderConfigurationError(
            config.chat_provider,
            "chat_context_window must be greater than zero.",
        )

    provider_config = build_chat_provider_config(config)
    if not provider_config.api_key.strip():
        raise ProviderConfigurationError(
            provider_config.provider,
            "chat_api_key must not be empty.",
        )
    if not provider_config.api_base.strip():
        raise ProviderConfigurationError(
            provider_config.provider,
            "chat_api_base must not be empty.",
        )
    if not provider_config.model.strip():
        raise ProviderConfigurationError(
            provider_config.provider,
            "chat_model must not be empty.",
        )

    dotted_path = _CHAT_PROVIDER_BUILDERS.get(provider_config.provider)
    if dotted_path is None:
        raise UnsupportedChatProviderError(
            provider_config.provider,
            f"Unsupported chat provider: {provider_config.provider!r}.",
        )

    builder = _load_chat_builder(dotted_path)
    try:
        adapter = builder(provider_config)
    except ProviderConfigurationError:
        raise
    except Exception as exc:
        raise ProviderConfigurationError(
            provider_config.provider,
            f"Failed to create chat provider {provider_config.provider!r}: {exc}",
        ) from exc

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

    try:
        await adapter.validate()
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderValidationError(
            provider_config.provider,
            f"Failed to validate chat provider {provider_config.provider!r}: {exc}",
        ) from exc

    return adapter
