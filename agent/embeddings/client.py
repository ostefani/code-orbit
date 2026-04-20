import warnings
from collections.abc import Sequence

from .adapters import (
    EmbeddingAdapter,
    EmbeddingAdapter as EmbeddingClient,
    EmbeddingProviderConfig,
)


class OpenAICompatibleEmbeddingClient:
    provider_name = "openai"

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        warnings.warn(
            "OpenAICompatibleEmbeddingClient is deprecated; use OpenAIEmbeddingAdapter via the provider factory.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .providers.openai import OpenAIEmbeddingAdapter

        self._adapter = OpenAIEmbeddingAdapter(
            EmbeddingProviderConfig(
                provider="openai",
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._adapter.embed(texts)

    async def aclose(self) -> None:
        await self._adapter.aclose()

    async def validate(self) -> None:
        await self._adapter.validate()

    async def probe(self) -> None:
        await self._adapter.probe()


__all__ = ["EmbeddingAdapter", "EmbeddingClient", "OpenAICompatibleEmbeddingClient"]
