import inspect
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingClient(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class _EmbeddingVectorsAPI(Protocol):
    async def create(self, *, model: str, input: list[str]) -> Any: ...


class _EmbeddingClientAPI(Protocol):
    @property
    def embeddings(self) -> _EmbeddingVectorsAPI: ...


class OpenAICompatibleEmbeddingClient:
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: _EmbeddingClientAPI | None = None

    def _get_client(self) -> _EmbeddingClientAPI:
        client = self._client
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise RuntimeError(
                    "The 'openai' package is required for embedding generation."
                ) from exc

            client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
            self._client = client
        return client

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=list(texts))
        return [tuple(item.embedding) for item in response.data]

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return

        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        self._client = None
