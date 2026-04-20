import asyncio
import types
import warnings

import pytest

from agent.config import Config
from agent.embeddings import factory as embeddings_factory
from agent.embeddings import (
    EmbeddingAdapter,
    EmbeddingProviderAuthenticationError,
    EmbeddingProviderConfig,
    EmbeddingProviderConfigurationError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderUnavailableError,
    OpenAICompatibleEmbeddingClient,
    UnsupportedEmbeddingProviderError,
    create_embedding_adapter,
)
from agent.embeddings.providers import openai as openai_provider
from agent.embeddings.providers.openai import OpenAIEmbeddingAdapter


class FakeEmbeddingAdapter:
    provider_name = "fake"

    def __init__(self) -> None:
        self.requests: list[list[str]] = []
        self.validates = 0
        self.probes = 0
        self.closed = False

    async def embed(self, texts):
        batch = list(texts)
        self.requests.append(batch)
        return [(float(index), float(len(text))) for index, text in enumerate(batch)]

    async def aclose(self) -> None:
        self.closed = True

    async def validate(self) -> None:
        self.validates += 1

    async def probe(self) -> None:
        self.probes += 1


async def _assert_embedding_adapter_contract(adapter) -> None:
    assert isinstance(adapter, EmbeddingAdapter)
    await adapter.validate()
    await adapter.probe()
    vectors = await adapter.embed(["alpha", "beta"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 2
    assert all(isinstance(value, float) for vector in vectors for value in vector)

    single = await adapter.embed(["hello world"])
    assert len(single) == 1
    assert len(single[0]) == 2

    await adapter.aclose()
    await adapter.aclose()


def _install_fake_openai(monkeypatch, *, error: Exception | None = None):
    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        def __init__(self, message: str = "api status error", status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(APIError):
        pass

    class AuthenticationError(APIError):
        pass

    class BadRequestError(APIError):
        pass

    class ConflictError(APIError):
        pass

    class InternalServerError(APIError):
        pass

    class NotFoundError(APIError):
        pass

    class PermissionDeniedError(APIError):
        pass

    class RateLimitError(APIError):
        pass

    class AsyncOpenAI:
        instances: list["AsyncOpenAI"] = []
        error: Exception | None = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[dict[str, object]] = []
            self.closed_via: str | None = None
            type(self).instances.append(self)

        @property
        def embeddings(self):
            owner = self

            class _EmbeddingsAPI:
                async def create(self, *, model: str, input: list[str]):
                    owner.calls.append({"model": model, "input": list(input)})
                    if owner.error is not None:
                        raise owner.error
                    return types.SimpleNamespace(
                        data=[
                            types.SimpleNamespace(embedding=(1.0, 0.0))
                            for _ in input
                        ]
                    )

            return _EmbeddingsAPI()

        async def close(self) -> None:
            self.closed_via = "close"

    AsyncOpenAI.error = error

    monkeypatch.setattr(openai_provider, "_OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider, "_AsyncOpenAI", AsyncOpenAI, raising=False)
    monkeypatch.setattr(openai_provider, "_APIConnectionError", APIConnectionError, raising=False)
    monkeypatch.setattr(openai_provider, "_APIError", APIError, raising=False)
    monkeypatch.setattr(openai_provider, "_APIStatusError", APIStatusError, raising=False)
    monkeypatch.setattr(openai_provider, "_APITimeoutError", APITimeoutError, raising=False)
    monkeypatch.setattr(openai_provider, "_AuthenticationError", AuthenticationError, raising=False)
    monkeypatch.setattr(openai_provider, "_BadRequestError", BadRequestError, raising=False)
    monkeypatch.setattr(openai_provider, "_ConflictError", ConflictError, raising=False)
    monkeypatch.setattr(openai_provider, "_InternalServerError", InternalServerError, raising=False)
    monkeypatch.setattr(openai_provider, "_NotFoundError", NotFoundError, raising=False)
    monkeypatch.setattr(openai_provider, "_PermissionDeniedError", PermissionDeniedError, raising=False)
    monkeypatch.setattr(openai_provider, "_RateLimitError", RateLimitError, raising=False)
    return AsyncOpenAI, AuthenticationError, RateLimitError


def test_embedding_adapter_contract_with_fake_adapter() -> None:
    adapter = FakeEmbeddingAdapter()

    asyncio.run(_assert_embedding_adapter_contract(adapter))

    assert adapter.requests == [["alpha", "beta"], ["hello world"]]
    assert adapter.probes == 1
    assert adapter.closed is True


def test_openai_embedding_adapter_contract(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    adapter = OpenAIEmbeddingAdapter(
        EmbeddingProviderConfig(
            provider="openai",
            api_base="http://embeddings.example/v1",
            api_key="embedding-secret",
            model="embed-model",
            options={"timeout": 9},
        )
    )

    asyncio.run(adapter.validate())
    assert len(AsyncOpenAI.instances) == 1
    assert AsyncOpenAI.instances[0].calls == []

    asyncio.run(_assert_embedding_adapter_contract(adapter))

    assert len(AsyncOpenAI.instances) == 1
    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["base_url"] == "http://embeddings.example/v1"
    assert instance.kwargs["api_key"] == "embedding-secret"
    assert instance.kwargs["timeout"] == 9
    assert instance.calls[0]["input"] == ["probe"]
    assert instance.calls[1]["input"] == ["alpha", "beta"]
    assert instance.calls[2]["input"] == ["hello world"]
    assert instance.calls[0]["model"] == "embed-model"
    assert instance.closed_via == "close"


def test_openai_validate_makes_no_network_call(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    adapter = OpenAIEmbeddingAdapter(
        EmbeddingProviderConfig(
            provider="openai",
            api_base="http://embeddings.example/v1",
            api_key="embedding-secret",
            model="embed-model",
            options={"timeout": 9},
        )
    )

    asyncio.run(adapter.validate())
    assert len(AsyncOpenAI.instances) == 1
    assert AsyncOpenAI.instances[0].calls == []

    asyncio.run(adapter.probe())
    vectors = asyncio.run(adapter.embed(["alpha", "beta"]))
    assert len(vectors) == 2
    assert len(vectors[0]) == 2
    assert all(isinstance(value, float) for vector in vectors for value in vector)

    single = asyncio.run(adapter.embed(["hello world"]))
    assert len(single) == 1
    assert len(single[0]) == 2

    asyncio.run(adapter.aclose())
    asyncio.run(adapter.aclose())


def test_embedding_factory_uses_embedding_provider_config(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    config = Config(
        embedding_provider="openai",
        embedding_api_base="http://embeddings.example/v1",
        embedding_api_key="embedding-secret",
        embedding_model="embed-model",
        embedding_provider_options={"timeout": 9},
    )

    adapter = asyncio.run(create_embedding_adapter(config))
    assert len(AsyncOpenAI.instances) == 1
    assert AsyncOpenAI.instances[0].calls == []
    vectors = asyncio.run(adapter.embed(["hello"]))

    assert vectors == [(1.0, 0.0)]
    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["base_url"] == "http://embeddings.example/v1"
    assert instance.kwargs["api_key"] == "embedding-secret"
    assert instance.kwargs["timeout"] == 9
    assert instance.calls[0]["input"] == ["hello"]
    assert instance.calls[0]["model"] == "embed-model"
    assert instance.closed_via is None


def test_embedding_factory_ignores_overrides_for_core_client_settings(
    monkeypatch,
) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    config = Config(
        embedding_provider="openai",
        embedding_api_base="http://embeddings.example/v1",
        embedding_api_key="embedding-secret",
        embedding_model="embed-model",
        embedding_provider_options={
            "base_url": "http://malicious.example/v1",
            "api_key": "wrong-secret",
            "model": "wrong-model",
            "timeout": 7,
        },
    )

    adapter = asyncio.run(create_embedding_adapter(config))
    assert AsyncOpenAI.instances[0].calls == []
    asyncio.run(adapter.embed(["hello"]))

    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["base_url"] == "http://embeddings.example/v1"
    assert instance.kwargs["api_key"] == "embedding-secret"
    assert instance.kwargs["timeout"] == 7
    assert "model" not in instance.kwargs
    assert instance.calls[0]["input"] == ["hello"]


def test_embedding_factory_falls_back_to_primary_api_key(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    config = Config(
        embedding_provider="openai",
        api_key="primary-secret",
        embedding_api_key="",
        embedding_api_base="http://embeddings.example/v1",
        embedding_model="embed-model",
    )

    adapter = asyncio.run(create_embedding_adapter(config))
    assert AsyncOpenAI.instances[0].calls == []
    asyncio.run(adapter.embed(["hello"]))

    instance = AsyncOpenAI.instances[0]
    assert instance.kwargs["api_key"] == "primary-secret"
    assert instance.calls[0]["input"] == ["hello"]


def test_embedding_factory_probes_when_configured(monkeypatch) -> None:
    AsyncOpenAI, _, _ = _install_fake_openai(monkeypatch)

    config = Config(
        embedding_provider="openai",
        embedding_api_base="http://embeddings.example/v1",
        embedding_api_key="embedding-secret",
        embedding_model="embed-model",
        embedding_probe_on_startup=True,
    )

    adapter = asyncio.run(create_embedding_adapter(config))
    instance = AsyncOpenAI.instances[0]
    assert instance.calls[0]["input"] == ["probe"]

    asyncio.run(adapter.embed(["hello"]))
    assert instance.calls[1]["input"] == ["hello"]


@pytest.mark.parametrize(
    ("error_factory", "expected_exception"),
    [
        ("AuthenticationError", EmbeddingProviderAuthenticationError),
        ("RateLimitError", EmbeddingProviderRateLimitError),
    ],
)
def test_embedding_factory_maps_provider_exceptions(
    monkeypatch,
    error_factory: str,
    expected_exception,
) -> None:
    AsyncOpenAI, AuthenticationError, RateLimitError = _install_fake_openai(monkeypatch)
    AsyncOpenAI.error = {
        "AuthenticationError": AuthenticationError("boom"),
        "RateLimitError": RateLimitError("boom"),
    }[error_factory]

    config = Config(
        embedding_provider="openai",
        embedding_api_base="http://embeddings.example/v1",
        embedding_api_key="embedding-secret",
        embedding_model="embed-model",
        embedding_probe_on_startup=True,
    )

    with pytest.raises(expected_exception) as excinfo:
        asyncio.run(create_embedding_adapter(config))

    assert excinfo.value.provider == "openai"


def test_embedding_factory_loads_builder_without_cache(monkeypatch) -> None:
    class FirstAdapter:
        provider_name = "first"

        def __init__(self, config: EmbeddingProviderConfig) -> None:
            self.config = config

        async def embed(self, texts):
            return [(1.0,)]

        async def aclose(self) -> None:
            return None

        async def validate(self) -> None:
            return None

        async def probe(self) -> None:
            return None

    class SecondAdapter:
        provider_name = "second"

        def __init__(self, config: EmbeddingProviderConfig) -> None:
            self.config = config

        async def embed(self, texts):
            return [(2.0,)]

        async def aclose(self) -> None:
            return None

        async def validate(self) -> None:
            return None

        async def probe(self) -> None:
            return None

    call_count = 0
    original_import_module = embeddings_factory.importlib.import_module

    def tracked_import_module(module_name: str, package: str | None = None):
        nonlocal call_count
        call_count += 1
        return original_import_module(module_name, package)

    monkeypatch.setattr(
        embeddings_factory.importlib,
        "import_module",
        tracked_import_module,
    )

    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingAdapter", FirstAdapter, raising=False)

    config = Config(embedding_provider="openai")
    adapter1 = asyncio.run(create_embedding_adapter(config))
    assert isinstance(adapter1, FirstAdapter)

    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingAdapter", SecondAdapter, raising=False)

    adapter2 = asyncio.run(create_embedding_adapter(config))
    assert isinstance(adapter2, SecondAdapter)
    assert call_count == 2


def test_embedding_factory_rejects_unknown_provider() -> None:
    config = Config(embedding_provider="anthropic")

    with pytest.raises(UnsupportedEmbeddingProviderError):
        asyncio.run(create_embedding_adapter(config))


def test_openai_adapter_reports_unavailable_package(monkeypatch) -> None:
    monkeypatch.setattr(openai_provider, "_OPENAI_AVAILABLE", False, raising=False)
    monkeypatch.setattr(openai_provider, "_AsyncOpenAI", None, raising=False)

    adapter = OpenAIEmbeddingAdapter(
        EmbeddingProviderConfig(
            provider="openai",
            api_base="http://embeddings.example/v1",
            api_key="embedding-secret",
            model="embed-model",
        )
    )

    with pytest.raises(EmbeddingProviderUnavailableError):
        asyncio.run(adapter.validate())


def test_openai_adapter_rejects_unknown_options(monkeypatch) -> None:
    _install_fake_openai(monkeypatch)

    adapter = OpenAIEmbeddingAdapter(
        EmbeddingProviderConfig(
            provider="openai",
            api_base="http://embeddings.example/v1",
            api_key="embedding-secret",
            model="embed-model",
            options={"workload_identity": "unexpected"},
        )
    )

    with pytest.raises(EmbeddingProviderConfigurationError):
        asyncio.run(adapter.validate())


def test_legacy_openai_client_alias_still_works(monkeypatch) -> None:
    _install_fake_openai(monkeypatch)

    with pytest.warns(DeprecationWarning):
        client = OpenAICompatibleEmbeddingClient(
            api_base="http://embeddings.example/v1",
            api_key="embedding-secret",
            model="embed-model",
        )

    asyncio.run(client.validate())
    asyncio.run(client.probe())
    vectors = asyncio.run(client.embed(["hello"]))
    assert vectors == [(1.0, 0.0)]
    asyncio.run(client.aclose())


def test_embedding_provider_config_options_are_immutable() -> None:
    provider_config = EmbeddingProviderConfig(
        provider="openai",
        api_base="http://embeddings.example/v1",
        api_key="embedding-secret",
        model="embed-model",
        options={"timeout": 9},
    )

    with pytest.raises(TypeError):
        provider_config.options["timeout"] = 1
