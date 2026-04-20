from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from agent.config import Config


def test_load_with_diagnostics_reports_profile_selection(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
default_profile: local
profiles:
  local:
    model: test-model
""".strip(),
        encoding="utf-8",
    )

    result = Config.load_with_diagnostics(config_path)

    assert result.config.model == "test-model"
    assert len(result.messages) == 1
    assert result.messages[0].level == "info"
    assert "Using profile 'local'" == result.messages[0].text


def test_load_with_diagnostics_reports_missing_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("profiles: {}\n", encoding="utf-8")

    result = Config.load_with_diagnostics(config_path, profile_name="missing")

    assert result.config.model == "local"
    assert len(result.messages) == 1
    assert result.messages[0].level == "warning"
    assert "Profile 'missing' not found" in result.messages[0].text


def test_config_defaults_include_embedding_settings() -> None:
    config = Config()

    assert config.embedding_model == "nomic-embed-text"
    assert config.embedding_api_base.startswith("http://localhost:")
    assert config.embedding_batch_size == 16
    assert config.embedding_max_concurrency == 4
    assert config.chat_model == "local"
    assert config.chat_api_base.startswith("http://localhost:")
    assert config.chat_context_window == 16384
    assert config.chat_streaming is True


def test_config_derives_chat_context_window_from_context_budget() -> None:
    config = Config(max_context_tokens=8192)

    assert config.chat_context_window == 8192


def test_config_preserves_explicit_chat_context_window() -> None:
    config = Config(max_context_tokens=8192, chat_context_window=4096)

    assert config.chat_context_window == 4096


def test_config_is_immutable_after_construction() -> None:
    config = Config()

    try:
        config.interactive = False
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Expected Config to be frozen")


def test_config_replace_preserves_original_instance() -> None:
    base = Config(interactive=True, auto_commit=False, allow_delete=False)

    overridden = replace(
        base,
        interactive=False,
        auto_commit=True,
        allow_delete=True,
    )

    assert base.interactive is True
    assert base.auto_commit is False
    assert base.allow_delete is False
    assert overridden.interactive is False
    assert overridden.auto_commit is True
    assert overridden.allow_delete is True
