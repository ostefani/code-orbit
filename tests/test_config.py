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
