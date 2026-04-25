import logging

from agent.config import Config
from agent import token_counter
from agent.token_counter import count_tokens


def test_count_tokens_reports_fallback_warning_for_missing_tokenizer_path() -> None:
    token_counter._warned.clear()
    config = Config(tokenizer_backend="tokenizers_json")

    result = count_tokens("hello world", config)

    assert result.count > 0
    assert result.tokenizer_name == "estimate:chars_div_3"
    assert result.warnings == (
        "tokenizer_backend='tokenizers_json' requires tokenizer_model_path in config. Falling back to estimate.",
    )


def test_count_tokens_logs_fallback_warning_once(caplog) -> None:
    token_counter._warned.clear()
    config = Config(tokenizer_backend="tokenizers_json")

    with caplog.at_level(logging.WARNING, logger="agent.token_counter"):
        count_tokens("hello world", config)
        count_tokens("hello again", config)

    assert [record.message for record in caplog.records] == [
        "tokenizer_backend='tokenizers_json' requires tokenizer_model_path in config. Falling back to estimate."
    ]
