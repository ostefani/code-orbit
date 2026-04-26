import warnings
from pathlib import Path

from agent.config import Config
from agent import token_counter
from agent.token_counter import count_tokens


def test_count_tokens_reports_fallback_warning_for_missing_tokenizer_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does-not-exist-tokenizer.json"
    config = Config(
        tokenizer_backend="tokenizers_json",
        tokenizer_model_path=missing_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once", token_counter.TokenCounterFallbackWarning)
        result = count_tokens("hello world", config)

    assert result.count > 0
    assert result.tokenizer_name == "estimate:chars_div_3"
    assert result.warnings == (
        f"tokenizer.json not found at '{missing_path.resolve()}'. Falling back to estimate.",
    )
    assert [str(item.message) for item in caught] == [
        f"tokenizer.json not found at '{missing_path.resolve()}'. Falling back to estimate."
    ]


def test_count_tokens_warns_once_for_repeated_fallbacks(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist-tokenizer.json"
    config = Config(
        tokenizer_backend="tokenizers_json",
        tokenizer_model_path=missing_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once", token_counter.TokenCounterFallbackWarning)
        count_tokens("hello world", config)
        count_tokens("hello again", config)

    assert [str(item.message) for item in caught] == [
        f"tokenizer.json not found at '{missing_path.resolve()}'. Falling back to estimate."
    ]


def test_count_tokens_reports_warning_for_missing_tokenizer_path_branch() -> None:
    config = Config(tokenizer_backend="tokenizers_json")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once", token_counter.TokenCounterFallbackWarning)
        result = count_tokens("hello world", config)

    assert result.count > 0
    assert result.tokenizer_name == "estimate:chars_div_3"
    assert result.warnings == (
        "tokenizer_backend='tokenizers_json' requires tokenizer_model_path in config. Falling back to estimate.",
    )
    assert [str(item.message) for item in caught] == [
        "tokenizer_backend='tokenizers_json' requires tokenizer_model_path in config. Falling back to estimate."
    ]
