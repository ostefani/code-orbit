"""Shared OpenAI SDK plumbing for the chat and embedding providers.

Single home for the client-option coercion table, the client-constructor
kwargs builder, and the exception-classification decision table.

Provider modules deliberately keep their own ``try/except ImportError`` SDK
import blocks: tests patch those module globals with fakes, so the shared
classifier receives the exception classes from its caller instead of
importing the SDK here.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from httpx import Timeout, URL
from openai import NOT_GIVEN


class OpenAIErrorKind(Enum):
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    UNAVAILABLE = "unavailable"
    REQUEST = "request"


@dataclass(frozen=True)
class OpenAIExceptionTypes:
    """SDK exception classes, sourced from the calling provider module."""

    available: bool
    authentication_error: type[BaseException] | None
    rate_limit_error: type[BaseException] | None
    connection_error: type[BaseException] | None
    timeout_error: type[BaseException] | None
    internal_server_error: type[BaseException] | None
    bad_request_error: type[BaseException] | None
    conflict_error: type[BaseException] | None
    not_found_error: type[BaseException] | None
    permission_denied_error: type[BaseException] | None
    api_error: type[BaseException] | None
    status_error: type[BaseException] | None


def _matches(exc: BaseException, cls: type[BaseException] | None) -> bool:
    return cls is not None and isinstance(exc, cls)


def _present(
    *classes: type[BaseException] | None,
) -> tuple[type[BaseException], ...]:
    return tuple(cls for cls in classes if cls is not None)


def classify_openai_status_code(status_code: object) -> OpenAIErrorKind:
    """Shared HTTP-status table used for status-bearing errors by both paths."""
    if isinstance(status_code, bool) or not isinstance(status_code, (int, str)):
        return OpenAIErrorKind.REQUEST
    try:
        status = int(status_code)
    except ValueError:
        return OpenAIErrorKind.REQUEST
    if status == 429:
        return OpenAIErrorKind.RATE_LIMIT
    if status in (401, 403):
        return OpenAIErrorKind.AUTHENTICATION
    if status >= 500:
        return OpenAIErrorKind.UNAVAILABLE
    return OpenAIErrorKind.REQUEST


def classify_openai_exception(
    exc: Exception, types: OpenAIExceptionTypes
) -> OpenAIErrorKind:
    if not types.available:
        return OpenAIErrorKind.REQUEST
    if _matches(exc, types.authentication_error):
        return OpenAIErrorKind.AUTHENTICATION
    if _matches(exc, types.rate_limit_error):
        return OpenAIErrorKind.RATE_LIMIT
    transient = _present(
        types.connection_error,
        types.timeout_error,
        types.internal_server_error,
    )
    if transient and isinstance(exc, transient):
        return OpenAIErrorKind.UNAVAILABLE
    # Status-bearing errors resolve through the shared table before the
    # generic API-error bucket, so e.g. a bare 403 classifies as auth rather
    # than a plain request error.
    if _matches(exc, types.status_error):
        return classify_openai_status_code(getattr(exc, "status_code", None))
    client_errors = _present(
        types.bad_request_error,
        types.conflict_error,
        types.not_found_error,
        types.permission_denied_error,
        types.api_error,
    )
    if client_errors and isinstance(exc, client_errors):
        return OpenAIErrorKind.REQUEST
    return OpenAIErrorKind.REQUEST


E = TypeVar("E", bound=Exception)

BASE_OPENAI_OPTION_KEYS = frozenset(
    {
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "organization",
        "project",
        "webhook_secret",
        "websocket_base_url",
    }
)


def coerce_openai_client_options(
    provider: str,
    options: Mapping[str, object],
    *,
    label: str,
    error_cls: type[E],
    allowed_keys: Collection[str],
) -> dict[str, Any]:
    """Coerce the client options shared by the chat and embedding providers.

    ``label`` ("chat"/"embedding") only steers error-message wording.
    ``allowed_keys`` is the caller's full allowlist: the base keys plus any
    provider-specific extras, which that provider coerces itself afterwards.
    """
    unknown = sorted(key for key in options if key not in allowed_keys)
    if unknown:
        raise error_cls(
            provider,
            f"Unsupported OpenAI {label} options: "
            + ", ".join(repr(key) for key in unknown)
            + ".",
        )

    result: dict[str, Any] = {}

    if "timeout" in options:
        timeout = options["timeout"]
        if timeout is None or isinstance(timeout, (int, float, Timeout)):
            result["timeout"] = timeout
        else:
            raise error_cls(
                provider,
                f"OpenAI {label} option 'timeout' must be a number, httpx.Timeout, or null.",
            )

    if "max_retries" in options:
        max_retries = options["max_retries"]
        if isinstance(max_retries, int) and not isinstance(max_retries, bool):
            result["max_retries"] = max_retries
        else:
            raise error_cls(
                provider,
                f"OpenAI {label} option 'max_retries' must be an integer.",
            )

    if "default_headers" in options:
        default_headers = options["default_headers"]
        if isinstance(default_headers, Mapping):
            headers: dict[str, str] = {}
            for header_key, header_value in default_headers.items():
                if not isinstance(header_key, str) or not isinstance(header_value, str):
                    raise error_cls(
                        provider,
                        f"OpenAI {label} option 'default_headers' must map strings to strings.",
                    )
                headers[header_key] = header_value
            result["default_headers"] = headers
        elif default_headers is not None:
            raise error_cls(
                provider,
                f"OpenAI {label} option 'default_headers' must be a mapping or null.",
            )

    if "default_query" in options:
        default_query = options["default_query"]
        if isinstance(default_query, Mapping):
            query: dict[str, object] = {}
            for query_key, query_value in default_query.items():
                if not isinstance(query_key, str):
                    raise error_cls(
                        provider,
                        f"OpenAI {label} option 'default_query' must use string keys.",
                    )
                query[query_key] = query_value
            result["default_query"] = query
        elif default_query is not None:
            raise error_cls(
                provider,
                f"OpenAI {label} option 'default_query' must be a mapping or null.",
            )

    for key in ("organization", "project", "webhook_secret"):
        if key in options:
            value = options[key]
            if isinstance(value, str):
                result[key] = value
            elif value is not None:
                raise error_cls(
                    provider,
                    f"OpenAI {label} option '{key}' must be a string or null.",
                )

    if "websocket_base_url" in options:
        websocket_base_url = options["websocket_base_url"]
        if isinstance(websocket_base_url, (str, URL)):
            result["websocket_base_url"] = websocket_base_url
        elif websocket_base_url is not None:
            raise error_cls(
                provider,
                f"OpenAI {label} option 'websocket_base_url' must be a string, httpx.URL, or null.",
            )

    return result


def base_openai_client_kwargs(
    openai_options: Mapping[str, Any],
    *,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    """Constructor kwargs shared by both adapters.

    The embedding adapter adds ``http_client`` and
    ``_strict_response_validation`` on top. Key set and defaults match the
    previous per-provider unpacking exactly.
    """
    return {
        "api_key": api_key,
        "organization": openai_options.get("organization"),
        "project": openai_options.get("project"),
        "webhook_secret": openai_options.get("webhook_secret"),
        "base_url": base_url,
        "websocket_base_url": openai_options.get("websocket_base_url"),
        "timeout": openai_options.get("timeout", NOT_GIVEN),
        "max_retries": openai_options.get("max_retries", 2),
        "default_headers": openai_options.get("default_headers"),
        "default_query": openai_options.get("default_query"),
    }
