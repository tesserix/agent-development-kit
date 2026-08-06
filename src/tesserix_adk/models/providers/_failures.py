"""One taxonomy over three vendors' ways of saying the same six things.

A rate limit is `rate_limit_error` at Anthropic, `rate_limit_exceeded` at OpenAI and
`RESOURCE_EXHAUSTED` at Google, under statuses that mostly but not always agree. A
consumer that branches on those strings has written three error handlers and will write a
fourth for the next endpoint, so the vendor's vocabulary is translated here, once, and
what leaves is the kit's own error with the retry decision already made.

The default is explicit rather than absent: a code nobody has mapped becomes a plain
`ProviderError` whose retryability follows its status. Guessing that an unknown failure is
transient is how a broken deployment becomes a burst of identical calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesserix_adk.core.errors import (
    AuthenticationError,
    ContentFilteredError,
    ContextWindowExceededError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

if TYPE_CHECKING:
    import httpx

    from tesserix_adk.core.errors import AdkError

__all__ = ["failure_for"]

_MESSAGE_IN_ERRORS = 200

# Each vendor's own word for the event, lowercased. Anthropic's `error.type`, OpenAI's
# `error.code`, Google's `error.status` — read from whichever of the three is present.
_RATE_LIMITED = frozenset(
    {"rate_limit_error", "rate_limit_exceeded", "resource_exhausted", "requests", "tokens"}
)
_QUOTA_SPENT = frozenset(
    {"insufficient_quota", "quota_exceeded", "billing_hard_limit_reached", "billing_not_active"}
)
_UNAUTHENTICATED = frozenset(
    {
        "authentication_error",
        "invalid_api_key",
        "invalid_authentication",
        "permission_denied",
        "permission_error",
        "unauthenticated",
    }
)
_FILTERED = frozenset(
    {"blocked", "content_filter", "content_policy_violation", "prompt_blocked", "safety"}
)
_TOO_LONG = frozenset(
    {"context_length_exceeded", "context_window_exceeded", "request_too_large", "too_many_tokens"}
)
_REJECTED = frozenset(
    {
        "failed_precondition",
        "invalid_argument",
        "invalid_request_error",
        "model_not_found",
        "not_found",
        "not_found_error",
    }
)
_NOT_THERE = frozenset({"overloaded_error", "unavailable"})

# Where the vendor sent no code the kit knows, its status says the same things.
_BY_STATUS: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: InvalidRequestError,
    405: InvalidRequestError,
    406: InvalidRequestError,
    408: ProviderTimeoutError,
    413: InvalidRequestError,
    415: InvalidRequestError,
    422: InvalidRequestError,
    429: RateLimitError,
    502: ProviderUnavailableError,
    503: ProviderUnavailableError,
    504: ProviderUnavailableError,
    529: ProviderUnavailableError,
}


def failure_for(
    response: httpx.Response,
    *,
    provider: str,
    model: str,
    window: int = 0,
    retry_after: float | None = None,
    request_id: str = "",
    redact: bool = True,
) -> AdkError:
    """Return the kit's error for one failed HTTP response.

    Args:
        response: What the vendor answered with.
        provider: Who answered.
        model: Which of its models was asked.
        window: The declared context window, carried where the vendor reports an overflow
            without saying how far over it was — which is what all three of them do.
        retry_after: The wait the vendor asked for, already parsed from its header.
        request_id: The vendor's own id for the call.
        redact: Whether to drop the vendor's free-text message. On by default: a 400 body
            quotes the request that caused it, and a request body is the prompt.

    Returns:
        The error to raise, never raised here — the caller owns the traceback.
    """
    code, message = _reported(response)
    detail = {"code": code, "status": str(response.status_code)}
    if not redact and message:
        detail["message"] = message[:_MESSAGE_IN_ERRORS]
    said = f"{provider} answered {response.status_code}" + (f" ({code})" if code else "")
    if code in _TOO_LONG:
        return ContextWindowExceededError(
            said, limit=window, provider=provider, model=model, details=detail
        )
    common: dict[str, Any] = {
        "status": response.status_code,
        "retry_after": retry_after,
        "provider": provider,
        "model": model,
        "request_id": request_id,
        "details": detail,
    }
    if code in _QUOTA_SPENT:
        return RateLimitError(said, quota=True, **common)
    return _kind_for(code, response.status_code)(said, **common)


def _kind_for(code: str, status: int) -> type[ProviderError]:
    """The taxonomy member for one vendor code, falling back to what the status says."""
    for codes, kind in (
        (_RATE_LIMITED, RateLimitError),
        (_UNAUTHENTICATED, AuthenticationError),
        (_FILTERED, ContentFilteredError),
        (_REJECTED, InvalidRequestError),
        (_NOT_THERE, ProviderUnavailableError),
    ):
        if code in codes:
            return kind
    return _BY_STATUS.get(status, ProviderError)


def _reported(response: httpx.Response) -> tuple[str, str]:
    """The vendor's own code and message, from wherever that vendor puts them."""
    try:
        body = response.json()
    except ValueError:
        return "", ""
    reported = body.get("error") if isinstance(body, dict) else None
    if isinstance(reported, str):
        return "", reported
    if not isinstance(reported, dict):
        return "", ""
    # Google's `code` is the numeric status and its word is in `status`, so a code that is
    # not a string is passed over rather than read as the vendor's name for the event.
    named = [reported.get(field) for field in ("code", "status", "type")]
    code = next((word for word in named if isinstance(word, str) and word), "")
    return code.lower(), str(reported.get("message") or "")
