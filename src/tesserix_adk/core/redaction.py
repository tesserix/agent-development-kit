"""The shapes that must not travel, and the masking applied before anything is emitted.

A credential, an address or a bearer token is not sensitive because of the key it sits
under — a deny-list of key names only catches the callers who already knew to be careful.
It is sensitive because of its shape, so the shapes live here, once, and both the run's
progress stream and the telemetry exporter mask against the same set.

They are in `core` because the runtime emits before an exporter exists: redaction that
happens at the transport has already been logged by whatever the transport wrote to.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MASK", "SENSITIVE_SHAPES", "looks_sensitive", "scrub"]

MASK = "[redacted]"

SENSITIVE_SHAPES: tuple[str, ...] = (
    r"[\w.+-]+@[\w-]+\.[\w.-]+",
    r"\b(?:sk|pk|rk|api)[-_][A-Za-z0-9\-_]{8,}",
    r"\bBearer\s+\S+",
    r"\beyJ[A-Za-z0-9_-]{10,}",
    r"\b[0-9a-f]{32,}\b",
    r"\b(?:\d{4}[ -]?){3}\d{1,4}\b",
)

_COMPILED = tuple(re.compile(shape) for shape in SENSITIVE_SHAPES)


def looks_sensitive(value: str, extra_patterns: Sequence[str] = ()) -> bool:
    """Whether `value` contains anything shaped like a secret.

    Example:
        >>> looks_sensitive("Bearer abc123")
        True
    """
    return any(pattern.search(value) for pattern in _patterns(extra_patterns))


def scrub(value: str, extra_patterns: Sequence[str] = ()) -> str:
    """Return `value` with each sensitive-looking run replaced by `MASK`.

    Substring masking rather than dropping the whole value: a tool argument with a token in
    one field is still what a consumer needs to render the call, and a field that vanished
    entirely reads as a bug rather than as a decision.

    Args:
        value: The text about to be emitted.
        extra_patterns: Shapes a deployment knows about, such as a local account reference.

    Example:
        >>> scrub("token sk-live-0123456789")
        'token [redacted]'
    """
    for pattern in _patterns(extra_patterns):
        value = pattern.sub(MASK, value)
    return value


def _patterns(extra: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return _COMPILED if not extra else (*_COMPILED, *(re.compile(p) for p in extra))
