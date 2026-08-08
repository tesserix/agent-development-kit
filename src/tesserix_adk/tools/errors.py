"""What a tool failing means, said precisely enough for the run loop to act on it.

A generic exception tells the runtime nothing it can use. "The supplier was briefly
unavailable" and "this booking is not cancellable" are both exceptions and want opposite
answers: one is worth another attempt, the other is the answer. Without the distinction a
run retries a refusal until the iteration cap fires, spending the budget to be told the
same thing and, worse, re-attempting an action the downstream already declined.

Codes are public API. See `docs/tool-errors.md` for the stability policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tesserix_adk.core import (
    ToolArgumentValidationError,
    ToolError,
    ToolFailure,
    ToolNotFoundError,
    ToolNotPermittedError,
    ToolRefusal,
    ToolResultError,
    ToolTimedOutError,
)
from tesserix_adk.core.redaction import scrub

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ToolArgumentValidationError",
    "ToolError",
    "ToolErrorMap",
    "ToolErrorRule",
    "ToolFailure",
    "ToolNotFoundError",
    "ToolNotPermittedError",
    "ToolRefusal",
    "ToolResultError",
    "ToolTimedOutError",
    "permanent",
    "refusal",
    "transient",
]

UNMAPPED = "unmapped_failure"
"""What an exception nobody classified is called. Permanent, because guessing costs money."""


@dataclass(frozen=True, slots=True)
class ToolErrorRule:
    """What one library exception means, in the taxonomy's terms.

    Args:
        code: The stable name to give it.
        kind: Whether it is a failure or the tool declining.
        transient: Whether it is worth another attempt. Only meaningful for failures.
        retry_after: A wait to honour, where the upstream declares one.
        message: What the model may be told. Required for a refusal, which exists to be
            explained.
    """

    code: str
    kind: Literal["failure", "refusal"] = "failure"
    transient: bool = False
    retry_after: float | None = None
    message: str = ""

    def raised_by(self, tool: str, detail: str) -> ToolError:
        """The typed error this rule turns `detail` into, for `tool`."""
        if self.kind == "refusal":
            return ToolRefusal(tool, self.code, self.message)
        return ToolFailure(
            tool,
            self.code,
            transient=self.transient,
            retry_after=self.retry_after,
            detail=self.message or detail,
        )


def transient(code: str, *, retry_after: float | None = None, message: str = "") -> ToolErrorRule:
    """A failure worth another attempt, because nothing landed."""
    return ToolErrorRule(code, transient=True, retry_after=retry_after, message=message)


def permanent(code: str, *, message: str = "") -> ToolErrorRule:
    """A failure no retry will fix."""
    return ToolErrorRule(code, message=message)


def refusal(code: str, message: str) -> ToolErrorRule:
    """The tool declining, with what the model may be told about it."""
    return ToolErrorRule(code, kind="refusal", message=message)


class ToolErrorMap:
    """Translates the exceptions a tool's libraries raise into the taxonomy.

    Declarative on purpose. A tool author writing `except` blocks by hand eventually writes
    a bare one, and a bare one classifies a bug as a retryable failure.

    Args:
        rules: What each exception type means. The most specific match on the raised type's
            MRO wins, so a base class can catch a family and a subclass can differ.
        statuses: What an HTTP status means, read from the exception's `status_code` or
            `status` where it carries one and no type rule matched.

    Example:
        >>> ToolErrorMap({TimeoutError: transient("upstream_slow")}).classify(
        ...     TimeoutError(), tool="book"
        ... ).retryable
        True
    """

    def __init__(
        self,
        rules: Mapping[type[BaseException], ToolErrorRule],
        *,
        statuses: Mapping[int, ToolErrorRule] | None = None,
    ) -> None:
        self._rules = dict(rules)
        self._statuses = dict(statuses or {})

    def classify(self, failure: BaseException, *, tool: str) -> ToolError:
        """The typed error `failure` means for `tool`.

        An exception already in the taxonomy is returned untouched: whoever raised it knew
        more than this map does. An unmapped exception is a permanent failure — the kit
        does not know whether repeating the call repeats a side effect, and guessing that
        it does not is how a run pays twice for one booking.

        Raises:
            BaseException: Cancellation and anything else outside `Exception` is
                re-raised. A run being torn down is not a fault to retry.
        """
        if isinstance(failure, ToolError):
            return failure
        if not isinstance(failure, Exception):
            raise failure
        detail = scrub(str(failure))
        rule = self._rule_for(failure)
        return rule.raised_by(tool, detail) if rule else ToolFailure(tool, UNMAPPED, detail=detail)

    def _rule_for(self, failure: Exception) -> ToolErrorRule | None:
        for ancestor in type(failure).__mro__:
            if (rule := self._rules.get(ancestor)) is not None:
                return rule
        return self._statuses.get(_status_of(failure), None) if self._statuses else None


def _status_of(failure: Exception) -> int:
    """The HTTP status an exception carries, or 0 where it carries none."""
    for attribute in ("status_code", "status"):
        value = getattr(failure, attribute, None)
        if isinstance(value, int):
            return value
    return 0
