"""What each class of activity is allowed to take, and what is never tried twice.

A reasoning call that streams for nine minutes is healthy; a default heartbeat timeout
reads it as hung. A tool that charges a card is not healthy to repeat; a default retry
policy repeats it three times. One set of numbers cannot serve both, so the numbers are
declared per activity class and derived from what the tool itself said about its timeout
and about repeating it.

Two questions stay separate, as they do in `RetryPlan`: whether a failure is a fault at
all, and how long to wait before trying again. A guardrail block and an approval denial
are answers. They are never faults, and no consumer override makes them one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tesserix_adk.core.config import RetryConfig
from tesserix_adk.core.errors import (
    AdkError,
    ApprovalDeniedError,
    BudgetExceededError,
    GuardrailError,
    SchemaViolationError,
    ScopeEscalationError,
)
from tesserix_adk.core.idempotency import Idempotency
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core.idempotency import IdempotencyPolicy

__all__ = [
    "DEFAULT_ACTIVITY_POLICIES",
    "NEVER_RETRYABLE",
    "ActivityClass",
    "ActivityPolicy",
    "Backoff",
    "Heartbeat",
]

MINIMUM_HEARTBEAT_SECONDS = 5.0
"""The shortest heartbeat window. Below it, a slow first token reads as a dead worker."""

MAXIMUM_HEARTBEAT_SECONDS = 60.0
"""The longest. Beyond it, a worker that died is not noticed for a minute either way."""


class ActivityClass(StrEnum):
    """What kind of work an activity does, which is what its numbers follow from.

    `MODEL` is slow and streams. `TOOL` touches the world and is assumed to change it.
    `RETRIEVAL` and `MEMORY` are reads, so repeating them costs latency and nothing else.
    """

    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"


NEVER_RETRYABLE: tuple[type[AdkError], ...] = (
    SchemaViolationError,
    GuardrailError,
    BudgetExceededError,
    ApprovalDeniedError,
    ScopeEscalationError,
)
"""Failures that are answers rather than faults, whatever a consumer's policy says.

Retrying a guardrail block is asking the guardrail until it gives in; retrying a budget
breach is spending past the ceiling one attempt at a time.
"""


class Heartbeat(AdkModel):
    """Progress from a running activity. Never a result, and never part of one.

    Carries counts and nothing else. A heartbeat that carried the partial text would be
    read as an answer by the first consumer in a hurry, and half a sentence rendered as a
    reply is worse than a spinner.

    Args:
        step: The step the activity is serving, so a reader can place the progress.
        tokens: Tokens seen so far.
        chunks: Stream chunks seen so far.
        at: When the beat was taken, on the run's clock.

    Example:
        >>> Heartbeat(step="step-3", tokens=120, chunks=8).is_result
        False
    """

    step: str = ""
    tokens: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    at: float = 0.0

    @property
    def is_result(self) -> bool:
        """Always false. It exists so the question has an answer in the type."""
        return False


class Backoff(AdkModel):
    """How long to wait, and whether the wait was cut to fit inside the activity.

    Args:
        seconds: What to wait. Zero means there is no room left for another attempt.
        truncated: Whether the computed window did not fit.
        reason: What it did not fit inside, where it was cut.

    Example:
        >>> Backoff(seconds=2.0).truncated
        False
    """

    seconds: float = Field(default=0.0, ge=0.0)
    truncated: bool = False
    reason: str = ""


class ActivityPolicy(AdkModel):
    """The numbers one activity runs under.

    Args:
        activity_class: What kind of work it is.
        retry: What to retry and how long to wait, before the never-retryable list is
            applied over the top of it.
        start_to_close_seconds: How long the whole activity may take, retries included.
        heartbeat_timeout_seconds: How long a silence is a dead worker. Zero for an
            activity that does not beat at all.

    Example:
        >>> ActivityPolicy().retryable(GuardrailError("blocked"))
        False
    """

    activity_class: ActivityClass = ActivityClass.TOOL
    retry: RetryConfig = RetryConfig()
    start_to_close_seconds: float = Field(default=60.0, gt=0.0)
    heartbeat_timeout_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _the_windows_can_be_breached(self) -> ActivityPolicy:
        """A heartbeat window wider than the activity is not a timeout at all."""
        if self.heartbeat_timeout_seconds >= self.start_to_close_seconds:
            raise ValueError(
                f"heartbeat_timeout_seconds must fit inside start_to_close_seconds, or "
                f"nothing could ever breach it; got {self.heartbeat_timeout_seconds} "
                f"against {self.start_to_close_seconds}"
            )
        return self

    @property
    def heartbeat_interval_seconds(self) -> float:
        """How often to beat: three per window, so one lost beat is not a death."""
        return self.heartbeat_timeout_seconds / 3

    def retryable(self, failure: BaseException) -> bool:
        """Whether `failure` is a fault worth another attempt.

        The never-retryable list wins over the policy: an answer does not become a fault
        because someone raised `max_attempts`. Anything outside the kit's hierarchy is
        left alone, because the kit does not know what repeating it repeats.
        """
        if isinstance(failure, NEVER_RETRYABLE):
            return False
        return isinstance(failure, AdkError) and failure.retryable

    def attempts_for(self, idempotency: IdempotencyPolicy | None, *, keyed: bool = False) -> int:
        """How many attempts a tool that declared `idempotency` gets.

        An effectful tool gets one, because the second attempt books a second seat. A key
        gives the attempts back: with one, the second call lands on the first call's
        result rather than beside it. A tool that declared nothing is treated as
        effectful, since assuming otherwise is assuming in the expensive direction.
        """
        if idempotency is None:
            return 1
        if idempotency.kind is Idempotency.EFFECTFUL and not keyed:
            return 1
        return self.retry.max_attempts

    def backoff(self, seconds: float, *, elapsed: float) -> Backoff:
        """Fit a computed wait inside what is left of the activity.

        A wait that would outlast `start_to_close_seconds` is cut to the remainder and
        reported as cut. Dropping it silently leaves an operator reading a schedule the
        run never followed.
        """
        left = self.start_to_close_seconds - elapsed
        if seconds <= left:
            return Backoff(seconds=seconds)
        return Backoff(
            seconds=max(left, 0.0),
            truncated=True,
            reason=(
                f"a {seconds:g}s wait does not fit in the {left:g}s left of start_to_close_seconds"
            ),
        )


DEFAULT_ACTIVITY_POLICIES: Mapping[ActivityClass, ActivityPolicy] = {
    ActivityClass.MODEL: ActivityPolicy(
        activity_class=ActivityClass.MODEL,
        retry=RetryConfig(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=30.0),
        start_to_close_seconds=900.0,
        heartbeat_timeout_seconds=MAXIMUM_HEARTBEAT_SECONDS,
    ),
    ActivityClass.TOOL: ActivityPolicy(
        activity_class=ActivityClass.TOOL,
        retry=RetryConfig(max_attempts=1),
        start_to_close_seconds=60.0,
    ),
    ActivityClass.RETRIEVAL: ActivityPolicy(
        activity_class=ActivityClass.RETRIEVAL,
        retry=RetryConfig(max_attempts=3, base_delay_seconds=0.25, max_delay_seconds=5.0),
        start_to_close_seconds=30.0,
    ),
    ActivityClass.MEMORY: ActivityPolicy(
        activity_class=ActivityClass.MEMORY,
        retry=RetryConfig(max_attempts=3, base_delay_seconds=0.25, max_delay_seconds=5.0),
        start_to_close_seconds=15.0,
    ),
}
"""The defaults, per class. Tuning one is a value change and never a breaking change."""
