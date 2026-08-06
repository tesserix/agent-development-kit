"""Deciding whether to try again, and when.

Two separate questions, kept separate: *is this failure a fault or an answer* is a
property of the error, and *how long to wait* is a property of the policy. Conflating
them is how a schema violation gets retried three times at full price.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import AdkError

if TYPE_CHECKING:
    from tesserix_adk.core.config import RetryConfig

__all__ = ["RetryPlan"]


class RetryPlan:
    """Applies one `RetryConfig`: what to retry, and how long to wait first.

    Args:
        config: The policy.
        random: The source of jitter. Injected so a test can seed it and assert the exact
            delays; left alone it is a fresh `Random`, seeded per plan rather than per
            process, so two runs in one process do not share a schedule.

    Example:
        >>> from tesserix_adk.core import RetryConfig
        >>> RetryPlan(RetryConfig(max_attempts=1)).delay_for(1) is None
        True
    """

    def __init__(self, config: RetryConfig, *, random: Random | None = None) -> None:
        self._config = config
        self._random = random or Random()  # noqa: S311 — jitter, not cryptography

    def retryable(self, failure: BaseException) -> bool:
        """Whether `failure` is a fault worth a second attempt.

        Anything outside the kit's hierarchy is not retried: the kit did not raise it, so
        it does not know whether repeating the call repeats a side effect.
        """
        return isinstance(failure, AdkError) and failure.retryable

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float | None:
        """Seconds to wait before attempt `attempt + 1`, or None if there is not to be one.

        Args:
            attempt: Which attempt just failed, counting from 1.
            retry_after: What the provider asked for, if it asked. Believed in preference
                to the computed window — it knows its own recovery better than a
                multiplier does — but refused outright beyond
                `max_retry_after_seconds`, because a provider asking for an hour is
                reporting a quota rather than a blip.
        """
        if attempt >= self._config.max_attempts:
            return None
        if retry_after is not None:
            return retry_after if retry_after <= self._config.max_retry_after_seconds else None
        window = min(
            self._config.base_delay_seconds * self._config.multiplier ** (attempt - 1),
            self._config.max_delay_seconds,
        )
        return self._random.uniform(0, window)
