"""Running one activity under its policy: attempts, waits, beats and the ceiling.

The jitter is drawn here, inside the activity, and never on the workflow path — two
replays that draw their own delays are two different histories. The ceiling is charged
before the attempt rather than after it, so a retry that the allowance cannot cover never
happens, and the ceiling is never widened to let it through.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.activities import Heartbeat
from tesserix_adk.core.errors import BudgetExceededError
from tesserix_adk.core.primitives import Usage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tesserix_adk.core.activities import ActivityClass, ActivityPolicy
    from tesserix_adk.core.budget import RunBudget
    from tesserix_adk.core.protocols import Clock

__all__ = ["ActivityAttempts", "AttemptBudget", "Heartbeater"]


class AttemptBudget:
    """One pool of retries, shared by every branch of a fan-out.

    Ten branches each retrying three times is thirty calls at a provider that is already
    struggling. The pool caps the total; it never refuses a first attempt, which is the
    work itself rather than a retry.

    Example:
        >>> pool = AttemptBudget(1)
        >>> pool.take(), pool.take()
        (True, False)
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self._left = total

    @property
    def left(self) -> int:
        """Retries still available to every branch together."""
        return self._left

    def take(self) -> bool:
        """Claim one retry, or report that the pool is empty."""
        if self._left <= 0:
            return False
        self._left -= 1
        return True


class Heartbeater:
    """Reports progress from a streaming activity, often enough and no oftener.

    Beats three times per heartbeat window, so one lost beat is not read as a dead
    worker, and never once per token, which would be a write per token.

    Args:
        emit: Where a beat goes. `None` for an activity with nowhere to report.
        policy: Whose heartbeat window sets the interval.
        clock: What decides that the interval has passed.
        step: The step being served, carried on every beat.

    Example:
        >>> from tesserix_adk.core import DEFAULT_ACTIVITY_POLICIES, ActivityClass
        >>> from tesserix_adk.testing import FakeClock
        >>> beating = Heartbeater(None, policy=DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL],
        ...                       clock=FakeClock())
        >>> beating.interval
        20.0
    """

    def __init__(
        self,
        emit: Callable[[Heartbeat], None] | None,
        *,
        policy: ActivityPolicy,
        clock: Clock,
        step: str = "",
    ) -> None:
        self._emit = emit
        self._clock = clock
        self._step = step
        self.interval = policy.heartbeat_interval_seconds
        self._tokens = 0
        self._chunks = 0
        self._last_at: float | None = None
        self.last: Heartbeat | None = None

    async def chunk(self, tokens: int = 1) -> None:
        """Take one stream chunk into account, beating where the interval has passed."""
        self._tokens += tokens
        self._chunks += 1
        now = self._clock.now()
        if self._last_at is not None and now - self._last_at < self.interval:
            return
        self._last_at = now
        beat = Heartbeat(step=self._step, tokens=self._tokens, chunks=self._chunks, at=now)
        if self._emit is None:
            return
        self.last = beat
        self._emit(beat)


class ActivityAttempts:
    """Runs one activity under one policy, until it answers or the policy runs out.

    Args:
        policy: The numbers this activity runs under.
        clock: What the waits happen on.
        seed: The jitter seed. Left alone, a fresh source per driver, so two workers
            recovering from one provider blip do not retry in unison.
        budget: The run's ceiling. Each attempt is charged before it is made.
        shared: A retry pool shared across a fan-out.

    Example:
        >>> from tesserix_adk.core import ActivityPolicy
        >>> from tesserix_adk.testing import FakeClock
        >>> ActivityAttempts(ActivityPolicy(), clock=FakeClock()).policy.retry.max_attempts
        1
    """

    def __init__(
        self,
        policy: ActivityPolicy,
        *,
        clock: Clock,
        seed: int | None = None,
        budget: RunBudget | None = None,
        shared: AttemptBudget | None = None,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._random = Random(seed)  # noqa: S311 — jitter, not cryptography
        self._budget = budget
        self._shared = shared

    async def run[T](
        self,
        call: Callable[[int], Awaitable[T]],
        *,
        repeatable: bool = True,
        idempotency_key: str = "",
        charge: ActivityClass | None = None,
    ) -> T:
        """Call `call` with its attempt number until it answers or the policy stops.

        Args:
            call: The work, taking the attempt number from one.
            repeatable: Whether repeating it repeats its effect. `False` means one
                attempt: a typed failure is a better outcome than a second charge.
            idempotency_key: The key the call is made under. Given, a non-repeatable call
                gets its attempts back, because the second one lands on the first's result.
            charge: What each attempt costs the run's ceiling, where there is one.

        Returns:
            Whatever `call` returned.

        Raises:
            BudgetExceededError: Where the next attempt would pass the ceiling. Reports
                the attempts already spent; the ceiling is never widened to fit one more.
            Exception: The last failure, once the attempts are spent.
        """
        allowed = self.policy.retry.max_attempts if repeatable or idempotency_key else 1
        started = self._clock.now()
        attempt = 1
        while True:
            await self._charged(charge, attempt)
            try:
                return await call(attempt)
            except Exception as failure:
                if not self._again(failure, attempt, allowed):
                    raise
                waited = self._wait_for(attempt, started)
                if waited is None:
                    raise
                await self._clock.sleep(waited)
                attempt += 1

    def _again(self, failure: Exception, attempt: int, allowed: int) -> bool:
        """Whether there is to be another attempt at all."""
        if attempt >= allowed or not self.policy.retryable(failure):
            return False
        return self._shared is None or self._shared.take()

    def _wait_for(self, attempt: int, started: float) -> float | None:
        """The jittered wait, cut to what is left of the activity, or `None` for no room."""
        config = self.policy.retry
        window = min(
            config.base_delay_seconds * config.multiplier ** (attempt - 1),
            config.max_delay_seconds,
        )
        backoff = self.policy.backoff(
            self._random.uniform(0, window), elapsed=self._clock.now() - started
        )
        if backoff.truncated and backoff.seconds <= 0:
            return None
        return backoff.seconds

    async def _charged(self, charge: ActivityClass | None, attempt: int) -> None:
        """Charge the attempt before it is made, so one that cannot be afforded is not."""
        if self._budget is None or charge is None:
            return
        counted: dict[str, Any] = {f"{charge.value}_calls": 1}
        try:
            await self._budget.record(Usage(input_tokens=0, output_tokens=0), **counted)
        except BudgetExceededError as stopped:
            raise _with_attempts(stopped, attempt - 1) from stopped


def _with_attempts(stopped: BudgetExceededError, attempts: int) -> BudgetExceededError:
    """The same breach, saying how many attempts were spent reaching it."""
    return BudgetExceededError(
        str(stopped),
        breached=stopped.breached,
        scope=stopped.scope,
        limit=stopped.limit,
        consumed=stopped.consumed,
        remaining=stopped.remaining,
        run_id=stopped.run_id,
        tenant=stopped.tenant,
        details={**dict(stopped.details), "attempts": str(attempts)},
    )
