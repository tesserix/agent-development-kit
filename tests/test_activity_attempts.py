"""Running an activity under its policy: attempts, backoff, heartbeats and the ceiling."""

from __future__ import annotations

from decimal import Decimal

import pytest

from tesserix_adk.core import (
    DEFAULT_ACTIVITY_POLICIES,
    ActivityClass,
    ActivityPolicy,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    GuardrailViolationError,
    Heartbeat,
    ProviderUnavailableError,
    ResolvedBudget,
    RetryConfig,
    RunBudget,
    ScopedLimits,
    most_restrictive,
)
from tesserix_adk.runtime import ActivityAttempts, AttemptBudget, Heartbeater
from tesserix_adk.testing import FakeClock

pytestmark = pytest.mark.anyio


class Flaky:
    """A call that fails `failures` times before it answers."""

    def __init__(self, failures: int, *, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or ProviderUnavailableError("503")
        self.attempts: list[int] = []

    async def __call__(self, attempt: int) -> str:
        self.attempts.append(attempt)
        if len(self.attempts) <= self.failures:
            raise self.error
        return "answered"


def attempts(
    *,
    policy: ActivityPolicy | None = None,
    clock: FakeClock | None = None,
    budget: RunBudget | None = None,
    shared: AttemptBudget | None = None,
) -> ActivityAttempts:
    """A driver whose waits and jitter are decided by a seeded, movable clock."""
    return ActivityAttempts(
        policy or ActivityPolicy(retry=RetryConfig(max_attempts=3, base_delay_seconds=1.0)),
        clock=clock or FakeClock(),
        seed=7,
        budget=budget,
        shared=shared,
    )


def budget_of(**limits: object) -> RunBudget:
    """A run ceiling over nothing but the limits named."""
    resolved: ResolvedBudget = most_restrictive(
        ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(**limits))  # type: ignore[arg-type]
    )
    return RunBudget(resolved, FakeClock())


class TestTryingAgain:
    async def test_a_fault_is_retried_until_it_answers(self) -> None:
        call = Flaky(failures=2)

        assert await attempts().run(call) == "answered"
        assert call.attempts == [1, 2, 3]

    async def test_the_attempts_run_out(self) -> None:
        call = Flaky(failures=9)

        with pytest.raises(ProviderUnavailableError):
            await attempts().run(call)

        assert len(call.attempts) == 3

    async def test_an_answer_is_never_retried(self) -> None:
        """A guardrail block is a decision. Repeating it is asking until it gives in."""
        call = Flaky(failures=1, error=GuardrailViolationError("blocked"))

        with pytest.raises(GuardrailViolationError):
            await attempts().run(call)

        assert call.attempts == [1]

    async def test_the_wait_happens_on_the_clock_rather_than_in_real_time(self) -> None:
        clock = FakeClock()
        call = Flaky(failures=1)

        await attempts(clock=clock).run(call)

        assert clock.now() > 0

    async def test_two_drivers_do_not_wait_in_unison(self) -> None:
        """Full jitter, so a fleet recovering from one blip does not cause the next."""
        one, other = FakeClock(), FakeClock()

        await ActivityAttempts(
            ActivityPolicy(retry=RetryConfig(max_attempts=3)), clock=one, seed=1
        ).run(Flaky(failures=1))
        await ActivityAttempts(
            ActivityPolicy(retry=RetryConfig(max_attempts=3)), clock=other, seed=2
        ).run(Flaky(failures=1))

        assert one.now() != other.now()


class TestATooLThatCannotBeRepeated:
    async def test_it_is_tried_once_and_the_failure_is_returned_typed(self) -> None:
        """A second charge is worse than a reported failure."""
        call = Flaky(failures=9)

        with pytest.raises(ProviderUnavailableError):
            await attempts().run(call, repeatable=False)

        assert call.attempts == [1]

    async def test_a_key_gives_the_attempts_back(self) -> None:
        call = Flaky(failures=1)

        assert await attempts().run(call, repeatable=False, idempotency_key="run-1:book") == (
            "answered"
        )
        assert call.attempts == [1, 2]


class TestTheCeiling:
    async def test_a_retry_is_charged_against_the_run_s_allowance(self) -> None:
        budget = budget_of(max_model_calls=5)
        call = Flaky(failures=2)

        await attempts(budget=budget).run(call, charge=ActivityClass.MODEL)

        assert budget.spent.model_calls == 3

    async def test_retries_that_would_pass_the_ceiling_fail_closed(self) -> None:
        budget = budget_of(max_model_calls=2)
        call = Flaky(failures=9)

        with pytest.raises(BudgetExceededError):
            await attempts(budget=budget).run(call, charge=ActivityClass.MODEL)

    async def test_the_ceiling_is_never_raised_to_let_the_next_retry_through(self) -> None:
        budget = budget_of(max_model_calls=2)

        with pytest.raises(BudgetExceededError) as stopped:
            await attempts(budget=budget).run(Flaky(failures=9), charge=ActivityClass.MODEL)

        assert stopped.value.limit == Decimal(2)

    async def test_the_failure_reports_how_many_attempts_were_spent(self) -> None:
        budget = budget_of(max_model_calls=2)

        with pytest.raises(BudgetExceededError) as stopped:
            await attempts(budget=budget).run(Flaky(failures=9), charge=ActivityClass.MODEL)

        assert stopped.value.details["attempts"] == "2"


class TestRetryStormsAcrossFanOut:
    async def test_the_branches_share_one_pool_of_attempts(self) -> None:
        """Ten branches retrying three times each is thirty calls at one provider."""
        shared = AttemptBudget(3)
        first, second = Flaky(failures=9), Flaky(failures=9)

        with pytest.raises(ProviderUnavailableError):
            await attempts(shared=shared).run(first)
        with pytest.raises(ProviderUnavailableError):
            await attempts(shared=shared).run(second)

        assert len(first.attempts) + len(second.attempts) == 5

    async def test_an_exhausted_pool_still_lets_the_first_attempt_through(self) -> None:
        """The pool caps retries. Refusing the work outright is a different failure."""
        shared = AttemptBudget(0)
        call = Flaky(failures=0)

        assert await attempts(shared=shared).run(call) == "answered"

    async def test_what_is_left_is_readable(self) -> None:
        shared = AttemptBudget(3)

        with pytest.raises(ProviderUnavailableError):
            await attempts(shared=shared).run(Flaky(failures=9))

        assert shared.left == 1


class TestHeartbeating:
    async def test_a_streaming_call_reports_progress(self) -> None:
        beats: list[Heartbeat] = []
        clock = FakeClock()
        beating = Heartbeater(beats.append, policy=_streaming(), clock=clock, step="step-3")

        for _ in range(3):
            clock.advance(30)
            await beating.chunk(tokens=10)

        assert [one.tokens for one in beats] == [10, 20, 30]

    async def test_it_does_not_beat_on_every_token(self) -> None:
        """A beat per token is a write per token."""
        beats: list[Heartbeat] = []
        beating = Heartbeater(beats.append, policy=_streaming(), clock=FakeClock())

        for _ in range(50):
            await beating.chunk(tokens=1)

        assert len(beats) <= 1

    async def test_it_beats_often_enough_that_one_lost_beat_is_not_a_death(self) -> None:
        """Three beats per window, so a dropped one does not read as a hung activity."""
        policy = _streaming()
        beating = Heartbeater(None, policy=policy, clock=FakeClock())

        assert beating.interval * 3 <= policy.heartbeat_timeout_seconds

    async def test_the_beat_carries_the_step_a_reader_needs_to_place_it(self) -> None:
        beats: list[Heartbeat] = []
        clock = FakeClock()
        beating = Heartbeater(beats.append, policy=_streaming(), clock=clock, step="step-3")

        clock.advance(30)
        await beating.chunk(tokens=4)

        assert beats[0].step == "step-3"

    async def test_a_call_with_nowhere_to_beat_still_runs(self) -> None:
        beating = Heartbeater(None, policy=_streaming(), clock=FakeClock())

        await beating.chunk(tokens=1)

        assert beating.last is None


class TestAWaitWithNoRoomLeft:
    async def test_a_retry_that_would_outlast_the_activity_is_not_waited_for(self) -> None:
        """The failure is raised now rather than after a wait the activity cannot afford."""
        clock = FakeClock()

        async def slow(attempt: int) -> str:  # noqa: ARG001
            clock.advance(5.0)
            raise ProviderUnavailableError("503")

        driver = attempts(
            policy=ActivityPolicy(
                start_to_close_seconds=1.0,
                retry=RetryConfig(max_attempts=3, base_delay_seconds=1.0),
            ),
            clock=clock,
        )

        with pytest.raises(ProviderUnavailableError):
            await driver.run(slow)

        assert clock.slept == []


def _streaming() -> ActivityPolicy:
    """The model policy, whose heartbeat window is what the beats have to fit inside."""
    return DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL]
