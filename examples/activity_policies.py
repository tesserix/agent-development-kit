"""What a slow model call and a card charge are each allowed to do.

Run it with `uv run python examples/activity_policies.py`.
"""

from __future__ import annotations

import asyncio
import contextlib

from tesserix_adk.core import (
    DEFAULT_ACTIVITY_POLICIES,
    ActivityClass,
    ActivityPolicy,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    GuardrailViolationError,
    Heartbeat,
    Idempotency,
    IdempotencyPolicy,
    ProviderUnavailableError,
    RetryConfig,
    RunBudget,
    ScopedLimits,
    most_restrictive,
)
from tesserix_adk.runtime import ActivityAttempts, AttemptBudget, Heartbeater
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import tool
from tesserix_adk.workflows import activity_policy_for, attempts_for_tool


@tool(idempotency=IdempotencyPolicy(kind=Idempotency.EFFECTFUL))
def charge_card(card: str) -> str:  # noqa: ARG001
    """Charge a card.

    Args:
        card: The card to charge.
    """
    return "charged"


def the_numbers_follow_the_kind_of_work() -> None:
    """A reasoning call gets its quarter hour; a tool does not get nine minutes."""
    model = DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL]
    tooling = DEFAULT_ACTIVITY_POLICIES[ActivityClass.TOOL]

    beats = model.heartbeat_interval_seconds
    print(f"model: {model.start_to_close_seconds}s, beats every {beats}s")  # noqa: T201
    print(f"tool:  {tooling.start_to_close_seconds}s, {tooling.retry.max_attempts} attempt")  # noqa: T201


def an_answer_is_not_a_fault() -> None:
    """No consumer override turns a guardrail block into something worth retrying."""
    eager = ActivityPolicy(retry=RetryConfig(max_attempts=9))

    print(f"retry a guardrail block? {eager.retryable(GuardrailViolationError('blocked'))}")  # noqa: T201
    print(f"retry a 503? {eager.retryable(ProviderUnavailableError('503'))}")  # noqa: T201


def what_the_tool_itself_declared() -> None:
    """A card charge is tried once, unless a key makes the second call land on the first."""
    tuned = ActivityPolicy(retry=RetryConfig(max_attempts=3))

    print(f"unkeyed: {attempts_for_tool(charge_card, base=tuned)} attempt")  # noqa: T201
    print(f"keyed:   {attempts_for_tool(charge_card, keyed=True, base=tuned)} attempts")  # noqa: T201

    narrowed = activity_policy_for(ActivityClass.MODEL, timeout=600.0)
    print(f"a 600s call beats every {narrowed.heartbeat_interval_seconds:g}s")  # noqa: T201


async def progress_is_never_a_result() -> None:
    """Three beats per window, carrying counts and nothing a reader could render."""
    beats: list[Heartbeat] = []
    clock = FakeClock()
    beating = Heartbeater(
        beats.append,
        policy=DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL],
        clock=clock,
        step="step-3",
    )

    for _ in range(100):
        clock.advance(1.0)
        await beating.chunk(tokens=8)

    print(f"{len(beats)} beats over 100 chunks, last at {beats[-1].tokens} tokens")  # noqa: T201


async def a_blip_is_worth_another_try() -> None:
    """The wait is jittered inside the activity, never on the workflow path."""
    seen: list[int] = []

    async def flaky(attempt: int) -> str:
        seen.append(attempt)
        if attempt < 3:
            raise ProviderUnavailableError("503")
        return "answered"

    clock = FakeClock()
    driver = ActivityAttempts(
        ActivityPolicy(retry=RetryConfig(max_attempts=3, base_delay_seconds=1.0)),
        clock=clock,
        seed=7,
    )

    print(f"{await driver.run(flaky)} on attempt {seen[-1]}, waits {clock.slept}")  # noqa: T201


async def the_ceiling_is_never_widened() -> None:
    """A retry the allowance cannot cover does not happen at all."""
    limits = ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_model_calls=2))
    budget = RunBudget(most_restrictive(limits), FakeClock())

    async def always_down(attempt: int) -> str:  # noqa: ARG001
        raise ProviderUnavailableError("503")

    driver = ActivityAttempts(
        ActivityPolicy(
            activity_class=ActivityClass.MODEL,
            retry=RetryConfig(max_attempts=5, base_delay_seconds=0.1),
        ),
        clock=FakeClock(),
        seed=7,
        budget=budget,
    )

    try:
        await driver.run(always_down, charge=ActivityClass.MODEL)
    except BudgetExceededError as stopped:
        print(f"stopped after {stopped.details['attempts']} attempts")  # noqa: T201


async def one_pool_for_the_whole_fan_out() -> None:
    """Ten branches retrying three times each is thirty calls at a struggling provider."""
    pool = AttemptBudget(2)
    calls = 0

    async def down(attempt: int) -> str:  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise ProviderUnavailableError("503")

    async def branch() -> None:
        driver = ActivityAttempts(
            ActivityPolicy(retry=RetryConfig(max_attempts=3, base_delay_seconds=0.1)),
            clock=FakeClock(),
            seed=7,
            shared=pool,
        )
        with contextlib.suppress(ProviderUnavailableError):
            await driver.run(down)

    for _ in range(5):
        await branch()

    print(f"5 branches, {calls} calls, {pool.left} retries left in the pool")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    the_numbers_follow_the_kind_of_work()
    an_answer_is_not_a_fault()
    what_the_tool_itself_declared()
    await progress_is_never_a_result()
    await a_blip_is_worth_another_try()
    await the_ceiling_is_never_widened()
    await one_pool_for_the_whole_fan_out()


if __name__ == "__main__":
    asyncio.run(main())
