"""Shaping the calls before the vendor has to.

A key's limit is shared by everything that holds it, so twenty concurrent runs on one key
do not each get the whole allowance — they get a twentieth of it, discover that as 429s,
and retry into the same wall. The limiter is the one place that knows the allowance, so
the calls are spaced before they are sent rather than rejected after.

Time here is a `FakeClock`: a limiter tested against the wall clock is a test suite that
sleeps for a minute to assert a minute.
"""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import ConfigurationError
from tesserix_adk.runtime import RateLimiter
from tesserix_adk.testing import FakeClock

A_MINUTE = 60.0


def limiter(clock: FakeClock, **limits: int | float) -> RateLimiter:
    return RateLimiter(clock=clock, **limits)


class TestAnAllowanceNobodyHasSpent:
    async def test_a_limiter_with_no_limits_never_waits(self) -> None:
        clock = FakeClock()
        for _ in range(100):
            await limiter(clock).acquire(tokens=10_000)
        assert clock.slept == []

    async def test_the_first_calls_go_straight_out(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=3)
        for _ in range(3):
            await shaped.acquire()
        assert clock.slept == []


class TestSpendingItFaster:
    async def test_the_call_past_the_allowance_waits_for_the_refill(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=2)
        for _ in range(3):
            await shaped.acquire()
        assert clock.slept == [pytest.approx(A_MINUTE / 2)]

    async def test_tokens_are_shaped_as_well_as_calls(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, tokens_per_minute=1000)
        await shaped.acquire(tokens=600)
        await shaped.acquire(tokens=600)
        assert clock.slept == [pytest.approx(A_MINUTE * 0.2)]

    async def test_the_longer_of_the_two_waits_is_the_one_taken(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=60, tokens_per_minute=60)
        await shaped.acquire(tokens=60)
        await shaped.acquire(tokens=30)
        assert clock.slept == [pytest.approx(A_MINUTE / 2)]

    async def test_a_bucket_refills_while_nobody_is_calling(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=2)
        await shaped.acquire()
        await shaped.acquire()
        clock.advance(A_MINUTE)
        await shaped.acquire()
        assert clock.slept == []

    async def test_it_never_refills_past_the_allowance_it_started_with(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=2)
        clock.advance(A_MINUTE * 10)
        for _ in range(3):
            await shaped.acquire()
        assert clock.slept == [pytest.approx(A_MINUTE / 2)]


class TestOneKeyManyCallers:
    async def test_concurrent_callers_share_one_allowance_rather_than_each_having_it(
        self,
    ) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=2)
        await asyncio.gather(*(shaped.acquire() for _ in range(4)))
        assert clock.slept == [pytest.approx(A_MINUTE / 2), pytest.approx(A_MINUTE / 2)]

    async def test_a_caller_cancelled_while_waiting_did_not_spend_anything(self) -> None:
        clock = FakeClock(auto_advance=False)
        shaped = limiter(clock, requests_per_minute=1)
        await shaped.acquire()
        waiting = asyncio.create_task(shaped.acquire())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        clock.advance(A_MINUTE)
        await shaped.acquire()
        assert clock.slept == [pytest.approx(A_MINUTE)]


class TestAnAllowanceNoRequestCouldEverFitIn:
    async def test_a_request_larger_than_the_whole_bucket_is_refused_rather_than_waited_on(
        self,
    ) -> None:
        shaped = limiter(FakeClock(), tokens_per_minute=100)
        with pytest.raises(ConfigurationError, match="larger than"):
            await shaped.acquire(tokens=101)

    async def test_a_limit_that_is_not_a_limit_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="above zero"):
            RateLimiter(requests_per_minute=0)


class TestBurstingOnPurpose:
    async def test_a_narrower_burst_releases_less_of_the_minute_at_once(self) -> None:
        clock = FakeClock()
        shaped = limiter(clock, requests_per_minute=60, burst=0.1)
        for _ in range(6):
            await shaped.acquire()
        assert clock.slept == []
        await shaped.acquire()
        assert clock.slept == [pytest.approx(1.0)]
