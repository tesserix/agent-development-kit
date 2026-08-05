"""The shipped fakes are the first consumers of the conformance suites.

If a suite cannot be inherited and run here, no third party can run it either.
"""

import pytest

from tesserix_adk.testing import (
    BudgetExceededError,
    BudgetPolicyConformance,
    ClockConformance,
    FakeBudgetPolicy,
    FakeClock,
    FakeMemoryStore,
    FakeTracer,
    MemoryStoreConformance,
    TracerConformance,
)


class TestFakeMemoryStore(MemoryStoreConformance):
    def make_store(self) -> FakeMemoryStore:
        return FakeMemoryStore()


class TestFakeClock(ClockConformance):
    def make_clock(self) -> FakeClock:
        return FakeClock()


class TestFakeBudgetPolicy(BudgetPolicyConformance):
    def make_policy(self) -> FakeBudgetPolicy:
        return FakeBudgetPolicy()


class TestFakeTracer(TracerConformance):
    def make_tracer(self) -> FakeTracer:
        return FakeTracer()


async def test_fake_clock_advances_without_real_sleeping() -> None:
    clock = FakeClock(start=100.0)
    await clock.sleep(30.0)
    assert clock.now() == 130.0
    assert clock.slept == [30.0]


async def test_fake_budget_refuses_a_reservation_past_the_limit() -> None:
    policy = FakeBudgetPolicy(limit=10)
    await policy.reserve(8)
    with pytest.raises(BudgetExceededError, match="exceed limit 10"):
        await policy.reserve(5)


async def test_recording_releases_the_outstanding_reservation() -> None:
    policy = FakeBudgetPolicy(limit=10)
    await policy.reserve(9)
    await policy.record(2)
    assert (policy.spent, policy.reserved) == (2, 0)
    await policy.reserve(8)


def test_fake_tracer_records_spans_and_events_in_order() -> None:
    tracer = FakeTracer()
    with tracer.span("outer", tenant="t1"):
        tracer.event("inner")
    assert tracer.names() == ["outer", "inner"]
    assert tracer.recorded[0].attributes == {"tenant": "t1"}


def test_fake_clock_can_be_advanced_without_recording_a_sleep() -> None:
    clock = FakeClock(start=5.0)
    clock.advance(2.5)
    assert (clock.now(), clock.slept) == (7.5, [])
