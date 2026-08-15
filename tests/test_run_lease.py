"""One worker advances a run. The other finds out it is not the one."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    DEFAULT_LEASE,
    LeasePolicy,
    LeaseStore,
    RunLease,
    RunLeaseError,
)
from tesserix_adk.runtime import Leaseholder, MemoryLeaseStore
from tesserix_adk.testing import FakeClock, LeaseStoreConformance

pytestmark = pytest.mark.anyio


def store(start: float = 0.0) -> tuple[MemoryLeaseStore, FakeClock]:
    """A lease store and the clock it decides expiry on."""
    clock = FakeClock(start)
    return MemoryLeaseStore(clock), clock


class TestOneHolderAtATime:
    """The whole point: two workers, one run, one of them dispatching."""

    async def test_the_first_worker_takes_the_run(self) -> None:
        leases, _ = store()

        lease = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        assert lease.holder == "w1"
        assert lease.fence == 1
        assert lease.expires_at == 60.0

    async def test_the_second_is_refused_and_told_who_has_it(self) -> None:
        leases, _ = store()
        await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        with pytest.raises(RunLeaseError) as refused:
            await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

        assert refused.value.holder == "w1"
        assert refused.value.requested_by == "w2"
        assert refused.value.run_id == "r1"
        assert refused.value.retryable is True

    async def test_the_same_worker_may_take_what_it_already_holds(self) -> None:
        leases, _ = store()
        await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        again = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        assert again.fence == 2

    async def test_a_lease_never_spans_a_tenant(self) -> None:
        leases, _ = store()
        await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        other = await leases.acquire("r1", tenant="globex", holder="w2", ttl_seconds=60.0)

        assert other.holder == "w2"

    async def test_two_runs_do_not_block_each_other(self) -> None:
        leases, _ = store()
        await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        assert (await leases.acquire("r2", tenant="acme", holder="w2", ttl_seconds=60.0)).fence == 1

    async def test_the_store_satisfies_the_protocol(self) -> None:
        leases, _ = store()

        assert isinstance(leases, LeaseStore)


class TestWhenAWorkerDies:
    """A hold nobody released must not strand the run forever."""

    async def test_an_expired_lease_is_takeable(self) -> None:
        leases, clock = store()
        await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        clock.advance(61.0)
        taken = await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

        assert taken.holder == "w2"

    async def test_taking_it_fences_the_previous_holder_out(self) -> None:
        leases, clock = store()
        first = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)
        clock.advance(61.0)
        second = await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

        assert first.superseded_by(second)
        with pytest.raises(RunLeaseError):
            await leases.renew(first, ttl_seconds=60.0)

    async def test_expiry_is_decided_on_the_stores_clock_not_the_callers(self) -> None:
        leases, clock = store()
        held = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        clock.advance(30.0)

        # A worker whose own clock reads far ahead would call this expired and steal it.
        assert held.held_at(clock.now()) is True
        with pytest.raises(RunLeaseError):
            await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

    async def test_releasing_frees_the_run_immediately(self) -> None:
        leases, _ = store()
        held = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        await leases.release(held)

        assert await leases.held("r1", tenant="acme") is None
        assert (await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)).fence == 1

    async def test_releasing_a_lease_that_has_moved_on_leaves_the_new_one_alone(self) -> None:
        leases, clock = store()
        first = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)
        clock.advance(61.0)
        second = await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

        await leases.release(first)

        assert await leases.held("r1", tenant="acme") == second

    async def test_a_run_nobody_has_held_has_no_lease(self) -> None:
        leases, _ = store()

        assert await leases.held("r1", tenant="acme") is None


class TestRenewal:
    """A turn that runs longer than a TTL must not lose the run halfway through."""

    async def test_renewing_moves_the_expiry_and_keeps_the_fence(self) -> None:
        leases, clock = store()
        held = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        clock.advance(30.0)
        renewed = await leases.renew(held, ttl_seconds=60.0)

        assert renewed.expires_at == 90.0
        assert renewed.fence == held.fence

    async def test_renewing_a_lease_nobody_holds_is_refused(self) -> None:
        leases, _ = store()
        lease = RunLease(run_id="r1", tenant="acme", holder="w1", expires_at=60.0)

        with pytest.raises(RunLeaseError) as refused:
            await leases.renew(lease, ttl_seconds=60.0)

        assert refused.value.holder == ""

    async def test_renewing_a_lease_another_worker_now_holds_is_refused(self) -> None:
        leases, clock = store()
        first = await leases.acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)
        clock.advance(61.0)
        await leases.acquire("r1", tenant="acme", holder="w2", ttl_seconds=60.0)

        with pytest.raises(RunLeaseError) as refused:
            await leases.renew(first, ttl_seconds=60.0)

        assert refused.value.holder == "w2"

    def test_the_policy_renews_before_expiry_not_at_it(self) -> None:
        lease = RunLease(run_id="r1", tenant="acme", holder="w1", expires_at=100.0)

        assert DEFAULT_LEASE.due_to_renew(lease, now=79.0) is False
        assert DEFAULT_LEASE.due_to_renew(lease, now=81.0) is True


class TestHoldingARunAroundATurn:
    """What a worker actually writes."""

    async def test_the_hold_is_released_however_the_turn_ended(self) -> None:
        leases, _ = store()

        async def failing_turn() -> None:
            async with Leaseholder(leases, holder="w1") as holder:
                await holder.acquire("r1", tenant="acme")
                raise RuntimeError("turn failed")

        with pytest.raises(RuntimeError, match="turn failed"):
            await failing_turn()

        assert await leases.held("r1", tenant="acme") is None

    async def test_the_fence_is_what_a_write_carries(self) -> None:
        leases, _ = store()
        holder = Leaseholder(leases, holder="w1")

        assert holder.fence == 0
        await holder.acquire("r1", tenant="acme")
        assert holder.fence == 1
        assert holder.lease is not None

    async def test_a_holder_renews_only_when_the_policy_says_to(self) -> None:
        leases, _ = store()
        holder = Leaseholder(leases, holder="w1", policy=LeasePolicy(ttl_seconds=60.0))
        await holder.acquire("r1", tenant="acme")

        unchanged = await holder.keep(now=10.0)
        renewed = await holder.keep(now=50.0)

        assert unchanged is not None
        assert unchanged.expires_at == 60.0
        assert renewed is not None
        assert renewed.expires_at == 60.0

    async def test_keeping_nothing_held_is_not_an_error(self) -> None:
        leases, _ = store()

        assert await Leaseholder(leases, holder="w1").keep(now=0.0) is None

    async def test_renewing_nothing_held_says_so(self) -> None:
        leases, _ = store()

        with pytest.raises(RunLeaseError, match="nothing is held"):
            await Leaseholder(leases, holder="w1").renew()

    async def test_a_second_release_is_not_an_error(self) -> None:
        leases, _ = store()
        holder = Leaseholder(leases, holder="w1")
        await holder.acquire("r1", tenant="acme")

        await holder.release()
        await holder.release()

        assert holder.lease is None


class Timed(MemoryLeaseStore):
    """A store that keeps the clock it decides expiry on, so the suite can move it."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        super().__init__(self.clock)


class TestTheMemoryStoreConforms(LeaseStoreConformance):
    def make_store(self) -> LeaseStore:
        return Timed()

    async def advance(self, store: LeaseStore, seconds: float) -> None:
        assert isinstance(store, Timed)
        store.clock.advance(seconds)
