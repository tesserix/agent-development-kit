"""A tenant ceiling that holds across replicas."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from tesserix_adk.adapters import CoalescingLedger, InMemoryLedger
from tesserix_adk.core.errors import BudgetExceededError, BudgetUnavailableError
from tesserix_adk.core.ledger import LedgerKey, Window, WindowKind
from tesserix_adk.testing import FakeClock, SpendLedgerConformance

HOUR = Window(kind=WindowKind.ROLLING, seconds=3_600)
CALENDAR_HOUR = Window(kind=WindowKind.CALENDAR, seconds=3_600)
CEILING = Decimal("10.00")


def key(tenant: str = "acme", agent: str | None = None) -> LedgerKey:
    return LedgerKey(tenant=tenant, agent=agent, window=HOUR)


def a_ledger(clock: FakeClock | None = None, **kwargs: object) -> InMemoryLedger:
    return InMemoryLedger(clock=clock or FakeClock(), **kwargs)  # type: ignore[arg-type]


class TestOneCeilingAcrossReplicas:
    async def test_spend_settled_by_one_holder_is_visible_to_another(self) -> None:
        """Replicas share a ledger or they share nothing, and the ceiling means nothing."""
        ledger = a_ledger()
        first = await ledger.reserve(key(), Decimal("4.00"), ceiling=CEILING)
        await ledger.settle(first, Decimal("4.00"))
        assert (await ledger.read_window(key())).settled == Decimal("4.00")

    async def test_the_ceiling_refuses_once_the_window_is_full(self) -> None:
        ledger = a_ledger()
        await ledger.settle(
            await ledger.reserve(key(), Decimal("9.50"), ceiling=CEILING), Decimal("9.50")
        )
        with pytest.raises(BudgetExceededError) as refused:
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        assert "10.00" in str(refused.value)

    async def test_a_reservation_counts_against_the_ceiling_before_it_settles(self) -> None:
        """Otherwise every replica reserves against the same empty window at once."""
        ledger = a_ledger()
        await ledger.reserve(key(), Decimal("9.50"), ceiling=CEILING)
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)

    async def test_concurrent_replicas_cannot_overshoot_the_ceiling(self) -> None:
        ledger = a_ledger()

        async def replica() -> bool:
            try:
                held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
            except BudgetExceededError:
                return False
            await ledger.settle(held, Decimal("1.00"))
            return True

        granted = await asyncio.gather(*(replica() for _ in range(40)))
        assert sum(granted) == 10
        assert (await ledger.read_window(key())).settled == CEILING

    async def test_settling_less_than_reserved_returns_the_difference(self) -> None:
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("6.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("1.00"))
        window = await ledger.read_window(key())
        assert (window.settled, window.reserved) == (Decimal("1.00"), Decimal(0))

    async def test_settling_more_than_reserved_records_what_was_actually_spent(self) -> None:
        """The vendor invoices for what it read, whatever the estimate said."""
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("3.00"))
        assert (await ledger.read_window(key())).settled == Decimal("3.00")

    async def test_a_released_reservation_frees_the_allowance(self) -> None:
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING)
        await ledger.release(held)
        assert (await ledger.read_window(key())).reserved == Decimal(0)

    async def test_settling_the_same_reservation_twice_is_refused(self) -> None:
        """A retried settlement that double-counts is a ceiling that quietly halves."""
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("1.00"))
        with pytest.raises(BudgetUnavailableError):
            await ledger.settle(held, Decimal("1.00"))

    async def test_releasing_a_settled_reservation_is_refused(self) -> None:
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("1.00"))
        with pytest.raises(BudgetUnavailableError):
            await ledger.release(held)


class TestTenantsCannotReachEachOther:
    async def test_one_tenant_s_spend_is_not_another_s(self) -> None:
        ledger = a_ledger()
        await ledger.settle(
            await ledger.reserve(key("acme"), Decimal("9.00"), ceiling=CEILING), Decimal("9.00")
        )
        assert (await ledger.read_window(key("globex"))).settled == Decimal(0)

    async def test_an_agent_scoped_window_is_not_the_tenant_window(self) -> None:
        ledger = a_ledger()
        await ledger.settle(
            await ledger.reserve(key("acme", "researcher"), Decimal("2.00"), ceiling=CEILING),
            Decimal("2.00"),
        )
        assert (await ledger.read_window(key("acme"))).settled == Decimal(0)

    def test_a_key_carries_identifiers_and_nothing_else(self) -> None:
        """Ledger contents are read by operators who were never cleared to read prompts."""
        assert key("acme", "researcher").name == "acme:researcher:rolling:3600"

    def test_a_tenant_cannot_forge_a_key_into_another_tenant_s_window(self) -> None:
        with pytest.raises(ValueError, match="separator"):
            LedgerKey(tenant="acme:globex", agent=None, window=HOUR)


class TestWindows:
    async def test_a_rolling_window_forgets_spend_older_than_its_length(self) -> None:
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.settle(
            await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING), Decimal("9.00")
        )
        clock.advance(3_601)
        assert (await ledger.read_window(key())).settled == Decimal(0)

    async def test_a_calendar_window_resets_on_its_boundary_not_on_first_spend(self) -> None:
        clock = FakeClock(start=3_500)
        ledger = a_ledger(clock)
        calendar = LedgerKey(tenant="acme", agent=None, window=CALENDAR_HOUR)
        await ledger.settle(
            await ledger.reserve(calendar, Decimal("9.00"), ceiling=CEILING), Decimal("9.00")
        )
        clock.advance(200)
        assert (await ledger.read_window(calendar)).settled == Decimal(0)

    async def test_a_run_crossing_a_boundary_is_not_granted_a_fresh_allowance(self) -> None:
        """The reservation was taken from the old window and is held until it settles."""
        clock = FakeClock()
        ledger = a_ledger(clock)
        held = await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=7_200)
        clock.advance(3_601)
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key(), Decimal("2.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("9.00"))

    async def test_a_clock_that_goes_backwards_does_not_open_a_second_window(self) -> None:
        clock = FakeClock(start=10_000)
        ledger = a_ledger(clock)
        await ledger.settle(
            await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING), Decimal("9.00")
        )
        clock.set(4_000)
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key(), Decimal("2.00"), ceiling=CEILING)

    async def test_a_window_says_when_the_allowance_next_returns(self) -> None:
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.settle(
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
        )
        assert (await ledger.read_window(key())).resets_at == 3_600


class TestOrphanedReservations:
    async def test_a_lease_that_expires_stops_holding_the_allowance(self) -> None:
        """A replica killed mid-run must not reduce the tenant's window for an hour."""
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(61)
        assert await ledger.reconcile() == 1
        assert (await ledger.read_window(key())).reserved == Decimal(0)

    async def test_an_expired_lease_is_not_credited_with_spend_that_happened(self) -> None:
        """Released, not settled: nothing recorded what the dead replica actually spent."""
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(61)
        await ledger.reconcile()
        assert (await ledger.read_window(key())).settled == Decimal(0)

    async def test_reconciliation_settles_an_expired_lease_against_recorded_usage(self) -> None:
        clock = FakeClock()
        ledger = a_ledger(clock)
        held = await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=60)
        await ledger.record_progress(held, Decimal("2.00"))
        clock.advance(61)
        await ledger.reconcile()
        window = await ledger.read_window(key())
        assert (window.settled, window.reserved) == (Decimal("2.00"), Decimal(0))

    async def test_an_expired_reservation_can_no_longer_be_settled(self) -> None:
        clock = FakeClock()
        ledger = a_ledger(clock)
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(61)
        await ledger.reconcile()
        with pytest.raises(BudgetUnavailableError, match="expired"):
            await ledger.settle(held, Decimal("1.00"))

    async def test_a_live_lease_is_left_alone(self) -> None:
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(30)
        assert await ledger.reconcile() == 0

    async def test_an_expired_lease_is_reclaimed_by_the_next_reservation(self) -> None:
        """Reconciliation is a sweeper, not the thing the guarantee rests on."""
        clock = FakeClock()
        ledger = a_ledger(clock)
        await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(61)
        await ledger.reserve(key(), Decimal("9.00"), ceiling=CEILING)


class TestFailingClosed:
    async def test_an_unreachable_ledger_refuses_the_call(self) -> None:
        ledger = a_ledger()
        ledger.break_with(BudgetUnavailableError("redis is unreachable"))
        with pytest.raises(BudgetUnavailableError):
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)

    async def test_degraded_mode_is_configured_explicitly_and_never_inferred(self) -> None:
        ledger = a_ledger(degraded_allowed=True)
        ledger.break_with(BudgetUnavailableError("redis is unreachable"))
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        assert held.degraded is True

    async def test_a_degraded_reservation_is_recorded_as_one(self) -> None:
        ledger = a_ledger(degraded_allowed=True)
        ledger.break_with(BudgetUnavailableError("redis is unreachable"))
        await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        assert ledger.degradations == 1


class TestErasure:
    async def test_a_tenant_s_records_reduce_to_an_aggregate(self) -> None:
        ledger = a_ledger()
        await ledger.settle(
            await ledger.reserve(key("acme"), Decimal("3.00"), ceiling=CEILING), Decimal("3.00")
        )
        forgotten = await ledger.forget("acme")
        assert forgotten.settled == Decimal("3.00")
        assert (await ledger.read_window(key("acme"))).settled == Decimal(0)

    async def test_forgetting_one_tenant_leaves_the_others_intact(self) -> None:
        ledger = a_ledger()
        await ledger.settle(
            await ledger.reserve(key("globex"), Decimal("2.00"), ceiling=CEILING), Decimal("2.00")
        )
        await ledger.forget("acme")
        assert (await ledger.read_window(key("globex"))).settled == Decimal("2.00")

    async def test_forgetting_drops_the_tenant_s_open_reservations_too(self) -> None:
        ledger = a_ledger()
        await ledger.reserve(key("acme"), Decimal("2.00"), ceiling=CEILING)
        await ledger.forget("acme")
        assert (await ledger.read_window(key("acme"))).reserved == Decimal(0)


class TestSharding:
    async def test_a_sharded_window_still_sums_to_what_was_spent(self) -> None:
        """A busy tenant's writes spread over shards; the ceiling reads all of them."""
        ledger = a_ledger(shards=8)
        for _ in range(8):
            await ledger.settle(
                await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
            )
        assert (await ledger.read_window(key())).settled == Decimal("8.00")

    async def test_a_sharded_ledger_refuses_at_the_same_ceiling(self) -> None:
        ledger = a_ledger(shards=8)
        for _ in range(10):
            await ledger.settle(
                await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
            )
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)


class TestCoalescing:
    async def test_small_reservations_are_drawn_from_one_round_trip(self) -> None:
        """A ledger call per model call is latency on every call."""
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        for _ in range(5):
            await ledger.settle(
                await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
            )
        assert inner.reservations == 1

    async def test_a_block_that_runs_out_takes_another(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        for _ in range(6):
            await ledger.settle(
                await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
            )
        assert inner.reservations == 2

    async def test_coalescing_does_not_lift_the_ceiling(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))

        async def eleven_calls() -> None:
            for _ in range(11):
                await ledger.settle(
                    await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
                )

        with pytest.raises(BudgetExceededError):
            await eleven_calls()

    async def test_what_a_block_did_not_spend_goes_back(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        await ledger.settle(
            await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING), Decimal("1.00")
        )
        await ledger.flush()
        assert (await inner.read_window(key())).reserved == Decimal(0)
        assert (await inner.read_window(key())).settled == Decimal("1.00")

    async def test_a_request_larger_than_the_block_goes_straight_through(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        held = await ledger.reserve(key(), Decimal("7.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("7.00"))
        assert (await inner.read_window(key())).settled == Decimal("7.00")

    async def test_released_spend_returns_to_the_block(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        held = await ledger.reserve(key(), Decimal("2.00"), ceiling=CEILING)
        await ledger.release(held)
        assert (await ledger.reserve(key(), Decimal("5.00"), ceiling=CEILING)) is not None
        assert inner.reservations == 1

    async def test_reading_a_window_through_a_block_counts_what_is_held(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        assert (await ledger.read_window(key())).reserved == Decimal("5.00")

    async def test_reconciliation_passes_through_to_the_ledger_underneath(self) -> None:
        clock = FakeClock()
        inner = a_ledger(clock)
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING, lease_seconds=60)
        clock.advance(61)
        assert await ledger.reconcile() == 1

    async def test_forgetting_a_tenant_drops_its_held_block(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        await ledger.reserve(key("acme"), Decimal("1.00"), ceiling=CEILING)
        await ledger.forget("acme")
        assert (await inner.read_window(key("acme"))).reserved == Decimal(0)

    async def test_a_reservation_that_bypassed_the_block_is_released_through_it(self) -> None:
        """One larger than a block was taken from the ledger, so it goes back there."""
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        held = await ledger.reserve(key(), Decimal("8.00"), ceiling=CEILING)
        await ledger.release(held)
        assert (await inner.read_window(key())).reserved == Decimal(0)

    async def test_a_draw_s_progress_is_recorded_against_the_block_that_holds_it(self) -> None:
        """The block is the hold the store knows about, so it is what a sweep can settle."""
        clock = FakeClock()
        inner = a_ledger(clock)
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        first = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(first, Decimal("1.00"))
        second = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.record_progress(second, Decimal("0.25"))
        clock.advance(301)
        assert await inner.reconcile() == 1
        assert (await inner.read_window(key())).settled == Decimal("1.25")

    async def test_progress_on_a_hold_that_bypassed_the_block_goes_underneath(self) -> None:
        inner = a_ledger()
        ledger = CoalescingLedger(inner, block=Decimal("5.00"))
        held = await ledger.reserve(key(), Decimal("8.00"), ceiling=CEILING)
        await ledger.record_progress(held, Decimal("2.00"))
        await ledger.settle(held, Decimal("2.00"))
        assert (await inner.read_window(key())).settled == Decimal("2.00")


class TestProgressAgainstALapsedLease:
    async def test_progress_on_a_hold_the_ledger_no_longer_has_is_refused(self) -> None:
        """Silently accepting it would credit a window that was already closed out."""
        ledger = a_ledger()
        held = await ledger.reserve(key(), Decimal("1.00"), ceiling=CEILING)
        await ledger.release(held)
        with pytest.raises(BudgetUnavailableError, match="not open"):
            await ledger.record_progress(held, Decimal("0.40"))


class TestTheInMemoryLedgerConforms(SpendLedgerConformance):
    """The reference implementation is held to the same suite as the shared stores."""

    def make_ledger(self) -> InMemoryLedger:
        return a_ledger()
