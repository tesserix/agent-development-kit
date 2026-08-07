"""A tenant ceiling that holds when the same agent runs on eight replicas.

Four scenarios: eight concurrent writers against one 10.00 USD hourly window; a replica
that dies holding a reservation and what reconciliation does about it; an unreachable
ledger refusing rather than permitting; and a tenant erased down to an aggregate.

Run it with `python examples/ledger.py`. The ledger is in-memory and the clock is fake, so
nothing here reaches the network and no server is needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.adapters import InMemoryLedger
from tesserix_adk.core import (
    BudgetExceededError,
    BudgetUnavailableError,
    LedgerKey,
    Window,
    WindowKind,
)
from tesserix_adk.testing import FakeClock

HOUR = Window(kind=WindowKind.ROLLING, seconds=3_600)
ACME = LedgerKey(tenant="acme", agent=None, window=HOUR)
CEILING = Decimal("10.00")


async def eight_replicas_share_one_ceiling() -> None:
    """Each replica reserves, spends, settles. The window is the only thing they share."""
    ledger = InMemoryLedger(clock=FakeClock())

    async def replica(calls: int) -> Decimal:
        spent = Decimal(0)
        for _ in range(calls):
            try:
                held = await ledger.reserve(ACME, Decimal("0.50"), ceiling=CEILING)
            except BudgetExceededError:
                break
            await ledger.settle(held, Decimal("0.50"))
            spent += Decimal("0.50")
        return spent

    spent = await asyncio.gather(*(replica(10) for _ in range(8)))
    window = await ledger.read_window(ACME)
    print(f"eight replicas asked for {Decimal('40.00')}, spent {sum(spent)}")  # noqa: T201
    print(f"window settled: {window.settled}, ceiling: {CEILING}")  # noqa: T201


async def a_replica_that_died_holding_an_allowance() -> None:
    """Its lease lapses, and the sweep settles what it admitted rather than guessing."""
    clock = FakeClock()
    ledger = InMemoryLedger(clock=clock)
    held = await ledger.reserve(ACME, Decimal("4.00"), ceiling=CEILING, lease_seconds=300)
    await ledger.record_progress(held, Decimal("1.20"))

    before = await ledger.read_window(ACME)
    clock.advance(301)
    closed = await ledger.reconcile()
    after = await ledger.read_window(ACME)
    print(f"\nheld {before.reserved} while the replica was alive")  # noqa: T201
    print(f"swept {closed} lapsed lease, settled {after.settled}, still held {after.reserved}")  # noqa: T201


async def a_ledger_that_cannot_be_reached() -> None:
    """Fail closed. Carrying on without it is how one outage becomes an unbounded bill."""
    ledger = InMemoryLedger(clock=FakeClock())
    ledger.break_with(BudgetUnavailableError("connection reset", tenant="acme"))
    try:
        await ledger.reserve(ACME, Decimal("0.50"), ceiling=CEILING)
    except BudgetUnavailableError as refusal:
        print(f"\nrefused rather than permitted: {refusal}")  # noqa: T201

    permissive = InMemoryLedger(clock=FakeClock(), degraded_allowed=True)
    permissive.break_with(BudgetUnavailableError("connection reset", tenant="acme"))
    waved_through = await permissive.reserve(ACME, Decimal("0.50"), ceiling=CEILING)
    print(f"degraded mode, configured in advance: degraded={waved_through.degraded}")  # noqa: T201


async def a_tenant_asking_to_be_forgotten() -> None:
    """What is left is a number, not a history."""
    ledger = InMemoryLedger(clock=FakeClock())
    await ledger.settle(
        await ledger.reserve(ACME, Decimal("3.00"), ceiling=CEILING), Decimal("3.00")
    )
    dropped = await ledger.forget("acme")
    remaining = await ledger.read_window(ACME)
    print(f"\ndropped an aggregate of {dropped.settled}; the window now holds {remaining.settled}")  # noqa: T201


async def main() -> None:
    """Run the four scenarios in order."""
    await eight_replicas_share_one_ceiling()
    await a_replica_that_died_holding_an_allowance()
    await a_ledger_that_cannot_be_reached()
    await a_tenant_asking_to_be_forgotten()


if __name__ == "__main__":
    asyncio.run(main())
