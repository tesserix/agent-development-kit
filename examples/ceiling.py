"""The three ways a ceiling leaks, and the one thing that closes all of them.

Two actions reading the same headroom, one action split into many, and a retry of a call
that may already have gone out. Each is answered by taking the headroom rather than
reading it.

Run it with `python examples/ceiling.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.core import (
    ActionClass,
    ActionRegistry,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    Ceiling,
    CeilingExceededError,
    InMemoryCeilingLedger,
    InMemoryGrants,
)
from tesserix_adk.runtime import AutonomyGate
from tesserix_adk.testing import FakeClock

NOW = 1_000.0
DAY = 86_400.0
LIMIT = Ceiling(amount=Decimal("10000"), currency="INR", window_seconds=DAY)

CLASSES = ActionRegistry(
    {
        "change_booking": ActionClass(
            name="booking.change", amount_field="amount", currency_field="currency"
        )
    }
)

GRANTED = AutonomyGrant(
    id="g1",
    tenant="acme",
    action_class="booking.change",
    level=AutonomyLevel.ACT_WITHIN_LIMITS,
    granted_by="ops@acme.example",
    issued_at=NOW,
    expires_at=NOW + DAY,
    ceiling=LIMIT,
)


def wired() -> tuple[AutonomyGate, InMemoryCeilingLedger]:
    """A gate that takes headroom from the same ledger the ladder reads."""
    clock = FakeClock(start=NOW)
    ledger = InMemoryCeilingLedger(clock=clock)
    ladder = AutonomyLadder(
        CLASSES, grants=InMemoryGrants([GRANTED]), commitments=ledger, clock=clock
    )
    return AutonomyGate(ladder, commitments=ledger), ledger


async def taking(ledger: InMemoryCeilingLedger, amount: str, key: str) -> str:
    """One reservation, as either the amount taken or the refusal."""
    try:
        await ledger.reserve(
            tenant="acme",
            action_class="booking.change",
            ceiling=LIMIT,
            amount=Decimal(amount),
            idempotency_key=key,
        )
    except CeilingExceededError as refused:
        return f"refused: {refused}"
    return f"held {amount}"


async def two_actions_cannot_both_fit_under_one_headroom() -> None:
    """Four concurrent 3000s under a 10000 ceiling: three fit, and the fourth does not."""
    _, ledger = wired()
    taken = await asyncio.gather(*(taking(ledger, "3000", f"call-{n}") for n in range(4)))
    print(f"  {sorted(taken)}")  # noqa: T201


async def one_action_split_into_many_meets_the_same_window() -> None:
    """Ten small ones are counted against the same tenant, class, currency and window."""
    _, ledger = wired()
    for n in range(4):
        print(f"  part {n}: {await taking(ledger, '3000', f'part-{n}')}")  # noqa: T201


async def a_retry_asks_about_the_same_action() -> None:
    """The reservation is keyed by the call, so a retry takes no second headroom."""
    gate, ledger = wired()
    for _ in range(3):
        await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 4000, "currency": "INR"},
            run_id="run_1",
            key="run_1:call-1",
        )
    spent = await ledger.committed(tenant="acme", action_class="booking.change", window_seconds=DAY)
    print(f"  three attempts at 4000 committed {spent}")  # noqa: T201


async def the_last_hundredth_is_not_rounded_away() -> None:
    """Decimal end to end: a ceiling a hundredth out is one nobody can reconcile."""
    _, ledger = wired()
    print(f"  {await taking(ledger, '9999.99', 'call-1')}")  # noqa: T201
    print(f"  {await taking(ledger, '0.02', 'call-2')}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    for scenario in (
        two_actions_cannot_both_fit_under_one_headroom,
        one_action_split_into_many_meets_the_same_window,
        a_retry_asks_about_the_same_action,
        the_last_hundredth_is_not_rounded_away,
    ):
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
