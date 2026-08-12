"""Taking authority back, and what happens to the work already under way.

One grant is issued and then withdrawn. The interesting part is when the withdrawal lands:
on the very next action, on a run that was asleep waiting for a human, and on a process
whose view of the bus has gone stale.

Run it with `python examples/revocation.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.core import (
    ActionClass,
    ActionRegistry,
    ActionRequest,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    Ceiling,
    InMemoryGrants,
    Revocation,
)
from tesserix_adk.runtime import AutonomyGate, RevocationWatch
from tesserix_adk.testing import FakeClock

NOW = 1_000.0
DAY = 86_400.0

CLASSES = ActionRegistry(
    {
        "change_booking": ActionClass(
            name="booking.change", amount_field="amount", currency_field="currency"
        ),
        "refund_payment": ActionClass(
            name="payment.refund",
            irreversible=True,
            amount_field="amount",
            currency_field="currency",
        ),
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
    ceiling=Ceiling(amount=Decimal("5000"), currency="INR", window_seconds=DAY),
)

ASKING = ActionRequest(
    tool="change_booking", tenant="acme", arguments={"amount": 10, "currency": "INR"}
)


def taken_back(**fields: object) -> Revocation:
    """One withdrawal, by somebody, at a moment."""
    named: dict[str, object] = {"revoked_by": "ops@acme.example", "revoked_at": NOW + 1}
    return Revocation.model_validate(named | fields)


async def it_lands_on_the_very_next_action() -> None:
    """The grant is read per action, so there is no window of stale authority."""
    grants = InMemoryGrants([GRANTED])
    ladder = AutonomyLadder(CLASSES, grants=grants, clock=FakeClock(start=NOW))
    print(f"  before: {(await ladder.decide(ASKING)).outcome.value}")  # noqa: T201
    await grants.revoke(taken_back(grant_id="g1"))
    print(f"  after:  {(await ladder.decide(ASKING)).outcome.value}")  # noqa: T201


async def one_withdrawal_can_cover_a_whole_class() -> None:
    """Naming a tenant and a class withdraws every grant under it at once."""
    grants = InMemoryGrants(
        [GRANTED, AutonomyGrant.model_validate(GRANTED.model_dump() | {"id": "g2"})]
    )
    ladder = AutonomyLadder(CLASSES, grants=grants, clock=FakeClock(start=NOW))
    await grants.revoke(taken_back(tenant="acme", action_class="booking.change"))
    decided = await ladder.decide(ASKING)
    print(f"  {decided.outcome.value} — {decided.reason}")  # noqa: T201


async def a_revoked_grant_is_never_put_back() -> None:
    """Re-granting mints a new id; the withdrawn one stays withdrawn and stays readable."""
    grants = InMemoryGrants([GRANTED])
    ladder = AutonomyLadder(CLASSES, grants=grants, clock=FakeClock(start=NOW))
    await grants.revoke(taken_back(grant_id="g1"))
    await grants.issue(AutonomyGrant.model_validate(GRANTED.model_dump() | {"id": "g1-again"}))
    decided = await ladder.decide(ASKING)
    print(f"  {decided.outcome.value} under {decided.grant_id}")  # noqa: T201
    print(f"  still on record: {[held.id for held in grants.all_grants()]}")  # noqa: T201


async def a_view_nobody_confirmed_is_not_authority() -> None:
    """A process cut off from the bus stops acting unattended rather than assuming."""
    clock = FakeClock(start=NOW)
    watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
    gate = AutonomyGate(
        AutonomyLadder(CLASSES, grants=InMemoryGrants([GRANTED]), clock=clock), revocations=watch
    )
    await clock.sleep(31.0)
    decided = await gate.decide(
        tool="change_booking",
        tenant="acme",
        arguments={"amount": 10, "currency": "INR"},
        run_id="run_1",
    )
    print(f"  {decided.outcome.value} — {decided.reason}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    for scenario in (
        it_lands_on_the_very_next_action,
        one_withdrawal_can_cover_a_whole_class,
        a_revoked_grant_is_never_put_back,
        a_view_nobody_confirmed_is_not_authority,
    ):
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
