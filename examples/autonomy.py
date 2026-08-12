"""How much an agent may do unattended, and where it has to stop and ask.

One grant is issued: booking changes up to 5000 INR a day. The interesting part is what
each attempt resolves to — inside the headroom, at the edge of it, in the wrong currency,
in a class nobody granted, and the one an agent may never have.

Run it with `python examples/autonomy.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.core import (
    RESERVED_ACTION_CLASS,
    ActionClass,
    ActionRegistry,
    ActionRequest,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    Ceiling,
    InMemoryGrants,
)
from tesserix_adk.runtime import AutonomyGate, InMemoryReports
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
        "grant_autonomy": ActionClass(name=RESERVED_ACTION_CLASS),
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


def ladder() -> AutonomyLadder:
    """The ladder over the one grant this example issues."""
    return AutonomyLadder(CLASSES, grants=InMemoryGrants([GRANTED]), clock=FakeClock(start=NOW))


async def one_grant_answers_four_attempts() -> None:
    """The same grant, four actions, and only one of them goes ahead."""
    held = ladder()
    attempts = (
        ("inside the headroom", "change_booking", {"amount": 900, "currency": "INR"}),
        ("over what is left", "change_booking", {"amount": 4200, "currency": "INR"}),
        ("another currency", "change_booking", {"amount": 10, "currency": "USD"}),
        ("a class nobody granted", "refund_payment", {"amount": 10, "currency": "INR"}),
    )
    for label, tool, arguments in attempts:
        decided = await held.decide(
            ActionRequest(tool=tool, tenant="acme", arguments=arguments, committed=Decimal("4100"))
        )
        print(f"  {label}: {decided.outcome.value} — {decided.reason}")  # noqa: T201


async def the_headroom_is_never_rounded_up_to_fit() -> None:
    """4200 committed leaves 800, and 900 does not fit in 800."""
    decided = await ladder().decide(
        ActionRequest(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            committed=Decimal("4200"),
        )
    )
    print(f"  {decided.outcome.value}, {decided.headroom} left, under {decided.grant_id}")  # noqa: T201


async def nobody_grants_themselves() -> None:
    """A tool that would issue autonomy is refused rather than put to a human."""
    decided = await ladder().decide(
        ActionRequest(tool="grant_autonomy", tenant="acme", arguments={})
    )
    print(f"  {decided.outcome.value} — {decided.reason}")  # noqa: T201


async def act_and_report_stops_when_nobody_reads_the_reports() -> None:
    """The first action goes ahead; the next one asks until the report is delivered."""
    reports = InMemoryReports()
    reporting = AutonomyGrant.model_validate(
        GRANTED.model_dump() | {"id": "g2", "level": AutonomyLevel.ACT_AND_REPORT}
    )
    gate = AutonomyGate(
        AutonomyLadder(CLASSES, grants=InMemoryGrants([reporting]), clock=FakeClock(start=NOW)),
        reports=reports,
    )
    for run in ("run_1", "run_2"):
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 10, "currency": "INR"},
            run_id=run,
        )
        print(f"  {run}: {decided.outcome.value} — {decided.reason}")  # noqa: T201
    await reports.delivered(tenant="acme", action_class="booking.change", run_id="run_1")
    decided = await gate.decide(
        tool="change_booking",
        tenant="acme",
        arguments={"amount": 10, "currency": "INR"},
        run_id="run_3",
    )
    print(f"  run_3, after the report: {decided.outcome.value}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    for scenario in (
        one_grant_answers_four_attempts,
        the_headroom_is_never_rounded_up_to_fit,
        nobody_grants_themselves,
        act_and_report_stops_when_nobody_reads_the_reports,
    ):
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
