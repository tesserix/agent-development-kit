"""What an agent did unattended, and what it declined to do.

Five scenarios: an action taken alone on record, a ceiling that held on record with the same
weight, an attempt to widen authority refused, what the payload turns into before it is
stored, and the question the trail exists to answer asked over a period. Run it with
`python examples/audit.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.core import (
    ActionClass,
    ActionRegistry,
    AuditDecision,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    AutonomyOutcome,
    Ceiling,
    InMemoryGrants,
)
from tesserix_adk.core.autonomy import RESERVED_ACTION_CLASS
from tesserix_adk.runtime import AuditTrail, AutonomyGate, MemoryAuditSink
from tesserix_adk.testing import FakeClock

NOW = 1_000.0
DAY = 86_400.0

CLASSES = ActionRegistry(
    {
        "change_booking": ActionClass(
            name="booking.change", amount_field="amount", currency_field="currency"
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


def wired() -> tuple[AutonomyGate, MemoryAuditSink, FakeClock]:
    """A gate that writes down what it decides, over an in-process trail."""
    clock = FakeClock(start=NOW)
    sink = MemoryAuditSink()
    ladder = AutonomyLadder(CLASSES, grants=InMemoryGrants([GRANTED]), clock=clock)
    return AutonomyGate(ladder, audit=AuditTrail(sink, clock=clock)), sink, clock


async def an_action_taken_alone_is_on_record() -> None:
    """The run loop records what it clears; here the trail is written directly."""
    gate, sink, _ = wired()
    decided = await gate.decide(
        tool="change_booking",
        tenant="acme",
        arguments={"amount": 900, "currency": "INR"},
        run_id="run_1",
        user="ops@acme.example",
    )
    await gate.record(
        decided,
        AuditDecision.EXECUTED,
        run_id="run_1",
        tenant="acme",
        tool="change_booking",
        arguments={"amount": 900, "currency": "INR"},
        agent="concierge",
    )
    [written] = await sink.records(tenant="acme")
    print(f"  {written.decision.value} under {written.grant_id}, unattended={written.unattended}")  # noqa: T201
    print(f"  headroom {written.headroom_before} -> {written.headroom_after}")  # noqa: T201


async def a_ceiling_that_held_is_on_record() -> None:
    """The record that shows the ceiling held. Nothing else keeps it."""
    gate, sink, _ = wired()
    await gate.decide(
        tool="change_booking",
        tenant="acme",
        arguments={"amount": 9_000, "currency": "INR"},
        run_id="run_2",
    )
    [written] = await sink.records(tenant="acme")
    print(f"  {written.decision.value} — {written.reason}")  # noqa: T201
    print(f"  headroom unchanged at {written.headroom_before}")  # noqa: T201


async def an_attempt_to_widen_its_own_authority_is_refused() -> None:
    """Nobody grants themselves, and the attempt is written down rather than dropped."""
    gate, sink, _ = wired()
    await gate.decide(
        tool="grant_autonomy", tenant="acme", arguments={"level": "act"}, run_id="run_3"
    )
    [written] = await sink.records(tenant="acme", decision=AuditDecision.REFUSED)
    print(f"  {written.decision.value} — {written.reason}")  # noqa: T201


async def the_payload_is_never_stored() -> None:
    """Redaction first, then a digest. The same call digests the same either way."""
    gate, sink, _ = wired()
    await gate.decide(
        tool="change_booking",
        tenant="acme",
        arguments={"amount": 9_000, "currency": "INR", "card": "4111 1111 1111 1111"},
        run_id="run_4",
    )
    [written] = await sink.records(tenant="acme")
    print(f"  digest {written.arguments_digest[:16]}… and nothing else of the payload")  # noqa: T201


async def what_did_this_agent_do_last_week() -> None:
    """One tenant, one period, declines included."""
    gate, sink, clock = wired()
    for run, amount in (("run_5", 900), ("run_6", 9_000), ("run_7", 400)):
        await clock.sleep(60.0)
        arguments = {"amount": amount, "currency": "INR"}
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments=arguments,
            run_id=run,
            agent="concierge",
        )
        if decided.outcome is AutonomyOutcome.ACT:
            await gate.record(
                decided,
                AuditDecision.EXECUTED,
                run_id=run,
                tenant="acme",
                tool="change_booking",
                arguments=arguments,
                agent="concierge",
            )
    every = await sink.records(tenant="acme", since=NOW, until=NOW + 300)
    declined = [held for held in every if held.decision is not AuditDecision.EXECUTED]
    print(f"  {len(declined)} declined, of {len(every)} decisions in the period")  # noqa: T201
    print(f"  first decline at {declined[0].recorded_at} in {declined[0].run_id}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    for scenario in (
        an_action_taken_alone_is_on_record,
        a_ceiling_that_held_is_on_record,
        an_attempt_to_widen_its_own_authority_is_refused,
        the_payload_is_never_stored,
        what_did_this_agent_do_last_week,
    ):
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
