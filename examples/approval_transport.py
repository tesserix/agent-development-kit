"""Where an approval question goes, and what happens when nobody answers it.

The gate holds the call; the transport only decides where the question is delivered. Silence
is a denial, a repeated answer settles nothing, and an agent cannot approve itself.

Run it with `python examples/approval_transport.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.adapters import ConsoleApprovals, NatsApprovals
from tesserix_adk.core import ApprovalDecision, ApprovalRecord
from tesserix_adk.runtime import TransportGate, self_granted
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Callable

NOW = 1_000.0


def quiet(line: str) -> None:
    """Swallow what the console transport would print."""


async def parked(ready: Callable[[], bool]) -> None:
    """Let the gate hold the call before anybody answers it."""
    for _ in range(100):
        if ready():
            return
        await asyncio.sleep(0)


def held(tool_name: str = "wire_funds") -> ApprovalRecord:
    """One call waiting on a human, as it goes onto the wire."""
    return ApprovalRecord.for_call(
        run_id="run_1",
        tenant="acme",
        agent_name="planner",
        tool_name=tool_name,
        arguments={"amount": 500, "iban": "GB33BUKB20201555555555"},
        reason=f"{tool_name} is declared to require approval",
        requested_at=NOW,
    )


class Desk:
    """A transport that reaches an approver who answers in the reply."""

    def __init__(self, *, by: str) -> None:
        self.by = by

    async def deliver(self, record: ApprovalRecord) -> ApprovalDecision:
        """Ask, and come back with what the desk said."""
        return ApprovalDecision(
            record_id=record.id, granted=True, decided_by=self.by, decided_at=NOW
        )


class Queue:
    """A transport that only posts the question; the answer arrives out of band."""

    async def deliver(self, record: ApprovalRecord) -> None:
        """Put it on the queue and return."""


async def an_answer_that_comes_back_on_the_wire() -> None:
    """A transport carrying the decision itself needs no second hop."""
    decision = await TransportGate(Desk(by="ada")).request(held())
    print(f"  granted={decision.granted} by {decision.decided_by}")  # noqa: T201


async def an_answer_that_arrives_out_of_band() -> None:
    """The run waits until somebody calls `decide`, and a second answer settles nothing."""
    gate = TransportGate(Queue())
    call = held()
    asked = asyncio.create_task(gate.request(call))
    await parked(lambda: bool(gate.waiting))

    accepted = gate.decide(
        ApprovalDecision(record_id=call.id, granted=True, decided_by="ada", decided_at=NOW)
    )
    await asked
    replayed = gate.decide(
        ApprovalDecision(record_id=call.id, granted=True, decided_by="mallory", decided_at=NOW)
    )
    print(f"  first answer accepted={accepted}, replay accepted={replayed}")  # noqa: T201


async def nobody_answering_is_a_refusal() -> None:
    """A gate that opens when the approver is asleep is not a gate."""
    clock = FakeClock(start=NOW, auto_advance=False)
    gate = TransportGate(Queue(), clock=clock, wait_seconds=900.0)
    asked = asyncio.create_task(gate.request(held()))
    await parked(lambda: bool(clock.slept))

    clock.advance(900.0)
    decision = await asked
    print(f"  granted={decision.granted} by {decision.decided_by}: {decision.reason}")  # noqa: T201


async def an_agent_cannot_approve_itself() -> None:
    """A grant from the agent's own service identity is not a second pair of eyes."""
    call = held()
    for who in ("ada", "agent:planner"):
        answer = ApprovalDecision(record_id=call.id, granted=True, decided_by=who, decided_at=NOW)
        print(f"  {who}: self_granted={self_granted(call, answer)}")  # noqa: T201


async def asking_whoever_is_at_the_terminal() -> None:
    """The console transport, answered here by a script rather than by a person."""
    console = ConsoleApprovals(approver="ada", ask=lambda: "y", show=quiet)
    decision = await console.deliver(held())
    print(f"  granted={decision.granted} by {decision.decided_by}")  # noqa: T201


async def publishing_the_question_on_a_tenants_subject() -> None:
    """NATS, with a client stubbed out so the example needs no server."""
    published: list[str] = []

    class Publisher:
        async def publish(self, subject: str, payload: bytes) -> None:
            published.append(f"{subject} ({len(payload)} bytes)")

    await NatsApprovals(Publisher()).deliver(held())
    print(f"  published on {published[0]}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    for scenario in (
        an_answer_that_comes_back_on_the_wire,
        an_answer_that_arrives_out_of_band,
        nobody_answering_is_a_refusal,
        an_agent_cannot_approve_itself,
        asking_whoever_is_at_the_terminal,
        publishing_the_question_on_a_tenants_subject,
    ):
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
