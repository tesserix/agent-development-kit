"""A conversation changing hands, carrying a declared payload rather than a transcript.

Five scenarios: a typed payload crossing to a specialist, a payload the target's contract
refuses, the allowlist the target ends up holding, a handoff to a human desk, and the
record a chain of handoffs leaves behind.
Run it with `python examples/handoff.py`.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from tesserix_adk.core import Agent, HandoffContractError, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    Delegation,
    DelegationScope,
    Handoff,
    HandoffContract,
    HandoffDesk,
    ModelResponse,
    Receiver,
)
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

HELD = frozenset({"read_account", "issue_credit"})


class Ticket(BaseModel):
    """What the billing agent accepts, and nothing else."""

    account: str = Field(min_length=1)
    complaint: str = Field(min_length=1)


class Escalation(BaseModel):
    """What the human desk accepts."""

    account: str = Field(min_length=1)
    urgency: int = Field(ge=1, le=5)


class ReviewDesk:
    """A queue standing in for the people who work it."""

    def __init__(self) -> None:
        self.taken: list[Handoff] = []

    async def receive(self, handoff: Handoff) -> None:
        """Take the conversation, and keep what came with it."""
        self.taken.append(handoff)


def _agent(name: str, *tools: str) -> Agent:
    """An agent that answers in prose, so the run needs nothing else declared."""
    fields: dict[str, object] = {
        "name": name,
        "instructions": f"You are {name}.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tools,
    }
    return Agent(**fields)  # type: ignore[arg-type]


def _desk(*answers: str, queue: ReviewDesk | None = None) -> HandoffDesk:
    """A triage desk that may hand to billing, to shipping, or to a person."""
    receivers = [
        Receiver(
            agent=_agent("billing", "read_account", "issue_credit", "close_account"),
            contract=HandoffContract(accepts=Ticket),
        ),
        Receiver(
            agent=_agent("shipping", "read_account"),
            contract=HandoffContract(accepts=Ticket),
        ),
    ]
    if queue is not None:
        receivers.append(
            Receiver(queue=queue, name="review_desk", contract=HandoffContract(accepts=Escalation))
        )
    runner = AgentRunner(
        provider=ScriptedProvider(
            *(
                ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))
                for text in answers
            )
        ),
        clock=FakeClock(),
        tools=FakeToolRegistry(
            dict.fromkeys(("read_account", "issue_credit", "close_account"), str)
        ),
    )
    return HandoffDesk(
        runner,
        receivers,
        agent=_agent("triage", *sorted(HELD)),
        delegation=Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="triage",
            user="ada",
            scope=DelegationScope(tools=HELD),
        ),
    )


def _ticket() -> Ticket:
    return Ticket(account="ac_9", complaint="charged twice in March")


async def what_crosses_with_the_conversation() -> None:
    """The declared payload, and not a word of the transcript that produced it."""
    desk = _desk("Credited, and the account is clear.")
    result = await desk.hand_off(
        "billing",
        reason="the customer disputes a charge",
        state=_ticket(),
        task="sort the double charge",
    )

    print("=== what crosses with the conversation ===")  # noqa: T201
    print(f"{result.handoff.from_agent} -> {result.handoff.to_agent}: {result.handoff.state}")  # noqa: T201


async def a_payload_the_target_does_not_accept() -> None:
    """Refused before the target is invoked, naming the fields it got wrong."""
    desk = _desk("Credited.")
    print("\n=== a payload the target does not accept ===")  # noqa: T201
    try:
        await desk.hand_off(
            "billing",
            reason="the customer disputes a charge",
            state={"account": "ac_9"},
            task="sort it",
        )
    except HandoffContractError as refused:
        print(f"{refused.reason} {refused.violations}: nothing crossed, {desk.handoffs}")  # noqa: T201


async def what_the_target_ends_up_holding() -> None:
    """`close_account` is billing's own and not triage's, so it does not cross."""
    desk = _desk("Credited.")
    result = await desk.hand_off(
        "billing", reason="the customer disputes a charge", state=_ticket(), task="sort it"
    )

    print("\n=== what the target ends up holding ===")  # noqa: T201
    print(f"declared three tools, ran with {result.handoff.scope}")  # noqa: T201
    print(f"tenant {result.handoff.tenant} and user {result.handoff.user} are unchanged")  # noqa: T201


async def handing_to_a_person() -> None:
    """A queue is a receiver like any other, held to the same contract."""
    queue = ReviewDesk()
    desk = _desk("Credited.", queue=queue)
    result = await desk.hand_off(
        "review_desk",
        reason="the customer asked for a person",
        state=Escalation(account="ac_9", urgency=4),
        task="review the credit",
    )

    print("\n=== handing to a person ===")  # noqa: T201
    print(f"queued: {result.queued}, waiting: {queue.taken[0].reason}")  # noqa: T201


async def the_chain_it_leaves_behind() -> None:
    """One conversation, one run, one trace — whoever has held it."""
    desk = _desk("Not a billing problem.", "It shipped on Tuesday.")
    await desk.hand_off(
        "billing", reason="the customer disputes a charge", state=_ticket(), task="sort it"
    )
    await desk.hand_off(
        "shipping", reason="it turned out to be a delivery", state=_ticket(), task="find it"
    )

    print("\n=== the chain it leaves behind ===")  # noqa: T201
    for one in desk.handoffs:
        print(f"{'/'.join(one.trace)} on {one.run_id}: {one.reason}")  # noqa: T201


def main() -> None:
    """Run every scenario in the order the docs describe them."""
    asyncio.run(what_crosses_with_the_conversation())
    asyncio.run(a_payload_the_target_does_not_accept())
    asyncio.run(what_the_target_ends_up_holding())
    asyncio.run(handing_to_a_person())
    asyncio.run(the_chain_it_leaves_behind())


if __name__ == "__main__":
    main()
