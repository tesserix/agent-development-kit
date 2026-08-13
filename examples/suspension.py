"""A run that stops for three days, and the one decision that carries it on.

An agent proposes a payment, nobody is at their desk, and the run stops holding nothing at
all. Sixty-two hours later somebody answers and the original run carries on from where it
was. The interesting parts are what a second presentation of the same token buys, and what
happens to a decision that arrives after the question closed.

Run it with `python examples/suspension.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.core import (
    Agent,
    ApprovalRecord,
    ApprovalToken,
    ApprovalTokenError,
    ModelCapabilities,
    PendingDecision,
    Run,
    RunEventKind,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import (
    AgentRunner,
    Checkpointer,
    DeferringGate,
    MemoryCheckpointStore,
    MemorySuspensionStore,
    ModelResponse,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

NOW = 1_000.0
HELD_FOR = 62 * 3_600.0
CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)

PAID: list[int] = []


@tool(requires_approval=True)
async def wire_funds(amount: int) -> str:
    """Send a payment.

    Args:
        amount: Minor units.
    """
    PAID.append(amount)
    return f"sent {amount}"


class Posting:
    """A transport that puts the question on a queue; the answer arrives elsewhere."""

    async def deliver(self, record: ApprovalRecord) -> None:
        """Put the question where the on-call team will see it."""
        print(f"  queued: {record.summary}")  # noqa: T201


class Desk:
    """Whoever is on rota, and the token they were handed."""

    def __init__(self) -> None:
        self.token = ""

    async def hand_to(self, token: ApprovalToken) -> None:
        """Take the token, as an approver's inbox would."""
        self.token = token.value


def calling(**arguments: object) -> ModelResponse:
    """A model that proposes the payment."""
    return ModelResponse(
        tool_calls=(ToolCall(id="c1", name="wire_funds", arguments=arguments),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def answering() -> ModelResponse:
    """A model that has nothing left to do."""
    return ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1))


class World:
    """One agent that moves money, one gate that defers, and a clock somebody else winds."""

    def __init__(self) -> None:
        self.clock = FakeClock(start=NOW, auto_advance=False)
        self.desk = Desk()
        self.suspensions = MemorySuspensionStore()
        self.gate = DeferringGate(
            Posting(), self.suspensions, hand_to=self.desk.hand_to, clock=self.clock
        )
        self.checkpoints = Checkpointer(MemoryCheckpointStore(), None, self.clock)
        self.registry = ToolRegistry((wire_funds,), clock=self.clock)
        self.agent: Agent[Any] = Agent(
            name="planner",
            instructions="Settle the invoice.",
            free_text=True,
            model="scripted-1",
            tools=("wire_funds",),
            approval_required_tools=("wire_funds",),
        )

    def runner(self, *responses: ModelResponse) -> AgentRunner:
        """A runner over this world, scripted with `responses`."""
        return AgentRunner(
            provider=ScriptedProvider(*responses, capabilities=CAPABLE),
            clock=self.clock,
            tools=self.registry.view(allow=("wire_funds",), agent="planner"),
            approvals=self.gate,
            checkpoints=self.checkpoints,
        )

    async def start(self) -> Run[Any]:
        """Drive the agent up to the point somebody has to decide."""
        return await self.runner(calling(amount=500), answering()).run(
            self.agent, "settle it", tenant="acme", run_id="run_1", user="ada"
        )

    async def decide(self, **fields: object) -> Run[Any]:
        """Answer the question, as whoever is answering it."""
        named: dict[str, object] = {
            "tenant": "acme",
            "token": self.desk.token,
            "decided_by": "ada",
            "user": "ada",
        }
        return await self.runner(answering()).resume_with_decision(
            self.agent, "run_1", **(named | fields)
        )


async def a_run_that_stops_holds_nothing() -> None:
    """No worker, no connection, no in-memory state — just a row and a token."""
    world = World()
    run = await world.start()
    print(f"  state: {run.state.value}, terminal: {run.state.is_terminal}")  # noqa: T201
    print(f"  tool ran: {PAID}")  # noqa: T201
    waiting = PendingDecision.of((await world.gate.pending(tenant="acme"))[0])
    print(f"  on the rota: {waiting.tool_name} for {waiting.agent_name}")  # noqa: T201


async def the_answer_arrives_sixty_two_hours_later() -> None:
    """The original run carries on from the iteration it stopped at."""
    world = World()
    await world.start()
    world.clock.set(NOW + HELD_FOR)
    run = await world.decide(granted=True)
    resumed = next(one for one in run.events if one.kind is RunEventKind.RUN_RESUMED)
    print(f"  {run.state.value}: {resumed.detail}")  # noqa: T201
    print(f"  tool ran: {PAID}")  # noqa: T201


async def the_same_token_twice_buys_nothing() -> None:
    """Single-use is the whole of the exactly-once guarantee, and refusals are recorded."""
    world = World()
    await world.start()
    await world.decide(granted=True)
    try:
        await world.decide(granted=True, decided_by="mallory")
    except ApprovalTokenError as refused:
        print(f"  refused: {refused}")  # noqa: T201
    attempt = world.suspensions.attempts[-1]
    print(f"  audited: {attempt.presented_by} — {attempt.reason}")  # noqa: T201
    print(f"  tool ran: {PAID}")  # noqa: T201


async def a_question_nobody_answered_closes_itself() -> None:
    """Past its expiry the token buys a denial, decided by nobody."""
    world = World()
    await world.start()
    world.clock.set(NOW + 4 * 86_400.0)
    run = await world.decide(granted=True)
    denied = next(one for one in run.events if one.kind is RunEventKind.APPROVAL_DENIED)
    print(f"  {denied.detail}")  # noqa: T201
    print(f"  tool ran: {PAID}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order, each against a fresh world."""
    for scenario in (
        a_run_that_stops_holds_nothing,
        the_answer_arrives_sixty_two_hours_later,
        the_same_token_twice_buys_nothing,
        a_question_nobody_answered_closes_itself,
    ):
        PAID.clear()
        print(f"\n{scenario.__name__.replace('_', ' ')}:")  # noqa: T201
        await scenario()


if __name__ == "__main__":
    asyncio.run(main())
