"""Calling a peer: typed both ways, scope narrowed, spend charged, answer offered as a tool.

Run it with `uv run python examples/peer_invocation.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.a2a import (
    AgentCard,
    AgentLimits,
    AgentSkill,
    PeerCall,
    PeerClient,
    PeerInvocationError,
    PeerReply,
)
from tesserix_adk.adapters import peer_tool
from tesserix_adk.core import (
    AgentIdentity,
    BudgetLimits,
    BudgetScope,
    CountSource,
    Principal,
    RunBudget,
    ScopedLimits,
    ToolArgumentValidationError,
    Usage,
    most_restrictive,
)
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CredentialBroker, CredentialRequest, ExchangedCredentials

READ = "itinerary:read"
WRITE = "payments:write"


def card() -> AgentCard:
    """What the peer publishes about itself, and everything a call is held to."""
    return AgentCard(
        agent="booker",
        audience="https://booker.example.gov",
        declared=(READ, WRITE),
        limits=AgentLimits(max_payload_bytes=4096),
        skills=(
            AgentSkill(
                name="price_leg",
                description="Price one leg.",
                input_schema={
                    "type": "object",
                    "properties": {"leg": {"type": "string"}},
                    "required": ["leg"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"eur": {"type": "number"}},
                    "required": ["eur"],
                },
                idempotent=True,
            ),
            AgentSkill(
                name="refund",
                description="Refund an order.",
                input_schema={"type": "object", "properties": {"order": {"type": "string"}}},
                required_scopes=(WRITE,),
            ),
        ),
    )


class Booker:
    """The other agent, as a transport that answers from a table."""

    def __init__(self) -> None:
        self.calls: list[PeerCall] = []

    async def invoke(self, call: PeerCall) -> PeerReply:
        """Answer, recording what actually travelled."""
        self.calls.append(call)
        return PeerReply(
            output={"eur": 412.0},
            usage=Usage(input_tokens=180, output_tokens=40, source=CountSource.PROVIDER),
        )

    async def cancel(self, call: PeerCall) -> None:
        """Stop work the caller no longer waits for."""
        del call


class Exchange:
    """A token endpoint, which in a deployment is the org's own."""

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Mint a token for the peer's audience alone."""
        return f"tok-{request.audience}", 300.0


def client(peer: Booker, held: tuple[str, ...], budget: RunBudget) -> PeerClient:
    """A client for one peer, acting for one person, against one ceiling."""
    clock = FakeClock()
    return PeerClient(
        card(),
        peer,
        credentials=CredentialBroker(ExchangedCredentials(Exchange(), clock=clock), clock=clock),
        identity=AgentIdentity.resolve(
            agent="desk",
            declared=(READ, WRITE),
            principal=Principal(subject="ada", tenant="acme", scopes=frozenset(held)),
        ),
        run_id="run_1",
        clock=clock,
        budget=budget,
    )


async def main() -> None:
    """Call a peer, watch the scope narrow, the budget move, and the refusals land."""
    peer = Booker()
    budget = RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_input_tokens=5000))
        ),
        clock=FakeClock(),
    )
    calling = client(peer, (READ,), budget)

    result = await calling.invoke("price_leg", {"leg": "LHR-JFK"})
    print("answer:", result.output)  # noqa: T201
    print("attributed to:", result.attributes()["a2a.peer"], result.chain)  # noqa: T201
    print("delegated scope:", peer.calls[0].meta["tesserix/adk/delegation/scopes"])  # noqa: T201
    print("charged to the run:", budget.spent.usage.input_tokens, "prompt tokens")  # noqa: T201

    try:
        await calling.invoke("price_leg", {"leg": "LHR-JFK", "cabin": "first"})
    except PeerInvocationError as refused:
        print("not sent:", refused.reason)  # noqa: T201

    try:
        await calling.invoke("refund", {"order": "o-1"})
    except PeerInvocationError as refused:
        print("not escalated:", refused.reason)  # noqa: T201

    offered = peer_tool(calling, "price_leg")
    print("offered to the model as:", offered.name, "idempotent:", offered.parallel_safe)  # noqa: T201
    answered: dict[str, Any] = await offered.invoke('{"leg": "CDG-JFK"}')
    print("through the tool:", answered)  # noqa: T201

    try:
        await offered.invoke({"leg": 7})
    except ToolArgumentValidationError as refused:
        print("the model corrected:", refused.feedback().splitlines()[1])  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
