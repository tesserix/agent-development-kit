"""An agent declaring two scopes, run for a caller who holds one of them.

Run it with `uv run python examples/agent_identity.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    AgentIdentity,
    AuthorisationError,
    Principal,
    ToolCall,
    Usage,
    principal_scope,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

READ = "bookings:read"
WRITE = "bookings:write"

DESK = Agent(
    name="desk",
    instructions="Help with bookings.",
    model="claude-sonnet-5",
    free_text=True,
    tools=("search", "refund"),
    scopes=(READ, WRITE),
    tool_scopes={"search": (READ,), "refund": (WRITE,)},
)


def _runner(*responses: ModelResponse) -> AgentRunner:
    return AgentRunner(
        provider=ScriptedProvider(*responses),
        tools=FakeToolRegistry({"search": lambda **_: "one seat", "refund": lambda **_: "done"}),
        clock=FakeClock(),
    )


def _asked_for(tool: str) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _said(text: str) -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=1, output_tokens=1))


async def main() -> None:
    """Show what the model is told, what a refusal says, and what a peer inherits."""
    reader = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}))
    provider = ScriptedProvider(_said("one seat"))
    runner = AgentRunner(
        provider=provider,
        tools=FakeToolRegistry({"search": lambda **_: "one seat", "refund": lambda **_: "done"}),
        clock=FakeClock(),
    )
    with principal_scope(reader):
        await runner.run(DESK, "find me a seat", tenant="acme", run_id="run_1")
    print(f"declared to the model: {[t.name for t in provider.requests[0].tools]}")  # noqa: T201

    with principal_scope(reader):
        run = await _runner(_asked_for("refund")).run(
            DESK, "refund my fare", tenant="acme", run_id="run_2"
        )
    print(f"asking for refund anyway: {run.state}")  # noqa: T201

    both = Principal(subject="ada", tenant="acme", scopes=frozenset({READ, WRITE}))
    with principal_scope(both):
        run = await _runner(_asked_for("refund"), _said("refunded")).run(
            DESK, "refund my fare", tenant="acme", run_id="run_3"
        )
    print(f"the same run for a caller who holds it: {run.state}")  # noqa: T201

    try:
        await _runner(_said("one seat")).run(DESK, "find a seat", tenant="acme", run_id="run_4")
    except AuthorisationError as refused:
        print(f"no caller at all: {refused}")  # noqa: T201

    identity = AgentIdentity.resolve(agent="desk", declared=(READ, WRITE), principal=reader)
    peer = identity.narrowed(agent="billing", declared=(READ, WRITE))
    print(f"a peer declaring both still holds: {peer.effective}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
