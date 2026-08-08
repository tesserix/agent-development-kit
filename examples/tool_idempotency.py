"""A booking tool that books once, across a retry, a replay and two concurrent calls.

Three scenarios: the key two spellings of one payload share; a run that replays against the
same store; and a call that fails without saying whether it landed. Run it with
`python examples/tool_idempotency.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    Claim,
    Idempotency,
    IdempotencyPolicy,
    ModelCapabilities,
    RunEventKind,
    ToolCall,
    Usage,
    idempotency_key,
)
from tesserix_adk.runtime import AgentRunner, MemoryIdempotencyStore, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext, ToolRegistry, tool

BOOKED: list[str] = []


@tool(idempotency=IdempotencyPolicy(Idempotency.EFFECTFUL, key_arguments=("flight",)))
async def book(flight: str, request_id: str, context: ToolContext) -> str:  # noqa: ARG001
    """Take a seat on a flight.

    Args:
        flight: Which flight.
        request_id: Fresh on every attempt, which is why it is not part of the key.
        context: Carries the key, for passing to a downstream that has one of its own.
    """
    BOOKED.append(flight)
    return f"booked {flight} under {(context.idempotency_key or '')[:8]}"


class Down(MemoryIdempotencyStore):
    """A store that is unreachable when the dispatcher asks it."""

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:  # noqa: ARG002
        """Fail, so the dispatcher has to decide what an unknown record means."""
        raise ConnectionError("the store is down")


def one_payload_one_key() -> None:
    """What the key is derived from, and what it deliberately ignores."""
    first = idempotency_key(
        tenant="acme",
        run_id="run_1",
        tool="book",
        arguments={"flight": "BA117", "request_id": "a"},
        key_arguments=("flight",),
    )
    second = idempotency_key(
        tenant="acme",
        run_id="run_1",
        tool="book",
        arguments={"flight": "BA117", "request_id": "b"},
        key_arguments=("flight",),
    )
    print("a retry that renumbers its request id is one key:", first == second)  # noqa: T201
    print("the key carries nothing of the payload:", "BA117" not in (first or ""))  # noqa: T201


def runner(store: MemoryIdempotencyStore) -> AgentRunner:
    """A runner over one booking tool, backed by `store`."""
    registry = ToolRegistry((book,), clock=FakeClock())
    return AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1", name="book", arguments={"flight": "BA117", "request_id": "a"}
                    ),
                ),
                usage=Usage(input_tokens=1, output_tokens=1),
            ),
            ModelResponse(content="Seat taken.", usage=Usage(input_tokens=1, output_tokens=1)),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("book",), agent="planner"),
        idempotency=store,
    )


def planner() -> Agent[str]:
    """The agent doing the booking."""
    return Agent(
        name="planner",
        instructions="Book it.",
        free_text=True,
        model="scripted-1",
        tools=("book",),
    )


async def replayed() -> None:
    """The same run twice against one store, which is what a worker restart looks like."""
    store = MemoryIdempotencyStore()
    for _ in range(2):
        run = await runner(store).run(planner(), "book BA117", tenant="acme", run_id="run_1")
    deduplicated = [e for e in run.events if e.kind is RunEventKind.TOOL_DEDUPLICATED]
    reused = bool(deduplicated)
    print("runs: 2 | bookings:", len(BOOKED), "| second run reused the record:", reused)  # noqa: T201


async def unknown() -> None:
    """A store nobody can reach: the call does not go out rather than going out twice."""
    run = await runner(Down()).run(planner(), "book BA117", tenant="acme", run_id="run_2")
    print("run state:", run.state.value, "| bookings still:", len(BOOKED))  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    one_payload_one_key()
    await replayed()
    await unknown()
    book.release()


if __name__ == "__main__":
    asyncio.run(main())
