"""An oversized tool result checked in, and fetched back only when it is actually wanted.

Four scenarios: what the model receives in place of a large document; what a run costs with
and without the substitution; redeeming a handle; and what a handle from another run gets.
Run it with `python examples/claim_check.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    ClaimCheckPolicy,
    ClaimTicket,
    ModelCapabilities,
    TextPart,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import (
    AgentRunner,
    ClaimCheck,
    MemoryClaimCheckStore,
    ModelResponse,
    ToolResult,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext, ToolRefusal, ToolRegistry, claim_check_tool, tool

CONTRACT = (
    "Clause 1. This agreement is governed by the laws of the stated jurisdiction.\n"
    "Clause 2. Either party may terminate on thirty days written notice.\n"
) + "".join(
    f"Clause {n}. Boilerplate that nobody needs in a prompt twice.\n" for n in range(3, 900)
)

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


@tool
async def read_contract() -> str:
    """Return the whole contract, because that is what reading a contract means."""
    return CONTRACT


async def what_the_model_receives() -> None:
    """The head answers most questions; the handle is there for the ones it does not."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    check = ClaimCheck(store=store, policy=ClaimCheckPolicy(head_chars=160))
    ticket = await _ticket(check)

    print("=== what replaces the contract ===")  # noqa: T201
    print(ticket.rendered())  # noqa: T201


async def what_a_turn_costs() -> None:
    """The saving is per turn, and every turn after the one that fetched it pays it."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    ticket = await _ticket(ClaimCheck(store=store))

    print("\n=== what a turn carries ===")  # noqa: T201
    print(f"without a claim check: {len(CONTRACT):,} characters, every iteration")  # noqa: T201
    print(f"with one:              {len(ticket.rendered()):,} characters")  # noqa: T201


async def redeeming_the_handle() -> None:
    """The model asks for the rest, a window at a time, and gets what the tool returned."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    ticket = await _ticket(ClaimCheck(store=store))
    fetch = claim_check_tool(store, max_chars=80)

    window = await fetch.invoke(
        {"handle": ticket.handle, "offset": 76}, ToolContext(run_id="run_1", tenant="acme")
    )

    print("\n=== one window of the rest ===")  # noqa: T201
    print(window.strip())  # noqa: T201
    fetch.release()


async def a_handle_from_another_run() -> None:
    """Refused, not approximated: a plausible substitute is worse than no answer."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    ticket = await _ticket(ClaimCheck(store=store))
    fetch = claim_check_tool(store)

    print("\n=== a handle another run made ===")  # noqa: T201
    try:
        await fetch.invoke({"handle": ticket.handle}, ToolContext(run_id="run_2", tenant="acme"))
    except ToolRefusal as refused:
        print(f"{refused.code}: the run that asked is not the run that stored it")  # noqa: T201
    fetch.release()


async def inside_a_run() -> None:
    """The same thing, wired into a runner, which is where it actually earns its keep."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    fetch = claim_check_tool(store)
    registry = ToolRegistry((read_contract, fetch), clock=FakeClock())
    provider = ScriptedProvider(
        ModelResponse(
            content="",
            tool_calls=(ToolCall(id="c1", name="read_contract", arguments={}),),
            usage=Usage(input_tokens=20, output_tokens=5),
        ),
        ModelResponse(content="Thirty days.", usage=Usage(input_tokens=20, output_tokens=5)),
        capabilities=CAPABLE,
    )
    runner = AgentRunner(
        provider=provider,
        clock=FakeClock(),
        tools=registry.view(allow=("read_contract", "fetch_result"), agent="counsel"),
        claim_check=ClaimCheck(store=store),
    )
    agent: Agent[None] = Agent(
        name="counsel",
        instructions="Answer from the contract.",
        free_text=True,
        model="scripted-1",
        tools=("read_contract", "fetch_result"),
    )

    run = await runner.run(agent, "What is the notice period?", tenant="acme", run_id="run_1")
    carried = sum(len(str(message.content)) for message in provider.requests[-1].messages)

    print("\n=== inside a run ===")  # noqa: T201
    answer = "".join(part.text for part in run.messages[-1].content if isinstance(part, TextPart))
    print(f"answer: {answer}")  # noqa: T201
    print(f"the final turn carried {carried:,} characters, not {len(CONTRACT):,}")  # noqa: T201
    fetch.release()


async def _ticket(check: ClaimCheck) -> ClaimTicket:
    """Check the contract in, and refuse to carry on if it was small enough not to be."""
    ticket = await check.stored(_result(CONTRACT), tenant="acme", run_id="run_1")
    if ticket is None:
        raise RuntimeError("the contract is below the threshold, so there is nothing to show")
    return ticket


def _result(text: str) -> ToolResult:
    return ToolResult(tool="read_contract", payload=text, text=text, tenant="acme")


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await what_the_model_receives()
    await what_a_turn_costs()
    await redeeming_the_handle()
    await a_handle_from_another_run()
    await inside_a_run()


if __name__ == "__main__":
    asyncio.run(main())
