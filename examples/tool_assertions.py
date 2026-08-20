"""Asserting what an agent called, under whose tenant, and what a retry did.

Run it with `uv run python examples/tool_assertions.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import ToolExecutionError, ToolNotPermittedError
from tesserix_adk.core.hooks import ApprovalRecord
from tesserix_adk.testing import (
    FakeToolRegistry,
    ToolSpy,
    assert_context_propagated,
    assert_idempotency_key_stable,
    assert_no_tool_called,
    assert_tool_sequence,
    denying,
    failing_tool,
    scoped_run,
)


def search_flights(**arguments: object) -> dict[str, object]:
    """Answer with one seat, so the second call has something to hold."""
    return {"seat": "12A", "query": arguments}


async def main() -> None:
    """Run a booking twice — once under a tenant, once with a scope it does not hold."""
    spy = ToolSpy(
        FakeToolRegistry(
            {
                "search_flights": search_flights,
                "hold_seat": lambda **kw: dict(kw),
                "charge": failing_tool(ToolExecutionError("the ledger is closed")),
            }
        )
    )

    with scoped_run(
        tenant="acme", user="ada", scopes=("search_flights", "hold_seat"), idempotency_key="hold-1"
    ):
        found = await spy.invoke("search_flights", {"from": "SYD", "to": "SIN"})
        await spy.invoke("hold_seat", {"seat": found["seat"]})

    assert_tool_sequence(spy, "search_flights", "hold_seat")
    assert_context_propagated(spy, tenant="acme", user="ada")
    print("the agent called what it should have, under the tenant it was given")  # noqa: T201

    retried = ToolSpy(FakeToolRegistry({"charge": failing_tool(ToolExecutionError("flaked"))}))
    with scoped_run(tenant="acme", idempotency_key="charge-1"):
        for _ in range(2):
            try:
                await retried.invoke("charge", {"amount": 4200})
            except ToolExecutionError as failed:
                print(f"attempt failed: {failed}")  # noqa: T201
    assert_idempotency_key_stable(retried, "charge")
    print("the retry reused its key, so the customer is charged once")  # noqa: T201

    refused = ToolSpy(FakeToolRegistry({"refund": lambda **kw: kw}))
    with scoped_run(tenant="acme", declares=("refund",), scopes=("search_flights",)) as run:
        try:
            run.allowlist.check("refund")
        except ToolNotPermittedError as denied:
            print(f"refused before dispatch: {denied}")  # noqa: T201
    assert_no_tool_called(refused)

    gate = denying(reason="too large for the desk")
    held = ApprovalRecord.for_call(
        run_id="run-1",
        tenant="acme",
        agent_name="booking",
        tool_name="refund",
        arguments={"amount": 4200},
        reason="effectful tool",
    )
    decision = await gate.request(held)
    print(f"the approval gate answered: granted={decision.granted}, {decision.reason}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
