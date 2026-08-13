"""The tenant a run belongs to, read by everything below it and passed by nobody.

Four scenarios: a tool body that reads the tenant it was never given; two runs racing
under different tenants without seeing each other; an unscoped read refused rather than
widened; and an administrative crossing that has to say why.

Run it with `python examples/tenancy.py`. A scripted provider stands in for the vendor, so
nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.core import (
    Agent,
    MissingTenantContextError,
    ModelCapabilities,
    Run,
    TenantContext,
    TenantCrossingError,
    ToolCall,
    Usage,
    current_tenant,
    tenant_here,
    tenant_scope,
)
from tesserix_adk.memory import MemoryScope
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)

read: list[str] = []


@tool
async def looking_up(what: str) -> str:
    """Look something up for whoever the run belongs to.

    Args:
        what: What is being looked up.
    """
    here = current_tenant(where="looking_up")
    offloaded = await asyncio.to_thread(lambda: current_tenant().tenant)
    read.append(f"{what} for {here.tenant} ({here.user}), {offloaded} on the pool thread")
    return read[-1]


async def a_run(tenant: str, user: str) -> Run[Any]:
    """One run, whose tool is never told which tenant it is working for."""
    registry = ToolRegistry((looking_up,), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="call_1", name="looking_up", arguments={"what": "fares"}),),
                usage=Usage(input_tokens=120, output_tokens=30),
            ),
            ModelResponse(content="40 EUR.", usage=Usage(input_tokens=120, output_tokens=30)),
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("looking_up",), agent="planner"),
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Plan trips.",
        free_text=True,
        model="scripted-1",
        tools=("looking_up",),
    )
    return await runner.run(agent, "look it up", tenant=tenant, user=user, run_id=f"run_{tenant}")


async def a_tool_reads_what_nobody_passed_it() -> None:
    """The tool signature says nothing about tenancy and the body still cannot be wrong."""
    read.clear()
    run = await a_run("acme", "ada")
    print("== a tool body reads the context ==")  # noqa: T201
    print(f"  run {run.id} for {run.tenant}: {run.state.value}")  # noqa: T201
    print(f"  the tool saw: {read[0]}")  # noqa: T201


async def two_tenants_at_once() -> None:
    """A fan-out for one tenant cannot observe the other, whatever the scheduler does."""
    read.clear()
    first, second = await asyncio.gather(a_run("acme", "ada"), a_run("globex", "grace"))
    print("\n== two runs, two tenants, one process ==")  # noqa: T201
    print(f"  {first.id}: {first.tenant}")  # noqa: T201
    print(f"  {second.id}: {second.tenant}")  # noqa: T201
    for line in sorted(read):
        print(f"  the tool saw: {line}")  # noqa: T201
    print(f"  nothing is bound once they return: {tenant_here()}")  # noqa: T201


def an_unscoped_read_is_refused() -> None:
    """Absence is never a default tenant: a wildcard scope reads everybody's memory."""
    print("\n== egress with nothing bound ==")  # noqa: T201
    try:
        MemoryScope.here()
    except MissingTenantContextError as refused:
        print(f"  refused at {refused.where}")  # noqa: T201

    with tenant_scope(TenantContext(tenant="acme", user="ada")):
        scope = MemoryScope.here(session_id="s-1")
    print(f"  inside a scope: tenant={scope.tenant_id}, user={scope.user_id}")  # noqa: T201


def a_crossing_has_to_say_why() -> None:
    """An administrative operation is fine; one nobody declared is the incident."""
    print("\n== crossing to another tenant ==")  # noqa: T201
    with tenant_scope("acme"):
        try:
            with tenant_scope("globex"):
                pass
        except TenantCrossingError as refused:
            print(f"  {refused.tenant} -> {refused.into}: refused")  # noqa: T201

        with tenant_scope("globex", crossing="registry backfill"):
            here = current_tenant()
            print(f"  {here.tenant} allowed, recorded as: {here.crossing}")  # noqa: T201
        print(f"  and back under {current_tenant().tenant} on the way out")  # noqa: T201


async def main() -> None:
    """Run the four scenarios in order."""
    await a_tool_reads_what_nobody_passed_it()
    await two_tenants_at_once()
    an_unscoped_read_is_refused()
    a_crossing_has_to_say_why()


if __name__ == "__main__":
    asyncio.run(main())
