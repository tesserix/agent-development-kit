"""A planner that only produces a plan, and an executor that refuses what it cannot check.

Six scenarios: a plan that runs, a step naming a tool nobody registered, arguments the tool
never declared, an irreversible step waiting for a person, a bounded replan loop, and a plan
picked up where a dead process left it.
Run it with `python examples/planner.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from tesserix_adk.core import Agent, ApprovalDeniedError, PlanValidationError
from tesserix_adk.core.hooks import ApprovalDecision, ApprovalRecord
from tesserix_adk.runtime import (
    Delegation,
    DelegationScope,
    InMemoryPlanStore,
    MemoryIdempotencyStore,
    Plan,
    PlanExecutor,
    PlanStep,
    ToolContract,
)
from tesserix_adk.testing import FakeClock, FakeToolRegistry

HELD = frozenset({"search_flights", "book_flight", "notify"})


class Search(BaseModel):
    """What the search tool declared it takes."""

    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class Booking(BaseModel):
    """What the booking tool declared it takes, seats and all."""

    flight: str = Field(min_length=1)
    seats: int = Field(ge=1)
    price: Decimal = Decimal("0")


class Note(BaseModel):
    """What the notify tool declared it takes."""

    text: str = Field(min_length=1)


class Registrar:
    """Whoever decides about a step that cannot be undone."""

    def __init__(self, *, granted: bool) -> None:
        self._granted = granted

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Decide, the one way this registrar was built to decide."""
        return ApprovalDecision(
            record_id=record.id,
            granted=self._granted,
            decided_by="ada",
            reason="fare holds for an hour" if self._granted else "the fare is wrong",
        )


class Drafts:
    """A planner that hands back what this file scripted, refusal feedback included."""

    def __init__(self, *plans: Plan) -> None:
        self._plans = list(plans)
        self.told: list[str] = []

    async def plan(self, task: str, *, feedback: str = "") -> Plan:
        """Hand back the next scripted plan, remembering what the last one was told."""
        self.told.append(feedback)
        return (self._plans.pop(0) if len(self._plans) > 1 else self._plans[0]).model_copy(
            update={"goal": task}
        )


def _agent() -> Agent[Any]:
    """The agent the plan is executed for. Its allowlist caps what a step may name."""
    fields: dict[str, object] = {
        "name": "courier",
        "instructions": "You are courier.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tuple(sorted(HELD)),
    }
    return Agent(**fields)  # type: ignore[arg-type]


def _executor(
    *,
    approvals: Registrar | None = None,
    plans: InMemoryPlanStore | None = None,
    idempotency: MemoryIdempotencyStore | None = None,
    max_replans: int = 1,
    broken: bool = False,
) -> PlanExecutor:
    """An executor over three tools, one of which cannot be undone."""
    tools = {
        "search_flights": lambda origin, destination: f"{origin}->{destination}: BA117",
        "book_flight": _breaks if broken else _books,
        "notify": lambda text: f"told them: {text}",
    }
    return PlanExecutor(
        FakeToolRegistry(tools),
        (
            ToolContract(tool="search_flights", accepts=Search),
            ToolContract(
                tool="book_flight", accepts=Booking, irreversible=True, key_arguments=("flight",)
            ),
            ToolContract(tool="notify", accepts=Note),
        ),
        agent=_agent(),
        delegation=Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="courier",
            user="ada",
            scope=DelegationScope(tools=HELD),
        ),
        approvals=approvals,
        plans=plans,
        idempotency=idempotency,
        clock=FakeClock(),
        max_replans=max_replans,
    )


def _books(flight: str, seats: int, price: Decimal = Decimal("0")) -> str:
    return f"booked {seats} on {flight} at {price}"


def _breaks(flight: str, **_: object) -> str:
    raise RuntimeError(f"the airline dropped the connection booking {flight}")


def _plan(*steps: PlanStep) -> Plan:
    return Plan(goal="get them to New York", steps=steps)


def _search() -> PlanStep:
    return PlanStep(
        id="s1", tool="search_flights", arguments={"origin": "LHR", "destination": "JFK"}
    )


def _booking(**arguments: object) -> PlanStep:
    return PlanStep(
        id="s2",
        tool="book_flight",
        arguments={"flight": "BA117", "seats": 2, **arguments},
        depends_on=("s1",),
        intent="hold the seats before the fare moves",
    )


async def a_plan_that_runs() -> None:
    """Steps run in dependency order, not in the order the planner wrote them."""
    done = await _executor(approvals=Registrar(granted=True)).execute(_plan(_booking(), _search()))

    print("=== a plan that runs ===")  # noqa: T201
    for result in done.results:
        print(f"{result.step_id} {result.tool}: {result.outcome}")  # noqa: T201


async def a_tool_nobody_registered() -> None:
    """Refused with the step and the tool named, before anything at all is called."""
    invalid = _plan(_search(), PlanStep(id="s2", tool="wire_transfer", arguments={"amount": "1"}))

    print("\n=== a tool nobody registered ===")  # noqa: T201
    try:
        await _executor().execute(invalid)
    except PlanValidationError as refused:
        print(f"{refused.reason} at {refused.step}: {refused}")  # noqa: T201


async def arguments_the_tool_never_declared() -> None:
    """Nothing is coerced or dropped: the raw payload comes back for whoever debugs it."""
    print("\n=== arguments the tool never declared ===")  # noqa: T201
    try:
        await _executor().execute(_plan(_search(), _booking(cabin="business")))
    except PlanValidationError as refused:
        print(f"{refused.reason}: {refused.violations} in {refused.payload}")  # noqa: T201


async def a_step_that_cannot_be_undone() -> None:
    """A denial halfway down the plan leaves nothing half-done: clearance comes first."""
    executor = _executor(approvals=Registrar(granted=False))

    print("\n=== a step that cannot be undone ===")  # noqa: T201
    try:
        await executor.execute(_plan(_search(), _booking(price=Decimal("412.50"))))
    except ApprovalDeniedError as refused:
        print(refused)  # noqa: T201
    print(f"record: {[(e.kind.value, e.name) for e in executor.events]}")  # noqa: T201


async def a_planner_that_keeps_getting_it_wrong() -> None:
    """Every attempt is told what the last one got wrong, and the allowance is finite."""
    drafts = Drafts(_plan(_search(), _booking(seats=0)))

    print("\n=== a planner that keeps getting it wrong ===")  # noqa: T201
    try:
        await _executor().planned(drafts, "get them to New York")
    except PlanValidationError as refused:
        print(f"{refused.reason} after {refused.attempts} attempts")  # noqa: T201
    print(f"told: {[told[:38] for told in drafts.told]}")  # noqa: T201


async def picking_up_where_a_dead_process_left_it() -> None:
    """The plan is revalidated against the contracts as they are now, then carried on."""
    plans, keys = InMemoryPlanStore(), MemoryIdempotencyStore()
    approvals = Registrar(granted=True)
    dying = _executor(approvals=approvals, plans=plans, idempotency=keys, broken=True)
    try:
        await dying.execute(_plan(_search(), _booking()))
    except RuntimeError as died:
        print(f"\n=== picking up where a dead process left it ===\n{died}")  # noqa: T201

    done = await _executor(approvals=approvals, plans=plans, idempotency=keys).resume()
    print(f"resumed: {done.outcomes}")  # noqa: T201
    print(f"complete: {done.complete}, and s1 was not run a second time")  # noqa: T201


def main() -> None:
    """Run every scenario in the order the docs describe them."""
    asyncio.run(a_plan_that_runs())
    asyncio.run(a_tool_nobody_registered())
    asyncio.run(arguments_the_tool_never_declared())
    asyncio.run(a_step_that_cannot_be_undone())
    asyncio.run(a_planner_that_keeps_getting_it_wrong())
    asyncio.run(picking_up_where_a_dead_process_left_it())


if __name__ == "__main__":
    main()
