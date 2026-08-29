"""Legacy agents remain bounded, typed and attributable when orchestrated by the kit."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import BaseModel

from tesserix_adk.adapters.foreign_agents import (
    ForeignAgentContext,
    ForeignAgentReply,
    WrappedAgentPolicy,
    wrap_agent_as_subagent,
    wrap_agent_as_tool,
)
from tesserix_adk.core import (
    Agent,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    CancelledError,
    Cost,
    Idempotency,
    IndeterminateOutcomeError,
    ProviderTimeoutError,
    RecursionLimitError,
    RunContext,
    RunState,
    SchemaViolationError,
    ScopedLimits,
    TenantContext,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.runtime import (
    AgentRunner,
    CancellationToken,
    Delegation,
    DelegationScope,
    Roster,
    Specialist,
    Supervisor,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext


class LegacyRequest(BaseModel):
    """Input declared at the wrapper boundary."""

    question: str


class LegacyAnswer(BaseModel):
    """Output that must validate before a parent may read it."""

    answer: str


ESTIMATE = Usage(
    input_tokens=8,
    output_tokens=4,
    cost=Cost(input=Decimal("0.02"), output=Decimal("0.01")),
)
POLICY = WrappedAgentPolicy(
    timeout_seconds=2,
    projected_usage=ESTIMATE,
    scopes=("legacy:read",),
    tools=("search",),
    requires_approval=False,
    idempotency=Idempotency.READ_ONLY,
)


class LegacyAgent:
    """A foreign agent exposing only an application-owned adapter call."""

    def __init__(self, reply: object | None = None) -> None:
        self.reply = reply or {"answer": "Kyoto"}
        self.calls: list[tuple[LegacyRequest, ForeignAgentContext]] = []

    async def __call__(
        self, request: LegacyRequest, context: ForeignAgentContext
    ) -> ForeignAgentReply[LegacyAnswer]:
        self.calls.append((request, context))
        return ForeignAgentReply(output=self.reply, usage=ESTIMATE)


def budget(*, max_cost: Decimal = Decimal("1")) -> RunBudget:
    """A real shared ledger, so preflight and roll-up are exercised together."""
    return RunBudget(
        most_restrictive(
            ScopedLimits(
                scope=BudgetScope.RUN,
                limits=BudgetLimits(
                    max_cost=max_cost,
                    max_input_tokens=100,
                    max_output_tokens=100,
                    max_model_calls=10,
                    max_peer_invocations=10,
                ),
            )
        ),
        FakeClock(),
    )


async def test_wrapped_tool_validates_output_and_narrows_context() -> None:
    foreign = LegacyAgent()
    wrapped = wrap_agent_as_tool(
        foreign,
        name="legacy_researcher",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY,
    )
    ledger = budget()
    context = ToolContext(
        run_id="run-1",
        tenant="acme",
        user="ada",
        scopes=("legacy:read", "admin"),
        trace={"traceparent": "00-abc-def-01"},
        budget=ledger,
    )

    result = await wrapped.invoke({"question": "Where?"}, context)

    assert result == LegacyAnswer(answer="Kyoto")
    assert foreign.calls[0][1].scopes == ("legacy:read",)
    assert foreign.calls[0][1].tenant == "acme"
    assert foreign.calls[0][1].trace == {"traceparent": "00-abc-def-01"}
    assert ledger.spent.usage == ESTIMATE


async def test_projected_cost_is_refused_before_a_foreign_call() -> None:
    foreign = LegacyAgent()
    wrapped = wrap_agent_as_tool(
        foreign,
        name="legacy_researcher",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY,
    )

    with pytest.raises(BudgetExceededError, match="max_cost") as raised:
        await wrapped.invoke(
            {"question": "Where?"},
            ToolContext(run_id="run-1", tenant="acme", budget=budget(max_cost=Decimal("0.01"))),
        )

    assert raised.value.breached == "max_cost"
    assert foreign.calls == []


async def test_invalid_foreign_prose_fails_closed_with_the_raw_reply() -> None:
    foreign = LegacyAgent("not structured JSON")
    wrapped = wrap_agent_as_tool(
        foreign,
        name="legacy_researcher",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY,
    )

    with pytest.raises(SchemaViolationError) as raised:
        await wrapped.invoke(
            {"question": "Where?"},
            ToolContext(run_id="run-1", tenant="acme", budget=budget()),
        )

    assert raised.value.payload == "not structured JSON"


async def test_a_wrapped_subagent_is_addressable_from_the_supervisor_roster() -> None:
    foreign = LegacyAgent()
    wrapped = wrap_agent_as_subagent(
        foreign,
        name="legacy_researcher",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY.model_copy(update={"scopes": ()}),
    )
    roster = Roster((Specialist(agent=wrapped, capabilities=frozenset({"research"})),))
    supervisor_agent = Agent(
        name="supervisor",
        instructions="Delegate research.",
        model="unused",
        free_text=True,
        tools=("search",),
    )
    supervisor = Supervisor(
        AgentRunner(provider=ScriptedProvider(), clock=FakeClock()),
        roster,
        agent=supervisor_agent,
        delegation=Delegation.root(
            run_id="root",
            tenant="acme",
            user="ada",
            agent="supervisor",
            scope=DelegationScope(tools=frozenset({"search"})),
        ),
        budget=budget(),
    )

    result = await supervisor.delegate('{"question":"Where?"}', needs={"research"})

    assert result.run.state is RunState.COMPLETED
    assert result.run.output == LegacyAnswer(answer="Kyoto")
    assert result.run.tenant == "acme"
    assert result.run.user == "ada"
    assert result.usage == ESTIMATE
    assert foreign.calls[0][1].depth == 1


async def test_a_raw_reply_uses_the_declared_estimate_and_labels_it() -> None:
    async def unmetered(request: LegacyRequest, context: ForeignAgentContext) -> dict[str, str]:
        assert context.tenant == "acme"
        return {"answer": request.question}

    wrapped = wrap_agent_as_subagent(
        unmetered,
        name="unmetered",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY.model_copy(update={"scopes": ()}),
    )

    run = await wrapped.run('{"question":"Kyoto"}', tenant="acme", user=None, budget=budget())

    assert run.usage == ESTIMATE
    assert any("estimated" in (event.detail or "") for event in run.events)


async def test_timeout_cancels_and_joins_the_foreign_async_task() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def waiting(_request: LegacyRequest, _context: ForeignAgentContext) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    wrapped = wrap_agent_as_subagent(
        waiting,
        name="waiting",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY.model_copy(update={"timeout_seconds": 0.01}),
    )

    with pytest.raises(ProviderTimeoutError):
        await wrapped.run('{"question":"Kyoto"}', tenant="acme", budget=budget())

    assert started.is_set()
    assert cleaned.is_set()


async def test_effectful_timeout_is_reported_as_an_indeterminate_outcome() -> None:
    effect_started = asyncio.Event()

    async def effectful(_request: LegacyRequest, _context: ForeignAgentContext) -> None:
        effect_started.set()
        await asyncio.Event().wait()

    wrapped = wrap_agent_as_subagent(
        effectful,
        name="effectful",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY.model_copy(
            update={"timeout_seconds": 0.01, "idempotency": Idempotency.EFFECTFUL}
        ),
    )

    with pytest.raises(IndeterminateOutcomeError, match="outcome is unknown"):
        await wrapped.run('{"question":"Kyoto"}', tenant="acme", budget=budget())

    assert effect_started.is_set()


async def test_parent_cancellation_cancels_and_joins_the_foreign_async_task() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    cancellation = CancellationToken()

    async def waiting(_request: LegacyRequest, _context: ForeignAgentContext) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    wrapped = wrap_agent_as_subagent(
        waiting,
        name="waiting",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY,
    )
    running = asyncio.create_task(
        wrapped.run(
            '{"question":"Kyoto"}',
            tenant="acme",
            budget=budget(),
            cancellation=cancellation,
        )
    )
    await started.wait()
    cancellation.cancel("caller left")

    with pytest.raises(CancelledError, match="caller left"):
        await running

    assert cleaned.is_set()


async def test_recursive_wrapped_agent_is_refused_before_foreign_work() -> None:
    foreign = LegacyAgent()
    wrapped = wrap_agent_as_subagent(
        foreign,
        name="legacy",
        input_type=LegacyRequest,
        output_type=LegacyAnswer,
        policy=POLICY,
    )
    parent = RunContext(
        run_id="root",
        tenant=TenantContext(tenant="acme"),
        path=("supervisor", "legacy"),
    )

    with pytest.raises(RecursionLimitError, match="already on"):
        await wrapped.run('{"question":"Kyoto"}', tenant="acme", budget=budget(), parent=parent)

    assert foreign.calls == []
