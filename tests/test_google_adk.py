"""Google Agent Development Kit assets cross Tesserix boundaries explicitly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("a2a")
pytest.importorskip("google.adk")

from google.adk.agents import BaseAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events import Event
from google.adk.tools.function_tool import FunctionTool

# FunctionTool resolves this annotation while building its schema.
from google.adk.tools.tool_context import ToolContext as GoogleToolContext  # noqa: TC002
from pydantic import BaseModel

from tesserix_adk.adapters import (
    GOOGLE_ADK_CONTEXT_KEY,
    A2AInterface,
    A2ASkill,
    ForeignAgentContext,
    ForeignAgentReply,
    ToolImportPolicy,
    WrappedAgentPolicy,
    a2a_card_for,
    google_adk_remote_agent,
    import_google_adk_toolset,
    wrap_google_adk_agent,
)
from tesserix_adk.core import (
    Agent,
    AgentDefinition,
    BudgetLimits,
    BudgetScope,
    Idempotency,
    Owner,
    RunState,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import ToolContext

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from a2a.types import AgentCard
    from google.adk.agents.invocation_context import InvocationContext


READ_ONLY = ToolImportPolicy(
    timeout_seconds=2,
    max_concurrency=2,
    requires_approval=False,
    idempotency=Idempotency.READ_ONLY,
)
PROJECTED = Usage(input_tokens=4, output_tokens=2)


class ResearchRequest(BaseModel):
    question: str


class ResearchAnswer(BaseModel):
    answer: str


class GoogleResearchAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        yield Event(invocation_id=ctx.invocation_id, author=self.name)


GOOGLE_TOOL_CONTEXTS: list[GoogleToolContext] = []


async def google_search(query: str, tool_context: GoogleToolContext) -> dict[str, str]:
    GOOGLE_TOOL_CONTEXTS.append(tool_context)
    caller = tool_context.state[GOOGLE_ADK_CONTEXT_KEY]
    return {"query": query, "tenant": caller["tenant"]}


def google_rank(query: str, limit: int) -> dict[str, object]:
    return {"query": query, "limit": limit}


def budget() -> RunBudget:
    return RunBudget(
        most_restrictive(
            ScopedLimits(
                scope=BudgetScope.RUN,
                limits=BudgetLimits(
                    max_input_tokens=100,
                    max_output_tokens=100,
                    max_model_calls=10,
                    max_peer_invocations=10,
                ),
            )
        ),
        FakeClock(),
    )


def card() -> AgentCard:
    definition = AgentDefinition(
        agent=Agent(
            name="trip-planner",
            version="1.0.0",
            instructions="Plan trips.",
            model="test-model",
            free_text=True,
        ),
        owner=Owner(team="Travel", contact="travel@example.test", service="planner"),
        evaluation_suite="evals/travel.jsonl",
    )
    return a2a_card_for(
        definition,
        description="Plans trips.",
        provider_url="https://agents.example.test",
        interfaces=(
            A2AInterface(
                url="https://agents.example.test/a2a/trip-planner",
                protocol_binding="JSONRPC",
            ),
        ),
        skills=(
            A2ASkill(
                id="plan-trip",
                name="Plan a trip",
                description="Creates an itinerary.",
                tags=("travel",),
            ),
        ),
    )


def test_google_adk_accepts_the_official_tesserix_agent_card() -> None:
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        remote = google_adk_remote_agent(
            name="tesserix_trip_planner",
            description="A Tesserix agent reached over official A2A.",
            agent_card=card(),
            timeout_seconds=30.0,
        )

    assert isinstance(remote, RemoteA2aAgent)
    assert remote.name == "tesserix_trip_planner"
    assert remote.description == "A Tesserix agent reached over official A2A."


async def test_google_function_tools_use_generic_policy_validation_and_context() -> None:
    GOOGLE_TOOL_CONTEXTS.clear()
    imported = import_google_adk_toolset(
        (FunctionTool(google_search), FunctionTool(google_rank)),
        policy=READ_ONLY,
    )

    result = await imported[0].invoke(
        {"query": "Kyoto"},
        ToolContext(
            run_id="run-1",
            tenant="acme",
            user="ada",
            scopes=("research:read",),
            trace={"traceparent": "00-abc-def-01"},
        ),
    )

    assert [tool.name for tool in imported] == ["google_search", "google_rank"]
    assert result == {"query": "Kyoto", "tenant": "acme"}
    assert imported[0].parameters_schema["additionalProperties"] is False
    assert imported[0].origin == "google-adk:function-tool:google_search"
    carried = GOOGLE_TOOL_CONTEXTS[0].state[GOOGLE_ADK_CONTEXT_KEY]
    assert carried == {
        "run_id": "run-1",
        "tenant": "acme",
        "user": "ada",
        "scopes": ["research:read"],
        "trace": {"traceparent": "00-abc-def-01"},
    }


async def test_google_agent_wrap_reuses_typed_budgeted_subagent_boundary() -> None:
    seen: list[tuple[object, ResearchRequest, ForeignAgentContext]] = []

    async def invoke(
        agent: object, request: ResearchRequest, context: ForeignAgentContext
    ) -> ForeignAgentReply[ResearchAnswer]:
        seen.append((agent, request, context))
        return ForeignAgentReply(output={"answer": request.question}, usage=PROJECTED)

    source = GoogleResearchAgent(name="google_research")
    wrapped = wrap_google_adk_agent(
        source,
        invoke=invoke,
        input_type=ResearchRequest,
        output_type=ResearchAnswer,
        policy=WrappedAgentPolicy(
            timeout_seconds=2,
            projected_usage=PROJECTED,
            scopes=("research:read",),
            tools=("search",),
            requires_approval=False,
            idempotency=Idempotency.READ_ONLY,
        ),
    )

    run = await wrapped.run(
        '{"question":"Kyoto"}',
        tenant="acme",
        user="ada",
        budget=budget(),
        scopes=("research:read", "admin"),
        trace={"traceparent": "00-abc-def-01"},
        tools=("search", "admin"),
    )

    assert run.state is RunState.COMPLETED
    assert run.output == ResearchAnswer(answer="Kyoto")
    assert seen[0][0] is source
    assert seen[0][2].tenant == "acme"
    assert seen[0][2].scopes == ("research:read",)
    assert seen[0][2].tools == ("search",)
    assert seen[0][2].trace == {"traceparent": "00-abc-def-01"}
