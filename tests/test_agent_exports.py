"""Kit agents leave through authenticated, stable and framework-neutral surfaces."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import BaseModel

from tesserix_adk.adapters.a2a import A2AInterface, A2ASkill
from tesserix_adk.adapters.agent_exports import (
    ExportDescriptorDriftError,
    ExportErrorCode,
    ExportInvocation,
    export_as_a2a,
    export_as_mcp_tool,
    export_as_tool,
)
from tesserix_adk.core import (
    Agent,
    AgentDefinition,
    BudgetLimits,
    BudgetScope,
    Cost,
    HookDecision,
    HookPoint,
    HookSubject,
    ModelCapabilities,
    ModelResponse,
    Owner,
    Principal,
    ScopedLimits,
    StopReason,
    ToolCall,
    TypedAgent,
    TypedAgentDefinition,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.mcp import META_PREFIX
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext, ToolRegistry, tool


class ResearchRequest(BaseModel):
    question: str


class ResearchAnswer(BaseModel):
    answer: str


SEEN_CONTEXTS: list[ToolContext] = []


@tool(name="export_capture_context")
async def capture_context(topic: str, context: ToolContext) -> dict[str, str]:
    """Capture the caller context that crossed the export boundary."""
    SEEN_CONTEXTS.append(context)
    return {"topic": topic}


class AttributionHook:
    def __init__(self) -> None:
        self.seen: list[HookSubject] = []

    @property
    def name(self) -> str:
        return "export-attribution"

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return (HookPoint.BEFORE_MODEL_CALL,)

    async def on(self, subject: HookSubject) -> HookDecision:
        self.seen.append(subject)
        return HookDecision.proceed()


def definition(*, with_tool: bool = False) -> TypedAgentDefinition[ResearchRequest, ResearchAnswer]:
    return TypedAgentDefinition(
        agent=TypedAgent(
            name="research-planner",
            version="2.1.0",
            instructions="Answer the research question.",
            model="scripted-model",
            input_type=ResearchRequest,
            output_type=ResearchAnswer,
            tools=(capture_context.name,) if with_tool else (),
            scopes=("research:read",) if with_tool else (),
            tool_scopes={capture_context.name: ("research:read",)} if with_tool else {},
        ),
        owner=Owner(team="Research", contact="research@example.test", service="research"),
        evaluation_suite="evals/research.jsonl",
    )


def invocation(*, tenant: str = "acme", authenticated_tenant: str = "acme") -> ExportInvocation:
    return ExportInvocation(
        tenant=tenant,
        user="ada",
        scopes=("research:read",),
        trace={"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"},
        principal=Principal(
            subject="ada",
            tenant=authenticated_tenant,
            scopes=frozenset({"research:read"}),
        ),
        run_id="foreign-parent-1",
    )


def native() -> ModelCapabilities:
    return CAPABLE.declaring(structured_output=True)


async def test_function_export_runs_with_authenticated_identity_trace_and_usage() -> None:
    SEEN_CONTEXTS.clear()
    hook = AttributionHook()
    provider = ScriptedProvider(
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name=capture_context.name,
                    arguments={"topic": "Kyoto"},
                ),
            ),
            stop_reason=StopReason.TOOL_CALLS,
            usage=Usage(input_tokens=3, output_tokens=1),
        ),
        ModelResponse(
            content='{"answer":"Kyoto"}',
            usage=Usage(
                input_tokens=5,
                output_tokens=2,
                cost=Cost(input=Decimal("0.02"), output=Decimal("0.01")),
            ),
        ),
        capabilities=native(),
    )
    registry = ToolRegistry((capture_context,))
    exported = export_as_tool(
        AgentRunner(
            provider=provider,
            tools=registry,
            hooks=(hook,),
            clock=FakeClock(),
        ),
        definition(with_tool=True),
        description="Answer a research question.",
    )

    result = await exported.invoke({"question": "Where?"}, invocation())

    assert result.ok
    assert result.output == ResearchAnswer(answer="Kyoto")
    assert result.tenant == "acme"
    assert result.user == "ada"
    assert result.usage.input_tokens == 8
    assert hook.seen[0].tenant == "acme"
    assert hook.seen[0].user == "ada"
    assert SEEN_CONTEXTS[0].scopes == ("research:read",)
    assert SEEN_CONTEXTS[0].trace == invocation().trace


async def test_missing_or_mismatched_authenticated_tenant_fails_before_provider() -> None:
    provider = ScriptedProvider(ModelResponse(content='{"answer":"unused"}'), capabilities=native())
    exported = export_as_tool(
        AgentRunner(provider=provider, clock=FakeClock()),
        definition(),
        description="Answer a research question.",
    )

    missing = await exported.invoke({"question": "Where?"}, None)
    mismatched = await exported.invoke(
        {"question": "Where?"}, invocation(tenant="globex", authenticated_tenant="acme")
    )

    assert missing.error is not None
    assert missing.error.code is ExportErrorCode.AUTHORISATION
    assert mismatched.error is not None
    assert mismatched.error.code is ExportErrorCode.AUTHORISATION
    assert not missing.error.retryable
    assert provider.requests == []


async def test_budget_and_schema_failures_are_stable_non_retryable_envelopes() -> None:
    budget = RunBudget(
        most_restrictive(
            ScopedLimits(
                scope=BudgetScope.RUN,
                limits=BudgetLimits(
                    max_cost=Decimal("1"),
                    max_input_tokens=1,
                    max_output_tokens=100,
                    max_model_calls=10,
                    max_peer_invocations=10,
                ),
            )
        ),
        FakeClock(),
    )
    budget_provider = ScriptedProvider(
        ModelResponse(content='{"answer":"unused"}'), capabilities=native()
    )
    budgeted = export_as_tool(
        AgentRunner(provider=budget_provider, clock=FakeClock()),
        definition(),
        description="Answer a research question.",
    )
    denied = await budgeted.invoke(
        {"question": "This prompt cannot fit."}, invocation().with_budget(budget)
    )

    invalid = export_as_tool(
        AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="not JSON"), capabilities=native()),
            clock=FakeClock(),
        ),
        definition(),
        description="Answer a research question.",
    )
    malformed = await invalid.invoke({"question": "Where?"}, invocation())

    assert denied.error is not None
    assert denied.error.code is ExportErrorCode.BUDGET_REFUSAL
    assert not denied.error.retryable
    assert budget_provider.requests == []
    assert malformed.error is not None
    assert malformed.error.code is ExportErrorCode.SCHEMA_VIOLATION
    assert not malformed.error.retryable


def test_descriptor_is_openai_compatible_pinnable_and_non_streaming() -> None:
    exported = export_as_tool(
        AgentRunner(
            provider=ScriptedProvider(capabilities=native()),
            clock=FakeClock(),
        ),
        definition(),
        description="Answer a research question.",
    )

    descriptor = exported.descriptor
    function = descriptor["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert descriptor["type"] == "function"
    assert function["strict"] is True
    assert parameters["additionalProperties"] is False
    assert not exported.streaming
    exported.assert_descriptor(exported.descriptor_fingerprint)
    with pytest.raises(ExportDescriptorDriftError):
        exported.assert_descriptor("sha256:stale")


async def test_mcp_export_delegates_to_the_authenticated_mcp_server() -> None:
    server = export_as_mcp_tool(
        AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(content='{"answer":"Kyoto"}'), capabilities=native()
            ),
            clock=FakeClock(),
        ),
        definition(),
        description="Answer a research question.",
    )
    session = server.connect(
        meta={
            f"{META_PREFIX}/tenant": "acme",
            f"{META_PREFIX}/subject": "ada",
            f"{META_PREFIX}/run": "foreign-parent-1",
        },
        authenticated="acme",
    )
    descriptor = (await session.list_tools())[0]
    result = await session.call_tool(
        descriptor.name,
        {"question": "Where?"},
        meta={},
        timeout_seconds=5,
    )

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["output"] == {"answer": "Kyoto"}
    assert result.structured_content["tenant"] == "acme"


def test_a2a_export_delegates_card_and_execution_to_the_official_adapter() -> None:
    pytest.importorskip("a2a")
    plain = AgentDefinition(
        agent=Agent(
            name="a2a-research",
            instructions="Answer research questions.",
            model="scripted-model",
            output_type=ResearchAnswer,
        ),
        owner=Owner(team="Research", contact="research@example.test", service="research"),
        evaluation_suite="evals/research.jsonl",
    )

    async def resolve(_context: object) -> Principal:
        return Principal(subject="ada", tenant="acme")

    exported = export_as_a2a(
        AgentRunner(
            provider=ScriptedProvider(capabilities=native()),
            clock=FakeClock(),
        ),
        plain,
        resolve=resolve,
        description="Answer research questions.",
        provider_url="https://agents.example.test",
        interfaces=(
            A2AInterface(
                url="https://agents.example.test/a2a",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ),
        skills=(
            A2ASkill(
                id="research",
                name="Research",
                description="Answer one research question.",
                tags=("research",),
                examples=("Where is Kyoto?",),
                input_modes=("text/plain",),
                output_modes=("application/json",),
            ),
        ),
    )

    assert exported.card.name == "a2a-research"
    assert exported.executor is not None
