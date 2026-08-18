"""Runtime-owned context contribution is observable and has explicit failure semantics."""

from __future__ import annotations

from tesserix_adk.core import Agent, RunEventKind, RunState, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    ContextContribution,
    ContextContributor,
    ContextRequest,
    ModelResponse,
)
from tesserix_adk.testing import ScriptedProvider


class Contributor:
    def __init__(self, *, required: bool = False, failure: Exception | None = None) -> None:
        self.name = "code-intelligence"
        self.required = required
        self.failure = failure
        self.requests: list[ContextRequest] = []

    async def contribute(self, request: ContextRequest) -> ContextContribution:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return ContextContribution(
            content=("AuthService.verify — src/auth.py:L40-L61",),
            keys=("checkout-1:auth-service",),
        )


AGENT = Agent(
    name="developer",
    instructions="Change the code safely.",
    model="test-model",
    free_text=True,
)


def answer() -> ModelResponse:
    return ModelResponse(content="done", usage=Usage(input_tokens=10, output_tokens=1))


def sent_text(provider: ScriptedProvider) -> str:
    return "\n".join(
        part.text
        for message in provider.requests[0].messages
        for part in message.content
        if hasattr(part, "text")
    )


async def test_contributed_context_is_untrusted_and_recorded() -> None:
    provider = ScriptedProvider(answer())
    contributor = Contributor()
    runner = AgentRunner(provider=provider, context_contributors=(contributor,))

    run = await runner.run(AGENT, "fix authorization", tenant="acme", run_id="run-1")

    assert run.state is RunState.COMPLETED
    assert '<untrusted-data source="retrieved">' in sent_text(provider)
    assert "AuthService.verify — src/auth.py:L40-L61" in sent_text(provider)
    assert contributor.requests == [
        ContextRequest(
            run_id="run-1",
            tenant="acme",
            agent_name="developer",
            query="fix authorization",
        )
    ]
    assert any(
        event.kind is RunEventKind.CONTEXT_RETRIEVED and event.name == "code-intelligence"
        for event in run.events
    )


async def test_duplicate_context_keys_are_admitted_once() -> None:
    provider = ScriptedProvider(answer())
    first = Contributor()
    second = Contributor()
    runner = AgentRunner(provider=provider, context_contributors=(first, second))

    run = await runner.run(AGENT, "fix authorization", tenant="acme")

    assert run.state is RunState.COMPLETED
    assert sent_text(provider).count("AuthService.verify — src/auth.py:L40-L61") == 1


async def test_optional_context_failure_degrades_to_a_cold_run() -> None:
    provider = ScriptedProvider(answer())
    contributor = Contributor(failure=OSError("index unavailable"))
    runner = AgentRunner(provider=provider, context_contributors=(contributor,))

    run = await runner.run(AGENT, "fix authorization", tenant="acme")

    assert run.state is RunState.COMPLETED
    assert len(provider.requests) == 1
    assert any(
        event.kind is RunEventKind.CONTEXT_DEGRADED
        and event.name == "code-intelligence"
        and "OSError" in (event.detail or "")
        and "index unavailable" not in (event.detail or "")
        for event in run.events
    )


async def test_required_context_failure_stops_before_the_model_call() -> None:
    provider = ScriptedProvider(answer())
    contributor = Contributor(required=True, failure=OSError("tenant source leaked here"))
    runner = AgentRunner(provider=provider, context_contributors=(contributor,))

    run = await runner.run(AGENT, "fix authorization", tenant="acme")

    assert run.state is RunState.FAILED
    assert provider.requests == []
    assert any(event.kind is RunEventKind.CONTEXT_DEGRADED for event in run.events)
    assert all("tenant source leaked here" not in (event.detail or "") for event in run.events)


async def test_context_contributor_protocol_is_structural() -> None:
    assert isinstance(Contributor(), ContextContributor)
