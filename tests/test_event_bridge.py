"""A watched run's progress, republished as events other systems can consume."""

from __future__ import annotations

from typing import Any

from tesserix_adk.adapters import payload_of, publishing
from tesserix_adk.core import (
    Agent,
    Delivery,
    Eventing,
    EventType,
    ToolCall,
    Usage,
    tenant_scope,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.runtime.progress import IterationStarted, ToolCallFailed
from tesserix_adk.testing import (
    CAPABLE,
    FakeClock,
    FakeToolRegistry,
    InMemoryEventPublisher,
    ScriptedProvider,
    assert_events,
)

TENANT = "acme"


def _answer(text: str = "Kyoto, four nights.", **overrides: Any) -> ModelResponse:
    fields: dict[str, Any] = {"content": text, "usage": Usage(input_tokens=10, output_tokens=5)}
    return ModelResponse(**{**fields, **overrides})


def _calling(*names: str) -> ModelResponse:
    calls = tuple(
        ToolCall(id=f"c{index}", name=name, arguments={"q": "kyoto"})
        for index, name in enumerate(names, start=1)
    )
    return _answer("", tool_calls=calls)


def _runner(*responses: ModelResponse, **overrides: Any) -> AgentRunner:
    fields: dict[str, Any] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})


async def _watched(runner: AgentRunner, agent: Agent) -> InMemoryEventPublisher:
    published = InMemoryEventPublisher()
    eventing = Eventing(published, clock=FakeClock(), delivery=Delivery.BEST_EFFORT)
    with tenant_scope(TENANT):
        stream = runner.stream(agent, "plan a trip", tenant=TENANT, run_id="run_1")
        async for _ in publishing(stream, eventing):
            pass
    return published


class TestARunRepublishedAsEvents:
    async def test_a_run_that_calls_two_tools_emits_the_whole_sequence_in_order(self) -> None:
        """Two parallel-safe calls are dispatched together, so both starts precede both finishes."""
        published = await _watched(
            _runner(
                _calling("lookup", "price"),
                _answer(),
                tools=FakeToolRegistry({"lookup": lambda q: q, "price": lambda q: q}),
            ),
            Agent(
                name="planner",
                instructions="Plan trips.",
                free_text=True,
                model="claude-sonnet-5",
                tools=("lookup", "price"),
            ),
        )
        assert_events(
            published.events,
            EventType.RUN_STARTED,
            EventType.TOOL_CALL_REQUESTED,
            EventType.TOOL_CALL_REQUESTED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.RUN_COMPLETED,
        )

    async def test_every_envelope_carries_the_tenant_and_the_run(self) -> None:
        published = await _watched(
            _runner(_answer()),
            Agent(
                name="planner", instructions="Plan trips.", free_text=True, model="claude-sonnet-5"
            ),
        )
        assert {(event.tenant, event.run_id) for event in published.events} == {(TENANT, "run_1")}

    async def test_no_payload_carries_the_answer_or_any_argument(self) -> None:
        published = await _watched(
            _runner(_calling("lookup"), _answer(), tools=FakeToolRegistry({"lookup": lambda q: q})),
            Agent(
                name="planner",
                instructions="Plan trips.",
                free_text=True,
                model="claude-sonnet-5",
                tools=("lookup",),
            ),
        )
        recorded = " ".join(
            value for event in published.events for value in event.attributes.values()
        )
        assert "kyoto" not in recorded.lower()
        assert "Kyoto" not in recorded

    async def test_the_events_of_one_run_form_a_causal_chain(self) -> None:
        published = await _watched(
            _runner(_answer()),
            Agent(
                name="planner", instructions="Plan trips.", free_text=True, model="claude-sonnet-5"
            ),
        )
        assert published.events[0].causation_id == ""
        assert published.events[1].causation_id == published.events[0].event_id


class TestWhatIsWorthPublishing:
    def test_a_progress_event_with_no_downstream_meaning_is_not_republished(self) -> None:
        assert payload_of(IterationStarted(run_id="run_1", iteration=1)) is None

    def test_a_failed_tool_call_is_still_a_completed_call_with_a_code(self) -> None:
        payload = payload_of(
            ToolCallFailed(run_id="run_1", call_id="c1", tool="lookup", error="timeout")
        )
        assert payload is not None
        assert payload.attributes()["state"] == "failed"
        assert payload.attributes()["error_code"] == "timeout"
