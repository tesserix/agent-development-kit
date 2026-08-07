"""What a consumer watching a run in flight is told, and in what order.

A stream of raw text chunks makes every UI guess: which tool is running, whether the
answer finished or the connection died, what it has cost so far. Here that is typed, so a
consumer renders progress from structure rather than from sniffing strings — and the
stream is the same run the buffered call returns, not a second account of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    ProviderError,
    Run,
    RunState,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.runtime.progress import (
    AnswerDelta,
    ApprovalRequired,
    GuardrailDecision,
    IterationStarted,
    ProgressEvent,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    SequenceCheck,
    StructuredDelta,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallStarted,
    UsageUpdated,
    decode_progress,
)
from tesserix_adk.testing import (
    CAPABLE,
    FakeClock,
    FakeGuardrail,
    FakeToolRegistry,
    ScriptedProvider,
    StallingProvider,
    estimate_tokens,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tesserix_adk.core import Message, ModelRequest, StreamEvent


class TripPlan(BaseModel):
    destination: str
    nights: int


class Dropping:
    """A provider whose stream ends where a connection died: text, and then nothing."""

    name = "dropping"
    capabilities = CAPABLE

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002 — scripted
        return answer()

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:  # noqa: ARG002
        return _cut_short()


async def _cut_short() -> AsyncIterator[StreamEvent]:
    yield TextDelta(text="Kyoto, ")


class Fragmenting:
    """A provider that sends one tool call's arguments in pieces, as a vendor does."""

    name = "fragmenting"
    capabilities = CAPABLE

    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002 — scripted
        return answer()

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:  # noqa: ARG002
        self.calls += 1
        return _in_fragments() if self.calls == 1 else _one_answer()


async def _in_fragments() -> AsyncIterator[StreamEvent]:
    call = ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"})
    for piece in ('{"q": ', '"kyo', 'to"}'):
        yield ToolCallDelta(index=0, id=call.id, name=call.name, arguments=piece)
    yield StreamEnd(response=answer("", tool_calls=(call,)))


async def _one_answer() -> AsyncIterator[StreamEvent]:
    yield StreamEnd(response=answer())


class Gate:
    """An approval gate that grants whatever it is asked."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        return ApprovalDecision(
            record_id=record.id,
            granted=True,
            decided_by="ada",
            decided_at=record.requested_at,
        )


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.", **overrides: object) -> ModelResponse:
    fields: dict[str, object] = {
        "content": text,
        "usage": Usage(input_tokens=10, output_tokens=5),
    }
    return ModelResponse(**{**fields, **overrides})  # type: ignore[arg-type]


def runner(*responses: ModelResponse | BaseException, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def watch(
    runner_: AgentRunner, agent_: Agent, text: str = "plan a trip"
) -> tuple[list[ProgressEvent], Run[Any]]:
    """Every event of one streamed run, and the run it produced."""
    stream = runner_.stream(agent_, text, tenant="acme", run_id="run_1")
    events = [event async for event in stream]
    return events, stream.run


def kinds(events: Sequence[ProgressEvent]) -> list[str]:
    return [event.kind for event in events]


class TestWhatAConsumerIsTold:
    async def test_a_run_opens_with_what_it_is_and_closes_with_how_it_went(self) -> None:
        events, _ = await watch(runner(answer()), agent())
        assert isinstance(events[0], RunStarted)
        assert (events[0].agent, events[0].model) == ("planner", "claude-sonnet-5")
        assert isinstance(events[-1], RunCompleted)

    async def test_the_answer_arrives_as_deltas_that_reassemble_to_the_buffered_one(self) -> None:
        events, run = await watch(runner(answer("Kyoto, four nights.")), agent())
        streamed = "".join(e.text for e in events if isinstance(e, AnswerDelta))
        assert streamed == "Kyoto, four nights."
        assert run.messages[-1].content[0].text == streamed  # type: ignore[union-attr]

    async def test_every_event_carries_the_run_it_belongs_to(self) -> None:
        events, _ = await watch(runner(answer()), agent())
        assert {event.run_id for event in events} == {"run_1"}

    async def test_sequence_numbers_are_contiguous_from_zero(self) -> None:
        """A consumer that cannot detect a gap cannot tell a slow stream from a lossy one."""
        events, _ = await watch(runner(answer()), agent())
        assert [event.sequence for event in events] == list(range(len(events)))

    async def test_each_iteration_is_announced(self) -> None:
        first = answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),))
        events, _ = await watch(
            runner(first, answer(), tools=FakeToolRegistry({"lookup": lambda q: q})),
            agent(tools=("lookup",)),
        )
        assert [e.iteration for e in events if isinstance(e, IterationStarted)] == [1, 2]

    async def test_spend_is_reported_while_the_run_is_still_going(self) -> None:
        """A long run attributable only at the end is unattributable while it matters."""
        events, run = await watch(runner(answer()), agent())
        reported = [event for event in events if isinstance(event, UsageUpdated)]
        assert reported
        assert reported[-1].usage == run.usage

    async def test_exactly_one_terminal_event_is_emitted_and_it_is_last(self) -> None:
        events, _ = await watch(runner(answer()), agent())
        terminal = [e for e in events if isinstance(e, RunCompleted | RunFailed | RunCancelled)]
        assert len(terminal) == 1
        assert terminal[0] is events[-1]

    async def test_the_streamed_run_is_the_run_the_buffered_call_would_have_returned(self) -> None:
        streamed, run = await watch(runner(answer()), agent())
        buffered = await runner(answer()).run(agent(), "plan a trip", tenant="acme", run_id="run_1")
        assert run.state is buffered.state
        assert run.usage == buffered.usage
        assert [e.kind for e in run.events] == [e.kind for e in buffered.events]
        assert kinds(streamed)[0] == "run_started"


class TestToolActivity:
    def _run(self) -> tuple[AgentRunner, Agent]:
        lookup = ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"})
        called = answer("", tool_calls=(lookup,))
        return (
            runner(called, answer(), tools=FakeToolRegistry({"lookup": lambda q: f"about {q}"})),
            agent(tools=("lookup",)),
        )

    async def test_a_tool_call_is_bracketed_by_a_start_and_a_finish(self) -> None:
        events, _ = await watch(*self._run())
        assert kinds(events).index("tool_call_started") < kinds(events).index("tool_call_finished")

    async def test_both_ends_name_the_same_call_id(self) -> None:
        """Two parallel calls to one tool differ only in arguments, so the id is the handle."""
        events, _ = await watch(*self._run())
        started = next(e for e in events if isinstance(e, ToolCallStarted))
        finished = next(e for e in events if isinstance(e, ToolCallFinished))
        assert started.call_id == finished.call_id == "c1"
        assert started.tool == finished.tool == "lookup"

    async def test_a_tool_that_raises_is_reported_as_a_failure_not_a_finish(self) -> None:
        def boom(**_: object) -> str:
            raise RuntimeError("the index is down")

        called = answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={}),))
        events, _ = await watch(
            runner(called, answer(), tools=FakeToolRegistry({"lookup": boom})),
            agent(tools=("lookup",)),
        )
        failed = next(e for e in events if isinstance(e, ToolCallFailed))
        assert failed.call_id == "c1"
        assert "the index is down" in failed.detail
        assert not [e for e in events if isinstance(e, ToolCallFinished)]

    async def test_a_truncated_result_still_finishes_and_says_so(self) -> None:
        called = answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={}),))
        events, _ = await watch(
            runner(
                called,
                answer(),
                tools=FakeToolRegistry({"lookup": lambda: "x" * 200}),
                max_tool_result_chars=10,
            ),
            agent(tools=("lookup",)),
        )
        finished = next(e for e in events if isinstance(e, ToolCallFinished))
        assert finished.truncated

    async def test_a_call_the_allowlist_refuses_never_reports_a_start(self) -> None:
        called = answer("", tool_calls=(ToolCall(id="c1", name="wire_money", arguments={}),))
        events, _ = await watch(
            runner(called, tools=FakeToolRegistry({"lookup": lambda: "ok"})),
            agent(tools=("lookup",)),
        )
        assert "tool_call_started" not in kinds(events)
        assert next(e for e in events if isinstance(e, ToolCallFailed)).tool == "wire_money"

    async def test_arguments_arriving_in_fragments_are_shown_only_once_complete(self) -> None:
        """Half an argument object is not an argument object, and rendering it says it is."""
        events, _ = await watch(
            runner(
                provider=Fragmenting(),
                tools=FakeToolRegistry({"lookup": lambda q: f"about {q}"}),
            ),
            agent(tools=("lookup",)),
        )
        started = next(e for e in events if isinstance(e, ToolCallStarted))
        assert started.arguments == '{"q": "kyoto"}'
        assert not [e for e in events if isinstance(e, AnswerDelta) and '{"q' in e.text]


class TestWhatNeverReachesAConsumer:
    async def test_a_credential_in_a_tool_argument_is_masked_before_it_is_emitted(self) -> None:
        """Redaction is the runtime's job: a transport that redacts has already logged it."""
        credential = "sk-live-01234567890abcdef"  # a fixture, not a credential; gitleaks:allow
        called = answer(
            "", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"token": credential}),)
        )
        events, _ = await watch(
            runner(called, answer(), tools=FakeToolRegistry({"lookup": lambda token: token})),
            agent(tools=("lookup",)),
        )
        started = next(e for e in events if isinstance(e, ToolCallStarted))
        assert credential not in started.arguments
        assert "[redacted]" in started.arguments

    async def test_the_answer_itself_is_not_scrubbed(self) -> None:
        """Deltas that no longer reassemble to the answer are a corrupted answer."""
        events, _ = await watch(runner(answer("write to ada@example.com")), agent())
        streamed = "".join(e.text for e in events if isinstance(e, AnswerDelta))
        assert streamed == "write to ada@example.com"


class TestATerminalOutcomeIsNeverGuessed:
    async def test_a_provider_that_drops_mid_stream_fails_the_run(self) -> None:
        events, run = await watch(runner(ProviderError("connection reset")), agent())
        assert run.state is RunState.FAILED
        assert isinstance(events[-1], RunFailed)
        assert "connection reset" in events[-1].detail

    async def test_nothing_accumulated_is_presented_as_a_finished_answer(self) -> None:
        events, _ = await watch(runner(ProviderError("connection reset")), agent())
        assert "run_completed" not in kinds(events)

    async def test_a_guardrail_refusal_is_reported_as_a_decision(self) -> None:
        events, _ = await watch(
            runner(answer(), guardrails={"pii": FakeGuardrail("pii", allow=False)}),
            agent(guardrails=("pii",)),
        )
        refusal = next(e for e in events if isinstance(e, GuardrailDecision))
        assert (refusal.guardrail, refusal.allowed) == ("pii", False)


class TestStructuredAnswers:
    async def test_a_structured_answer_streams_as_structured_deltas(self) -> None:
        """A consumer rendering a form needs to know it is being sent JSON, not prose."""
        payload = '{"destination": "Kyoto", "nights": 4}'
        events, run = await watch(
            runner(answer(payload)), agent(free_text=False, output_type=TripPlan)
        )
        assert "".join(e.fragment for e in events if isinstance(e, StructuredDelta)) == payload
        assert not [e for e in events if isinstance(e, AnswerDelta)]
        assert run.output == TripPlan(destination="Kyoto", nights=4)


class TestForwardCompatibility:
    def test_a_variant_this_version_has_never_heard_of_is_ignored(self) -> None:
        """Adding a variant is a minor release, so an older consumer must not break on one."""
        assert decode_progress({"kind": "quantum_delta", "run_id": "r", "sequence": 0}) is None

    def test_a_known_variant_decodes_to_its_own_type(self) -> None:
        payload = AnswerDelta(run_id="r", sequence=3, text="hi").model_dump()
        assert decode_progress(payload) == AnswerDelta(run_id="r", sequence=3, text="hi")

    def test_a_payload_that_is_not_an_event_at_all_is_ignored(self) -> None:
        assert decode_progress({"sequence": 0}) is None

    def test_a_known_variant_with_a_broken_payload_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ValueError, match="answer_delta"):
            decode_progress({"kind": "answer_delta", "run_id": "r", "sequence": "third"})


class TestDetectingLoss:
    def test_contiguous_events_are_accepted(self) -> None:
        check = SequenceCheck()
        assert all(check.accept(AnswerDelta(run_id="r", sequence=n, text="")) for n in range(3))
        assert check.missing == 0

    def test_a_gap_is_counted_rather_than_swallowed(self) -> None:
        check = SequenceCheck()
        check.accept(AnswerDelta(run_id="r", sequence=0, text=""))
        assert check.accept(AnswerDelta(run_id="r", sequence=3, text=""))
        assert check.missing == 2

    def test_an_event_that_arrives_late_is_rejected_rather_than_reordered(self) -> None:
        """Reordering a delta into place is how a UI renders an answer nobody wrote."""
        check = SequenceCheck()
        check.accept(AnswerDelta(run_id="r", sequence=1, text=""))
        assert not check.accept(AnswerDelta(run_id="r", sequence=0, text=""))

    def test_a_duplicate_is_rejected(self) -> None:
        check = SequenceCheck()
        check.accept(AnswerDelta(run_id="r", sequence=0, text=""))
        assert not check.accept(AnswerDelta(run_id="r", sequence=0, text=""))


class TestTheStreamItself:
    async def test_the_run_is_available_once_the_stream_is_drained(self) -> None:
        _, run = await watch(runner(answer()), agent())
        assert run.state is RunState.COMPLETED

    async def test_asking_for_the_run_before_it_finished_says_so_rather_than_lying(self) -> None:
        stream = runner(answer()).stream(agent(), "plan a trip", tenant="acme", run_id="run_1")
        with pytest.raises(RuntimeError, match="still running"):
            _ = stream.run
        assert [event async for event in stream]

    async def test_an_approval_hold_is_visible_to_whoever_has_to_answer_it(self) -> None:
        called = answer("", tool_calls=(ToolCall(id="c1", name="wire_money", arguments={}),))
        events, _ = await watch(
            runner(
                called,
                answer(),
                tools=FakeToolRegistry({"wire_money": lambda: "sent"}),
                approvals=Gate(),
            ),
            agent(tools=("wire_money",), approval_required_tools=("wire_money",)),
        )
        held = next(e for e in events if isinstance(e, ApprovalRequired))
        assert (held.tool, held.call_id) == ("wire_money", "c1")

    async def test_a_stream_is_watched_once_rather_than_replayed(self) -> None:
        """Events are consumed as they are read, so a second reader would see a part-run."""
        stream = runner(answer()).stream(agent(), "plan a trip", tenant="acme", run_id="run_1")
        assert [event async for event in stream]
        with pytest.raises(RuntimeError, match="already been consumed"):
            _ = [event async for event in stream]

    async def test_reasoning_reaches_the_accumulator_without_becoming_the_answer(self) -> None:
        events, _ = await watch(runner(answer(reasoning="weighing Kyoto against Osaka")), agent())
        streamed = "".join(e.text for e in events if isinstance(e, AnswerDelta))
        assert streamed == "Kyoto, four nights."

    async def test_a_stream_that_drops_before_the_end_is_a_failure_not_an_answer(self) -> None:
        """Accumulated text from a dropped connection is not an answer."""
        events, run = await watch(runner(provider=Dropping()), agent())
        assert run.state is RunState.FAILED
        assert isinstance(events[-1], RunFailed)

    async def test_a_provider_that_stalls_after_answering_still_streams_the_answer(self) -> None:
        events, run = await watch(runner(provider=StallingProvider(answer())), agent())
        assert run.state is RunState.COMPLETED
        streamed = "".join(e.text for e in events if isinstance(e, AnswerDelta))
        assert streamed == "Kyoto, four nights."

    async def test_a_cancelled_run_closes_with_a_cancellation_not_a_failure(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        stream = runner(provider=provider).stream(
            agent(), "plan a trip", tenant="acme", run_id="run_1", cancellation=token
        )
        events = []
        async for event in stream:
            events.append(event)
            if isinstance(event, IterationStarted) and event.iteration == 1:
                token.cancel("caller went away")
        assert stream.run.state is RunState.CANCELLED
        assert isinstance(events[-1], RunCancelled)
        provider.release()
