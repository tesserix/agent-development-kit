"""One event vocabulary for every vendor's stream, and one taxonomy for why it stopped.

Each vendor emits its own event shapes and its own stop strings. A consumer that branches
on them is a consumer that has integrated a vendor rather than the kit, so the translation
happens once, here, and what leaves a provider is the same either way.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    ModelResponse,
    ModelResponseError,
    ReasoningDelta,
    StopReason,
    StreamAccumulator,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageDelta,
    telemetry_dump,
)


class TestAStopReasonIsATaxonomyNotAVendorString:
    def test_every_reason_the_kit_knows_is_a_member(self) -> None:
        assert {reason.value for reason in StopReason} == {
            "end_turn",
            "max_tokens",
            "tool_calls",
            "refusal",
            "safety",
            "unknown",
        }

    def test_a_response_that_says_nothing_else_ended_its_turn(self) -> None:
        assert ModelResponse(content="hello").stop_reason is StopReason.END_TURN

    def test_a_reason_no_vendor_map_covers_is_named_unknown(self) -> None:
        """Guessing `end_turn` for a string nobody has seen reports a truncation as an answer."""
        assert ModelResponse(content="", stop_reason=StopReason.UNKNOWN).stop_reason.value == (
            "unknown"
        )


class TestReasoningIsNeitherReplayedNorReported:
    def test_reasoning_is_kept_apart_from_the_answer(self) -> None:
        response = ModelResponse(content="42", reasoning="first I counted")
        assert response.content == "42"
        assert response.reasoning == "first I counted"

    def test_reasoning_never_reaches_telemetry(self) -> None:
        dumped = telemetry_dump(ModelResponse(content="42", reasoning="first I counted"))
        assert "reasoning" not in dumped
        assert dumped["content"] == "42"


class TestEveryEventIsOneOfFive:
    def test_each_event_is_a_stream_event(self) -> None:
        events: list[StreamEvent] = [
            TextDelta(text="a"),
            ReasoningDelta(text="b"),
            ToolCallDelta(index=0, id="call_1", name="lookup", arguments='{"q":'),
            UsageDelta(usage=Usage(input_tokens=1, output_tokens=2)),
            StreamEnd(response=ModelResponse(content="a")),
        ]
        assert all(isinstance(event, StreamEvent) for event in events)

    def test_a_text_delta_carries_text(self) -> None:
        assert TextDelta(text="a").text == "a"

    def test_a_reasoning_delta_never_reaches_telemetry_either(self) -> None:
        assert "text" not in telemetry_dump(ReasoningDelta(text="thinking"))


class TestAStreamIsAssembledIntoTheAnswerItWas:
    def test_text_deltas_concatenate_in_arrival_order(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(TextDelta(text="hel"))
        accumulator.feed(TextDelta(text="lo"))
        assert accumulator.finish(StopReason.END_TURN).content == "hello"

    def test_reasoning_deltas_stay_out_of_the_content(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(ReasoningDelta(text="hmm"))
        accumulator.feed(TextDelta(text="hello"))
        response = accumulator.finish(StopReason.END_TURN)
        assert response.content == "hello"
        assert response.reasoning == "hmm"

    def test_tool_call_arguments_arrive_in_fragments_and_are_parsed_once(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, id="call_1", name="lookup", arguments='{"q": "'))
        accumulator.feed(ToolCallDelta(index=0, arguments='rain"}'))
        (call,) = accumulator.finish(StopReason.TOOL_CALLS).tool_calls
        assert (call.id, call.name, call.arguments) == ("call_1", "lookup", {"q": "rain"})

    def test_parallel_calls_are_kept_apart_by_index(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, id="a", name="lookup", arguments="{}"))
        accumulator.feed(ToolCallDelta(index=1, id="b", name="search", arguments="{}"))
        assert [call.id for call in accumulator.finish(StopReason.TOOL_CALLS).tool_calls] == [
            "a",
            "b",
        ]

    def test_usage_replaces_rather_than_accumulates(self) -> None:
        """Vendors report a running total, so adding them up bills the last token twice."""
        accumulator = StreamAccumulator()
        accumulator.feed(UsageDelta(usage=Usage(input_tokens=10, output_tokens=1)))
        accumulator.feed(UsageDelta(usage=Usage(input_tokens=10, output_tokens=4)))
        assert accumulator.finish(StopReason.END_TURN).usage.output_tokens == 4

    def test_the_stop_reason_is_carried_onto_the_response(self) -> None:
        assert (
            StreamAccumulator().finish(StopReason.MAX_TOKENS).stop_reason is StopReason.MAX_TOKENS
        )

    def test_arguments_that_never_became_json_are_refused(self) -> None:
        """A truncated argument object completed by guessing runs a tool nobody asked for."""
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, id="a", name="lookup", arguments='{"q": "rai'))
        with pytest.raises(ModelResponseError) as refused:
            accumulator.finish(StopReason.TOOL_CALLS)
        assert refused.value.payload == '{"q": "rai'

    def test_arguments_that_are_json_but_not_an_object_are_refused(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, id="a", name="lookup", arguments='"rain"'))
        with pytest.raises(ModelResponseError):
            accumulator.finish(StopReason.TOOL_CALLS)

    def test_a_call_with_no_arguments_at_all_is_an_empty_object(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, id="a", name="lookup"))
        (call,) = accumulator.finish(StopReason.TOOL_CALLS).tool_calls
        assert call.arguments == {}

    def test_a_fragment_that_names_no_tool_is_refused(self) -> None:
        """An id with no name is a call the kit cannot route, and inventing one is worse."""
        accumulator = StreamAccumulator()
        accumulator.feed(ToolCallDelta(index=0, arguments="{}"))
        with pytest.raises(ModelResponseError):
            accumulator.finish(StopReason.TOOL_CALLS)

    def test_a_terminal_event_adds_nothing_to_what_it_terminates(self) -> None:
        """`StreamEnd` carries the assembled answer, so feeding it back would double it."""
        accumulator = StreamAccumulator()
        accumulator.feed(TextDelta(text="it rained"))
        settled = accumulator.finish(StopReason.END_TURN)
        accumulator.feed(StreamEnd(response=settled))
        assert accumulator.finish(StopReason.END_TURN).content == "it rained"
