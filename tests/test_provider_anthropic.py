"""The Anthropic adapter, against recorded wire traffic.

Two things are under test and the first is the one a provider-level recording cannot see:
what the adapter *sent*. A response translated correctly from a request that dropped the
system prompt is still a broken adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    BinaryPart,
    CapabilityError,
    ConfigurationError,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ProviderError,
    ReasoningDelta,
    StopReason,
    StreamEnd,
    StreamInterruptedError,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDeclaration,
)
from tesserix_adk.models import ModelCapabilities
from tesserix_adk.models.providers import AnthropicProvider
from tesserix_adk.testing import HttpCassette, HttpReplay
from tesserix_adk.testing.http_cassette import HttpExchange

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core import StreamEvent

MODEL = "claude-sonnet-4-5"


class Secrets:
    def __init__(self, key: str | None = "sk-test-key") -> None:
        self._key = key

    def secret(self, name: str) -> str | None:
        return self._key if name == "ANTHROPIC_API_KEY" else None


def answered(**body: Any) -> HttpCassette:
    return HttpCassette(
        provider="anthropic",
        exchanges=(HttpExchange(path="/v1/messages", body=_message(**body)),),
    )


def _message(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "it rained"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    } | overrides


def provider(cassette: HttpCassette, **kwargs: Any) -> tuple[AnthropicProvider, HttpReplay]:
    replay = HttpReplay(cassette, expect_provider="anthropic")
    return (
        AnthropicProvider(MODEL, secrets=Secrets(), transport=replay.transport, **kwargs),
        replay,
    )


def asked(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(model=MODEL, messages=messages or (user("did it rain"),), **kwargs)


def user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


async def collected(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class TestWhatTheAdapterSends:
    async def test_the_key_and_the_api_version_travel_in_headers(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked())
        headers = replay.sent[0].headers
        assert headers["x-api-key"] == "sk-test-key"
        assert headers["anthropic-version"]

    async def test_a_missing_key_is_refused_before_the_request(self) -> None:
        replay = HttpReplay(answered())
        model = AnthropicProvider(MODEL, secrets=Secrets(None), transport=replay.transport)
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            await model.complete(asked())
        assert replay.sent == []

    async def test_the_system_prompt_becomes_the_system_field_not_a_turn(self) -> None:
        """Anthropic takes no system role, and a system prompt sent as a user turn is a lie."""
        model, replay = provider(answered())
        await model.complete(
            asked(Message(role="system", content=[TextPart(text="Be brief.")]), user("hi"))
        )
        body = replay.sent[0].body or {}
        assert body["system"] == "Be brief."
        assert [turn["role"] for turn in body["messages"]] == ["user"]

    async def test_text_becomes_a_text_block(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked())
        body = replay.sent[0].body or {}
        assert body["messages"][0]["content"] == [{"type": "text", "text": "did it rain"}]

    async def test_an_output_ceiling_is_always_sent(self) -> None:
        """The vendor rejects a request without one, so it is never left to a default."""
        model, replay = provider(answered())
        await model.complete(asked())
        assert (replay.sent[0].body or {})["max_tokens"] > 0

    async def test_an_image_becomes_a_base64_image_block(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(Message(role="user", content=[BinaryPart(media_type="image/png", data=b"PNG")]))
        )
        block = (replay.sent[0].body or {})["messages"][0]["content"][0]
        assert block["type"] == "image"
        assert block["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": "UE5H",
        }

    async def test_an_image_for_a_model_without_vision_is_refused(self) -> None:
        model, replay = provider(answered(), capabilities=ModelCapabilities(tool_calling=True))
        with pytest.raises(CapabilityError, match="vision"):
            await model.complete(
                asked(Message(role="user", content=[BinaryPart(media_type="image/png", data=b"x")]))
            )
        assert replay.sent == []

    async def test_tools_are_declared_with_their_schema(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(
                tools=(
                    ToolDeclaration(
                        name="lookup",
                        description="Look it up",
                        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                    ),
                )
            )
        )
        declared = (replay.sent[0].body or {})["tools"][0]
        assert declared["name"] == "lookup"
        assert declared["input_schema"]["properties"] == {"q": {"type": "string"}}

    async def test_an_assistant_tool_call_is_replayed_as_a_tool_use_block(self) -> None:
        """The vendor matches a result to the call by id, so the call must be in the history."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                user("did it rain"),
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),),
                ),
                Message(role="tool", tool_call_id="call_1", content=[TextPart(text="yes")]),
            )
        )
        turns = (replay.sent[0].body or {})["messages"]
        assert turns[1]["content"][0] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "lookup",
            "input": {"q": "rain"},
        }
        assert turns[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "yes"}],
        }

    async def test_parallel_results_share_one_turn(self) -> None:
        """The vendor requires every result for a turn in a single user message."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                user("two things"),
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(id="call_1", name="lookup"),
                        ToolCall(id="call_2", name="lookup"),
                    ),
                ),
                Message(role="tool", tool_call_id="call_1", content=[TextPart(text="a")]),
                Message(role="tool", tool_call_id="call_2", content=[TextPart(text="b")]),
            )
        )
        turns = (replay.sent[0].body or {})["messages"]
        assert len(turns) == 3
        assert len(turns[2]["content"]) == 2

    async def test_a_structured_request_forces_the_vendors_own_mechanism(self) -> None:
        """Anthropic's native structured output is a forced tool; nothing is parsed out of prose."""
        model, replay = provider(answered())
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        await model.complete(asked(output_schema=schema))
        body = replay.sent[0].body or {}
        assert body["tool_choice"]["type"] == "tool"
        assert body["tools"][-1]["input_schema"] == schema


class TestWhatTheAdapterReadsBack:
    async def test_text_blocks_become_the_content(self) -> None:
        model, _ = provider(answered())
        assert (await model.complete(asked())).content == "it rained"

    async def test_thinking_becomes_reasoning_and_not_content(self) -> None:
        model, _ = provider(
            answered(
                content=[
                    {"type": "thinking", "thinking": "checking the record"},
                    {"type": "text", "text": "it rained"},
                ]
            )
        )
        response = await model.complete(asked())
        assert response.reasoning == "checking the record"
        assert response.content == "it rained"

    async def test_a_tool_use_block_becomes_a_tool_call(self) -> None:
        model, _ = provider(
            answered(
                content=[
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "rain"}}
                ],
                stop_reason="tool_use",
            )
        )
        response = await model.complete(
            asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
        )
        assert response.tool_calls == (
            ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),
        )
        assert response.stop_reason is StopReason.TOOL_CALLS

    async def test_a_forced_structured_answer_comes_back_as_content(self) -> None:
        """The caller asked for a shape, not for a tool call it never declared."""
        model, _ = provider(
            answered(
                content=[
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "structured_output",
                        "input": {"answer": "yes"},
                    }
                ],
                stop_reason="tool_use",
            )
        )
        response = await model.complete(
            asked(output_schema={"type": "object", "properties": {"answer": {"type": "string"}}})
        )
        assert response.content == '{"answer": "yes"}'
        assert response.tool_calls == ()

    async def test_cache_reads_are_counted_as_input_and_recorded_as_cached(self) -> None:
        model, _ = provider(
            answered(usage={"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 90})
        )
        usage = (await model.complete(asked())).usage
        assert (usage.input_tokens, usage.cached_tokens) == (100, 90)

    async def test_a_priced_model_reports_what_the_call_cost(self) -> None:
        model, _ = provider(answered())
        usage = (await model.complete(asked())).usage
        assert usage.cost is not None
        assert usage.currency == "USD"

    @pytest.mark.parametrize(
        ("vendor", "expected"),
        [
            ("end_turn", StopReason.END_TURN),
            ("stop_sequence", StopReason.END_TURN),
            ("max_tokens", StopReason.MAX_TOKENS),
            ("tool_use", StopReason.TOOL_CALLS),
            ("refusal", StopReason.REFUSAL),
            ("something_new", StopReason.UNKNOWN),
            (None, StopReason.UNKNOWN),
        ],
    )
    async def test_every_stop_reason_lands_in_the_kits_taxonomy(
        self, vendor: str | None, expected: StopReason
    ) -> None:
        """A reason no map covers is unknown, never the nearest guess."""
        model, _ = provider(answered(stop_reason=vendor, content=[{"type": "text", "text": "x"}]))
        assert (await model.complete(asked())).stop_reason is expected

    async def test_a_body_that_is_not_a_message_is_refused(self) -> None:
        model, _ = provider(
            HttpCassette(
                provider="anthropic",
                exchanges=(HttpExchange(path="/v1/messages", body={"content": "a string"}),),
            )
        )
        with pytest.raises(ModelResponseError, match="content"):
            await model.complete(asked())

    async def test_a_tool_call_with_no_name_is_refused(self) -> None:
        model, _ = provider(answered(content=[{"type": "tool_use", "id": "call_1", "input": {}}]))
        with pytest.raises(ModelResponseError, match="name"):
            await model.complete(asked())


class TestWhenTheVendorFails:
    async def test_a_rate_limit_carries_the_wait_the_vendor_asked_for(self) -> None:
        model, _ = provider(
            HttpCassette(
                provider="anthropic",
                exchanges=(
                    HttpExchange(
                        path="/v1/messages",
                        status=429,
                        headers={"retry-after": "7", "request-id": "req_1"},
                        body={"error": {"type": "rate_limit_error"}},
                    ),
                ),
            )
        )
        with pytest.raises(ProviderError) as failure:
            await model.complete(asked())
        assert failure.value.status == 429
        assert failure.value.retry_after == 7.0


class TestStreaming:
    def _stream(self, *lines: str) -> HttpCassette:
        return HttpCassette(
            provider="anthropic",
            exchanges=(HttpExchange(path="/v1/messages", stream=lines),),
        )

    async def test_text_arrives_in_order_and_the_last_event_carries_the_whole_answer(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it "}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"rained"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":4}}',
                'data: {"type":"message_stop"}',
            )
        )
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it ", "rained"]
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.content == "it rained"
        assert end.response.stop_reason is StopReason.END_TURN

    async def test_the_streamed_answer_matches_the_buffered_one(self) -> None:
        """Two paths that disagree mean a consumer's choice of path changes the answer."""
        model, _ = provider(
            self._stream(
                'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it rained"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":4}}',
            )
        )
        streamed = (await collected(await model.stream(asked())))[-1]
        buffered = await provider(answered())[0].complete(asked())
        assert isinstance(streamed, StreamEnd)
        assert streamed.response == buffered

    async def test_tool_arguments_arrive_a_fragment_at_a_time(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                'data: {"type":"content_block_start","index":0,"content_block":'
                '{"type":"tool_use","id":"call_1","name":"lookup"}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"input_json_delta","partial_json":"\\"rain\\"}"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
                '"usage":{"output_tokens":4}}',
            )
        )
        events = await collected(
            await model.stream(
                asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
            )
        )
        assert any(isinstance(event, ToolCallDelta) for event in events)
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.tool_calls == (
            ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),
        )

    async def test_a_stream_that_stops_before_the_end_is_refused(self) -> None:
        """A truncated answer handed back whole is a wrong answer with nothing to show it."""
        model, _ = provider(
            self._stream(
                'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it rai"}}',
            )
        )
        with pytest.raises(StreamInterruptedError) as broken:
            await collected(await model.stream(asked()))
        assert broken.value.partial == "it rai"

    async def test_an_error_frame_becomes_a_provider_error(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}',
            )
        )
        with pytest.raises(ProviderError, match="overloaded_error"):
            await collected(await model.stream(asked()))

    async def test_streaming_a_model_that_never_declared_it_is_refused(self) -> None:
        model, replay = provider(answered(), capabilities=ModelCapabilities(tool_calling=True))
        with pytest.raises(CapabilityError, match="streaming"):
            await model.stream(asked())
        assert replay.sent == []


class TestTheProviderIsAWellBehavedClient:
    async def test_it_closes_its_pool(self) -> None:
        async with provider(answered())[0] as model:
            assert isinstance(await model.complete(asked()), ModelResponse)

    def test_a_model_the_catalogue_does_not_know_needs_its_capabilities_declared(self) -> None:
        """Guessing a capability moves the failure to the first request that needed it."""
        model = AnthropicProvider("claude-from-the-future", secrets=Secrets())
        assert model.capabilities == ModelCapabilities()

    def test_it_counts_tokens_from_the_text_it_was_given(self) -> None:
        model, _ = provider(answered())
        assert model.count_tokens((user("hello there, this is longer"),)) > model.count_tokens(
            (user("hi"),)
        )


class TestTheStreamedThingsThatAreNotText:
    def _stream(self, *lines: str) -> HttpCassette:
        return HttpCassette(
            provider="anthropic",
            exchanges=(HttpExchange(path="/v1/messages", stream=lines),),
        )

    async def test_a_thinking_delta_is_reasoning_and_not_content(self) -> None:
        """Replayed as content it becomes something the model never said to the user."""
        model, _ = provider(
            self._stream(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"thinking_delta","thinking":"weighing it up"}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it rained"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
            )
        )
        events = await collected(await model.stream(asked()))
        reasoning = [event for event in events if isinstance(event, ReasoningDelta)]
        assert [event.text for event in reasoning] == ["weighing it up"]
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.content == "it rained"

    async def test_a_frame_type_the_kit_does_not_know_is_passed_over(self) -> None:
        """A vendor adds event types, and a stream that dies on one is a fragile stream."""
        model, _ = provider(
            self._stream(
                'data: {"type":"ping"}',
                'data: {"type":"content_block_stop","index":0}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it rained"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
            )
        )
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it rained"]

    async def test_a_content_block_the_kit_does_not_know_is_passed_over(self) -> None:
        model, _ = provider(
            answered(
                content=[
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "text", "text": "it rained"},
                ]
            )
        )
        assert (await model.complete(asked())).content == "it rained"

    async def test_a_delta_type_the_kit_does_not_know_is_passed_over(self) -> None:
        """`signature_delta` carries a signature, not words, and is not part of the answer."""
        model, _ = provider(
            self._stream(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"signature_delta","signature":"abc"}}',
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"it rained"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
            )
        )
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it rained"]
