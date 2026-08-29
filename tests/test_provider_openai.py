"""The OpenAI adapter, against recorded wire traffic.

Structured output goes through `response_format`, which is the vendor's own mechanism.
`strict` is asserted only for a schema that actually meets the vendor's strict subset:
claiming it for a schema that does not is a 400 on the first real request.
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
    ToolDeclaration,
)
from tesserix_adk.models import ModelCapabilities
from tesserix_adk.models.providers import OpenAIProvider
from tesserix_adk.testing import HttpCassette, HttpReplay
from tesserix_adk.testing.http_cassette import HttpExchange

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core import StreamEvent

MODEL = "gpt-4o"
COMPLETIONS = "/v1/chat/completions"

STRICT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class Secrets:
    def __init__(self, key: str | None = "sk-test-key") -> None:
        self._key = key

    def secret(self, name: str) -> str | None:
        return self._key if name == "OPENAI_API_KEY" else None


def _completion(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "it rained"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    } | overrides


def answered(**body: Any) -> HttpCassette:
    return HttpCassette(
        provider="openai", exchanges=(HttpExchange(path=COMPLETIONS, body=_completion(**body)),)
    )


def choice(message: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}


def provider(cassette: HttpCassette, **kwargs: Any) -> tuple[OpenAIProvider, HttpReplay]:
    replay = HttpReplay(cassette, expect_provider="openai")
    return (
        OpenAIProvider(MODEL, secrets=Secrets(), transport=replay.transport, **kwargs),
        replay,
    )


def asked(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(model=MODEL, messages=messages or (user("did it rain"),), **kwargs)


def user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


async def collected(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class TestWhatTheAdapterSends:
    async def test_the_key_travels_as_a_bearer_token(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked())
        assert replay.sent[0].headers["authorization"] == "Bearer sk-test-key"

    async def test_a_missing_key_is_refused_before_the_request(self) -> None:
        replay = HttpReplay(answered())
        model = OpenAIProvider(MODEL, secrets=Secrets(None), transport=replay.transport)
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY") as refused:
            await model.complete(asked())
        assert "FakeModelProvider" in str(refused.value)
        assert "offline" in str(refused.value)
        assert replay.sent == []

    async def test_a_system_prompt_stays_a_turn_of_its_own(self) -> None:
        """OpenAI takes a system role, so nothing is folded into the user's words."""
        model, replay = provider(answered())
        await model.complete(
            asked(Message(role="system", content=[TextPart(text="Be brief.")]), user("hi"))
        )
        turns = (replay.sent[0].body or {})["messages"]
        assert [turn["role"] for turn in turns] == ["system", "user"]
        assert turns[0]["content"] == "Be brief."

    async def test_an_image_travels_as_a_data_url_part(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(
                Message(
                    role="user",
                    content=[
                        TextPart(text="what is this"),
                        BinaryPart(media_type="image/png", data=b"PNG"),
                    ],
                )
            )
        )
        parts = (replay.sent[0].body or {})["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "what is this"}
        assert parts[1]["image_url"] == {"url": "data:image/png;base64,UE5H"}

    async def test_an_image_for_a_model_without_vision_is_refused(self) -> None:
        model, replay = provider(answered(), capabilities=ModelCapabilities(tool_calling=True))
        with pytest.raises(CapabilityError, match="vision"):
            await model.complete(
                asked(Message(role="user", content=[BinaryPart(media_type="image/png", data=b"x")]))
            )
        assert replay.sent == []

    async def test_tools_are_declared_as_functions(self) -> None:
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
        assert declared["type"] == "function"
        assert declared["function"]["name"] == "lookup"

    async def test_an_assistant_tool_call_is_replayed_with_its_arguments_as_json_text(self) -> None:
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
        assert turns[1]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "rain"}'},
            }
        ]
        assert turns[2] == {"role": "tool", "tool_call_id": "call_1", "content": "yes"}

    async def test_a_structured_request_uses_response_format(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked(output_schema=STRICT_SCHEMA))
        response_format = (replay.sent[0].body or {})["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == STRICT_SCHEMA

    async def test_strict_is_claimed_only_for_a_schema_that_meets_the_strict_subset(self) -> None:
        """The vendor rejects a schema that says strict and is not, on the first request."""
        model, replay = provider(answered())
        await model.complete(asked(output_schema=STRICT_SCHEMA))
        assert (replay.sent[0].body or {})["response_format"]["json_schema"]["strict"] is True

    async def test_a_schema_that_allows_extra_keys_does_not_claim_strict(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(output_schema={"type": "object", "properties": {"answer": {"type": "string"}}})
        )
        assert (replay.sent[0].body or {})["response_format"]["json_schema"]["strict"] is False


class TestWhatTheAdapterReadsBack:
    async def test_the_message_content_becomes_the_content(self) -> None:
        model, _ = provider(answered())
        assert (await model.complete(asked())).content == "it rained"

    async def test_a_tool_call_is_read_back_with_its_arguments_parsed(self) -> None:
        model, _ = provider(
            answered(
                **choice(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q": "rain"}'},
                            }
                        ],
                    },
                    "tool_calls",
                )
            )
        )
        response = await model.complete(
            asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
        )
        assert response.tool_calls == (
            ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),
        )
        assert response.stop_reason is StopReason.TOOL_CALLS

    async def test_arguments_that_never_became_json_are_refused(self) -> None:
        """Guessing at half an object runs a tool with arguments nobody sent."""
        model, _ = provider(
            answered(
                **choice(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": '{"q": '},
                            }
                        ],
                    },
                    "tool_calls",
                )
            )
        )
        with pytest.raises(ModelResponseError, match="lookup"):
            await model.complete(asked())

    async def test_a_refusal_is_reported_as_a_refusal(self) -> None:
        model, _ = provider(
            answered(**choice({"role": "assistant", "content": None, "refusal": "I can't help."}))
        )
        response = await model.complete(asked())
        assert response.stop_reason is StopReason.REFUSAL
        assert response.content == "I can't help."

    async def test_cached_prompt_tokens_are_recorded(self) -> None:
        model, _ = provider(
            answered(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 90},
                }
            )
        )
        usage = (await model.complete(asked())).usage
        assert (usage.input_tokens, usage.cached_tokens) == (100, 90)

    async def test_a_priced_model_reports_what_the_call_cost(self) -> None:
        model, _ = provider(answered())
        assert (await model.complete(asked())).usage.cost is not None

    @pytest.mark.parametrize(
        ("vendor", "expected"),
        [
            ("stop", StopReason.END_TURN),
            ("length", StopReason.MAX_TOKENS),
            ("tool_calls", StopReason.TOOL_CALLS),
            ("function_call", StopReason.TOOL_CALLS),
            ("content_filter", StopReason.SAFETY),
            ("something_new", StopReason.UNKNOWN),
        ],
    )
    async def test_every_finish_reason_lands_in_the_kits_taxonomy(
        self, vendor: str, expected: StopReason
    ) -> None:
        model, _ = provider(answered(**choice({"role": "assistant", "content": "x"}, vendor)))
        assert (await model.complete(asked())).stop_reason is expected

    async def test_a_body_with_no_choices_is_refused(self) -> None:
        model, _ = provider(answered(choices=[]))
        with pytest.raises(ModelResponseError, match="choice"):
            await model.complete(asked())


class TestWhenTheVendorFails:
    async def test_a_rate_limit_carries_the_wait_the_vendor_asked_for(self) -> None:
        model, _ = provider(
            HttpCassette(
                provider="openai",
                exchanges=(
                    HttpExchange(
                        path=COMPLETIONS,
                        status=429,
                        headers={"retry-after": "12"},
                        body={"error": {"type": "rate_limit_exceeded"}},
                    ),
                ),
            )
        )
        with pytest.raises(ProviderError) as failure:
            await model.complete(asked())
        assert (failure.value.status, failure.value.retry_after) == (429, 12.0)


class TestStreaming:
    def _stream(self, *lines: str) -> HttpCassette:
        return HttpCassette(
            provider="openai", exchanges=(HttpExchange(path=COMPLETIONS, stream=lines),)
        )

    async def test_usage_is_asked_for_so_a_streamed_call_is_still_costed(self) -> None:
        """Without it the vendor sends no usage at all and a streamed run reports no spend."""
        model, replay = provider(
            self._stream(
                'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            )
        )
        await collected(await model.stream(asked()))
        assert (replay.sent[0].body or {})["stream_options"] == {"include_usage": True}

    async def test_text_arrives_in_order_and_the_last_event_carries_the_whole_answer(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"it "}}]}',
                'data: {"choices":[{"index":0,"delta":{"content":"rained"},'
                '"finish_reason":"stop"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4}}',
                "data: [DONE]",
            )
        )
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it ", "rained"]
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.content == "it rained"
        assert end.response.usage.input_tokens == 10

    async def test_the_streamed_answer_matches_the_buffered_one(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"choices":[{"index":0,"delta":{"content":"it rained"},'
                '"finish_reason":"stop"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4}}',
                "data: [DONE]",
            )
        )
        streamed = (await collected(await model.stream(asked())))[-1]
        buffered = await provider(answered())[0].complete(asked())
        assert isinstance(streamed, StreamEnd)
        assert streamed.response == buffered

    async def test_tool_arguments_arrive_a_fragment_at_a_time(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                '"function":{"name":"lookup","arguments":""}}]}}]}',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"q\\":"}}]}}]}',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"rain\\"}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            )
        )
        end = (
            await collected(
                await model.stream(
                    asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
                )
            )
        )[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.tool_calls == (
            ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),
        )

    async def test_a_stream_that_stops_before_the_end_is_refused(self) -> None:
        model, _ = provider(
            self._stream('data: {"choices":[{"index":0,"delta":{"content":"it rai"}}]}')
        )
        with pytest.raises(StreamInterruptedError) as broken:
            await collected(await model.stream(asked()))
        assert broken.value.partial == "it rai"

    async def test_an_error_frame_becomes_a_provider_error(self) -> None:
        model, _ = provider(
            self._stream('data: {"error":{"type":"server_error","message":"upstream fell over"}}')
        )
        with pytest.raises(ProviderError, match="server_error"):
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
        assert OpenAIProvider("gpt-from-the-future", secrets=Secrets()).capabilities == (
            ModelCapabilities()
        )

    async def test_a_compatible_gateway_is_reachable_by_base_url(self) -> None:
        """The same wire format is served by proxies, and the provider name stays honest."""
        replay = HttpReplay(answered())
        model = OpenAIProvider(
            MODEL,
            secrets=Secrets(),
            transport=replay.transport,
            base_url="https://gateway.internal.test",
        )
        assert (await model.complete(asked())).content == "it rained"
        assert model.name == "openai"


class TestTheAnswersThatCannotBeUsed:
    async def test_a_streamed_reasoning_delta_is_reasoning_and_not_content(self) -> None:
        cassette = HttpCassette(
            provider="openai",
            exchanges=(
                HttpExchange(
                    path=COMPLETIONS,
                    stream=(
                        'data: {"choices":[{"index":0,'
                        '"delta":{"reasoning_content":"weighing it up"}}]}',
                        'data: {"choices":[{"index":0,"delta":{"content":"it rained"},'
                        '"finish_reason":"stop"}]}',
                    ),
                ),
            ),
        )
        model, _ = provider(cassette)
        events = await collected(await model.stream(asked()))
        reasoning = [event for event in events if isinstance(event, ReasoningDelta)]
        assert [event.text for event in reasoning] == ["weighing it up"]

    async def test_a_tool_call_with_no_id_is_refused(self) -> None:
        """A result is matched back by id, so a call without one can never be answered."""
        model, _ = provider(
            answered(
                **choice(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                    "tool_calls",
                )
            )
        )
        with pytest.raises(ModelResponseError, match="no id or no name"):
            await model.complete(asked())

    async def test_arguments_that_are_not_json_are_refused(self) -> None:
        model, _ = provider(_call_answering('{"q": '))
        with pytest.raises(ModelResponseError, match="not JSON"):
            await model.complete(asked())

    async def test_arguments_that_are_not_an_object_are_refused(self) -> None:
        """A bare list is not a keyword argument, and guessing which one it is invents a call."""
        model, _ = provider(_call_answering('["rain"]'))
        with pytest.raises(ModelResponseError, match="not an object"):
            await model.complete(asked())


class TestWhichSchemasCanClaimStrict:
    async def test_a_schema_that_is_not_an_object_cannot(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked(output_schema={"type": "string"}))
        assert _strict(replay) is False

    async def test_a_schema_that_allows_extra_keys_cannot(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            )
        )
        assert _strict(replay) is False

    async def test_a_nested_object_that_allows_extra_keys_cannot(self) -> None:
        """The vendor's rule is recursive, and a check that stops at the top is a 400."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                output_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["place"],
                    "properties": {
                        "place": {"type": "object", "properties": {"city": {"type": "string"}}}
                    },
                }
            )
        )
        assert _strict(replay) is False

    async def test_an_array_of_open_objects_cannot(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(
                output_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["places"],
                    "properties": {
                        "places": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    },
                }
            )
        )
        assert _strict(replay) is False

    async def test_a_tool_called_with_no_arguments_at_all_is_called_with_none(self) -> None:
        """A no-argument tool arrives as an empty string, which is not a broken object."""
        model, _ = provider(_call_answering(""))
        response = await model.complete(asked())
        assert [(call.name, call.arguments) for call in response.tool_calls] == [("lookup", {})]

    async def test_a_schema_that_allows_extra_properties_is_not_strict(self) -> None:
        """The vendor's strict mode forbids them, so claiming it would have the call refused."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                output_schema={
                    "type": "object",
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}},
                }
            )
        )
        assert _strict(replay) is False

    async def test_a_schema_with_an_optional_property_is_not_strict(self) -> None:
        """Strict mode requires every property, so an optional one puts the schema outside it."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                output_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}, "note": {"type": "string"}},
                }
            )
        )
        assert _strict(replay) is False


def _call_answering(arguments: str) -> HttpCassette:
    return answered(
        **choice(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            },
            "tool_calls",
        )
    )


def _strict(replay: HttpReplay) -> bool:
    body = replay.sent[0].body or {}
    strict = body["response_format"]["json_schema"]["strict"]
    assert isinstance(strict, bool)
    return strict
