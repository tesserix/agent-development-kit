"""The Gemini adapter, against recorded wire traffic.

Gemini is the awkward one. It gives function calls no id and matches a result back by tool
name, so the adapter mints ids on the way out and resolves them again on the way in — and
a history with two calls to one tool is the case that proves it.
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
    UsageDelta,
)
from tesserix_adk.models import ModelCapabilities
from tesserix_adk.models.providers import GeminiProvider
from tesserix_adk.testing import HttpCassette, HttpReplay
from tesserix_adk.testing.http_cassette import HttpExchange

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core import StreamEvent

MODEL = "gemini-2.5-pro"
GENERATE = f"/v1beta/models/{MODEL}:generateContent"
STREAM = f"/v1beta/models/{MODEL}:streamGenerateContent"


class Secrets:
    def __init__(self, key: str | None = "test-key") -> None:
        self._key = key

    def secret(self, name: str) -> str | None:
        return self._key if name == "GEMINI_API_KEY" else None


def _generated(**overrides: Any) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "it rained"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4},
    } | overrides


def answered(**body: Any) -> HttpCassette:
    return HttpCassette(
        provider="gemini", exchanges=(HttpExchange(path=GENERATE, body=_generated(**body)),)
    )


def candidate(*parts: dict[str, Any], finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}]
    }


def provider(cassette: HttpCassette, **kwargs: Any) -> tuple[GeminiProvider, HttpReplay]:
    replay = HttpReplay(cassette, expect_provider="gemini")
    return GeminiProvider(MODEL, secrets=Secrets(), transport=replay.transport, **kwargs), replay


def asked(*messages: Message, **kwargs: Any) -> ModelRequest:
    return ModelRequest(model=MODEL, messages=messages or (user("did it rain"),), **kwargs)


def user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


async def collected(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class TestWhatTheAdapterSends:
    async def test_the_key_travels_in_a_header_and_never_in_the_url(self) -> None:
        """A key in a query string is a key in every access log between here and there."""
        model, replay = provider(answered())
        await model.complete(asked())
        assert replay.sent[0].headers["x-goog-api-key"] == "test-key"
        assert "test-key" not in replay.sent[0].path

    async def test_a_missing_key_is_refused_before_the_request(self) -> None:
        replay = HttpReplay(answered())
        model = GeminiProvider(MODEL, secrets=Secrets(None), transport=replay.transport)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            await model.complete(asked())
        assert replay.sent == []

    async def test_the_model_is_named_in_the_path(self) -> None:
        model, replay = provider(answered())
        await model.complete(asked())
        assert replay.sent[0].path == GENERATE

    async def test_the_system_prompt_becomes_a_system_instruction(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(Message(role="system", content=[TextPart(text="Be brief.")]), user("hi"))
        )
        body = replay.sent[0].body or {}
        assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
        assert [turn["role"] for turn in body["contents"]] == ["user"]

    async def test_an_assistant_turn_is_named_model(self) -> None:
        """Gemini has no assistant role, and an unknown role is a 400."""
        model, replay = provider(answered())
        await model.complete(
            asked(user("hi"), Message(role="assistant", content=[TextPart(text="hello")]))
        )
        assert [turn["role"] for turn in (replay.sent[0].body or {})["contents"]] == [
            "user",
            "model",
        ]

    async def test_an_image_becomes_inline_data(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(Message(role="user", content=[BinaryPart(media_type="image/png", data=b"PNG")]))
        )
        part = (replay.sent[0].body or {})["contents"][0]["parts"][0]
        assert part["inlineData"] == {"mimeType": "image/png", "data": "UE5H"}

    async def test_an_image_for_a_model_without_vision_is_refused(self) -> None:
        model, replay = provider(answered(), capabilities=ModelCapabilities(tool_calling=True))
        with pytest.raises(CapabilityError, match="vision"):
            await model.complete(
                asked(Message(role="user", content=[BinaryPart(media_type="image/png", data=b"x")]))
            )
        assert replay.sent == []

    async def test_tools_are_declared_as_function_declarations(self) -> None:
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
        declared = (replay.sent[0].body or {})["tools"][0]["functionDeclarations"][0]
        assert declared["name"] == "lookup"
        assert declared["parameters"]["properties"] == {"q": {"type": "string"}}

    async def test_schema_keywords_the_vendor_rejects_are_stripped(self) -> None:
        """The vendor 400s on `additionalProperties`, valid JSON Schema everywhere else."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                tools=(
                    ToolDeclaration(
                        name="lookup",
                        parameters={
                            "type": "object",
                            "additionalProperties": False,
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "properties": {"q": {"type": "string", "title": "Q"}},
                        },
                    ),
                )
            )
        )
        declared = (replay.sent[0].body or {})["tools"][0]["functionDeclarations"][0]
        assert "additionalProperties" not in declared["parameters"]
        assert "$schema" not in declared["parameters"]
        assert "title" not in declared["parameters"]["properties"]["q"]

    async def test_a_tool_result_is_matched_back_by_the_name_the_vendor_uses(self) -> None:
        """Gemini matches a result to its call by tool name, not by the id the kit holds."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                user("did it rain"),
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="lookup-0", name="lookup", arguments={"q": "rain"}),),
                ),
                Message(role="tool", tool_call_id="lookup-0", content=[TextPart(text="yes")]),
            )
        )
        turns = (replay.sent[0].body or {})["contents"]
        assert turns[1]["parts"][0]["functionCall"] == {"name": "lookup", "args": {"q": "rain"}}
        assert turns[2]["parts"][0]["functionResponse"] == {
            "name": "lookup",
            "response": {"result": "yes"},
        }

    async def test_a_structured_request_asks_for_json_with_a_schema(self) -> None:
        model, replay = provider(answered())
        await model.complete(
            asked(output_schema={"type": "object", "properties": {"answer": {"type": "string"}}})
        )
        config = (replay.sent[0].body or {})["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"]["properties"] == {"answer": {"type": "string"}}


class TestWhatTheAdapterReadsBack:
    async def test_text_parts_become_the_content(self) -> None:
        model, _ = provider(answered())
        assert (await model.complete(asked())).content == "it rained"

    async def test_a_thought_part_becomes_reasoning_and_not_content(self) -> None:
        model, _ = provider(
            answered(**candidate({"text": "weighing it", "thought": True}, {"text": "it rained"}))
        )
        response = await model.complete(asked())
        assert response.reasoning == "weighing it"
        assert response.content == "it rained"

    async def test_a_function_call_is_given_an_id_the_kit_can_match_a_result_to(self) -> None:
        """The vendor sends none, and a result with nothing to match cannot be placed."""
        model, _ = provider(
            answered(
                **candidate(
                    {"functionCall": {"name": "lookup", "args": {"q": "rain"}}}, finish="STOP"
                )
            )
        )
        response = await model.complete(
            asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
        )
        assert response.tool_calls[0].name == "lookup"
        assert response.tool_calls[0].id

    async def test_parallel_calls_to_one_tool_get_ids_of_their_own(self) -> None:
        model, _ = provider(
            answered(
                **candidate(
                    {"functionCall": {"name": "lookup", "args": {"q": "rain"}}},
                    {"functionCall": {"name": "lookup", "args": {"q": "snow"}}},
                )
            )
        )
        response = await model.complete(
            asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
        )
        identities = {call.id for call in response.tool_calls}
        assert len(identities) == 2

    async def test_a_call_reported_with_no_name_is_refused(self) -> None:
        model, _ = provider(answered(**candidate({"functionCall": {"args": {}}})))
        with pytest.raises(ModelResponseError, match="name"):
            await model.complete(asked())

    async def test_cached_content_is_counted_as_input_and_recorded_as_cached(self) -> None:
        model, _ = provider(
            answered(
                usageMetadata={
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 4,
                    "cachedContentTokenCount": 90,
                }
            )
        )
        usage = (await model.complete(asked())).usage
        assert (usage.input_tokens, usage.cached_tokens) == (100, 90)

    @pytest.mark.parametrize(
        ("vendor", "expected"),
        [
            ("STOP", StopReason.END_TURN),
            ("MAX_TOKENS", StopReason.MAX_TOKENS),
            ("SAFETY", StopReason.SAFETY),
            ("PROHIBITED_CONTENT", StopReason.SAFETY),
            ("RECITATION", StopReason.SAFETY),
            ("OTHER", StopReason.UNKNOWN),
        ],
    )
    async def test_every_finish_reason_lands_in_the_kits_taxonomy(
        self, vendor: str, expected: StopReason
    ) -> None:
        model, _ = provider(answered(**candidate({"text": "x"}, finish=vendor)))
        assert (await model.complete(asked())).stop_reason is expected

    async def test_a_turn_that_called_a_tool_reports_it_stopped_to_call_one(self) -> None:
        """The vendor says STOP either way, and the caller needs to know a tool is pending."""
        model, _ = provider(answered(**candidate({"functionCall": {"name": "lookup", "args": {}}})))
        response = await model.complete(
            asked(tools=(ToolDeclaration(name="lookup", parameters={"type": "object"}),))
        )
        assert response.stop_reason is StopReason.TOOL_CALLS

    async def test_a_body_with_no_candidate_is_refused(self) -> None:
        model, _ = provider(answered(candidates=[]))
        with pytest.raises(ModelResponseError, match="candidate"):
            await model.complete(asked())


class TestWhenTheVendorFails:
    async def test_a_rate_limit_carries_the_wait_the_vendor_asked_for(self) -> None:
        model, _ = provider(
            HttpCassette(
                provider="gemini",
                exchanges=(
                    HttpExchange(
                        path=GENERATE,
                        status=429,
                        headers={"retry-after": "30"},
                        body={"error": {"status": "RESOURCE_EXHAUSTED"}},
                    ),
                ),
            )
        )
        with pytest.raises(ProviderError) as failure:
            await model.complete(asked())
        assert (failure.value.status, failure.value.retry_after) == (429, 30.0)


class TestStreaming:
    def _stream(self, *lines: str) -> HttpCassette:
        return HttpCassette(provider="gemini", exchanges=(HttpExchange(path=STREAM, stream=lines),))

    async def test_it_asks_for_server_sent_events(self) -> None:
        model, replay = provider(
            self._stream(
                'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}'
            )
        )
        await collected(await model.stream(asked()))
        assert replay.sent[0].path == f"{STREAM}?alt=sse"

    async def test_text_arrives_in_order_and_the_last_event_carries_the_whole_answer(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"candidates":[{"content":{"parts":[{"text":"it "}]}}]}',
                'data: {"candidates":[{"content":{"parts":[{"text":"rained"}]},'
                '"finishReason":"STOP"}],'
                '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}}',
            )
        )
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it ", "rained"]
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.content == "it rained"

    async def test_the_streamed_answer_matches_the_buffered_one(self) -> None:
        model, _ = provider(
            self._stream(
                'data: {"candidates":[{"content":{"parts":[{"text":"it rained"}]},'
                '"finishReason":"STOP"}],'
                '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}}',
            )
        )
        streamed = (await collected(await model.stream(asked())))[-1]
        buffered = await provider(answered())[0].complete(asked())
        assert isinstance(streamed, StreamEnd)
        assert streamed.response == buffered

    async def test_a_function_call_arrives_whole_and_is_still_a_tool_call(self) -> None:
        """Gemini sends the arguments in one piece where the others send fragments."""
        model, _ = provider(
            self._stream(
                'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup",'
                '"args":{"q":"rain"}}}]},"finishReason":"STOP"}]}',
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
        assert end.response.tool_calls[0].arguments == {"q": "rain"}

    async def test_a_stream_that_stops_before_the_end_is_refused(self) -> None:
        model, _ = provider(
            self._stream('data: {"candidates":[{"content":{"parts":[{"text":"it rai"}]}}]}')
        )
        with pytest.raises(StreamInterruptedError) as broken:
            await collected(await model.stream(asked()))
        assert broken.value.partial == "it rai"

    async def test_an_error_frame_becomes_a_provider_error(self) -> None:
        model, _ = provider(
            self._stream('data: {"error":{"status":"UNAVAILABLE","message":"overloaded"}}')
        )
        with pytest.raises(ProviderError, match="UNAVAILABLE"):
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
        assert GeminiProvider("gemini-from-the-future", secrets=Secrets()).capabilities == (
            ModelCapabilities()
        )


class TestTheThingsOnlyGeminiDoes:
    async def test_parallel_results_share_one_turn(self) -> None:
        """Two results for one turn belong in one user content, as the vendor reads them."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                user("two things"),
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(id="lookup-0", name="lookup"),
                        ToolCall(id="weather-1", name="weather"),
                    ),
                ),
                Message(role="tool", tool_call_id="lookup-0", content=[TextPart(text="yes")]),
                Message(role="tool", tool_call_id="weather-1", content=[TextPart(text="clear")]),
            )
        )
        contents = (replay.sent[0].body or {})["contents"]
        assert [part["functionResponse"]["name"] for part in contents[-1]["parts"]] == [
            "lookup",
            "weather",
        ]

    async def test_a_result_after_text_starts_a_turn_of_its_own(self) -> None:
        """Appending it to a user's words would put a tool result in the user's mouth."""
        model, replay = provider(answered())
        await model.complete(
            asked(
                user("did it rain"),
                Message(role="assistant", tool_calls=(ToolCall(id="lookup-0", name="lookup"),)),
                Message(role="tool", tool_call_id="lookup-0", content=[TextPart(text="yes")]),
            )
        )
        contents = (replay.sent[0].body or {})["contents"]
        assert [turn["role"] for turn in contents] == ["user", "model", "user"]
        assert "functionResponse" in contents[-1]["parts"][0]

    async def test_a_streamed_thought_is_reasoning_and_not_content(self) -> None:
        cassette = HttpCassette(
            provider="gemini",
            exchanges=(
                HttpExchange(
                    path=STREAM,
                    stream=(
                        'data: {"candidates":[{"content":{"parts":'
                        '[{"thought":true,"text":"weighing it up"}]}}]}',
                        'data: {"candidates":[{"content":{"parts":[{"text":"it rained"}]},'
                        '"finishReason":"STOP"}]}',
                    ),
                ),
            ),
        )
        model, _ = provider(cassette)
        events = await collected(await model.stream(asked()))
        reasoning = [event for event in events if isinstance(event, ReasoningDelta)]
        assert [event.text for event in reasoning] == ["weighing it up"]

    async def test_a_frame_carrying_only_usage_is_still_read(self) -> None:
        """Gemini reports the count in a frame of its own, and dropping it loses the cost."""
        cassette = HttpCassette(
            provider="gemini",
            exchanges=(
                HttpExchange(
                    path=STREAM,
                    stream=(
                        'data: {"usageMetadata":{"promptTokenCount":9}}',
                        'data: {"candidates":[{"content":{"parts":[{"text":"it rained"}]},'
                        '"finishReason":"STOP"}]}',
                    ),
                ),
            ),
        )
        model, _ = provider(cassette)
        events = await collected(await model.stream(asked()))
        usage = [event for event in events if isinstance(event, UsageDelta)]
        assert usage[0].usage.input_tokens == 9

    async def test_a_part_that_is_neither_words_nor_a_call_is_passed_over(self) -> None:
        """Gemini can return an image part, and it belongs in neither the text nor the calls."""
        model, _ = provider(
            answered(
                **candidate(
                    {"inlineData": {"mimeType": "image/png", "data": "iVBOR"}},
                    {"text": "it rained"},
                )
            )
        )
        response = await model.complete(asked())
        assert response.content == "it rained"
        assert response.tool_calls == ()

    async def test_a_streamed_part_that_is_neither_is_passed_over_too(self) -> None:
        cassette = HttpCassette(
            provider="gemini",
            exchanges=(
                HttpExchange(
                    path=STREAM,
                    stream=(
                        'data: {"candidates":[{"content":{"parts":'
                        '[{"inlineData":{"mimeType":"image/png","data":"iVBOR"}},'
                        '{"text":"it rained"}]},"finishReason":"STOP"}]}',
                    ),
                ),
            ),
        )
        model, _ = provider(cassette)
        events = await collected(await model.stream(asked()))
        assert [event.text for event in events if isinstance(event, TextDelta)] == ["it rained"]
