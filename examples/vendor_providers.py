"""Three vendors behind one interface, from recorded traffic rather than live keys.

Four scenarios: the same agent answered by Anthropic, OpenAI and Gemini in turn; the wire
requests those three runs actually sent, which is where they differ; a streamed answer
read as events; and a vendor failure arriving as one of the kit's own errors.

Run it with `python examples/vendor_providers.py`. Nothing here reaches the network — the
recordings stand in for it. Against the real endpoints, drop `transport` and `secrets` and
let each provider read its key from the environment.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from tesserix_adk.core import Agent, Message, ModelRequest, ProviderError, TextDelta, TextPart
from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import (
    FakeClock,
    FakeSecrets,
    FakeToolRegistry,
    HttpCassette,
    HttpExchange,
    HttpReplay,
)

ANSWER = {"city": "Delhi", "summary": "clear"}
SECRETS = FakeSecrets(
    {"ANTHROPIC_API_KEY": "test-key", "OPENAI_API_KEY": "test-key", "GEMINI_API_KEY": "test-key"}
)


class Weather(BaseModel):
    """The shape the agent must answer in.

    Args:
        city: Where the reading is from.
        summary: The reading, in a few words.
    """

    city: str
    summary: str


def anthropic(exchanges: tuple[HttpExchange, ...]) -> tuple[AnthropicProvider, HttpReplay]:
    """An Anthropic provider answered by `exchanges` instead of by the network."""
    replay = HttpReplay(HttpCassette(provider="anthropic", exchanges=exchanges))
    return (
        AnthropicProvider("claude-sonnet-4-5", secrets=SECRETS, transport=replay.transport),
        replay,
    )


def openai(exchanges: tuple[HttpExchange, ...]) -> tuple[OpenAIProvider, HttpReplay]:
    """An OpenAI provider answered by `exchanges` instead of by the network."""
    replay = HttpReplay(HttpCassette(provider="openai", exchanges=exchanges))
    return OpenAIProvider("gpt-4o", secrets=SECRETS, transport=replay.transport), replay


def gemini(exchanges: tuple[HttpExchange, ...]) -> tuple[GeminiProvider, HttpReplay]:
    """A Gemini provider answered by `exchanges` instead of by the network."""
    replay = HttpReplay(HttpCassette(provider="gemini", exchanges=exchanges))
    return GeminiProvider("gemini-2.5-flash", secrets=SECRETS, transport=replay.transport), replay


def anthropic_turns() -> tuple[HttpExchange, ...]:
    """A tool call, then the forced structured-output tool Anthropic answers shapes with."""
    return (
        _message(
            [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Delhi"}}]
        ),
        _message(
            [{"type": "tool_use", "id": "toolu_2", "name": "structured_output", "input": ANSWER}]
        ),
    )


def _message(content: list[dict[str, Any]]) -> HttpExchange:
    return HttpExchange(
        path="/v1/messages",
        body={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 412, "output_tokens": 38},
        },
    )


def openai_turns() -> tuple[HttpExchange, ...]:
    """A tool call, then JSON in the content, which is where `response_format` puts it."""
    return (
        _completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city": "Delhi"}'},
                    }
                ],
            },
            "tool_calls",
        ),
        _completion({"role": "assistant", "content": json.dumps(ANSWER)}, "stop"),
    )


def _completion(message: dict[str, Any], finish_reason: str) -> HttpExchange:
    return HttpExchange(
        path="/v1/chat/completions",
        body={
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 389, "completion_tokens": 21},
        },
    )


def gemini_turns() -> tuple[HttpExchange, ...]:
    """A function call the vendor gave no id, then JSON text."""
    return (
        _candidate([{"functionCall": {"name": "lookup", "args": {"city": "Delhi"}}}]),
        _candidate([{"text": json.dumps(ANSWER)}]),
    )


def _candidate(parts: list[dict[str, Any]]) -> HttpExchange:
    return HttpExchange(
        path="/v1beta/models/gemini-2.5-flash:generateContent",
        body={
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 358, "candidatesTokenCount": 15},
        },
    )


def forecaster(model: str) -> Agent[Weather]:
    """One agent, and the only vendor detail in it is how each spells its model."""
    return Agent(
        name="forecaster",
        instructions="Answer with the weather.",
        model=model,
        output_type=Weather,
        tools=("lookup",),
    )


async def same_agent_three_vendors() -> None:
    """The run is the same run: same tool, same arguments, same typed answer."""
    for provider, _ in (
        anthropic(anthropic_turns()),
        openai(openai_turns()),
        gemini(gemini_turns()),
    ):
        tools = FakeToolRegistry({"lookup": lambda city: f"{city}: clear"})
        finished = await AgentRunner(provider=provider, tools=tools, clock=FakeClock()).run(
            forecaster(provider.model), "what is the weather in Delhi", tenant="acme"
        )
        print(f"{provider.name:<10}", finished.output, tools.calls)  # noqa: T201


async def what_went_on_the_wire() -> None:
    """One request, three shapes. A system prompt is where the vendors diverge first."""
    for provider, replay in (
        anthropic(anthropic_turns()[1:]),
        openai(openai_turns()[1:]),
        gemini(gemini_turns()[1:]),
    ):
        await AgentRunner(provider=provider, clock=FakeClock()).run(
            Agent(
                name="asker",
                instructions="Be brief.",
                model=provider.model,
                output_type=Weather,
            ),
            "what is the weather in Delhi",
            tenant="acme",
        )
        body = replay.sent[0].body or {}
        print(f"{provider.name:<10}", sorted(body))  # noqa: T201


async def streamed() -> None:
    """Events as they arrive, then one `StreamEnd` carrying the settled response."""
    provider, _ = anthropic(
        (
            HttpExchange(
                path="/v1/messages",
                stream=(
                    'data: {"type":"message_start","message":{"usage":{"input_tokens":9}}}',
                    'data: {"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"it "}}',
                    'data: {"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"rained"}}',
                    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                    '"usage":{"output_tokens":4}}',
                ),
            ),
        )
    )
    async for event in await provider.stream(_asked()):
        if isinstance(event, TextDelta):
            print("delta:    ", event.text)  # noqa: T201
        else:
            print("event:    ", type(event).__name__)  # noqa: T201


async def a_vendor_failure() -> None:
    """A 500 arrives as `ProviderError`, so nothing above the adapter reads vendor JSON."""
    provider, _ = anthropic(
        (
            HttpExchange(
                path="/v1/messages",
                status=500,
                body={"type": "error", "error": {"type": "api_error", "message": "upstream"}},
            ),
        )
    )
    try:
        await provider.complete(_asked())
    except ProviderError as failed:
        print("refused:  ", failed)  # noqa: T201


def _asked() -> ModelRequest:
    return ModelRequest(
        model="claude-sonnet-4-5",
        messages=(Message(role="user", content=[TextPart(text="did it rain")]),),
    )


async def main() -> None:
    """Run every scenario in order."""
    await same_agent_three_vendors()
    await what_went_on_the_wire()
    await streamed()
    await a_vendor_failure()


if __name__ == "__main__":
    asyncio.run(main())
