"""Anthropic's Messages API, translated into the kit's own types.

Structured output goes through a forced tool call, which is the vendor's own mechanism for
it. Nothing is parsed out of prose: a JSON object fished out of a sentence is a shape that
holds until the model writes a preamble.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.capabilities import Capability
from tesserix_adk.core.errors import ModelResponseError, ProviderError
from tesserix_adk.core.primitives import BinaryPart, Message, TextPart, ToolCall, Usage
from tesserix_adk.core.provider import ModelResponse, StopReason
from tesserix_adk.core.streaming import (
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    UsageDelta,
)
from tesserix_adk.models.providers._http import HttpProvider, VendorStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

    from tesserix_adk.core.provider import ModelRequest, ToolDeclaration

__all__ = ["AnthropicProvider"]

_API_VERSION = "2023-06-01"
_MESSAGES = "/v1/messages"
_STRUCTURED_TOOL = "structured_output"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "tool_use": StopReason.TOOL_CALLS,
    "refusal": StopReason.REFUSAL,
}


class AnthropicProvider(HttpProvider):
    """Calls Anthropic's Messages API and answers in the kit's types.

    Args:
        model: A Claude model id, as Anthropic spells it.
        **options: See `HttpProvider` — capabilities, secrets, base URL, transport.
    """

    provider_name = "anthropic"
    default_base_url = "https://api.anthropic.com"
    default_key_variable = "ANTHROPIC_API_KEY"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the whole answer.

        Raises:
            CapabilityError: If the request needs something the model has not declared.
            ProviderError: On any transport or upstream failure, after translation.
            ModelResponseError: If the body cannot be read as a message.
        """
        payload = self._payload(request)
        return self._settled(_read(await self._post(_MESSAGES, payload)), request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Send one request and return its events as they arrive.

        Raises:
            CapabilityError: If the model does not declare streaming.
            StreamInterruptedError: If the stream ends before the model had finished.
        """
        self._capabilities.require(
            Capability.STREAMING, provider=self.provider_name, model=request.model
        )
        payload = {**self._payload(request), "stream": True}
        return self._streamed(_MESSAGES, payload, request=request, state=_Stream())

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._credential.value(),
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        self._refuse_what_the_model_cannot_do(request)
        system, turns = _conversation(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": self._output_cap(),
            "messages": turns,
        }
        if system:
            payload["system"] = system
        tools = [_declared(tool) for tool in request.tools]
        if request.output_schema is not None:
            tools.append(
                {
                    "name": _STRUCTURED_TOOL,
                    "description": "Return the final answer in the required shape.",
                    "input_schema": request.output_schema,
                }
            )
            payload["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL}
        if tools:
            payload["tools"] = tools
        return payload

    def _refuse_what_the_model_cannot_do(self, request: ModelRequest) -> None:
        required = {
            Capability.VISION: any(
                isinstance(part, BinaryPart)
                for message in request.messages
                for part in message.content
            ),
            Capability.TOOL_CALLING: bool(request.tools),
            Capability.STRUCTURED_OUTPUT: request.output_schema is not None,
        }
        for capability, needed in required.items():
            if needed:
                self._capabilities.require(
                    capability, provider=self.provider_name, model=request.model
                )

    def _settled(self, response: ModelResponse, request: ModelRequest) -> ModelResponse:
        """Unwrap the forced structured-output tool before the shared checks run."""
        structured = [call for call in response.tool_calls if call.name == _STRUCTURED_TOOL]
        if request.output_schema is None or not structured:
            return super()._settled(response, request)
        return super()._settled(
            response.model_copy(
                update={
                    "content": json.dumps(structured[0].arguments),
                    "tool_calls": tuple(
                        call for call in response.tool_calls if call.name != _STRUCTURED_TOOL
                    ),
                }
            ),
            request,
        )


class _Stream(VendorStream):
    """The part of a streamed message that arrives outside the deltas themselves."""

    def __init__(self) -> None:
        super().__init__()
        self._input = 0
        self._cached = 0

    def read(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        """Translate one server-sent frame into the kit's events."""
        self.frames += 1
        kind = frame.get("type")
        # Kept rather than raised: an error frame ends the vendor's stream, so the body
        # is drained first and the connection released before the failure surfaces.
        if kind == "error":
            self.failure = ProviderError(f"anthropic sent {_error(frame)}", details=_details(frame))
            return []
        if kind == "message_start":
            return self._usage(_dict(frame.get("message")).get("usage"))
        if kind == "content_block_start":
            return self._started(frame)
        if kind == "content_block_delta":
            return self._delta(frame)
        if kind == "message_delta":
            self.stop_reason = _stop_reason(_dict(frame.get("delta")).get("stop_reason"))
            return self._usage(frame.get("usage"))
        return []

    def _started(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        block = _dict(frame.get("content_block"))
        if block.get("type") != "tool_use":
            return []
        return [
            ToolCallDelta(
                index=int(frame.get("index", 0)),
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
            )
        ]

    def _delta(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        delta = _dict(frame.get("delta"))
        kind = delta.get("type")
        if kind == "text_delta":
            self.text += str(delta.get("text", ""))
            return [TextDelta(text=str(delta.get("text", "")))]
        if kind == "thinking_delta":
            return [ReasoningDelta(text=str(delta.get("thinking", "")))]
        if kind == "input_json_delta":
            return [
                ToolCallDelta(
                    index=int(frame.get("index", 0)),
                    arguments=str(delta.get("partial_json", "")),
                )
            ]
        return []

    def _usage(self, reported: object) -> list[StreamEvent]:
        """Anthropic reports input once and output as it goes, so the total is carried."""
        counted = _dict(reported)
        self._input = int(counted.get("input_tokens", self._input) or self._input)
        self._cached = int(counted.get("cache_read_input_tokens", self._cached) or self._cached)
        return [
            UsageDelta(
                usage=Usage(
                    input_tokens=self._input + self._cached,
                    output_tokens=int(counted.get("output_tokens", 0) or 0),
                    cached_tokens=self._cached,
                )
            )
        ]


def _read(body: Mapping[str, Any]) -> ModelResponse:
    blocks = body.get("content")
    if not isinstance(blocks, list):
        raise ModelResponseError(
            "anthropic answered without a list of content blocks",
            payload=body.get("content"),
            provider="anthropic",
            request_id=str(body.get("id", "")),
        )
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for block in blocks:
        entry = _dict(block)
        kind = entry.get("type")
        if kind == "text":
            text.append(str(entry.get("text", "")))
        elif kind == "thinking":
            reasoning.append(str(entry.get("thinking", "")))
        elif kind == "tool_use":
            calls.append(_call(entry, request_id=str(body.get("id", ""))))
    return ModelResponse(
        content="".join(text),
        reasoning="".join(reasoning),
        tool_calls=tuple(calls),
        usage=_usage(body.get("usage")),
        stop_reason=_stop_reason(body.get("stop_reason")),
    )


def _call(block: Mapping[str, Any], *, request_id: str) -> ToolCall:
    identity, name = str(block.get("id", "")), str(block.get("name", ""))
    if not identity or not name:
        raise ModelResponseError(
            f"a tool_use block arrived with no id or no name (id {identity!r}, "
            f"name {name!r}); a result cannot be matched back to it",
            payload=dict(block),
            provider="anthropic",
            request_id=request_id,
        )
    arguments = block.get("input")
    return ToolCall(
        id=identity, name=name, arguments=dict(arguments) if isinstance(arguments, dict) else {}
    )


def _usage(reported: object) -> Usage:
    """Cache reads count as input: the vendor reports them apart, and they were sent."""
    counted = _dict(reported)
    cached = int(counted.get("cache_read_input_tokens", 0) or 0)
    written = int(counted.get("cache_creation_input_tokens", 0) or 0)
    return Usage(
        input_tokens=int(counted.get("input_tokens", 0) or 0) + cached,
        output_tokens=int(counted.get("output_tokens", 0) or 0),
        cached_tokens=cached,
        extras={"cache_creation_input_tokens": written} if written else {},
    )


def _stop_reason(reported: object) -> StopReason:
    return _STOP_REASONS.get(str(reported), StopReason.UNKNOWN)


def _conversation(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system.extend(part.text for part in message.content if isinstance(part, TextPart))
        elif message.role == "tool":
            _append_result(turns, message)
        else:
            turns.append({"role": message.role, "content": _blocks(message)})
    return "\n\n".join(system), turns


def _append_result(turns: list[dict[str, Any]], message: Message) -> None:
    """Every result for one turn goes in a single user message, as the vendor requires."""
    result = {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": "".join(part.text for part in message.content if isinstance(part, TextPart)),
    }
    if turns and turns[-1]["role"] == "user" and _all_results(turns[-1]["content"]):
        turns[-1]["content"].append(result)
        return
    turns.append({"role": "user", "content": [result]})


def _all_results(content: Iterable[Mapping[str, Any]]) -> bool:
    blocks = list(content)
    return bool(blocks) and all(block.get("type") == "tool_result" for block in blocks)


def _blocks(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": base64.b64encode(part.data).decode("ascii"),
                    },
                }
            )
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return blocks


def _declared(tool: ToolDeclaration) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters or {"type": "object", "properties": {}},
    }


def _error(frame: Mapping[str, Any]) -> str:
    failure = _dict(frame.get("error"))
    return f"{failure.get('type', 'an error')}: {failure.get('message', '')}".strip()


def _details(frame: Mapping[str, Any]) -> dict[str, str]:
    return {"type": str(_dict(frame.get("error")).get("type", ""))}


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
