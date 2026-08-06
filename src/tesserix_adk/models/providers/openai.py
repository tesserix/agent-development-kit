"""OpenAI's Chat Completions API, translated into the kit's own types.

Structured output goes through `response_format`, and `strict` is claimed only for a
schema that actually meets the vendor's strict subset — every property required, no extra
keys, all the way down. Claiming it otherwise is a 400 on the first real request, which is
a deployment failure standing in for a check that costs nothing here.

The same wire format is served by several gateways and proxies. `base_url` reaches them;
the provider name stays `openai`, because traffic attributed to the vendor that a proxy
served is traffic recorded against the wrong bill.
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
    from collections.abc import AsyncIterator, Mapping, Sequence

    from tesserix_adk.core.provider import ModelRequest, ToolDeclaration

__all__ = ["OpenAIProvider"]

_COMPLETIONS = "/v1/chat/completions"
_STRUCTURED_NAME = "structured_output"

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_CALLS,
    "function_call": StopReason.TOOL_CALLS,
    "content_filter": StopReason.SAFETY,
}


class OpenAIProvider(HttpProvider):
    """Calls OpenAI's Chat Completions API and answers in the kit's types.

    Args:
        model: An OpenAI model id, as OpenAI spells it.
        **options: See `HttpProvider` — capabilities, secrets, base URL, transport.
    """

    provider_name = "openai"
    default_base_url = "https://api.openai.com"
    default_key_variable = "OPENAI_API_KEY"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the whole answer.

        Raises:
            CapabilityError: If the request needs something the model has not declared.
            ProviderError: On any transport or upstream failure, after translation.
            ModelResponseError: If the body cannot be read as a completion.
        """
        body = await self._post(_COMPLETIONS, self._payload(request))
        return self._settled(_read(body), request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Send one request and return its events as they arrive.

        Raises:
            CapabilityError: If the model does not declare streaming.
            StreamInterruptedError: If the stream ends before the model had finished.
        """
        self._capabilities.require(
            Capability.STREAMING, provider=self.provider_name, model=request.model
        )
        payload = {
            **self._payload(request),
            "stream": True,
            # Without it the vendor sends no usage at all, and a streamed run reports no
            # spend rather than an unknown one.
            "stream_options": {"include_usage": True},
        }
        return self._streamed(_COMPLETIONS, payload, request=request, state=_Stream())

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._credential.value()}",
            "content-type": "application/json",
        }

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        self._refuse_what_the_model_cannot_do(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _conversation(request.messages),
        }
        if request.tools:
            payload["tools"] = [_declared(tool) for tool in request.tools]
        if request.output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _STRUCTURED_NAME,
                    "schema": request.output_schema,
                    "strict": _is_strict(request.output_schema),
                },
            }
        return payload

    def _refuse_what_the_model_cannot_do(self, request: ModelRequest) -> None:
        needed = {
            Capability.VISION: any(
                isinstance(part, BinaryPart)
                for message in request.messages
                for part in message.content
            ),
            Capability.TOOL_CALLING: bool(request.tools),
            Capability.STRUCTURED_OUTPUT: request.output_schema is not None,
        }
        for capability, required in needed.items():
            if required:
                self._capabilities.require(
                    capability, provider=self.provider_name, model=request.model
                )


class _Stream(VendorStream):
    """Chat Completions streams one choice's deltas, and the usage after the last of them."""

    def __init__(self) -> None:
        super().__init__()
        self._usage = Usage(input_tokens=0, output_tokens=0)

    def read(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        """Translate one chunk into the kit's events."""
        self.frames += 1
        if frame.get("error"):
            self.failure = ProviderError(
                f"openai sent {_error(frame)}",
                details={"type": str(_dict(frame.get("error")).get("type", ""))},
            )
            return []
        events: list[StreamEvent] = []
        if frame.get("usage"):
            events.append(UsageDelta(usage=_usage(frame.get("usage"))))
        choices = frame.get("choices")
        if not isinstance(choices, list) or not choices:
            return events
        choice = _dict(choices[0])
        if choice.get("finish_reason"):
            self.stop_reason = _stop_reason(choice.get("finish_reason"))
        return [*self._deltas(_dict(choice.get("delta"))), *events]

    def _deltas(self, delta: Mapping[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.text += content
            events.append(TextDelta(text=content))
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            events.append(ReasoningDelta(text=reasoning))
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            events.extend(_fragment(_dict(call)) for call in calls)
        return events


def _fragment(call: Mapping[str, Any]) -> ToolCallDelta:
    function = _dict(call.get("function"))
    return ToolCallDelta(
        index=int(call.get("index", 0) or 0),
        id=str(call.get("id", "") or ""),
        name=str(function.get("name", "") or ""),
        arguments=str(function.get("arguments", "") or ""),
    )


def _read(body: Mapping[str, Any]) -> ModelResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelResponseError(
            "openai answered with no choice to read",
            payload=body.get("choices"),
            provider="openai",
            request_id=str(body.get("id", "")),
        )
    choice = _dict(choices[0])
    message = _dict(choice.get("message"))
    refusal = message.get("refusal")
    calls = tuple(
        _call(_dict(call), request_id=str(body.get("id", "")))
        for call in message.get("tool_calls") or []
    )
    return ModelResponse(
        content=str(refusal or message.get("content") or ""),
        reasoning=str(message.get("reasoning_content") or ""),
        tool_calls=calls,
        usage=_usage(body.get("usage")),
        stop_reason=StopReason.REFUSAL if refusal else _stop_reason(choice.get("finish_reason")),
    )


def _call(call: Mapping[str, Any], *, request_id: str) -> ToolCall:
    function = _dict(call.get("function"))
    identity, name = str(call.get("id", "")), str(function.get("name", ""))
    if not identity or not name:
        raise ModelResponseError(
            f"a tool call arrived with no id or no name (id {identity!r}, name {name!r}); "
            f"a result cannot be matched back to it",
            payload=dict(call),
            provider="openai",
            request_id=request_id,
        )
    return ToolCall(
        id=identity, name=name, arguments=_arguments(str(function.get("arguments", "")), name)
    )


def _arguments(raw: str, tool: str) -> dict[str, Any]:
    """Arguments arrive as JSON text, and half an object is not most of an argument."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as broken:
        raise ModelResponseError(
            f"arguments for {tool} are not JSON: {broken}", payload=raw, provider="openai"
        ) from broken
    if not isinstance(parsed, dict):
        raise ModelResponseError(
            f"arguments for {tool} are {type(parsed).__name__}, not an object",
            payload=raw,
            provider="openai",
        )
    return parsed


def _usage(reported: object) -> Usage:
    counted = _dict(reported)
    cached = int(_dict(counted.get("prompt_tokens_details")).get("cached_tokens", 0) or 0)
    reasoning = int(_dict(counted.get("completion_tokens_details")).get("reasoning_tokens", 0) or 0)
    return Usage(
        input_tokens=int(counted.get("prompt_tokens", 0) or 0),
        output_tokens=int(counted.get("completion_tokens", 0) or 0),
        cached_tokens=cached,
        extras={"reasoning_tokens": reasoning} if reasoning else {},
    )


def _stop_reason(reported: object) -> StopReason:
    return _STOP_REASONS.get(str(reported), StopReason.UNKNOWN)


def _conversation(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [_turn(message) for message in messages]


def _turn(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text(message),
        }
    turn: dict[str, Any] = {"role": message.role, "content": _content(message)}
    if message.tool_calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return turn


def _content(message: Message) -> str | list[dict[str, Any]] | None:
    """Plain text stays a string, which is the shape the vendor documents for a text turn."""
    if all(isinstance(part, TextPart) for part in message.content):
        return _text(message) or None
    return [_part(part) for part in message.content]


def _part(part: TextPart | BinaryPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    encoded = base64.b64encode(part.data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{part.media_type};base64,{encoded}"},
    }


def _text(message: Message) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _declared(tool: ToolDeclaration) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }


def _is_strict(schema: Mapping[str, Any]) -> bool:
    """Whether `schema` meets the vendor's strict subset, all the way down."""
    if schema.get("type") != "object":
        return False
    properties = _dict(schema.get("properties"))
    required = schema.get("required")
    if schema.get("additionalProperties") is not False:
        return False
    if not isinstance(required, list) or set(required) != set(properties):
        return False
    return all(_is_strict_value(value) for value in properties.values())


def _is_strict_value(value: object) -> bool:
    entry = _dict(value)
    if entry.get("type") == "object":
        return _is_strict(entry)
    if entry.get("type") == "array":
        return _is_strict_value(entry.get("items"))
    return True


def _error(frame: Mapping[str, Any]) -> str:
    failure = _dict(frame.get("error"))
    return f"{failure.get('type', 'an error')}: {failure.get('message', '')}".strip()


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
