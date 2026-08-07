"""Google's Gemini API, translated into the kit's own types.

Gemini is the awkward one, in three ways the adapter has to absorb. It gives function
calls no identity, and matches a result back by tool name — so ids are minted here and
resolved again from the history, and a turn with two calls to one tool is the case that
decides the design. It reports `STOP` whether or not it asked for a tool, so the stop
reason is read off the parts rather than believed. And it rejects JSON Schema keywords
that are valid everywhere else, so a schema is pruned before it is sent.
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

__all__ = ["GeminiProvider"]

_STOP_REASONS = {
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "SAFETY": StopReason.SAFETY,
    "RECITATION": StopReason.SAFETY,
    "PROHIBITED_CONTENT": StopReason.SAFETY,
    "BLOCKLIST": StopReason.SAFETY,
    "SPII": StopReason.SAFETY,
}

# Valid JSON Schema that the vendor answers 400 to, so a schema is pruned before it is sent.
_UNSUPPORTED_KEYWORDS = frozenset(
    {"additionalProperties", "$schema", "$id", "title", "default", "examples"}
)


class GeminiProvider(HttpProvider):
    """Calls Gemini's `generateContent` API and answers in the kit's types.

    Args:
        model: A Gemini model id, as Google spells it.
        **options: See `HttpProvider` — capabilities, secrets, base URL, transport.
    """

    provider_name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com"
    default_key_variable = "GEMINI_API_KEY"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the whole answer.

        Raises:
            CapabilityError: If the request needs something the model has not declared.
            ProviderError: On any transport or upstream failure, after translation.
            ModelResponseError: If the body cannot be read as a candidate.
        """
        body = await self._post(
            _generate(request.model),
            self._payload(request),
            cost=self.count_tokens(request.messages),
        )
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
        return self._streamed(
            _stream_path(request.model),
            self._payload(request),
            request=request,
            state=_Stream(),
        )

    def _headers(self) -> dict[str, str]:
        # In a header, never the query string: a key in a URL is a key in every access log
        # between here and the vendor.
        return {"x-goog-api-key": self._credential.value(), "content-type": "application/json"}

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        self._refuse_what_the_model_cannot_do(request)
        instruction, contents = _conversation(request.messages)
        payload: dict[str, Any] = {"contents": contents}
        if instruction:
            payload["systemInstruction"] = {"parts": [{"text": instruction}]}
        if request.tools:
            payload["tools"] = [
                {"functionDeclarations": [_declared(tool) for tool in request.tools]}
            ]
        if request.output_schema is not None:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": _pruned(request.output_schema),
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
    """Each frame is a whole candidate, so the parts are deltas and the rest is a snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def read(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        """Translate one candidate snapshot into the kit's events."""
        self.frames += 1
        if frame.get("error"):
            failure = _dict(frame.get("error"))
            self.failure = ProviderError(
                f"gemini sent {failure.get('status', 'an error')}: {failure.get('message', '')}",
                details={"status": str(failure.get("status", ""))},
            )
            return []
        events: list[StreamEvent] = []
        candidates = frame.get("candidates")
        if isinstance(candidates, list) and candidates:
            events.extend(self._parts(_dict(candidates[0])))
        if frame.get("usageMetadata"):
            events.append(UsageDelta(usage=_usage(frame.get("usageMetadata"))))
        return events

    def _parts(self, candidate: Mapping[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        called = False
        for part in _dict(candidate.get("content")).get("parts") or []:
            entry = _dict(part)
            if entry.get("functionCall"):
                events.append(self._call(_dict(entry.get("functionCall"))))
                called = True
            elif entry.get("thought"):
                events.append(ReasoningDelta(text=str(entry.get("text", ""))))
            elif isinstance(entry.get("text"), str):
                self.text += str(entry["text"])
                events.append(TextDelta(text=str(entry["text"])))
        if candidate.get("finishReason"):
            reported = _stop_reason(candidate.get("finishReason"))
            self.stop_reason = StopReason.TOOL_CALLS if called else reported
        return events

    def _call(self, call: Mapping[str, Any]) -> ToolCallDelta:
        """Whole, not fragmented: Gemini sends the arguments in one piece."""
        index = self._calls
        self._calls += 1
        name = str(call.get("name", ""))
        return ToolCallDelta(
            index=index,
            id=_minted(name, index),
            name=name,
            arguments=json.dumps(_dict(call.get("args"))),
        )


def _generate(model: str) -> str:
    return f"/v1beta/models/{model}:generateContent"


def _stream_path(model: str) -> str:
    return f"/v1beta/models/{model}:streamGenerateContent?alt=sse"


def _minted(name: str, index: int) -> str:
    """An id the vendor never sent, so a result can be matched to the call it answers."""
    return f"{name}-{index}"


def _read(body: Mapping[str, Any]) -> ModelResponse:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ModelResponseError(
            "gemini answered with no candidate to read",
            payload=body.get("candidates") if "candidates" in body else body.get("promptFeedback"),
            provider="gemini",
        )
    candidate = _dict(candidates[0])
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for part in _dict(candidate.get("content")).get("parts") or []:
        entry = _dict(part)
        if entry.get("functionCall"):
            calls.append(_call(_dict(entry.get("functionCall")), index=len(calls)))
        elif entry.get("thought"):
            reasoning.append(str(entry.get("text", "")))
        elif isinstance(entry.get("text"), str):
            text.append(str(entry["text"]))
    reported = _stop_reason(candidate.get("finishReason"))
    return ModelResponse(
        content="".join(text),
        reasoning="".join(reasoning),
        tool_calls=tuple(calls),
        usage=_usage(body.get("usageMetadata")),
        # The vendor says STOP either way, and a caller that reads that as a finished turn
        # never runs the tool it asked for.
        stop_reason=StopReason.TOOL_CALLS if calls else reported,
    )


def _call(call: Mapping[str, Any], *, index: int) -> ToolCall:
    name = str(call.get("name", ""))
    if not name:
        raise ModelResponseError(
            "a function call arrived with no name; there is nothing to run",
            payload=dict(call),
            provider="gemini",
        )
    return ToolCall(id=_minted(name, index), name=name, arguments=_dict(call.get("args")))


def _usage(reported: object) -> Usage:
    counted = _dict(reported)
    thoughts = int(counted.get("thoughtsTokenCount", 0) or 0)
    return Usage(
        input_tokens=int(counted.get("promptTokenCount", 0) or 0),
        output_tokens=int(counted.get("candidatesTokenCount", 0) or 0),
        cached_tokens=int(counted.get("cachedContentTokenCount", 0) or 0),
        reasoning_tokens=thoughts,
    )


def _stop_reason(reported: object) -> StopReason:
    return _STOP_REASONS.get(str(reported), StopReason.UNKNOWN)


def _conversation(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    instruction: list[str] = []
    contents: list[dict[str, Any]] = []
    names = _tool_names(messages)
    for message in messages:
        if message.role == "system":
            instruction.extend(part.text for part in message.content if isinstance(part, TextPart))
        elif message.role == "tool":
            _append_result(contents, message, names)
        else:
            contents.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": _parts(message),
                }
            )
    return "\n\n".join(instruction), contents


def _tool_names(messages: Sequence[Message]) -> dict[str, str]:
    """Which tool each call id belongs to, since a result is matched back by name."""
    return {call.id: call.name for message in messages for call in message.tool_calls}


def _append_result(
    contents: list[dict[str, Any]], message: Message, names: Mapping[str, str]
) -> None:
    result = {
        "functionResponse": {
            "name": names.get(message.tool_call_id or "", message.tool_call_id or ""),
            "response": {
                "result": "".join(
                    part.text for part in message.content if isinstance(part, TextPart)
                )
            },
        }
    }
    if contents and contents[-1]["role"] == "user" and _all_results(contents[-1]["parts"]):
        contents[-1]["parts"].append(result)
        return
    contents.append({"role": "user", "parts": [result]})


def _all_results(parts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(parts) and all("functionResponse" in part for part in parts)


def _parts(message: Message) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
        else:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": part.media_type,
                        "data": base64.b64encode(part.data).decode("ascii"),
                    }
                }
            )
    parts.extend(
        {"functionCall": {"name": call.name, "args": call.arguments}} for call in message.tool_calls
    )
    return parts


def _declared(tool: ToolDeclaration) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": _pruned(tool.parameters or {"type": "object", "properties": {}}),
    }


def _pruned(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the keywords the vendor rejects, at every depth."""
    return {
        key: _pruned_value(value)
        for key, value in schema.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }


def _pruned_value(value: object) -> object:
    if isinstance(value, dict):
        return _pruned(value)
    if isinstance(value, list):
        return [_pruned_value(entry) for entry in value]
    return value


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
