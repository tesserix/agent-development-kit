"""One vocabulary for an incremental answer, whichever vendor produced it.

Every vendor streams a different envelope: named SSE events, numbered choice deltas,
whole-candidate snapshots. A consumer that reads those shapes has integrated a vendor
rather than the kit, so each provider translates into the events here and nothing above
the provider layer sees a vendor payload.

These live in `core` for the same reason `ModelRequest` does: the provider protocol is
typed over them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from tesserix_adk.core.errors import ModelResponseError
from tesserix_adk.core.models import AdkModel, Sensitive
from tesserix_adk.core.primitives import ToolCall, Usage
from tesserix_adk.core.provider import ModelResponse

if TYPE_CHECKING:
    from tesserix_adk.core.provider import StopReason

__all__ = [
    "ReasoningDelta",
    "StreamAccumulator",
    "StreamEnd",
    "StreamEvent",
    "TextDelta",
    "ToolCallDelta",
    "UsageDelta",
]


class StreamEvent(AdkModel):
    """Base for every event a provider may emit while an answer is being generated."""


class TextDelta(StreamEvent):
    """More of the answer. Concatenating every delta in arrival order gives the content."""

    text: str


class ReasoningDelta(StreamEvent):
    """More of the model's own working out.

    Marked sensitive: reasoning is not the answer, it is not replayed into the next turn,
    and it does not travel to telemetry, where it would outlive the run under a retention
    policy written for metrics.
    """

    text: Annotated[str, Sensitive("model reasoning is not user-visible content")] = ""


class ToolCallDelta(StreamEvent):
    """Part of one tool call. Vendors send the arguments as JSON text, a fragment at a time.

    Args:
        index: Which call of the turn this belongs to. Parallel calls to one tool differ
            only in their arguments, so position is what keeps them apart, never the name.
        id: Provider-assigned id, sent once, usually with the first fragment.
        name: The tool being called, sent once.
        arguments: A fragment of the JSON argument object, never parsed on its own.
    """

    index: int = Field(ge=0)
    id: str = ""
    name: str = ""
    arguments: str = ""


class UsageDelta(StreamEvent):
    """The running total, as the provider reports it. Later events replace earlier ones."""

    usage: Usage


class StreamEnd(StreamEvent):
    """The last event, carrying the assembled answer.

    A consumer that wants only the finished response reads this one and ignores the rest,
    which is how the streamed and buffered paths stay the same code.
    """

    response: ModelResponse


class _PartialCall:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []


class StreamAccumulator:
    """Assembles a `ModelResponse` from the events of one stream.

    Providers use it so that the streamed and buffered paths produce the same record, and
    consumers can use it to collapse a stream they only partly care about.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._usage = Usage(input_tokens=0, output_tokens=0)
        self._calls: dict[int, _PartialCall] = {}

    def feed(self, event: StreamEvent) -> None:
        """Take one event into the answer being assembled."""
        if isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ReasoningDelta):
            self._reasoning.append(event.text)
        elif isinstance(event, UsageDelta):
            self._usage = event.usage
        elif isinstance(event, ToolCallDelta):
            self._fragment(event)

    def finish(self, stop_reason: StopReason) -> ModelResponse:
        """Return the assembled response.

        Raises:
            ModelResponseError: If a tool call never became a JSON object, or named no
                tool. Completing a truncated argument object by guessing runs a tool with
                arguments nobody sent.
        """
        return ModelResponse(
            content="".join(self._text),
            reasoning="".join(self._reasoning),
            tool_calls=tuple(self._call(partial) for _, partial in sorted(self._calls.items())),
            usage=self._usage,
            stop_reason=stop_reason,
        )

    def _fragment(self, event: ToolCallDelta) -> None:
        partial = self._calls.setdefault(event.index, _PartialCall())
        partial.id = event.id or partial.id
        partial.name = event.name or partial.name
        partial.arguments.append(event.arguments)

    def _call(self, partial: _PartialCall) -> ToolCall:
        raw = "".join(partial.arguments)
        if not partial.name:
            raise ModelResponseError(
                f"a tool call arrived with no tool name (id {partial.id!r})", payload=raw
            )
        return ToolCall(id=partial.id, name=partial.name, arguments=_arguments(raw, partial.name))


def _arguments(raw: str, tool: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as broken:
        raise ModelResponseError(
            f"arguments for {tool} never became JSON: {broken}", payload=raw
        ) from broken
    if not isinstance(parsed, dict):
        raise ModelResponseError(
            f"arguments for {tool} are {type(parsed).__name__}, not an object", payload=raw
        )
    return parsed
