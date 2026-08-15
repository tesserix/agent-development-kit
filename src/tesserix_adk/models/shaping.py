"""Spending less on the turns that do not need deliberation, and never more on the ones that do.

Input compression is the well-known half of the problem and the smaller one. Output is billed
at several times the input rate on reasoning-capable models, and a large share of what is
generated is preamble, restated context and narration of work nobody asked to see. A turn that
merely resumes after a successful tool result rarely needs full effort, yet most gateways send
the same setting for every call in a run.

Two levers, both deliberately conservative. Effort is clamped downward and only downward, so a
policy can cost less than what the caller asked for and never more. Terseness is steered by
appending to the end of the system prompt, never by editing what is already there, because a
rewrite invalidates the cached prefix and gives back more than it saves.

Both are off until switched on, per agent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import model_validator

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Message, TextPart

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.provider import ModelRequest

__all__ = [
    "TOOL_ERROR",
    "Effort",
    "Shaped",
    "Shaping",
    "TurnKind",
    "classify",
    "errored",
    "provider_effort",
]

TOOL_ERROR = "adk.tool_error"
"""Metadata key marking a tool result as a failure. Failures are where deliberation pays."""

_ATTRIBUTES = "adk.shaping."


class Effort(StrEnum):
    """How much deliberation a call is worth, expressed once and mapped per provider."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TurnKind(StrEnum):
    """What a turn is for, which is what decides whether thinking less is safe."""

    NEW_QUESTION = "new_question"
    TOOL_RESUMPTION = "tool_resumption"
    ERROR_RECOVERY = "error_recovery"


_RANK = {Effort.MINIMAL: 0, Effort.LOW: 1, Effort.MEDIUM: 2, Effort.HIGH: 3}

_PARAMETERS = {"openai": "reasoning_effort", "anthropic": "effort", "gemini": "thinking_level"}


def errored(message: Message) -> Message:
    """Return `message` marked as a failed tool result, so no policy clamps the recovery.

    Args:
        message: The tool result that failed.

    Returns:
        A copy carrying the mark. The original is untouched.
    """
    return message.model_copy(update={"metadata": {**message.metadata, TOOL_ERROR: "true"}})


def classify(messages: Sequence[Message]) -> TurnKind:
    """What the next call is for, judged from what the conversation ends with.

    Args:
        messages: The conversation as it will be sent.

    Returns:
        `TOOL_RESUMPTION` where the last message is a successful tool result,
        `ERROR_RECOVERY` where it is a failed one, and `NEW_QUESTION` otherwise — including
        for an empty conversation, which is the least aggressive answer available.
    """
    if not messages or messages[-1].role != "tool":
        return TurnKind.NEW_QUESTION
    if TOOL_ERROR in messages[-1].metadata:
        return TurnKind.ERROR_RECOVERY
    return TurnKind.TOOL_RESUMPTION


def provider_effort(effort: Effort, *, provider: str) -> dict[str, str]:
    """How to say `effort` to `provider`, or nothing at all where it has no such parameter.

    Args:
        effort: What was decided, in the kit's vocabulary.
        provider: Which provider is being called, by its `provider_name`.

    Returns:
        The parameter to merge into the request body. Empty for a provider with no effort
        setting — a local llama.cpp build has none, and that is not an error.
    """
    parameter = _PARAMETERS.get(provider)
    return {parameter: effort.value} if parameter else {}


class Shaped(AdkModel):
    """What shaping decided for one call, for the model call record.

    Args:
        requested: What the caller asked for, before any policy touched it.
        effort: What will be sent. Never above `requested`.
        turn: What the turn was judged to be.
        clamped: Whether the policy lowered anything.
        reason: Why, in a sentence a cost report can quote.
    """

    requested: Effort = Effort.MEDIUM
    effort: Effort = Effort.MEDIUM
    turn: TurnKind = TurnKind.NEW_QUESTION
    clamped: bool = False
    reason: str = ""

    def attributes(self) -> dict[str, str]:
        """The decision as span attributes. Settings and outcomes, never prompt content."""
        return {
            f"{_ATTRIBUTES}effort": self.effort.value,
            f"{_ATTRIBUTES}requested": self.requested.value,
            f"{_ATTRIBUTES}clamped": "true" if self.clamped else "false",
            f"{_ATTRIBUTES}turn": self.turn.value,
        }


class Shaping(AdkModel):
    """A clamp-only output shaping policy, off until an agent turns it on.

    Args:
        enabled: Whether either lever applies at all.
        baseline: What the agent would ask for where a call names nothing.
        resumption: What a tool-resumption turn is clamped to. May not exceed `baseline`:
            a policy that can increase spend or latency is rejected here, at configuration
            time, rather than discovered on a bill.
        terseness: The instruction appended to the end of the system prompt. Byte-stable, so
            enabling it costs exactly one prefix invalidation and none after that.
    """

    enabled: bool = False
    baseline: Effort = Effort.MEDIUM
    resumption: Effort = Effort.LOW
    terseness: str = ""

    @model_validator(mode="after")
    def _clamp_only(self) -> Shaping:
        """Refuse a policy whose shaping would raise effort rather than lower it."""
        if _RANK[self.resumption] > _RANK[self.baseline]:
            raise ConfigurationError(
                f"shaping is clamp-only: resumption effort {self.resumption.value!r} "
                f"is above the baseline {self.baseline.value!r}",
                details={"baseline": self.baseline.value, "resumption": self.resumption.value},
            )
        return self

    def shape(
        self, request: ModelRequest, *, effort: Effort | None = None, final: bool = False
    ) -> Shaped:
        """Decide what effort this call should carry, and record why.

        Args:
            request: The call about to go out. Read, never modified — the clamp is a
                parameter, and the cacheable prefix stays exactly as it was assembled.
            effort: What the caller asked for. The policy's baseline where unstated.
            final: Whether this turn must produce the user-visible answer. Those are not
                clamped: the saving is small and the thing it degrades is what the user reads.

        Returns:
            The decision, with `effort` never above what was asked for.
        """
        requested = effort or self.baseline
        turn = classify(request.messages)
        held = Shaped(requested=requested, effort=requested, turn=turn)
        if not self.enabled:
            return held.model_copy(update={"reason": "shaping is not enabled"})
        if request.output_schema is not None:
            return held.model_copy(update={"reason": "the output schema requires completeness"})
        if final:
            return held.model_copy(update={"reason": "the final turn answers the user"})
        if turn is not TurnKind.TOOL_RESUMPTION:
            return held.model_copy(update={"reason": f"{turn.value} turns are not clamped"})
        if _RANK[self.resumption] >= _RANK[requested]:
            return held.model_copy(update={"reason": "the caller asked for no more than the bound"})
        return held.model_copy(
            update={
                "effort": self.resumption,
                "clamped": True,
                "reason": f"a tool resumption clamped to {self.resumption.value}",
            }
        )

    def steer(self, request: ModelRequest) -> ModelRequest:
        """Append the terseness instruction to the end of the system prompt.

        Args:
            request: The call about to go out.

        Returns:
            The request with the instruction appended after everything already in the
            system prompt, or unchanged where there is nothing to append or it is already
            there. Nothing existing is rewritten and nothing below the system prompt is
            touched, so the invalidation is paid once rather than every turn.
        """
        if not self.enabled or not self.terseness:
            return request
        messages = list(request.messages)
        first = messages[0] if messages else None
        if first is None or first.role != "system":
            messages.insert(0, Message(role="system", content=[TextPart(text=self.terseness)]))
            return request.model_copy(update={"messages": tuple(messages)})
        parts = list(first.content)
        if parts and isinstance(parts[-1], TextPart) and parts[-1].text == self.terseness:
            return request
        messages[0] = first.model_copy(update={"content": [*parts, TextPart(text=self.terseness)]})
        return request.model_copy(update={"messages": tuple(messages)})
