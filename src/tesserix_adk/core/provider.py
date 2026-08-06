"""One call to a model, and what came back, as data.

These live in `core` because the provider protocol is typed over them: a protocol whose
request and response are `Any` is a protocol an implementation cannot be checked against,
and moving them any further out would make `core` depend on the layer above it.

Nothing here names a vendor type. A provider translates its own wire format into these on
the way in and out; that translation is the provider's whole job.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field

from tesserix_adk.core.models import AdkModel, Sensitive
from tesserix_adk.core.primitives import Message, ToolCall, Usage

__all__ = ["ModelRequest", "ModelResponse", "StopReason", "ToolDeclaration"]


class StopReason(StrEnum):
    """Why a model stopped generating, in the kit's words rather than a vendor's.

    Vendors spell the same handful of outcomes a dozen ways, and a consumer branching on
    the spelling changes when a vendor renames one. A reason no vendor map covers is
    `UNKNOWN` rather than the nearest guess: reporting a truncation as a finished answer
    is the failure that guess produces.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_CALLS = "tool_calls"
    REFUSAL = "refusal"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class ToolDeclaration(AdkModel):
    """A tool as the model is told about it.

    Provisional until the Tools epic lands a registry type (#130). It carries the JSON
    Schema as data so that a declaration can be hashed into the prompt version and diffed
    in review.
    """

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(AdkModel):
    """One call to a provider, as data."""

    model: str = Field(min_length=1)
    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...] = ()
    output_schema: dict[str, Any] | None = None
    output_schema_hash: str | None = None


class ModelResponse(AdkModel):
    """What a provider returned for one call.

    A response with neither content nor tool calls is not retried: asking again for the
    same nothing is how a loop wedges.

    Args:
        content: The answer, as the caller may show it.
        reasoning: The model's own working out, where it emitted any. Marked sensitive:
            it is not the answer, it is never replayed into the next turn as though the
            assistant had said it, and it does not travel to telemetry.
        stop_reason: Why generation ended, translated into the kit's taxonomy.
    """

    content: str = ""
    reasoning: Annotated[str, Sensitive("model reasoning is not user-visible content")] = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    stop_reason: StopReason = StopReason.END_TURN
