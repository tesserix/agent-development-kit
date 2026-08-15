"""What a message, a tool call and the cost of a step are, everywhere in the kit.

Every type here is frozen and validates at construction, so a partially-populated
primitive never reaches a provider, a budget or a tracer. Every type round-trips through
JSON without loss, because a run is checkpointed by one process and rehydrated by
another — nothing here may hold a client, a socket or a callable.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/primitives.md`.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from tesserix_adk.core.cost import Cost, CountSource, weaker_source
from tesserix_adk.core.models import AdkModel, Sensitive
from tesserix_adk.core.provenance import TrustLevel, weakest

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BinaryPart",
    "ContentPart",
    "Message",
    "Role",
    "TextPart",
    "ToolCall",
    "Usage",
    "deduplicate",
]

Role = Literal["system", "user", "assistant", "tool"]

# A turn's trust follows the role that produced it, and nothing the turn says about itself.
_TRUST_BY_ROLE: Mapping[Role, TrustLevel] = {
    "system": TrustLevel.SYSTEM,
    "user": TrustLevel.CALLER,
    "assistant": TrustLevel.CALLER,
    "tool": TrustLevel.UNTRUSTED,
}


class TextPart(AdkModel):
    """Text in a message.

    Its repr shows the text: redacting prompt content is the telemetry exporter's job,
    and a type that hides it from a debugger helps nobody.
    """

    kind: Literal["text"] = "text"
    text: str


class BinaryPart(AdkModel):
    """Bytes in a message — an image, an audio clip, a scanned document.

    The payload round-trips through JSON but never appears in the repr, which is the
    form that reaches a log line or a span attribute.
    """

    kind: Literal["binary"] = "binary"
    media_type: str
    data: Annotated[bytes, Sensitive("an exhibit, a scan or a recording is not a span attribute")]

    # Base64 on the wire, raw bytes in Python: a checkpoint is JSON, and JSON has no bytes.
    @field_serializer("data")
    def _encode(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @field_validator("data", mode="before")
    @classmethod
    def _decode(cls, value: object) -> object:
        if isinstance(value, str):
            return base64.b64decode(value, validate=True)
        return value

    def __repr__(self) -> str:
        """The size and type, never the payload: this is the form a log line gets."""
        return f"BinaryPart(media_type={self.media_type!r}, {len(self.data)} bytes withheld)"

    __str__ = __repr__


ContentPart = Annotated[TextPart | BinaryPart, Field(discriminator="kind")]


class Usage(AdkModel):
    """What one step consumed, and what it cost if the cost is knowable.

    Args:
        input_tokens: Tokens sent, including any that were served from cache.
        cached_tokens: Of the input, how many the provider served from its cache.
        cache_write_tokens: Prompt tokens the provider charged to write into its cache.
            Priced apart from a read, and often at a premium, so a total that hides it
            makes caching look free.
        output_tokens: Tokens generated and shown, not counting hidden reasoning.
        reasoning_tokens: Hidden reasoning tokens, recorded beside the visible answer
            rather than inside it. Vendors that bill them within the completion total
            have that total split by their adapter, so one workload reads the same way
            whoever answered it.
        image_units: Images, tiles or audio seconds, which are priced per unit rather
            than per token.
        cost: What it came to, or `None` where nothing has priced it. A self-hosted model
            costs something, so recording zero would be a false statement.
        extras: Usage fields a provider reports that the kit does not model, kept rather
            than dropped. Nothing in the kit reads them; they are evidence.
        source: Who counted. A ledger that cannot tell a count from a guess presents a
            guess as a count.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    image_units: int = Field(default=0, ge=0)
    cost: Cost | None = None
    extras: dict[str, int] = Field(default_factory=dict)
    source: CountSource = CountSource.PROVIDER

    @property
    def estimated(self) -> bool:
        """Whether the counts were worked out by the kit rather than reported."""
        return self.source.is_estimate

    @property
    def fresh_input_tokens(self) -> int:
        """Input the server actually evaluated, cache reads taken out.

        Never negative: vendors disagree about whether a cache read is counted inside the
        input total, and a negative token count would be believed by whatever divides it.

        Example:
            >>> Usage(input_tokens=1000, output_tokens=50, cached_tokens=800).fresh_input_tokens
            200
        """
        return max(self.input_tokens - self.cached_tokens, 0)

    @property
    def measured(self) -> bool:
        """Whether anything was sent, so a hit ratio over this means something.

        Zero over zero is "nobody looked", which a dashboard must not draw as "no hits".
        """
        return self.input_tokens > 0

    @property
    def cache_hit_ratio(self) -> float:
        """How much of the input the provider served from its own cache, from 0 to 1.

        Zero rather than a division error where nothing was sent. Read it beside
        `measured`, and beside `estimated` where the counts are the kit's own.

        Example:
            >>> Usage(input_tokens=1000, output_tokens=50, cached_tokens=800).cache_hit_ratio
            0.8
        """
        if not self.measured:
            return 0.0
        return min(self.cached_tokens / self.input_tokens, 1.0)

    @property
    def _is_nothing(self) -> bool:
        """Whether this records no spend at all, as opposed to spend at an unknown price."""
        return (
            self.input_tokens == 0
            and self.output_tokens == 0
            and self.cached_tokens == 0
            and self.cache_write_tokens == 0
            and self.reasoning_tokens == 0
            and self.image_units == 0
            and self.cost is None
            and not self.extras
        )

    def __add__(self, other: Usage) -> Usage:
        """Total two steps.

        A known cost plus an unknown one is unknown, never the known part alone: a total
        that silently omits a step understates the bill. Nothing spent yet is not an
        unknown price, though — a run starts on an empty usage, and treating that as
        unknown would leave every run unable to report a cost.

        Raises:
            ValueError: If both carry a cost in different currencies.

        Example:
            >>> from decimal import Decimal
            >>> priced = Usage(input_tokens=1, output_tokens=1, cost=Cost(input=Decimal("0.5")))
            >>> (priced + Usage(input_tokens=1, output_tokens=1)).cost is None
            True
        """
        if self._is_nothing:
            return other
        if other._is_nothing:
            return self
        priced = self.cost is not None and other.cost is not None
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            image_units=self.image_units + other.image_units,
            cost=self.cost + other.cost if priced and self.cost and other.cost else None,
            extras={**self.extras, **other.extras},
            # One guessed step makes the total a guess; reporting it as counted would
            # hide the one part of the bill nobody measured.
            source=weaker_source(self.source, other.source),
        )


class Message(AdkModel):
    """One turn in a conversation.

    Args:
        role: Who produced it. A `tool` message is a result and must name the call it
            answers; every other role must not, since there is nothing to answer.
        content: The parts of what was said. Empty only on an assistant turn that asked
            for a tool and said nothing else, which is a turn with no words in it.
        tool_calls: What the assistant asked for on this turn. Recorded because every
            vendor wants the call beside the result that answers it, and a history of
            results alone is one no vendor accepts and no reader can follow.
        tool_call_id: The `ToolCall.id` this result belongs to.
        trust: How far the turn may act. Stamped from the role at construction. It may be
            set lower — an assistant summary of a retrieved page is still that page — but
            never higher: a tool result relabelled as a system turn is the injection,
            written in Python.
        metadata: Consumer-owned annotations. The kit reads nothing from it.
    """

    role: Role
    content: list[ContentPart] = Field(default_factory=list)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    trust: TrustLevel | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _the_role_decides_the_trust(self) -> Message:
        stamped = _TRUST_BY_ROLE[self.role]
        if self.trust is None:
            object.__setattr__(self, "trust", stamped)
        elif weakest(self.trust, stamped) is not self.trust:
            raise ValueError(
                f"a {self.role} message is at most {stamped.value} content; it cannot carry "
                f"trust={self.trust.value}"
            )
        return self

    @model_validator(mode="after")
    def _a_message_says_something(self) -> Message:
        if not self.content and not self.tool_calls:
            raise ValueError("a message with no content and no tool calls says nothing")
        if self.tool_calls and self.role != "assistant":
            raise ValueError(f"a {self.role} message cannot ask for a tool; only an assistant can")
        return self

    @model_validator(mode="after")
    def _tool_results_name_their_call(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "tool_call_id is required on a tool message: without it a result cannot "
                "be matched to its call once several run in parallel"
            )
        if self.role != "tool" and self.tool_call_id:
            raise ValueError(f"tool_call_id is meaningless on a {self.role} message")
        return self


class ToolCall(AdkModel):
    """A model's request to run a tool.

    Args:
        id: Provider-assigned identity. Deduplication and result matching use it, so it
            may not be empty.
        name: The registered tool name.
        arguments: Arguments as the provider produced them, before the tool's own schema
            has validated them.
        idempotent: Whether repeating the call is known to be safe. Defaults to `False`:
            a retry that re-sends a payment is worse than a retry that does nothing.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotent: bool = False


def deduplicate(calls: list[ToolCall]) -> tuple[ToolCall, ...]:
    """Drop repeated calls, keeping the first of each id.

    A retried provider response repeats calls it already sent. Matching is by id and
    never by position: parallel calls to one tool differ only in their arguments.

    Example:
        >>> a = ToolCall(id="call_1", name="search")
        >>> deduplicate([a, a]) == (a,)
        True
    """
    seen: dict[str, ToolCall] = {}
    for call in calls:
        seen.setdefault(call.id, call)
    return tuple(seen.values())
