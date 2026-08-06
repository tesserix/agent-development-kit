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
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

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

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class TextPart(BaseModel):
    """Text in a message.

    Its repr shows the text: redacting prompt content is the telemetry exporter's job,
    and a type that hides it from a debugger helps nobody.
    """

    model_config = _FROZEN

    kind: Literal["text"] = "text"
    text: str


class BinaryPart(BaseModel):
    """Bytes in a message — an image, an audio clip, a scanned document.

    The payload round-trips through JSON but never appears in the repr, which is the
    form that reaches a log line or a span attribute.
    """

    model_config = _FROZEN

    kind: Literal["binary"] = "binary"
    media_type: str
    data: bytes

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


class Usage(BaseModel):
    """What one step consumed, and what it cost if the cost is knowable.

    Args:
        input_tokens: Tokens sent, including any that were served from cache.
        output_tokens: Tokens generated.
        cached_tokens: Of the input, how many the provider served from its cache.
        cost: Money, in `currency`. `None` when the model has no price — a self-hosted
            model costs something, so recording zero would be a false statement.
        currency: ISO 4217 code. Required whenever `cost` is set.
        extras: Usage fields a provider reports that the kit does not model, kept rather
            than dropped. Nothing in the kit reads them; they are evidence.
    """

    model_config = _FROZEN

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cost: float | None = Field(default=None, ge=0)
    currency: str | None = None
    extras: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cost_needs_a_currency(self) -> Usage:
        if self.cost is not None and not self.currency:
            raise ValueError("currency is required when cost is set; a bare number is not money")
        return self

    @property
    def _is_nothing(self) -> bool:
        """Whether this records no spend at all, as opposed to spend at an unknown price."""
        return (
            self.input_tokens == 0
            and self.output_tokens == 0
            and self.cached_tokens == 0
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
            >>> priced = Usage(input_tokens=1, output_tokens=1, cost=0.5, currency="USD")
            >>> (priced + Usage(input_tokens=1, output_tokens=1)).cost is None
            True
        """
        if self._is_nothing:
            return other
        if other._is_nothing:
            return self
        priced = self.cost is not None and other.cost is not None
        if priced and self.currency != other.currency:
            raise ValueError(
                f"cannot total {self.currency} and {other.currency} usage: the result "
                f"would be a number that is true in neither currency"
            )
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost=(self.cost or 0) + (other.cost or 0) if priced else None,
            currency=self.currency if priced else None,
            extras={**self.extras, **other.extras},
        )


class Message(BaseModel):
    """One turn in a conversation.

    Args:
        role: Who produced it. A `tool` message is a result and must name the call it
            answers; every other role must not, since there is nothing to answer.
        content: One or more parts. A message with none says nothing and is refused.
        tool_call_id: The `ToolCall.id` this result belongs to.
        metadata: Consumer-owned annotations. The kit reads nothing from it.
    """

    model_config = _FROZEN

    role: Role
    content: list[ContentPart] = Field(min_length=1)
    tool_call_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

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


class ToolCall(BaseModel):
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

    model_config = _FROZEN

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
