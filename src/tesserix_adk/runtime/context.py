"""Admission and eviction for the context window.

The largest waste in a retrieval loop is re-injecting a chunk the model already has, and
on CPU that waste is seconds of prefill on every turn. So admission is keyed: a segment
whose key is already held is refused, whichever layer holds it.

When there is no room, what leaves is decided here rather than by whoever appended last.
Conversation goes oldest-first, then retrieval goes lowest-scored-first, and the cacheable
prefix never goes at all — dropping it would refill every cache downstream, which is a
cost dressed as a saving. A prefix that cannot fit alone is refused, not trimmed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.errors import AdkError, ContextWindowExceededError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.runtime.prompt import PROMPT_LAYERS, PromptLayer, approximate_tokens

if TYPE_CHECKING:
    from tesserix_adk.runtime.prompt import Tokenizer

__all__ = [
    "ContextContribution",
    "ContextContributionError",
    "ContextContributor",
    "ContextRequest",
    "ContextWindow",
    "Segment",
]

# The prefix. Everything below is assembled fresh each turn and so can be given up.
_PROTECTED = (PromptLayer.SYSTEM, PromptLayer.TOOLS, PromptLayer.PINNED)


class ContextContributionError(AdkError):
    """A required context contributor could not answer."""


class ContextRequest(AdkModel):
    """Facts a context contributor receives before prompt assembly."""

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    user: str | None = None
    agent_name: str = Field(min_length=1)
    query: str = Field(min_length=1)


class ContextContribution(AdkModel):
    """Retrieved context and optional stable keys used for admission deduplication."""

    content: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _one_key_per_segment(self) -> ContextContribution:
        if self.keys and len(self.keys) != len(self.content):
            raise ValueError("context keys must name every contributed segment")
        return self


@runtime_checkable
class ContextContributor(Protocol):
    """A context source attached once to every run of a runner."""

    @property
    def name(self) -> str:
        """Stable name recorded on retrieval events."""
        ...

    @property
    def required(self) -> bool:
        """Whether an outage stops the run instead of degrading it."""
        ...

    async def contribute(self, request: ContextRequest) -> ContextContribution:
        """Return context for one authorized run."""
        ...


class Segment(AdkModel):
    """One admitted piece of context.

    Args:
        text: The content itself.
        layer: Which band of the prompt it belongs to, which is what decides whether it
            can be evicted.
        key: A content key — a chunk id, a document hash. A segment whose key is already
            held is refused. `None` means never deduplicated: two turns of conversation
            can read identically and both still happened.
        score: Retrieval relevance. Higher survives eviction longer. Meaningless outside
            the retrieved layer, and ignored there.
    """

    text: str
    layer: PromptLayer
    key: str | None = None
    score: float = 0.0


class ContextWindow:
    """What the model is holding, and what leaves when there is no more room.

    Args:
        limit_tokens: How much fits. Positive.
        tokenizer: How to measure it. `None` uses `approximate_tokens`, which is an
            estimate; pass the server's own where the boundary has to be exact.

    Raises:
        ValueError: If `limit_tokens` is not positive.

    Example:
        >>> from tesserix_adk.runtime import PromptLayer, Segment
        >>> window = ContextWindow(limit_tokens=100)
        >>> window.admit(Segment(text="page 12", layer=PromptLayer.RETRIEVED, key="p12"))
        True
        >>> window.admit(Segment(text="page 12", layer=PromptLayer.RETRIEVED, key="p12"))
        False
    """

    def __init__(self, *, limit_tokens: int, tokenizer: Tokenizer | None = None) -> None:
        if limit_tokens <= 0:
            raise ValueError(f"limit_tokens must be positive, got {limit_tokens}")
        self._limit = limit_tokens
        self._count = tokenizer or approximate_tokens
        self._admitted: list[Segment] = []

    def admit(self, segment: Segment) -> bool:
        """Take `segment` in unless its key is already held.

        Returns:
            Whether it was admitted. `False` means the model already has this content and
            sending it again would be prefill spent on nothing.
        """
        if segment.key is not None and self.holds(segment.key):
            return False
        self._admitted.append(segment)
        return True

    def holds(self, key: str) -> bool:
        """Whether a segment with this key is in the window right now."""
        return any(segment.key == key for segment in self._admitted)

    @property
    def segments(self) -> tuple[Segment, ...]:
        """Everything held, in layer order, and in admission order within a layer."""
        return tuple(sorted(self._admitted, key=lambda segment: PROMPT_LAYERS.index(segment.layer)))

    @property
    def tokens(self) -> int:
        """How much is held, by the tokenizer this window was given."""
        return sum(self._count(segment.text) for segment in self._admitted)

    def texts(self, layer: PromptLayer) -> tuple[str, ...]:
        """The text of one layer, in the form `assemble_prompt` takes."""
        return tuple(segment.text for segment in self._admitted if segment.layer is layer)

    def fit(self) -> tuple[Segment, ...]:
        """Evict until what is held fits the limit.

        Conversation goes first, oldest first; then retrieval, lowest-scored first. An
        evicted segment releases its key, so the same chunk can be admitted again on a
        later turn — it is no longer content the model has.

        Returns:
            What was evicted, in the order it left, so a caller can log or re-rank it.

        Raises:
            ContextWindowExceededError: If the prefix alone does not fit. Trimming it
                would refill every cache downstream, so the window refuses instead.
        """
        evicted: list[Segment] = []
        while self.tokens > self._limit:
            giving_up = self._next_to_go()
            if giving_up is None:
                protected = sum(
                    self._count(segment.text)
                    for segment in self._admitted
                    if segment.layer in _PROTECTED
                )
                raise ContextWindowExceededError(
                    f"the prefix alone is {protected} tokens against a limit of "
                    f"{self._limit}; it is never evicted, because refilling every cache "
                    f"downstream costs more than it saves",
                    counted=protected,
                    limit=self._limit,
                )
            self._admitted.remove(giving_up)
            evicted.append(giving_up)
        return tuple(evicted)

    def _next_to_go(self) -> Segment | None:
        conversation = [
            segment for segment in self._admitted if segment.layer is PromptLayer.CONVERSATION
        ]
        if conversation:
            return conversation[0]
        retrieved = [
            segment for segment in self._admitted if segment.layer is PromptLayer.RETRIEVED
        ]
        if retrieved:
            return min(retrieved, key=lambda segment: segment.score)
        return None
