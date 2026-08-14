"""Folding older turns away without folding away where their claims came from.

A long conversation has to be made smaller, and the honest way to do it is to replace old
turns with a summary. Prose is what a summary is allowed to lose. Provenance is not: a
summary of turns that each cited a policy, arriving with no citations, reads as a set of
claims the agent made up itself, and the consumer holding it cannot check any of them.

So the summariser writes the words and this module checks the sources: every citation id
carried by a folded turn must be carried by what replaces it, or compaction fails and the
conversation is left as it was. Only the conversation layer is touched, so the cacheable
prefix is byte-identical afterwards and no cache downstream is refilled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.errors import ProvenanceLostError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Message, TextPart
from tesserix_adk.runtime.prompt import approximate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.runtime.prompt import Tokenizer

__all__ = [
    "CITATIONS",
    "COMPACTED",
    "Compaction",
    "CompactionEvent",
    "Summariser",
    "citations_of",
    "cited",
    "compact_conversation",
]

CITATIONS = "adk.citations"
"""Message metadata key: the citation ids that turn rests on, comma-separated."""

COMPACTED = "adk.compacted"
"""Message metadata key marking a message that stands for turns no longer present."""


def cited(message: Message, citations: Sequence[str]) -> Message:
    """Return `message` carrying `citations`.

    The ids travel in the message's own metadata rather than in a parallel list, because
    the two go through different hands and a history is what gets persisted, replayed and
    handed to a provider.
    """
    return message.model_copy(
        update={"metadata": {**message.metadata, CITATIONS: ",".join(citations)}}
    )


def citations_of(message: Message) -> tuple[str, ...]:
    """The citation ids `message` rests on, in the order they were recorded."""
    return tuple(id_ for id_ in message.metadata.get(CITATIONS, "").split(",") if id_)


@runtime_checkable
class Summariser(Protocol):
    """Writes one message standing for a span of turns.

    It is handed the turns and returns the replacement, rather than returning prose, so a
    caller can carry across whatever else its messages hold. Carrying the citations is the
    one part not left to it: `compact_conversation` checks and refuses.
    """

    async def __call__(self, messages: tuple[Message, ...]) -> Message:
        """Return the message that will stand for `messages`."""
        ...


class CompactionEvent(AdkModel):
    """What one compaction did, in terms an audit trail can repeat.

    Args:
        run_id: The run it happened in.
        folded: How many turns were replaced.
        citations: The ids carried across, sorted.
        tokens_before: What the conversation cost before.
        tokens_after: What it costs now.
    """

    run_id: str = ""
    folded: int = 0
    citations: tuple[str, ...] = ()
    tokens_before: int = 0
    tokens_after: int = 0

    def attributes(self) -> dict[str, str]:
        """Span attributes: what it cost and what it kept, never what was said."""
        return {
            "adk.compaction.folded": str(self.folded),
            "adk.compaction.citations": str(len(self.citations)),
            "adk.compaction.tokens_before": str(self.tokens_before),
            "adk.compaction.tokens_after": str(self.tokens_after),
        }


class Compaction(AdkModel):
    """A conversation after compaction, and what compaction did to it.

    Args:
        history: The turns to send, in order. Unchanged where nothing was folded.
        ran: Whether anything was folded. False both below the threshold and where there
            was nothing old enough to fold, which are the same thing to a caller.
        folded: How many turns the summary stands for.
        citations: The ids carried across, sorted.
        event: What to write down, or `None` where there was nothing to write down.
    """

    history: tuple[Message, ...] = ()
    ran: bool = False
    folded: int = 0
    citations: tuple[str, ...] = ()
    event: CompactionEvent | None = None


async def compact_conversation(
    history: Sequence[Message],
    *,
    summarise: Summariser,
    threshold_tokens: int,
    keep_recent: int = 4,
    tokenizer: Tokenizer | None = None,
    run_id: str = "",
) -> Compaction:
    """Fold the oldest turns into one summary once the conversation gets too long.

    Args:
        history: The conversation so far, oldest first.
        summarise: Who writes the replacement message.
        threshold_tokens: The size above which compaction runs. Positive.
        keep_recent: How many newest turns stay verbatim whatever happens. Not negative.
        tokenizer: How to measure a turn. `None` uses `approximate_tokens`, which is an
            estimate; pass the server's own where the threshold has to be exact.
        run_id: Recorded on the event.

    Returns:
        The conversation to send, and what was done to get it. Below the threshold the
        history is returned untouched, which makes a second pass over an already
        compacted conversation a no-op.

    Raises:
        ValueError: If `threshold_tokens` is not positive or `keep_recent` is negative.
        ProvenanceLostError: If the summary does not carry every citation the folded
            turns did. The original history is left as it was.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.core import Message, TextPart
        >>> async def fold(messages):
        ...     return Message(role="system", content=[TextPart(text="Earlier: refunds.")])
        >>> turns = [
        ...     Message(role="user", content=[TextPart(text="A refund question. " * 20)])
        ...     for _ in range(6)
        ... ]
        >>> done = asyncio.run(
        ...     compact_conversation(
        ...         turns, summarise=fold, threshold_tokens=50, keep_recent=2
        ...     )
        ... )
        >>> done.ran, done.folded, len(done.history)
        (True, 4, 3)
    """
    if threshold_tokens <= 0:
        raise ValueError(f"threshold_tokens must be positive, got {threshold_tokens}")
    if keep_recent < 0:
        raise ValueError(f"keep_recent must not be negative, got {keep_recent}")

    count = tokenizer or approximate_tokens
    before = _tokens(history, count)
    spanned = tuple(history[: max(len(history) - keep_recent, 0)])
    # A span that is nothing but a previous summary is already as folded as it gets;
    # folding it again would cost a call and a little more fidelity for no tokens.
    if not spanned or before <= threshold_tokens or all(_is_summary(m) for m in spanned):
        return Compaction(history=tuple(history))

    summary = await summarise(spanned)
    carried = _sources(spanned)
    lost = tuple(id_ for id_ in carried if id_ not in citations_of(summary))
    if lost:
        raise ProvenanceLostError(
            "compaction would drop the sources of turns it folded away",
            lost=lost,
            folded=len(spanned),
        )

    folded = summary.model_copy(update={"metadata": {**summary.metadata, COMPACTED: "true"}})
    compacted = (folded, *history[len(spanned) :])
    return Compaction(
        history=compacted,
        ran=True,
        folded=len(spanned),
        citations=carried,
        event=CompactionEvent(
            run_id=run_id,
            folded=len(spanned),
            citations=carried,
            tokens_before=before,
            tokens_after=_tokens(compacted, count),
        ),
    )


def _is_summary(message: Message) -> bool:
    """Whether this message already stands for turns that are gone."""
    return message.metadata.get(COMPACTED) == "true"


def _sources(messages: Sequence[Message]) -> tuple[str, ...]:
    """Every citation id the span carries, deduplicated and sorted."""
    return tuple(sorted({id_ for message in messages for id_ in citations_of(message)}))


def _tokens(messages: Sequence[Message], count: Tokenizer) -> int:
    """What a conversation costs, counting only the words in it."""
    return sum(
        count(part.text)
        for message in messages
        for part in message.content
        if isinstance(part, TextPart)
    )
