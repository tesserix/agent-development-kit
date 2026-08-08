"""Ways of making a section smaller that are not "cut the front off and hope".

A provider that truncates drops whichever tokens fall off the end of its window, which
in a long conversation is the constraint stated in turn three rather than the small talk
in turn ninety. A strategy decides deliberately, and says what it decided.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.errors import ContextBudgetError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Message, Role, TextPart
from tesserix_adk.core.provider import ModelRequest

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import ModelProvider

type TokenCounter = Callable[[Sequence[Message]], int]
"""How many tokens some messages occupy, answered by the provider that will read them."""

__all__ = [
    "CompactionOutcome",
    "CompactionStrategy",
    "ContextEntry",
    "DropOldest",
    "PinAndFold",
    "SummariseSpan",
    "TokenCounter",
    "pinned",
]

PIN = "adk.pinned"
"""Metadata key marking a message the caller does not want evicted."""

SUMMARY_OF = "adk.summary_of"
"""Metadata key on a summary, listing the entry ids it stands for."""


def pinned(message: Message) -> Message:
    """Return `message` marked non-evictable.

    A pin travels on the message rather than in a separate list because the two go
    through different hands and only one of them survives a `model_copy`.
    """
    return message.model_copy(update={"metadata": {**message.metadata, PIN: "true"}})


class ContextEntry(AdkModel):
    """One candidate message, with the identity a report can name it by.

    Args:
        id: What the outcome calls it. The caller's `metadata["id"]` where there is one,
            otherwise the section name and position, so a report is never about "the
            third message" in a list the reader cannot see.
        message: The message itself.
        pinned: Whether it may be evicted or folded away.
    """

    id: str
    message: Message
    pinned: bool = False


class CompactionOutcome(AdkModel):
    """What a strategy did, in terms a section report can repeat.

    Args:
        entries: What survives, in order.
        evicted: Ids dropped outright.
        summarised: Ids replaced by something shorter that stands for them.
    """

    entries: tuple[ContextEntry, ...] = ()
    evicted: tuple[str, ...] = ()
    summarised: tuple[str, ...] = ()


@runtime_checkable
class CompactionStrategy(Protocol):
    """How a section is reduced when it does not fit the tokens allowed for it."""

    async def compact(
        self,
        entries: Sequence[ContextEntry],
        *,
        budget_tokens: int,
        count: Callable[[Sequence[Message]], int],
    ) -> CompactionOutcome:
        """Return entries fitting `budget_tokens`, and what it cost to get there.

        Pinned entries are never removed. Where they alone exceed the budget, the
        strategy returns them anyway and the assembler fails closed — a strategy is not
        the place to decide that a caller's constraint was optional after all.

        Raises:
            ContextBudgetError: If reduction needed something that did not work, such as
                a summarisation call that failed.
        """
        ...


def _fits(entries: Sequence[ContextEntry], budget: int, count: TokenCounter) -> bool:
    return count([entry.message for entry in entries]) <= budget


class DropOldest:
    """Drop unpinned entries from the front until what is left fits.

    The cheapest strategy and the only one that needs nothing: no model call, no
    network, no failure mode beyond running out of things to drop.
    """

    async def compact(
        self,
        entries: Sequence[ContextEntry],
        *,
        budget_tokens: int,
        count: TokenCounter,
    ) -> CompactionOutcome:
        """Return the newest entries that fit, and the ids of the older ones dropped."""
        kept = list(entries)
        evicted: list[str] = []
        for entry in entries:
            if _fits(kept, budget_tokens, count):
                break
            if entry.pinned:
                continue
            kept.remove(entry)
            evicted.append(entry.id)
        return CompactionOutcome(entries=tuple(kept), evicted=tuple(evicted))


class PinAndFold:
    """Keep pinned entries verbatim and fold the rest into one short note.

    Model-free, so it cannot fail or cost anything, and lossy in a way the reader can
    see: the note says what it stands for rather than pretending nothing was there.
    """

    def __init__(self, *, width: int = 80) -> None:
        self._width = width

    async def compact(
        self,
        entries: Sequence[ContextEntry],
        *,
        budget_tokens: int,
        count: TokenCounter,
    ) -> CompactionOutcome:
        """Return the pinned entries plus a folded note for everything else."""
        pins = [entry for entry in entries if entry.pinned]
        folded = [entry for entry in entries if not entry.pinned]
        if not folded or _fits(entries, budget_tokens, count):
            return CompactionOutcome(entries=tuple(entries))

        note = _entry(
            "folded",
            _text("Earlier, in brief: " + " / ".join(_gist(e, self._width) for e in folded)),
            [entry.id for entry in folded],
        )
        kept = [*pins, note]
        if not _fits(kept, budget_tokens, count):
            kept = pins
        return CompactionOutcome(
            entries=tuple(kept), summarised=tuple(entry.id for entry in folded)
        )


class SummariseSpan:
    """Replace the oldest unpinned span with a model-written summary of it.

    The summary is produced by a real call, so it can fail; when it does the assembly
    fails with it. A half-written summary presented as the record of a conversation is
    worse than a conversation that would not fit.

    Args:
        provider: Who writes the summary. Often a smaller, cheaper model than the run's.
        model: The model to ask for.
        keep_recent: How many newest entries stay verbatim regardless.
    """

    def __init__(self, *, provider: ModelProvider, model: str, keep_recent: int = 1) -> None:
        self._provider = provider
        self._model = model
        self._keep_recent = keep_recent

    async def compact(
        self,
        entries: Sequence[ContextEntry],
        *,
        budget_tokens: int,
        count: TokenCounter,
    ) -> CompactionOutcome:
        """Return a summary of the oldest span plus whatever stays verbatim.

        Raises:
            ContextBudgetError: If the summarisation call failed, or answered with
                nothing usable.
        """
        verbatim = [entry for entry in entries if entry.pinned]
        recent = list(entries[len(entries) - self._keep_recent :])
        verbatim += [entry for entry in recent if entry not in verbatim]
        spanned = [entry for entry in entries if entry not in verbatim]
        if not spanned:
            return await DropOldest().compact(entries, budget_tokens=budget_tokens, count=count)

        summary = await self._summarise(spanned)
        kept = [_entry("summary", summary, [entry.id for entry in spanned]), *verbatim]
        outcome = await DropOldest().compact(kept, budget_tokens=budget_tokens, count=count)
        return CompactionOutcome(
            entries=outcome.entries,
            evicted=outcome.evicted,
            summarised=tuple(entry.id for entry in spanned),
        )

    async def _summarise(self, spanned: Sequence[ContextEntry]) -> Message:
        request = ModelRequest(
            model=self._model,
            messages=(
                _text(
                    "Summarise these turns in a few sentences, keeping every constraint,"
                    " preference and commitment stated in them.",
                    role="system",
                ),
                *(entry.message for entry in spanned),
            ),
        )
        try:
            response = await self._provider.complete(request)
        except Exception as failure:
            raise ContextBudgetError(
                "the span could not be summarised, so the prompt cannot be made to fit",
                required_tokens=0,
            ) from failure
        if not response.content.strip():
            raise ContextBudgetError("the summariser returned no summary")
        return _text(response.content)


def _text(text: str, *, role: Role = "system") -> Message:
    return Message(role=role, content=[TextPart(text=text)])


def _entry(id_: str, message: Message, stands_for: Sequence[str]) -> ContextEntry:
    marked = message.model_copy(
        update={"metadata": {**message.metadata, SUMMARY_OF: ",".join(stands_for)}}
    )
    return ContextEntry(id=id_, message=marked)


def _gist(entry: ContextEntry, width: int) -> str:
    said = " ".join(
        part.text for part in entry.message.content if isinstance(part, TextPart)
    ).strip()
    return said if len(said) <= width else said[: width - 1] + "…"
