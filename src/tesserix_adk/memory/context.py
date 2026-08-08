"""Building the prompt from what is remembered, under a budget that is not a guess.

Concatenating history until the provider complains means the provider decides what is
lost, and it decides by position. A plan says what the prompt is made of, what share of
the window each part may have, and what happens to the part that does not fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tesserix_adk.core.errors import CapabilityError, ContextBudgetError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import (
    Message,
    TextPart,
)
from tesserix_adk.memory.compaction import (
    PIN,
    SUMMARY_OF,
    CompactionStrategy,
    ContextEntry,
    DropOldest,
    PinAndFold,
)
from tesserix_adk.memory.records import MemoryKind, MemoryRecord

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.protocols import ModelProvider
    from tesserix_adk.memory.protocol import MemoryStore
    from tesserix_adk.memory.scope import MemoryScope

__all__ = [
    "AssembledContext",
    "ContextAssembler",
    "ContextPlan",
    "SectionOutcome",
    "SectionPlan",
]


class SectionPlan(AdkModel):
    """One part of the prompt, and what may happen to it under pressure.

    Args:
        name: What the caller passes its messages under, and what the report calls it.
        share: The fraction of the budget this section may occupy. Shares are of the
            whole budget, not of what previous sections left, so adding a section cannot
            silently shrink the ones after it.
        priority: Higher survives longer where the budget is short. Sections are
            reported in plan order regardless.
        pinned: Whether the whole section is non-evictable. Pinned sections take the
            room they need before anything else is allocated.
        compaction: Which strategy reduces it. Must be one the assembler was given.
    """

    name: str = Field(min_length=1)
    share: float = Field(gt=0.0, le=1.0)
    priority: int = 0
    pinned: bool = False
    compaction: str = "drop-oldest"


class ContextPlan(AdkModel):
    """What the prompt is made of, and how much room there is for it.

    Args:
        sections: The parts, in the order they appear in the prompt.
        budget_tokens: The ceiling. Where None, the provider's declared context window
            is used, so swapping a model changes the budget rather than breaking it.
        reserve_output_tokens: Room kept back for the answer. A prompt that exactly
            fills the window leaves the model nothing to say.
    """

    sections: tuple[SectionPlan, ...]
    budget_tokens: int | None = Field(default=None, gt=0)
    reserve_output_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_of_each_and_no_more_than_all(self) -> ContextPlan:
        if not self.sections:
            return self._refuse("a plan needs at least one section")
        names = [section.name for section in self.sections]
        repeated = {name for name in names if names.count(name) > 1}
        if repeated:
            return self._refuse(f"{', '.join(sorted(repeated))} appears twice")
        if sum(section.share for section in self.sections) > 1.0:
            return self._refuse("the shares add up to more than the budget")
        return self

    def _refuse(self, why: str) -> ContextPlan:
        raise ValueError(why)


class SectionOutcome(AdkModel):
    """What became of one section, in terms a reader can act on.

    Args:
        name: The section this is about.
        tokens: What it occupies in the assembled prompt.
        kept: How many messages survived.
        evicted: Ids dropped outright.
        summarised: Ids replaced by something that stands for them.
    """

    name: str
    tokens: int
    kept: int
    evicted: tuple[str, ...] = ()
    summarised: tuple[str, ...] = ()


class AssembledContext(AdkModel):
    """A prompt that fits, and the account of what it took to make it fit.

    Args:
        messages: The prompt, in plan order.
        tokens: What it occupies, by the provider's own count.
        budget_tokens: What there was room for.
        sections: One outcome per planned section, in plan order.
    """

    messages: tuple[Message, ...]
    tokens: int
    budget_tokens: int
    sections: tuple[SectionOutcome, ...]

    def span_attributes(self) -> dict[str, int]:
        """Return counts for a span, and nothing that was said.

        A trace of what a prompt cost is useful on every run; a trace of what was in it
        is a copy of the conversation in whatever system holds the traces.
        """
        attributes = {
            "context.tokens": self.tokens,
            "context.budget_tokens": self.budget_tokens,
            "context.evicted": sum(len(section.evicted) for section in self.sections),
            "context.summarised": sum(len(section.summarised) for section in self.sections),
        }
        for section in self.sections:
            attributes[f"context.section.{section.name}.tokens"] = section.tokens
            attributes[f"context.section.{section.name}.kept"] = section.kept
        return attributes


class ContextAssembler:
    """Builds a prompt from a plan, a provider's token count and a set of strategies.

    Args:
        plan: What the prompt is made of.
        provider: Whose window is being filled and whose tokeniser is doing the counting.
            Read on every assembly, so a model swapped mid-session is budgeted for
            correctly rather than overflowed once.
        strategies: Compaction strategies by name, merged over the model-free built-ins
            `drop-oldest` and `pin-and-fold`. `summarise-span` needs a provider, so it is
            supplied here rather than assumed.
        memory: Where summaries are written back, with provenance to the turns they
            replace. Optional: a summary nobody kept is a summary paid for twice.
        scope: Whose memory to write them under. Required alongside `memory`.

    Raises:
        ValueError: If a section names a compaction strategy the assembler does not have,
            or `memory` is given without a `scope`.
    """

    def __init__(
        self,
        plan: ContextPlan,
        *,
        provider: ModelProvider,
        strategies: Mapping[str, CompactionStrategy] | None = None,
        memory: MemoryStore | None = None,
        scope: MemoryScope | None = None,
    ) -> None:
        self._plan = plan
        self._provider = provider
        self.strategies: dict[str, CompactionStrategy] = {
            "drop-oldest": DropOldest(),
            "pin-and-fold": PinAndFold(),
            **(strategies or {}),
        }
        if memory is not None and scope is None:
            raise ValueError("a store to write summaries to needs a scope to write them under")
        self._memory = memory
        self._scope = scope
        for section in plan.sections:
            if section.compaction not in self.strategies:
                raise ValueError(f"there is no compaction strategy called {section.compaction!r}")

    async def assemble(self, sections: Mapping[str, Sequence[Message]]) -> AssembledContext:
        """Return a prompt within the budget, and what it cost to get there.

        Raises:
            CapabilityError: If no budget was set and the provider declares no window.
            ContextBudgetError: If pinned content alone exceeds the budget, or a
                compaction step failed. The kit does not emit an over-budget prompt.
            ValueError: If `sections` carries a name the plan does not declare.
        """
        unplanned = set(sections) - {section.name for section in self._plan.sections}
        if unplanned:
            raise ValueError(f"{', '.join(sorted(unplanned))} is not in the plan")

        budget = self._budget()
        entries = {
            section.name: _entries(section, sections.get(section.name, ()))
            for section in self._plan.sections
        }
        held = self._pinned_cost(entries, budget)
        allowances = self._allowances(budget - held)

        outcomes: list[SectionOutcome] = []
        messages: list[Message] = []
        for section in self._plan.sections:
            candidates = entries[section.name]
            allowance = budget - held if section.pinned else allowances[section.name]
            outcome = await self.strategies[section.compaction].compact(
                candidates, budget_tokens=allowance, count=self._provider.count_tokens
            )
            kept = [entry.message for entry in outcome.entries]
            messages += kept
            outcomes.append(
                SectionOutcome(
                    name=section.name,
                    tokens=self._provider.count_tokens(kept),
                    kept=len(kept),
                    evicted=outcome.evicted,
                    summarised=outcome.summarised,
                )
            )
            await self._remember(section.name, outcome.entries)

        total = self._provider.count_tokens(messages)
        if total > budget:
            raise ContextBudgetError(
                "the prompt could not be reduced to the budget",
                budget_tokens=budget,
                required_tokens=total,
            )
        return AssembledContext(
            messages=tuple(messages),
            tokens=total,
            budget_tokens=budget,
            sections=tuple(outcomes),
        )

    def _budget(self) -> int:
        if self._plan.budget_tokens is not None:
            return self._plan.budget_tokens - self._plan.reserve_output_tokens
        window = self._provider.capabilities.context_window_tokens
        if window is None:
            raise CapabilityError(
                "this provider declares no context window, so the plan must set a budget",
                capability="context window",
                provider=self._provider.name,
            )
        return window - self._plan.reserve_output_tokens

    def _pinned_cost(self, entries: Mapping[str, Sequence[ContextEntry]], budget: int) -> int:
        held = self._provider.count_tokens(
            [
                entry.message
                for section in self._plan.sections
                if section.pinned
                for entry in entries[section.name]
            ]
        )
        if held > budget:
            raise ContextBudgetError(
                "what is pinned does not fit on its own",
                budget_tokens=budget,
                required_tokens=held,
            )
        return held

    def _allowances(self, room: int) -> dict[str, int]:
        evictable = [section for section in self._plan.sections if not section.pinned]
        total = sum(section.share for section in evictable)
        if not total:
            return {}
        return {
            section.name: int(room * section.share / total)
            for section in sorted(evictable, key=lambda section: section.name)
        }

    async def _remember(self, section: str, entries: Sequence[ContextEntry]) -> None:
        if self._memory is None or self._scope is None:
            return
        for entry in entries:
            replaced = entry.message.metadata.get(SUMMARY_OF)
            if not replaced:
                continue
            await self._memory.log(
                self._scope,
                MemoryRecord(
                    id=f"summary:{section}:{replaced}",
                    kind=MemoryKind.EPISODIC,
                    scope=self._scope,
                    key=f"summary:{section}:{replaced}",
                    value=_said(entry.message),
                    source=f"compaction:{section}",
                ),
            )


def _entries(section: SectionPlan, messages: Sequence[Message]) -> tuple[ContextEntry, ...]:
    return tuple(
        ContextEntry(
            id=message.metadata.get("id") or f"{section.name}:{at}",
            message=message,
            pinned=section.pinned or message.metadata.get(PIN) == "true",
        )
        for at, message in enumerate(messages)
    )


def _said(message: Message) -> str:
    return " ".join(part.text for part in message.content if isinstance(part, TextPart)).strip()
