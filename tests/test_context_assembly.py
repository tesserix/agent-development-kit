"""Building a prompt that fits, and saying what it cost to make it fit."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

import pytest

from tesserix_adk.core import (
    CapabilityError,
    ContextBudgetError,
    Message,
    ModelResponse,
    Role,
    TextPart,
)
from tesserix_adk.core.capabilities import ModelCapabilities
from tesserix_adk.memory import (
    AssembledContext,
    CompactionOutcome,
    ContextAssembler,
    ContextEntry,
    ContextPlan,
    DropOldest,
    MemoryKind,
    MemoryQuery,
    MemoryScope,
    PinAndFold,
    SectionPlan,
    SummariseSpan,
    TokenCounter,
    pinned,
)
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1")


def said(text: str, *, role: Role = "user", mid: str | None = None) -> Message:
    """One turn of `text`, identified where the caller wants to track it."""
    return Message(role=role, content=[TextPart(text=text)], metadata={"id": mid} if mid else {})


def provider(
    *, window: int | None = None, responses: Sequence[ModelResponse] = ()
) -> ScriptedProvider:
    """A provider that counts tokens and declares the window the test needs."""
    return ScriptedProvider(
        *responses,
        capabilities=ModelCapabilities(context_window_tokens=window)
        if window
        else ModelCapabilities(),
    )


def plan(*sections: SectionPlan, budget: int | None = None, reserve: int = 0) -> ContextPlan:
    """A plan over `sections`, budgeted where the test says so."""
    return ContextPlan(sections=sections, budget_tokens=budget, reserve_output_tokens=reserve)


RECENT = SectionPlan(name="recent", share=1.0)


class TestTheBudgetIsTheProviderSAnswerNotAnEstimate:
    async def test_the_window_the_provider_declares_is_the_budget(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=64)).assemble(
            {"recent": [said("hello")]}
        )

        assert assembled.budget_tokens == 64

    async def test_room_reserved_for_the_answer_comes_off_the_budget(self) -> None:
        assembled = await ContextAssembler(
            plan(RECENT, reserve=24), provider=provider(window=64)
        ).assemble({"recent": [said("hello")]})

        assert assembled.budget_tokens == 40

    async def test_an_explicit_budget_beats_the_declared_window(self) -> None:
        assembled = await ContextAssembler(
            plan(RECENT, budget=30), provider=provider(window=64)
        ).assemble({"recent": [said("hello")]})

        assert assembled.budget_tokens == 30

    async def test_a_provider_that_declares_no_window_cannot_be_budgeted_for(self) -> None:
        with pytest.raises(CapabilityError, match="context window"):
            await ContextAssembler(plan(RECENT), provider=provider()).assemble({"recent": []})

    async def test_counting_is_the_provider_s_own(self) -> None:
        counted = provider(window=64)
        history = [said("x" * 40)]

        assembled = await ContextAssembler(plan(RECENT), provider=counted).assemble(
            {"recent": history}
        )

        assert assembled.tokens == counted.count_tokens(history)

    async def test_a_smaller_model_mid_session_is_read_again(self) -> None:
        class Shrinking(ScriptedProvider):
            """A provider swapped for one with a smaller window between calls."""

            windows: ClassVar[list[int]] = [400, 40]

            @property
            def capabilities(self) -> ModelCapabilities:
                """Declare the next window in the list, so the second read is smaller."""
                return ModelCapabilities(context_window_tokens=self.windows.pop(0))

        assembler = ContextAssembler(plan(RECENT), provider=Shrinking())
        history = [said("x" * 200, mid="a"), said("y" * 200, mid="b")]

        first = await assembler.assemble({"recent": history})
        second = await assembler.assemble({"recent": history})

        assert first.tokens > second.tokens
        assert second.tokens <= 40


class TestWhatIsPinnedSurvives:
    async def test_a_pinned_section_appears_verbatim_when_history_overflows(self) -> None:
        assembled = await ContextAssembler(
            plan(
                SectionPlan(name="constraints", share=0.2, pinned=True),
                SectionPlan(name="recent", share=0.8),
            ),
            provider=provider(window=30),
        ).assemble(
            {
                "constraints": [said("window seat only"), said("no peanuts")],
                "recent": [said("z" * 400, mid="chatter")],
            }
        )

        texts = [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ]
        assert "window seat only" in texts
        assert "no peanuts" in texts

    async def test_a_pinned_message_inside_an_evictable_section_is_not_evicted(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=20)).assemble(
            {
                "recent": [
                    pinned(said("allergic to peanuts", mid="allergy")),
                    said("z" * 400, mid="chatter"),
                ]
            }
        )

        texts = [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ]
        assert "allergic to peanuts" in texts
        assert assembled.sections[0].evicted == ("chatter",)

    async def test_pinned_content_alone_over_budget_fails_closed(self) -> None:
        with pytest.raises(ContextBudgetError) as refused:
            await ContextAssembler(
                plan(SectionPlan(name="constraints", share=1.0, pinned=True)),
                provider=provider(window=10),
            ).assemble({"constraints": [said("q" * 400)]})

        assert refused.value.budget_tokens == 10
        assert refused.value.required_tokens > 10


class TestWhatDoesNotFitIsCompactedNotTruncated:
    async def test_drop_oldest_drops_from_the_front(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=30)).assemble(
            {"recent": [said("a" * 200, mid="old"), said("b" * 40, mid="new")]}
        )

        assert assembled.sections[0].evicted == ("old",)
        assert [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ] == ["b" * 40]

    async def test_a_message_larger_than_its_whole_section_is_evicted(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=5)).assemble(
            {"recent": [said("c" * 400, mid="huge")]}
        )

        assert assembled.messages == ()
        assert assembled.sections[0].evicted == ("huge",)

    async def test_a_summary_replaces_the_span_it_stands_for(self) -> None:
        summariser = provider(responses=[ModelResponse(content="they discussed baggage")])
        assembled = await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
            provider=provider(window=40),
            strategies={"summarise-span": SummariseSpan(provider=summariser, model="m")},
        ).assemble(
            {
                "recent": [
                    said("a" * 200, mid="t1"),
                    said("b" * 200, mid="t2"),
                    said("and the seat?", mid="t3"),
                ]
            }
        )

        texts = [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ]
        assert any("they discussed baggage" in text for text in texts)
        assert assembled.sections[0].summarised == ("t1", "t2")

    async def test_a_summary_is_written_back_to_episodic_memory_with_its_provenance(self) -> None:
        store = InMemoryMemoryStore(clock=FakeClock())
        summariser = provider(responses=[ModelResponse(content="they discussed baggage")])
        await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
            provider=provider(window=40),
            strategies={"summarise-span": SummariseSpan(provider=summariser, model="m")},
            memory=store,
            scope=SCOPE,
        ).assemble({"recent": [said("a" * 400, mid="t1"), said("ok", mid="t2")]})

        kept = await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
        assert [hit.record.value for hit in kept] == ["they discussed baggage"]
        assert kept[0].record.source == "compaction:recent"

    async def test_folding_needs_no_model_call(self) -> None:
        assembled = await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="pin-and-fold")),
            provider=provider(window=30),
        ).assemble(
            {
                "recent": [
                    pinned(said("allergic to peanuts", mid="allergy")),
                    said("d" * 400, mid="chatter"),
                ]
            }
        )

        texts = [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ]
        assert "allergic to peanuts" in texts
        assert assembled.sections[0].summarised == ("chatter",)

    async def test_a_summary_sees_only_the_turns_it_was_given(self) -> None:
        summariser = provider(responses=[ModelResponse(content="a card was used")])
        await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
            provider=provider(window=40),
            strategies={"summarise-span": SummariseSpan(provider=summariser, model="m")},
        ).assemble({"recent": [said("card [redacted]", mid="t1"), said("ok", mid="t2")]})

        asked = "".join(
            part.text
            for request in summariser.requests
            for message in request.messages
            for part in message.content
            if isinstance(part, TextPart)
        )
        assert "[redacted]" in asked


class TestAnOverBudgetPromptIsNeverEmitted:
    async def test_a_failed_summary_fails_the_assembly(self) -> None:
        summariser = provider(responses=[])
        broken = SummariseSpan(provider=ScriptedProvider(TimeoutError("upstream")), model="m")
        assert summariser is not None

        with pytest.raises(ContextBudgetError) as refused:
            await ContextAssembler(
                plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
                provider=provider(window=20),
                strategies={"summarise-span": broken},
            ).assemble({"recent": [said("e" * 400, mid="t1"), said("ok", mid="t2")]})

        assert isinstance(refused.value.__cause__, TimeoutError)

    async def test_a_partial_summary_is_not_a_summary(self) -> None:
        empty = SummariseSpan(provider=ScriptedProvider(ModelResponse(content="")), model="m")

        with pytest.raises(ContextBudgetError, match="no summary"):
            await ContextAssembler(
                plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
                provider=provider(window=20),
                strategies={"summarise-span": empty},
            ).assemble({"recent": [said("f" * 400, mid="t1"), said("ok", mid="t2")]})

    async def test_cancellation_is_cancellation_not_a_budget_failure(self) -> None:
        started = asyncio.Event()

        class Stalls(ScriptedProvider):
            """A summariser that never answers, so the wait can be cancelled."""

            async def complete(self, request: object) -> ModelResponse:  # noqa: ARG002 — it never reads it
                """Signal that the call began, then wait to be cancelled."""
                started.set()
                await asyncio.Event().wait()
                raise AssertionError

        assembling = asyncio.create_task(
            ContextAssembler(
                plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
                provider=provider(window=20),
                strategies={"summarise-span": SummariseSpan(provider=Stalls(), model="m")},
            ).assemble({"recent": [said("g" * 400, mid="t1"), said("ok", mid="t2")]})
        )
        await started.wait()
        assembling.cancel()

        with pytest.raises(asyncio.CancelledError):
            await assembling

    async def test_an_every_section_pinned_plan_that_fits_still_assembles(self) -> None:
        assembled = await ContextAssembler(
            plan(SectionPlan(name="constraints", share=1.0, pinned=True)),
            provider=provider(window=200),
        ).assemble({"constraints": [said("window seat only")]})

        assert assembled.sections[0].kept == 1

    async def test_the_assembled_prompt_is_within_the_budget(self) -> None:
        assembled = await ContextAssembler(
            plan(
                SectionPlan(name="system", share=0.1),
                SectionPlan(name="recent", share=0.9),
            ),
            provider=provider(window=50),
        ).assemble(
            {
                "system": [said("be helpful", role="system")],
                "recent": [said("h" * 400, mid=f"t{n}") for n in range(6)],
            }
        )

        assert assembled.tokens <= assembled.budget_tokens


class TestTheStrategiesOnTheirOwn:
    async def test_folding_leaves_a_section_that_already_fits_alone(self) -> None:
        assembled = await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="pin-and-fold")),
            provider=provider(window=200),
        ).assemble({"recent": [said("hi", mid="t1"), said("hello", mid="t2")]})

        assert assembled.sections[0].kept == 2
        assert assembled.sections[0].summarised == ()

    async def test_a_note_too_long_for_the_room_leaves_only_the_pins(self) -> None:
        assembled = await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="pin-and-fold")),
            provider=provider(window=12),
        ).assemble(
            {
                "recent": [
                    pinned(said("no peanuts", mid="allergy")),
                    *(said("k" * 200, mid=f"t{n}") for n in range(4)),
                ]
            }
        )

        texts = [
            part.text
            for m in assembled.messages
            for part in m.content
            if isinstance(part, TextPart)
        ]
        assert texts == ["no peanuts"]

    async def test_a_span_with_nothing_old_in_it_is_not_summarised(self) -> None:
        summariser = provider(responses=[])
        assembled = await ContextAssembler(
            plan(SectionPlan(name="recent", share=1.0, compaction="summarise-span")),
            provider=provider(window=200),
            strategies={"summarise-span": SummariseSpan(provider=summariser, model="m")},
        ).assemble({"recent": [said("hi", mid="t1")]})

        assert assembled.sections[0].summarised == ()
        assert summariser.requests == []

    async def test_a_strategy_that_reduces_nothing_does_not_get_a_prompt_sent(self) -> None:
        class Stubborn:
            """A strategy that hands back everything it was given."""

            async def compact(
                self,
                entries: Sequence[ContextEntry],
                *,
                budget_tokens: int,  # noqa: ARG002 — that is the point of it
                count: TokenCounter,  # noqa: ARG002 — it never counts
            ) -> CompactionOutcome:
                """Return the entries unchanged, budget or no budget."""
                return CompactionOutcome(entries=tuple(entries))

        with pytest.raises(ContextBudgetError, match="could not be reduced"):
            await ContextAssembler(
                plan(SectionPlan(name="recent", share=1.0, compaction="stubborn")),
                provider=provider(window=10),
                strategies={"stubborn": Stubborn()},
            ).assemble({"recent": [said("l" * 400, mid="t1")]})

    def test_a_store_with_no_scope_has_nowhere_to_write(self) -> None:
        with pytest.raises(ValueError, match="needs a scope"):
            ContextAssembler(
                plan(RECENT),
                provider=provider(window=40),
                memory=InMemoryMemoryStore(clock=FakeClock()),
            )


class TestTheResultSaysWhatHappened:
    async def test_it_reports_the_sections_in_plan_order(self) -> None:
        assembled = await ContextAssembler(
            plan(
                SectionPlan(name="system", share=0.5),
                SectionPlan(name="recent", share=0.5),
            ),
            provider=provider(window=200),
        ).assemble({"system": [said("be helpful")], "recent": [said("hi")]})

        assert [section.name for section in assembled.sections] == ["system", "recent"]

    async def test_a_section_the_caller_left_out_is_still_reported(self) -> None:
        assembled = await ContextAssembler(
            plan(SectionPlan(name="profile", share=0.5), SectionPlan(name="recent", share=0.5)),
            provider=provider(window=200),
        ).assemble({"recent": [said("hi")]})

        assert assembled.sections[0].kept == 0

    async def test_span_attributes_carry_no_content(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=200)).assemble(
            {"recent": [said("the secret is hunter2", mid="t1")]}
        )

        attributes = assembled.span_attributes()
        assert attributes["context.tokens"] == assembled.tokens
        assert "hunter2" not in " ".join(str(value) for value in attributes.values())

    async def test_it_is_a_frozen_record_of_one_assembly(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=200)).assemble(
            {"recent": [said("hi")]}
        )

        with pytest.raises(ValueError, match="frozen"):
            assembled.tokens = 0

    async def test_the_same_inputs_assemble_the_same_prompt(self) -> None:
        history = {"recent": [said("i" * 200, mid="t1"), said("j" * 40, mid="t2")]}
        assembler = ContextAssembler(plan(RECENT), provider=provider(window=40))

        first = await assembler.assemble(history)
        second = await assembler.assemble(history)

        assert first == second


class TestAPlanIsCheckedBeforeItIsUsed:
    def test_two_sections_cannot_share_a_name(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            plan(SectionPlan(name="recent", share=0.5), SectionPlan(name="recent", share=0.5))

    def test_shares_over_the_whole_budget_are_not_shares(self) -> None:
        with pytest.raises(ValueError, match="more than the budget"):
            plan(SectionPlan(name="a", share=0.8), SectionPlan(name="b", share=0.8))

    def test_a_plan_with_no_sections_assembles_nothing(self) -> None:
        with pytest.raises(ValueError, match="at least one section"):
            plan()

    def test_a_strategy_the_assembler_does_not_have_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="no compaction strategy called"):
            ContextAssembler(
                plan(SectionPlan(name="recent", share=1.0, compaction="hand-wave")),
                provider=provider(window=40),
            )

    def test_the_built_ins_are_there_without_being_asked_for(self) -> None:
        assembler = ContextAssembler(plan(RECENT), provider=provider(window=40))

        assert isinstance(assembler.strategies["drop-oldest"], DropOldest)
        assert isinstance(assembler.strategies["pin-and-fold"], PinAndFold)

    async def test_a_section_the_plan_never_declared_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the plan"):
            await ContextAssembler(plan(RECENT), provider=provider(window=40)).assemble(
                {"recent": [], "smuggled": [said("hi")]}
            )


class TestTheResultIsAContext:
    async def test_it_is_an_assembled_context(self) -> None:
        assembled = await ContextAssembler(plan(RECENT), provider=provider(window=40)).assemble(
            {"recent": [said("hi")]}
        )

        assert isinstance(assembled, AssembledContext)
