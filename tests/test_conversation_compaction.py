"""What a compacted conversation still owes the reader: every source it had before."""

from __future__ import annotations

import pytest

from tesserix_adk.core import Agent, Message, ProvenanceLostError, TextPart
from tesserix_adk.runtime.compaction import (
    COMPACTED,
    Compaction,
    CompactionEvent,
    Summariser,
    citations_of,
    cited,
    compact_conversation,
)
from tesserix_adk.runtime.prompt import assemble_prompt

pytestmark = pytest.mark.anyio

AGENT = Agent(name="desk", instructions="Answer refunds.", model="fake", free_text=True)


def said(text: str, *, role: str = "user", sources: tuple[str, ...] = ()) -> Message:
    """One turn, carrying the citation ids it rests on."""
    message = Message(role=role, content=[TextPart(text=text)])  # type: ignore[arg-type]
    return cited(message, sources) if sources else message


def conversation(turns: int = 12) -> tuple[Message, ...]:
    """A conversation long enough to need compacting, sourced every other turn."""
    return tuple(
        said(
            f"Turn {index}. " + "The claim is stated at some length here. " * 4,
            role="user" if index % 2 == 0 else "assistant",
            sources=(f"c{index}",) if index % 2 else (),
        )
        for index in range(turns)
    )


class Paraphrase:
    """A summariser that keeps the provenance it was handed."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, messages: tuple[Message, ...]) -> Message:
        self.calls += 1
        sources = tuple(dict.fromkeys(id_ for m in messages for id_ in citations_of(m)))
        return cited(said(f"Earlier: {len(messages)} turns.", role="system"), sources)


class Forgetful:
    """A summariser that writes prose and drops the sources it was built from."""

    async def __call__(self, messages: tuple[Message, ...]) -> Message:
        return said(f"Earlier: {len(messages)} turns about refunds.", role="system")


class TestCompactionRuns:
    """Rolling, above a threshold the caller sets."""

    async def test_a_short_conversation_is_left_alone(self) -> None:
        history = conversation(2)

        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=10_000
        )

        assert result.ran is False
        assert result.history == history

    async def test_a_long_one_is_folded_at_the_front(self) -> None:
        history = conversation()

        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        assert result.ran is True
        assert len(result.history) < len(history)
        assert result.history[-4:] == history[-4:]

    async def test_the_summary_says_what_it_stands_for(self) -> None:
        result = await compact_conversation(
            conversation(), summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        assert result.history[0].metadata[COMPACTED] == "true"
        assert result.folded == 8

    async def test_the_recent_turns_are_never_folded(self) -> None:
        result = await compact_conversation(
            conversation(), summarise=Paraphrase(), threshold_tokens=1, keep_recent=2
        )

        assert result.folded == 10

    async def test_a_conversation_of_only_recent_turns_cannot_be_folded(self) -> None:
        history = conversation(3)

        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=1, keep_recent=4
        )

        assert result.ran is False
        assert result.history == history


class TestProvenanceSurvives:
    """The primary scenario: nothing loses its source."""

    async def test_every_citation_before_is_present_after(self) -> None:
        history = conversation()
        before = {id_ for message in history for id_ in citations_of(message)}

        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        after = {id_ for message in result.history for id_ in citations_of(message)}
        assert before == after

    async def test_the_carried_ones_are_reported(self) -> None:
        result = await compact_conversation(
            conversation(), summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        assert result.citations == ("c1", "c3", "c5", "c7")

    async def test_a_summariser_that_drops_a_source_fails_closed(self) -> None:
        history = conversation()

        with pytest.raises(ProvenanceLostError) as refused:
            await compact_conversation(
                history, summarise=Forgetful(), threshold_tokens=100, keep_recent=4
            )

        assert refused.value.lost == ("c1", "c3", "c5", "c7")

    async def test_nothing_is_emitted_when_it_fails(self) -> None:
        history = conversation()

        with pytest.raises(ProvenanceLostError):
            await compact_conversation(
                history, summarise=Forgetful(), threshold_tokens=100, keep_recent=4
            )

        assert {id_ for message in history for id_ in citations_of(message)} == {
            "c1",
            "c3",
            "c5",
            "c7",
            "c9",
            "c11",
        }

    async def test_an_unsourced_span_needs_no_sources(self) -> None:
        history = tuple(said(f"Turn {index}. " + "words " * 20) for index in range(12))

        result = await compact_conversation(
            history, summarise=Forgetful(), threshold_tokens=100, keep_recent=4
        )

        assert result.ran is True
        assert result.citations == ()


class TestCompactionIsIdempotent:
    """Running it twice is running it once."""

    async def test_a_second_pass_changes_nothing(self) -> None:
        summarise = Paraphrase()
        once = await compact_conversation(
            conversation(), summarise=summarise, threshold_tokens=100, keep_recent=4
        )

        twice = await compact_conversation(
            once.history, summarise=summarise, threshold_tokens=100, keep_recent=4
        )

        assert twice.history == once.history
        assert twice.ran is False
        assert summarise.calls == 1

    async def test_a_further_pass_folds_the_earlier_summary_in(self) -> None:
        summarise = Paraphrase()
        once = await compact_conversation(
            conversation(), summarise=summarise, threshold_tokens=100, keep_recent=4
        )

        twice = await compact_conversation(
            (*once.history, *conversation(8)),
            summarise=summarise,
            threshold_tokens=100,
            keep_recent=4,
        )

        carried = {id_ for message in twice.history for id_ in citations_of(message)}
        assert {"c1", "c3", "c5", "c7"} <= carried


class TestTheCacheablePrefix:
    """Compaction is a conversation-layer operation and touches nothing above it."""

    async def test_the_fingerprint_is_unchanged(self) -> None:
        history = conversation()
        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        before = assemble_prompt(AGENT, "and now?", history=history, pinned=("a case file",))
        after = assemble_prompt(AGENT, "and now?", history=result.history, pinned=("a case file",))

        assert after.fingerprint == before.fingerprint
        assert after.prefix == before.prefix

    async def test_the_prompt_is_shorter(self) -> None:
        history = conversation()
        result = await compact_conversation(
            history, summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )

        before = assemble_prompt(AGENT, "and now?", history=history)
        after = assemble_prompt(AGENT, "and now?", history=result.history)

        assert len(after.messages) < len(before.messages)


class TestWhatItRecords:
    """An auditable event, and span attributes that carry no conversation."""

    async def test_the_event_names_what_was_folded(self) -> None:
        result = await compact_conversation(
            conversation(),
            summarise=Paraphrase(),
            threshold_tokens=100,
            keep_recent=4,
            run_id="run-1",
        )

        event = result.event
        assert isinstance(event, CompactionEvent)
        assert event.run_id == "run-1"
        assert event.folded == 8
        assert event.tokens_after < event.tokens_before
        assert event.citations == ("c1", "c3", "c5", "c7")

    async def test_a_pass_that_did_nothing_records_nothing(self) -> None:
        result = await compact_conversation(
            conversation(2), summarise=Paraphrase(), threshold_tokens=10_000
        )

        assert result.event is None

    async def test_the_attributes_carry_no_prose(self) -> None:
        result = await compact_conversation(
            conversation(), summarise=Paraphrase(), threshold_tokens=100, keep_recent=4
        )
        assert result.event is not None

        attributes = result.event.attributes()

        assert attributes["adk.compaction.folded"] == "8"
        assert attributes["adk.compaction.citations"] == "4"
        assert not any("Turn" in value for value in attributes.values())


class TestTheArguments:
    """The settings a caller gets wrong first."""

    @pytest.mark.parametrize(("threshold", "keep"), [(0, 4), (-1, 4), (100, -1)])
    async def test_a_nonsense_setting_is_refused(self, threshold: int, keep: int) -> None:
        with pytest.raises(ValueError, match=r"threshold_tokens|keep_recent"):
            await compact_conversation(
                conversation(), summarise=Paraphrase(), threshold_tokens=threshold, keep_recent=keep
            )

    async def test_an_empty_conversation_is_nothing_to_do(self) -> None:
        result = await compact_conversation((), summarise=Paraphrase(), threshold_tokens=1)

        assert result == Compaction(history=())

    async def test_the_caller_may_count_tokens_its_own_way(self) -> None:
        counted: list[str] = []

        def tokenizer(text: str) -> int:
            counted.append(text)
            return len(text)

        await compact_conversation(
            conversation(),
            summarise=Paraphrase(),
            threshold_tokens=100,
            keep_recent=4,
            tokenizer=tokenizer,
        )

        assert counted

    async def test_the_summariser_is_a_protocol(self) -> None:
        assert isinstance(Paraphrase(), Summariser)
