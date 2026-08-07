"""What goes into the context, and what leaves it when there is no room.

The largest waste in a retrieval loop is re-injecting a chunk the model already has. On
CPU that waste is paid in seconds of prefill, every turn, so admission is keyed and
eviction is ordered rather than left to whatever the caller happened to append.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import ContextWindowExceededError
from tesserix_adk.runtime import ContextWindow, PromptLayer, Segment


def chunk(text: str, *, key: str | None = None, score: float = 0.0) -> Segment:
    return Segment(text=text, layer=PromptLayer.RETRIEVED, key=key, score=score)


def turn(text: str) -> Segment:
    return Segment(text=text, layer=PromptLayer.CONVERSATION)


def pinned(text: str) -> Segment:
    return Segment(text=text, layer=PromptLayer.PINNED, key=f"pinned:{text}")


def window(limit: int = 1_000) -> ContextWindow:
    return ContextWindow(limit_tokens=limit, tokenizer=lambda text: len(text.split()))


class TestKeyedAdmission:
    def test_the_same_chunk_retrieved_twice_is_admitted_once(self) -> None:
        """Re-injecting what the model already has is prefill paid for nothing."""
        held = window()
        assert held.admit(chunk("the lease was signed on a Tuesday", key="p12")) is True
        assert held.admit(chunk("the lease was signed on a Tuesday", key="p12")) is False
        assert len(held.segments) == 1

    def test_a_key_is_matched_across_layers_not_within_one(self) -> None:
        """The same page pinned and then retrieved is still the same page."""
        held = window()
        held.admit(Segment(text="page 12", layer=PromptLayer.PINNED, key="p12"))
        assert held.admit(chunk("page 12", key="p12")) is False

    def test_an_unkeyed_segment_is_never_deduped(self) -> None:
        """Two turns of conversation can read identically and both still happened."""
        held = window()
        assert held.admit(turn("yes")) is True
        assert held.admit(turn("yes")) is True
        assert len(held.segments) == 2

    def test_a_repeat_does_not_move_what_was_already_admitted(self) -> None:
        held = window()
        held.admit(chunk("first", key="a"))
        held.admit(chunk("second", key="b"))
        held.admit(chunk("first again", key="a"))
        assert [segment.text for segment in held.segments] == ["first", "second"]

    def test_it_says_whether_a_key_is_held(self) -> None:
        held = window()
        held.admit(chunk("first", key="a"))
        assert held.holds("a")
        assert not held.holds("b")


class TestEviction:
    def test_nothing_is_evicted_while_it_fits(self) -> None:
        held = window(limit=100)
        held.admit(turn("a short question"))
        assert held.fit() == ()

    def test_the_oldest_conversation_goes_first(self) -> None:
        held = window(limit=4)
        for text in ("one two", "three four", "five six"):
            held.admit(turn(text))
        evicted = held.fit()
        assert [segment.text for segment in evicted] == ["one two"]
        assert [segment.text for segment in held.segments] == ["three four", "five six"]

    def test_then_the_lowest_scored_retrieval(self) -> None:
        """Conversation is dropped before retrieval, which was scored for this turn."""
        held = window(limit=3)
        held.admit(chunk("relevant chunk", key="a", score=0.9))
        held.admit(chunk("marginal chunk", key="b", score=0.1))
        held.admit(turn("old turn"))
        evicted = held.fit()
        assert [segment.text for segment in evicted] == ["old turn", "marginal chunk"]
        assert [segment.text for segment in held.segments] == ["relevant chunk"]

    def test_the_prefix_is_never_evicted_even_where_dropping_it_would_be_quicker(self) -> None:
        """Evicting the prefix refills every cache; it is not a saving, it is a cost."""
        held = window(limit=6)
        held.admit(pinned("a very long pinned file indeed"))
        held.admit(turn("one two"))
        held.fit()
        assert [segment.layer for segment in held.segments] == [PromptLayer.PINNED]

    def test_a_prefix_that_cannot_fit_alone_is_refused_rather_than_trimmed(self) -> None:
        held = window(limit=2)
        held.admit(pinned("five words in this file"))
        with pytest.raises(ContextWindowExceededError) as refused:
            held.fit()
        assert refused.value.limit == 2
        assert refused.value.counted == 5

    def test_eviction_frees_the_key_for_readmission(self) -> None:
        """A chunk dropped for room is not a chunk the model has; it can come back."""
        held = window(limit=2)
        held.admit(chunk("one two", key="a"))
        held.admit(chunk("three four", key="b", score=1.0))
        held.fit()
        assert not held.holds("a")
        assert held.admit(chunk("one two", key="a")) is True

    def test_what_left_is_returned_so_it_can_be_logged(self) -> None:
        held = window(limit=2)
        held.admit(turn("one two"))
        held.admit(turn("three four"))
        assert [segment.text for segment in held.fit()] == ["one two"]


class TestWhatItHolds:
    def test_segments_come_back_in_layer_order_whatever_order_they_arrived(self) -> None:
        held = window()
        held.admit(turn("the question"))
        held.admit(chunk("a document", key="d"))
        held.admit(pinned("the file"))
        assert [segment.layer for segment in held.segments] == [
            PromptLayer.PINNED,
            PromptLayer.RETRIEVED,
            PromptLayer.CONVERSATION,
        ]

    def test_the_texts_of_one_layer_are_what_assembly_takes(self) -> None:
        held = window()
        held.admit(chunk("first", key="a"))
        held.admit(chunk("second", key="b"))
        assert held.texts(PromptLayer.RETRIEVED) == ("first", "second")

    def test_it_counts_with_the_tokenizer_it_was_given(self) -> None:
        held = window()
        held.admit(turn("one two three"))
        assert held.tokens == 3

    def test_the_default_tokenizer_is_the_documented_estimate(self) -> None:
        held = ContextWindow(limit_tokens=100)
        held.admit(turn("a" * 40))
        assert held.tokens == 10

    def test_a_limit_has_to_be_positive(self) -> None:
        with pytest.raises(ValueError, match="limit_tokens"):
            ContextWindow(limit_tokens=0)
