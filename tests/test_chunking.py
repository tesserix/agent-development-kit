"""Pieces a retriever can return, sized by the tokeniser of the model that will read them.

Chunking is the one decision in a retrieval pipeline nothing downstream can undo: a
passage split through the middle is two hits that are each wrong, and no reranker recovers
the sentence that was left in the other one. So the strategy is configured per collection,
every chunk knows exactly which characters of which document it is, and a run of text that
will not divide under the limit is an error rather than a chunk that breaks the window.
"""

from __future__ import annotations

import re
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tesserix_adk.core import ChunkingError, ConfigurationError
from tesserix_adk.rag import (
    Chunk,
    Chunker,
    ChunkerRegistry,
    ChunkingSpec,
    CodeAware,
    Document,
    FixedTokens,
    Overflow,
    SentenceWindow,
    Structural,
    chunk_id,
    tokens_via,
)
from tesserix_adk.testing import ScriptedProvider

WORDS = re.compile(r"\S+")


def words(text: str) -> int:
    """Count a word as a token, which is wrong by a constant and right about boundaries."""
    return len(WORDS.findall(text))


def characters(text: str) -> int:
    return len(text)


def spec(**overrides: object) -> ChunkingSpec:
    return ChunkingSpec(**{"strategy": "fixed", "max_tokens": 10, **overrides})  # type: ignore[arg-type]


def document(text: str, *, doc_id: str = "doc-1", **metadata: str) -> Document:
    return Document(id=doc_id, text=text, metadata=metadata)


PROSE = (
    "# Trains\n\n"
    "The overnight service leaves at ten. It arrives before dawn.\n\n"
    "## Sleepers\n\n"
    "A berth must be booked ahead. There are four to a compartment.\n"
)


class TestSplittingADocumentIntoPiecesThatFit:
    def test_no_chunk_is_longer_than_the_limit(self) -> None:
        """The limit is the whole point: a chunk over it is a context error later."""
        chunker = FixedTokens(spec(max_tokens=5), words)

        chunks = chunker.chunk(document(" ".join(f"word{n}" for n in range(37))))

        assert chunks
        assert all(words(chunk.text) <= 5 for chunk in chunks)

    def test_chunks_come_back_in_the_order_they_appear(self) -> None:
        chunker = FixedTokens(spec(max_tokens=4), words)

        chunks = chunker.chunk(document(" ".join(f"word{n}" for n in range(20))))

        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        assert [chunk.start for chunk in chunks] == sorted(chunk.start for chunk in chunks)

    def test_the_document_is_covered_with_nothing_left_between_the_pieces(self) -> None:
        """Whitespace dropped between two chunks is an offset every citation is wrong by."""
        text = "  leading space, then a good many words that will not fit in one chunk.  "
        chunker = FixedTokens(spec(max_tokens=4), words)

        chunks = chunker.chunk(document(text))

        assert "".join(chunk.text for chunk in chunks) == text

    def test_an_empty_document_produces_no_chunks_rather_than_one_empty_one(self) -> None:
        assert FixedTokens(spec(), words).chunk(document("")) == ()
        assert Structural(spec(), words).chunk(document("")) == ()
        assert SentenceWindow(spec(), words).chunk(document("")) == ()
        assert CodeAware(spec(), words).chunk(document("")) == ()

    def test_a_document_that_already_fits_is_left_whole(self) -> None:
        chunker = FixedTokens(spec(max_tokens=100), words)

        chunks = chunker.chunk(document("Short enough."))

        assert [chunk.text for chunk in chunks] == ["Short enough."]

    def test_the_documents_metadata_travels_with_every_chunk(self) -> None:
        """Retrieval filters on it, so a chunk without it is a chunk nobody can restrict."""
        chunker = FixedTokens(spec(max_tokens=3), words)

        chunks = chunker.chunk(document("one two three four five six", tenant="acme"))

        assert all(chunk.metadata == {"tenant": "acme"} for chunk in chunks)


class TestWhereAChunkCameFrom:
    def test_a_chunk_is_exactly_the_characters_its_span_names(self) -> None:
        doc = document(PROSE)

        for chunk in Structural(spec(max_tokens=6), words).chunk(doc):
            assert doc.text[chunk.start : chunk.end] == chunk.text

    def test_a_span_that_does_not_describe_its_text_is_refused(self) -> None:
        """The model is the last place this can be caught before a citation is wrong."""
        with pytest.raises(ValueError, match="would point at the wrong place"):
            Chunk(
                id="c",
                document_id="doc-1",
                ordinal=0,
                text="four",
                start=0,
                end=99,
                tokens=1,
            )

    def test_a_span_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot end at 2 having started at 9"):
            Chunk(id="c", document_id="doc-1", ordinal=0, text="", start=9, end=2, tokens=0)

    def test_offsets_are_code_points_so_they_survive_scripts_that_are_not_latin(self) -> None:
        """A byte offset into mixed script text cuts a character in half and reads as mojibake."""
        text = "مرحبا بالعالم — 日本語のテキスト and some English words here too."
        doc = document(text)

        chunks = FixedTokens(spec(max_tokens=3), words).chunk(doc)

        assert len(chunks) > 1
        assert "".join(chunk.text for chunk in chunks) == text
        assert all(doc.text[chunk.start : chunk.end] == chunk.text for chunk in chunks)

    def test_the_heading_a_chunk_sits_under_travels_with_it(self) -> None:
        chunks = Structural(spec(max_tokens=8), words).chunk(document(PROSE))

        sections = {chunk.section for chunk in chunks}
        assert ("Trains",) in sections
        assert ("Trains", "Sleepers") in sections


class TestChunkIdentity:
    def test_the_same_document_chunks_to_the_same_ids_every_time(self) -> None:
        """Re-indexing an unchanged document must not rewrite the whole collection."""
        chunker = FixedTokens(spec(max_tokens=4), words)
        doc = document(PROSE)

        assert [c.id for c in chunker.chunk(doc)] == [c.id for c in chunker.chunk(doc)]

    def test_ids_are_derived_from_the_document_so_two_documents_do_not_collide(self) -> None:
        assert chunk_id("doc-1", "same text") != chunk_id("doc-2", "same text")

    def test_editing_one_passage_leaves_the_other_chunks_ids_alone(self) -> None:
        chunker = FixedTokens(spec(max_tokens=4), words)
        first = " ".join(f"word{n}" for n in range(20))
        edited = first.replace("word19", "changed")

        before = {chunk.id for chunk in chunker.chunk(document(first))}
        after = {chunk.id for chunk in chunker.chunk(document(edited))}

        assert len(before & after) >= len(before) - 2


class TestTextThatWillNotDivide:
    def test_an_indivisible_run_is_refused_by_default_naming_where_it_is(self) -> None:
        """Emitting it anyway moves the failure to a model call nobody can trace back."""
        doc = document("short bit " + "x" * 400, doc_id="manifest")

        with pytest.raises(ChunkingError, match="manifest") as raised:
            FixedTokens(spec(max_tokens=10), characters).chunk(doc)

        assert raised.value.document == "manifest"
        assert raised.value.offset == doc.text.index("x")

    def test_a_refusal_is_not_worth_retrying(self) -> None:
        assert ChunkingError("no", document="d").retryable is False

    def test_an_explicit_overflow_policy_cuts_it_at_the_limit_instead(self) -> None:
        blob = "x" * 50
        chunker = FixedTokens(spec(max_tokens=8, overflow=Overflow.SPLIT), characters)

        chunks = chunker.chunk(document(blob))

        assert all(len(chunk.text) <= 8 for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks) == blob

    def test_a_cut_lands_on_the_limit_even_where_the_counter_is_not_linear(self) -> None:
        """A real tokeniser answers in steps, so the boundary is searched for, not computed."""
        blob = "x" * 200
        chunker = FixedTokens(
            spec(max_tokens=5, overflow=Overflow.SPLIT), lambda text: -(-len(text) // 3)
        )

        chunks = chunker.chunk(document(blob))

        assert all(len(chunk.text) <= 15 for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks) == blob

    def test_an_overlap_that_does_not_leave_room_to_advance_is_refused(self) -> None:
        """Each chunk would repeat the last entirely, which is a loop rather than a split."""
        with pytest.raises(ValueError, match="does not fit inside"):
            spec(max_tokens=10, overlap_tokens=10)

    def test_a_stride_past_the_window_would_leave_sentences_in_no_chunk(self) -> None:
        with pytest.raises(ValueError, match="in no chunk at all"):
            spec(strategy="sentence-window", window=2, stride=3)


class TestOverlap:
    def test_each_chunk_repeats_the_end_of_the_one_before_it(self) -> None:
        """A sentence split across a boundary is retrievable from at least one side of it."""
        text = " ".join(f"word{n}" for n in range(30))
        chunker = FixedTokens(spec(max_tokens=6, overlap_tokens=2), words)

        chunks = chunker.chunk(document(text))

        assert len(chunks) > 1
        assert all(later.start < earlier.end for earlier, later in pairwise(chunks))

    def test_overlapping_chunks_still_reach_the_end_of_the_document(self) -> None:
        text = " ".join(f"word{n}" for n in range(30))
        chunks = FixedTokens(spec(max_tokens=6, overlap_tokens=2), words).chunk(document(text))

        assert chunks[0].start == 0
        assert chunks[-1].end == len(text)


class TestSplittingOnTheDocumentsOwnBoundaries:
    def test_a_paragraph_that_fits_is_kept_whole(self) -> None:
        chunks = Structural(spec(max_tokens=12), words).chunk(document(PROSE))

        assert any("The overnight service leaves at ten." in chunk.text for chunk in chunks)

    def test_a_paragraph_that_does_not_fit_falls_back_to_sentences(self) -> None:
        chunks = Structural(spec(max_tokens=6), words).chunk(document(PROSE))

        assert all(words(chunk.text) <= 6 for chunk in chunks)
        assert any(chunk.text.strip() == "It arrives before dawn." for chunk in chunks)

    def test_a_document_with_no_structure_at_all_still_chunks(self) -> None:
        """No headings, no paragraphs, no sentence ends: the word boundary is what is left."""
        text = " ".join(f"word{n}" for n in range(40))

        chunks = Structural(spec(max_tokens=5), words).chunk(document(text))

        assert "".join(chunk.text for chunk in chunks) == text
        assert all(words(chunk.text) <= 5 for chunk in chunks)

    def test_a_document_that_is_one_table_is_split_by_its_rows(self) -> None:
        table = "\n".join(f"| row {n} | value {n} |" for n in range(20))

        chunks = Structural(spec(max_tokens=9), words).chunk(document(table))

        assert "".join(chunk.text for chunk in chunks) == table
        assert all(words(chunk.text) <= 9 for chunk in chunks)


class TestWindowsOfWholeSentences:
    def test_a_window_is_several_sentences_and_the_next_one_overlaps_it(self) -> None:
        text = "One. Two. Three. Four. Five. Six."
        chunker = SentenceWindow(spec(window=3, stride=1, max_tokens=20), words)

        chunks = chunker.chunk(document(text))

        assert chunks[0].text.strip() == "One. Two. Three."
        assert chunks[1].text.strip() == "Two. Three. Four."

    def test_a_window_is_cut_short_rather_than_pushed_over_the_limit(self) -> None:
        text = "One two three. Four five six. Seven eight nine."
        chunker = SentenceWindow(spec(window=3, stride=3, max_tokens=6), words)

        chunks = chunker.chunk(document(text))

        assert all(words(chunk.text) <= 6 for chunk in chunks)

    def test_every_sentence_appears_in_some_window(self) -> None:
        text = "One. Two. Three. Four. Five."
        chunks = SentenceWindow(spec(window=2, stride=2, max_tokens=20), words).chunk(
            document(text)
        )

        assert "".join(chunk.text for chunk in chunks[::1]).count("Five") >= 1
        assert chunks[-1].end == len(text)

    def test_a_sentence_longer_than_the_limit_is_split_within_itself(self) -> None:
        """One sentence nobody can fit is still indexed, on the words it is made of."""
        text = "One two three four five six seven eight nine ten eleven twelve."

        chunks = SentenceWindow(spec(window=2, stride=2, max_tokens=4), words).chunk(document(text))

        assert all(words(chunk.text) <= 4 for chunk in chunks)
        assert "".join(chunk.text for chunk in chunks) == text


class TestChunkingSource:
    def test_a_top_level_definition_is_not_cut_in_half(self) -> None:
        source = (
            "def first():\n    return 1\n\n\ndef second():\n    return 2\n\n\n"
            "def third():\n    return 3\n"
        )

        chunks = CodeAware(spec(max_tokens=5), words).chunk(document(source))

        assert [chunk.text.count("def ") for chunk in chunks] == [1, 1, 1]
        assert all(chunk.text.startswith("def ") for chunk in chunks)

    def test_a_definition_too_long_for_the_limit_falls_back_to_lines(self) -> None:
        source = "def wide():\n" + "".join(f"    step_{n}()\n" for n in range(30))

        chunks = CodeAware(spec(max_tokens=6), words).chunk(document(source))

        assert "".join(chunk.text for chunk in chunks) == source
        assert all(chunk.text.endswith("\n") or chunk.end == len(source) for chunk in chunks)

    def test_a_document_that_is_one_code_block_still_produces_covering_chunks(self) -> None:
        source = "```\n" + "\n".join(f"line {n} of output" for n in range(30)) + "\n```\n"

        chunks = CodeAware(spec(max_tokens=8), words).chunk(document(source))

        assert "".join(chunk.text for chunk in chunks) == source


class TestChoosingAStrategyPerCollection:
    def test_each_collection_is_chunked_the_way_it_was_configured(self) -> None:
        registry = ChunkerRegistry(
            count=words,
            collections={
                "handbook": spec(strategy="structural", max_tokens=12),
                "repo": spec(strategy="code", max_tokens=12),
            },
        )

        assert isinstance(registry.chunker_for("handbook"), Structural)
        assert isinstance(registry.chunker_for("repo"), CodeAware)

    def test_a_registered_chunker_satisfies_the_protocol(self) -> None:
        registry = ChunkerRegistry(count=words, default=spec())

        assert isinstance(registry.chunker_for("anything"), Chunker)

    def test_a_collection_nobody_configured_is_a_configuration_error(self) -> None:
        """Guessing would index it differently from everything already in it."""
        registry = ChunkerRegistry(count=words, collections={"handbook": spec()})

        with pytest.raises(ConfigurationError, match="no chunking settings"):
            registry.chunker_for("invoices")

    def test_a_default_covers_the_collections_not_named(self) -> None:
        registry = ChunkerRegistry(count=words, default=spec(strategy="fixed", max_tokens=4))

        chunks = registry.chunker_for("invoices").chunk(document("one two three four five"))

        assert all(words(chunk.text) <= 4 for chunk in chunks)

    def test_a_deployment_can_register_a_strategy_of_its_own(self) -> None:
        registry = ChunkerRegistry(count=words, default=spec(strategy="ours", max_tokens=4))
        registry.register("ours", lambda spec, count: FixedTokens(spec, count))

        assert isinstance(registry.chunker_for("invoices"), FixedTokens)

    def test_a_strategy_nobody_registered_says_what_is_registered(self) -> None:
        registry = ChunkerRegistry(count=words, default=spec(strategy="magic", max_tokens=4))

        with pytest.raises(ConfigurationError, match="sentence-window"):
            registry.chunker_for("invoices")

    def test_a_strategy_needs_a_name_to_be_selected_by(self) -> None:
        registry = ChunkerRegistry(count=words)

        with pytest.raises(ConfigurationError, match="needs a name"):
            registry.register("", FixedTokens)


class TestCountingWithTheTokeniserThatWillRead:
    def test_the_count_comes_from_the_provider_rather_than_from_a_character_guess(self) -> None:
        """A character heuristic is wrong by a factor of three the first time it is not English."""
        count = tokens_via(ScriptedProvider())

        assert count("one two three four") == count("one two three four")
        assert count("") <= count("a longer piece of text than that one")

    def test_a_chunker_sized_by_a_provider_stays_under_that_providers_window(self) -> None:
        count = tokens_via(ScriptedProvider())
        chunker = FixedTokens(spec(max_tokens=12), count)

        chunks = chunker.chunk(document(PROSE * 4))

        assert chunks
        assert all(count(chunk.text) <= 12 for chunk in chunks)


class TestDocumentsBigEnoughToMatter:
    def test_a_large_document_chunks_without_counting_it_a_word_at_a_time(self) -> None:
        """Growing a chunk one unit at a time counts the same prefix once per word."""
        counted = 0

        def count(text: str) -> int:
            nonlocal counted
            counted += 1
            return words(text)

        text = " ".join(f"word{n}" for n in range(50_000))
        chunks = FixedTokens(spec(max_tokens=200), count).chunk(document(text))

        assert "".join(chunk.text for chunk in chunks) == text
        assert counted < 20 * len(chunks)


class TestTheseHoldForAnyDocument:
    @settings(max_examples=50, deadline=None)
    @given(st.text(min_size=1, max_size=400))
    def test_no_chunk_ever_exceeds_the_limit(self, text: str) -> None:
        chunks = FixedTokens(spec(max_tokens=8, overflow=Overflow.SPLIT), characters).chunk(
            document(text)
        )

        assert all(len(chunk.text) <= 8 for chunk in chunks)

    @settings(max_examples=50, deadline=None)
    @given(st.text(min_size=1, max_size=400))
    def test_the_chunks_of_a_partitioning_strategy_reassemble_the_document(self, text: str) -> None:
        doc = document(text)

        chunks = Structural(spec(max_tokens=8, overflow=Overflow.SPLIT), characters).chunk(doc)

        assert "".join(chunk.text for chunk in chunks) == text
        assert all(doc.text[chunk.start : chunk.end] == chunk.text for chunk in chunks)

    @settings(max_examples=50, deadline=None)
    @given(st.text(min_size=1, max_size=400))
    def test_spans_run_end_to_end_with_no_gap_and_no_overlap(self, text: str) -> None:
        chunks = FixedTokens(spec(max_tokens=8, overflow=Overflow.SPLIT), characters).chunk(
            document(text)
        )

        assert chunks[0].start == 0
        assert chunks[-1].end == len(text)
        assert all(earlier.end == later.start for earlier, later in pairwise(chunks))

    @settings(max_examples=50, deadline=None)
    @given(st.text(min_size=1, max_size=400))
    def test_chunking_twice_gives_the_same_answer(self, text: str) -> None:
        chunker = Structural(spec(max_tokens=8, overflow=Overflow.SPLIT), characters)
        doc = document(text)

        assert chunker.chunk(doc) == chunker.chunk(doc)
