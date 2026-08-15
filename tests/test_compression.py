"""What content is worth compressing, which compressor gets it, and when none does."""

from __future__ import annotations

import json

import pytest

from tesserix_adk.memory import (
    CodeCompressor,
    Compressed,
    ContentKind,
    ContentRouter,
    PassThrough,
    ProseCompressor,
    StructuredCompressor,
    TabularCompressor,
    classify,
    estimate_tokens,
)

ROWS = [
    {"id": index, "region": "apac", "status": "active", "name": f"host-{index:03d}"}
    for index in range(500)
]

CODE = '''
"""A module the model is reading to write the next call."""

import json


class Ledger:
    """Money in, money out."""

    def credit(self, account: str, amount: int) -> int:
        """Add to an account."""
        total = amount * 2
        for _ in range(3):
            total += 1
        if amount < 0:
            raise ValueError("amount must be positive")
        return total
'''

PROSE = " ".join(
    ["The deployment finished at nine and the queue drained by ten." for _ in range(40)]
)

TABLE = "id   region   status\n" + "\n".join(f"{index:<4} apac     active" for index in range(200))


def router(**fields: object) -> ContentRouter:
    """A router over the default compressors, with a low threshold for small fixtures."""
    return ContentRouter(threshold_tokens=8, **fields)  # type: ignore[arg-type]


class TestWhatTheContentIs:
    """Classification reads the content. It never runs it and never follows it."""

    def test_a_json_array_is_json(self) -> None:
        assert classify(json.dumps(ROWS)) is ContentKind.JSON

    def test_python_source_is_code(self) -> None:
        assert classify(CODE) is ContentKind.CODE

    def test_an_aligned_table_is_tabular(self) -> None:
        assert classify(TABLE) is ContentKind.TABULAR

    def test_sentences_are_prose(self) -> None:
        assert classify(PROSE) is ContentKind.PROSE

    def test_something_that_starts_like_json_and_is_not_stays_unknown(self) -> None:
        assert classify('{"id": 1, "region": ') is ContentKind.UNKNOWN

    def test_a_line_of_symbols_is_not_guessed_at(self) -> None:
        assert classify("<<<<>>>> ||| ??? ###") is ContentKind.UNKNOWN

    def test_nothing_at_all_is_unknown(self) -> None:
        assert classify("   \n  ") is ContentKind.UNKNOWN


class TestRowsOfTheSameShape:
    """The primary scenario: 500 rows of identical keys cost a fraction of 500 rows."""

    def test_the_structured_compressor_is_chosen_and_the_ratio_is_recorded(self) -> None:
        admitted = router().admit(json.dumps(ROWS), budget_tokens=4_000)

        assert admitted.kind is ContentKind.JSON
        assert admitted.compressor == "structured"
        assert admitted.ratio < 0.6
        assert admitted.saved_tokens > 0

    def test_every_field_name_and_every_value_survives(self) -> None:
        admitted = router().admit(json.dumps(ROWS), budget_tokens=4_000)

        for field in ("id", "region", "status", "name"):
            assert field in admitted.content
        assert "host-499" in admitted.content
        assert "apac" in admitted.content

    def test_a_field_the_same_in_every_row_is_written_once(self) -> None:
        admitted = router().admit(json.dumps(ROWS), budget_tokens=4_000)

        assert admitted.content.count("apac") == 1

    def test_identical_rows_are_folded_and_counted(self) -> None:
        repeated = json.dumps([{"code": 500, "path": "/orders"}] * 12)

        admitted = router().admit(repeated, budget_tokens=1_000)

        assert admitted.content.count("/orders") == 1
        assert "12" in admitted.content

    def test_json_that_is_not_rows_at_least_loses_its_whitespace(self) -> None:
        padded = json.dumps({"nested": {"deep": [1, 2, 3]}}, indent=4)

        admitted = router().admit(padded, budget_tokens=1_000)

        assert admitted.compressor == "structured"
        assert json.loads(admitted.content) == {"nested": {"deep": [1, 2, 3]}}

    def test_the_same_bytes_in_produce_the_same_bytes_out(self) -> None:
        once = router().admit(json.dumps(ROWS), budget_tokens=4_000)
        twice = router().admit(json.dumps(ROWS), budget_tokens=4_000)

        assert once.content == twice.content


class TestCodeAndTablesAndProse:
    """Each type keeps the part of itself the model was actually using."""

    def test_code_keeps_its_signatures_and_loses_its_bodies(self) -> None:
        admitted = router().admit(CODE, budget_tokens=1_000)

        assert admitted.compressor == "code"
        assert "def credit(self, account: str, amount: int) -> int" in admitted.content
        assert "class Ledger" in admitted.content
        assert "total = amount * 2" not in admitted.content

    def test_code_keeps_what_it_raises(self) -> None:
        admitted = router().admit(CODE, budget_tokens=1_000)

        assert "ValueError" in admitted.content

    def test_a_table_loses_its_alignment_and_its_repeated_rows(self) -> None:
        admitted = router().admit(TABLE, budget_tokens=1_000)

        assert admitted.compressor == "tabular"
        assert "     " not in admitted.content
        assert admitted.ratio < 1.0

    def test_prose_is_deduplicated_down_towards_the_budget(self) -> None:
        admitted = router().admit(PROSE, budget_tokens=64)

        assert admitted.compressor == "prose"
        assert admitted.compressed_tokens <= admitted.original_tokens

    def test_a_compressor_can_be_used_on_its_own(self) -> None:
        assert StructuredCompressor().compress("[1, 2]", budget_tokens=10) == "[1,2]"
        assert PassThrough().compress("as it was", budget_tokens=1) == "as it was"


class TestWhenNothingShouldTouchIt:
    """A wrong compressor destroying content is worse than paying full price for it."""

    def test_unclassifiable_content_passes_through_and_says_so(self) -> None:
        opaque = "<<<<>>>> ||| ??? ###" * 40

        admitted = router().admit(opaque, budget_tokens=1_000)

        assert admitted.content == opaque
        assert admitted.compressor == "pass-through"
        assert admitted.kind is ContentKind.UNKNOWN
        assert admitted.reason != ""

    def test_content_below_the_threshold_never_reaches_the_router(self) -> None:
        admitted = ContentRouter(threshold_tokens=64).admit("[1, 2, 3]", budget_tokens=100)

        assert admitted.compressor == "pass-through"
        assert "threshold" in admitted.reason

    def test_a_compressor_that_expands_its_input_is_discarded(self) -> None:
        class Longer:
            """A compressor that makes things worse."""

            name = "longer"

            def compress(self, content: str, *, budget_tokens: int) -> str:
                """Return more than was given."""
                del budget_tokens
                return content * 3

        admitted = router(compressors={ContentKind.PROSE: Longer()}).admit(
            PROSE, budget_tokens=1_000
        )

        assert admitted.content == PROSE
        assert admitted.compressor == "pass-through"
        assert "expanded" in admitted.reason

    def test_a_compressor_that_raises_falls_back_rather_than_propagating(self) -> None:
        class Broken:
            """A compressor that cannot cope with what it was handed."""

            name = "broken"

            def compress(self, content: str, *, budget_tokens: int) -> str:
                """Fail the way a parser fails on malformed input."""
                del content, budget_tokens
                raise RuntimeError("malformed")

        admitted = router(compressors={ContentKind.PROSE: Broken()}).admit(
            PROSE, budget_tokens=1_000
        )

        assert admitted.content == PROSE
        assert admitted.compressor == "pass-through"
        assert "broken" in admitted.reason

    def test_a_kind_with_no_compressor_configured_passes_through(self) -> None:
        admitted = router(compressors={}).admit(json.dumps(ROWS), budget_tokens=1_000)

        assert admitted.content == json.dumps(ROWS)
        assert admitted.compressor == "pass-through"

    def test_compression_is_not_sanitisation(self) -> None:
        admitted = router().admit(json.dumps(ROWS), budget_tokens=4_000, untrusted=True)

        assert admitted.untrusted is True

    def test_trust_is_carried_rather_than_assumed(self) -> None:
        assert router().admit(PROSE, budget_tokens=1_000).untrusted is False


class TestWhatTheRecordSays:
    """An admission that cannot be accounted for cannot be tuned."""

    def test_a_ratio_of_one_is_reported_for_content_left_alone(self) -> None:
        admitted = ContentRouter(threshold_tokens=10_000).admit(PROSE, budget_tokens=100)

        assert admitted.ratio == pytest.approx(1.0)
        assert admitted.saved_tokens == 0

    def test_an_empty_admission_has_a_ratio_rather_than_a_division_by_zero(self) -> None:
        admitted = router().admit("", budget_tokens=100)

        assert admitted.ratio == pytest.approx(1.0)

    def test_the_estimate_is_stable_and_never_zero_for_real_content(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1
        assert estimate_tokens("x" * 400) == estimate_tokens("y" * 400)

    def test_a_counter_of_the_callers_own_is_used_where_given(self) -> None:
        admitted = ContentRouter(threshold_tokens=1, count=len).admit(
            json.dumps(ROWS), budget_tokens=4_000
        )

        assert admitted.original_tokens == len(json.dumps(ROWS))

    def test_the_record_reads_as_what_happened(self) -> None:
        admitted = Compressed(
            content="x", kind=ContentKind.PROSE, compressor="prose", original_tokens=10
        )

        assert admitted.compressed_tokens == 0
        assert admitted.ratio == pytest.approx(0.0)


class TestCompressorsInIsolation:
    """The parts a caller can reach for without the router."""

    def test_the_tabular_compressor_needs_more_than_a_header(self) -> None:
        assert TabularCompressor().compress("id region", budget_tokens=10) == "id region"

    def test_the_code_compressor_refuses_what_does_not_parse(self) -> None:
        with pytest.raises(SyntaxError):
            CodeCompressor().compress("def (:", budget_tokens=10)

    def test_rows_of_different_shapes_are_left_as_objects(self) -> None:
        mixed = json.dumps([{"id": 1, "region": "apac"}, {"id": 2, "zone": "b"}])

        compressed = StructuredCompressor().compress(mixed, budget_tokens=100)

        assert json.loads(compressed) == json.loads(mixed)

    def test_rows_that_share_nothing_get_no_shared_block(self) -> None:
        varying = json.dumps([{"id": 1, "host": "a"}, {"id": 2, "host": "b"}])

        compressed = StructuredCompressor().compress(varying, budget_tokens=100)

        assert "shared" not in json.loads(compressed)

    def test_a_table_with_a_row_repeated_three_times_says_so(self) -> None:
        repeated = "id  host\n1   a\n1   a\n1   a\n2   b"

        compressed = TabularCompressor().compress(repeated, budget_tokens=100)

        assert "1 a x3" in compressed

    def test_a_function_without_a_docstring_still_loses_its_body(self) -> None:
        source = (
            "def add(left: int, right: int) -> int:\n    total = left + right\n    return total\n"
        )

        compressed = CodeCompressor().compress(source, budget_tokens=100)

        assert "def add(left: int, right: int) -> int" in compressed
        assert "total = left + right" not in compressed

    def test_a_coroutine_is_elided_like_anything_else(self) -> None:
        source = (
            "async def fetch(url: str) -> str:\n"
            '    """Get it."""\n'
            "    body = await get(url)\n"
            "    return body\n"
        )

        compressed = CodeCompressor().compress(source, budget_tokens=100)

        assert "async def fetch(url: str) -> str" in compressed
        assert "await get(url)" not in compressed

    def test_prose_over_the_budget_is_cut_at_a_sentence_and_marked(self) -> None:
        many = " ".join(f"Sentence number {index} of the report." for index in range(60))

        compressed = ProseCompressor().compress(many, budget_tokens=20)

        assert compressed.endswith("…")
        assert estimate_tokens(compressed) <= 20 + len("Sentence number 59 of the report.")

    def test_nothing_compresses_to_nothing(self) -> None:
        assert ProseCompressor().compress("", budget_tokens=10) == ""

    def test_the_prose_compressor_keeps_the_first_of_a_repeated_sentence(self) -> None:
        compressed = ProseCompressor().compress("Once.\nOnce.\nTwice.\n", budget_tokens=100)

        assert compressed.count("Once.") == 1
        assert "Twice." in compressed
