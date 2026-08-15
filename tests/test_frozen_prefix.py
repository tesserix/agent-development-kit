"""The boundary compression may not cross, and the assertion that proves it held."""

from __future__ import annotations

import pytest

from tesserix_adk.core import Message, PrefixDriftError, TextPart
from tesserix_adk.memory import COMPRESSED, SECTION, FrozenPrefix, compressed

pytestmark = pytest.mark.anyio


def said(text: str, *, role: str = "user", section: str = "") -> Message:
    """One message, optionally naming the layer of the prompt it belongs to."""
    return Message(
        role=role,  # type: ignore[arg-type]
        content=[TextPart(text=text)],
        metadata={SECTION: section} if section else {},
    )


TURN_ONE = [
    said("You answer from tool output only.", role="system", section="instructions"),
    said("Which hosts are down?"),
]
TURN_TWO = [*TURN_ONE, said("host-004 is unreachable", role="assistant"), said("Since when?")]


class TestWhatIsFrozen:
    """Everything above the boundary is byte-identical across turns, or the cache is lost."""

    def test_nothing_is_frozen_before_the_first_turn_completes(self) -> None:
        prefix = FrozenPrefix()

        assert prefix.boundary.position == 0
        assert prefix.live(TURN_ONE) == tuple(TURN_ONE)

    def test_a_completed_turn_freezes_what_it_sent(self) -> None:
        prefix = FrozenPrefix()

        boundary = prefix.advance(TURN_ONE, turn=1)

        assert boundary.position == len(TURN_ONE)
        assert boundary.fingerprint != ""

    def test_only_what_arrived_since_is_live(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        assert prefix.live(TURN_TWO) == tuple(TURN_TWO[len(TURN_ONE) :])

    def test_the_frozen_bytes_are_the_same_bytes_a_turn_later(self) -> None:
        prefix = FrozenPrefix()
        first = prefix.advance(TURN_ONE, turn=1)

        prefix.verify(TURN_TWO)

        assert prefix.boundary.fingerprint == first.fingerprint

    def test_a_turn_that_sent_nothing_freezes_nothing(self) -> None:
        prefix = FrozenPrefix()

        boundary = prefix.advance([], turn=1)

        assert boundary.position == 0
        assert boundary.fingerprint == ""

    def test_an_empty_live_zone_means_the_prompt_did_not_move(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        assert prefix.live(TURN_ONE) == ()
        prefix.verify(TURN_ONE)


class TestWhatMustFailTheBuild:
    """A silent doubling of prefill latency must fail a build rather than reach production."""

    def test_rewriting_a_frozen_message_is_caught_and_named(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)
        rewritten = [
            said("You answer briefly.", role="system", section="instructions"),
            TURN_ONE[1],
        ]

        with pytest.raises(PrefixDriftError) as drift:
            prefix.verify(rewritten)

        assert drift.value.layer == "instructions"
        assert drift.value.position == 0

    def test_reordering_the_frozen_region_is_caught(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        with pytest.raises(PrefixDriftError):
            prefix.verify([TURN_ONE[1], TURN_ONE[0]])

    def test_losing_frozen_content_is_caught_rather_than_read_as_a_short_prompt(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_TWO, turn=1)

        with pytest.raises(PrefixDriftError) as drift:
            prefix.verify(TURN_ONE)

        assert drift.value.position == len(TURN_ONE)

    def test_a_layer_with_no_name_is_reported_by_its_role(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance([said("Which hosts are down?")], turn=1)

        with pytest.raises(PrefixDriftError) as drift:
            prefix.verify([said("Which hosts are up?")])

        assert drift.value.layer == "user"


class TestWhatMayBeCompressed:
    """Compression that rewrites cached bytes costs more prefill than it saves tokens."""

    def test_only_live_content_is_offered_to_a_compressor(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        assert prefix.compressible(TURN_TWO) == (2, 3)

    def test_content_already_compressed_is_never_compressed_again(self) -> None:
        prefix = FrozenPrefix()
        arrived = [*TURN_ONE, compressed(said("rows"), by="structured")]

        assert prefix.compressible(arrived) == (0, 1)

    def test_a_compressed_message_says_what_compressed_it(self) -> None:
        marked = compressed(said("rows"), by="structured")

        assert marked.metadata[COMPRESSED] == "structured"

    def test_marking_a_message_leaves_the_original_alone(self) -> None:
        original = said("rows")

        compressed(original, by="structured")

        assert COMPRESSED not in original.metadata


class TestWhenTheBoundaryMoves:
    """A reset is recorded rather than hidden: it is a deploy-time cost, not a mystery."""

    def test_a_reset_records_why_and_counts(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        boundary = prefix.reset("eviction removed frozen content")

        assert boundary.position == 0
        assert boundary.resets == 1
        assert "eviction" in boundary.reason

    def test_after_a_reset_everything_is_live_again(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)
        prefix.reset("the system prompt was deployed again")

        assert prefix.live(TURN_ONE) == tuple(TURN_ONE)
        prefix.verify(TURN_ONE)

    def test_two_concurrent_turns_do_not_advance_the_boundary_twice(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_TWO, turn=1)

        boundary = prefix.advance(TURN_ONE, turn=1)

        assert boundary.position == len(TURN_TWO)
        assert boundary.turn == 1

    def test_a_later_turn_still_advances(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        boundary = prefix.advance(TURN_TWO, turn=2)

        assert boundary.position == len(TURN_TWO)
        assert boundary.turn == 2

    def test_two_contexts_know_nothing_of_each_other(self) -> None:
        one = FrozenPrefix()
        other = FrozenPrefix()

        one.advance(TURN_TWO, turn=1)

        assert other.boundary.position == 0
        assert other.live(TURN_ONE) == tuple(TURN_ONE)


class TestWhatAMetricReads:
    """The boundary is worth watching next to the cache hit ratio it exists to protect."""

    def test_the_boundary_reports_itself_as_attributes(self) -> None:
        prefix = FrozenPrefix()
        prefix.advance(TURN_ONE, turn=1)

        attributes = prefix.boundary.attributes()

        assert attributes["adk.frozen_prefix_position"] == len(TURN_ONE)
        assert attributes["adk.frozen_prefix_resets"] == 0

    def test_a_fresh_boundary_reads_as_nothing_frozen(self) -> None:
        assert FrozenPrefix().boundary.attributes()["adk.frozen_prefix_position"] == 0
