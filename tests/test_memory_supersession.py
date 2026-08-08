"""What happens to the old fact when a new one contradicts it."""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import MemoryConflictError, MemoryContradictionError
from tesserix_adk.memory import (
    ConfidenceFloor,
    Contradiction,
    ContradictionPolicy,
    DecayPolicy,
    HalfLife,
    MemoryCapabilities,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    Resolution,
    SupersedeMatching,
)
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore

SCOPE = MemoryScope(tenant_id="acme", user_id="u1")
LAST_MONTH = 1_000.0
TODAY = 100_000.0


def fact(
    value: str,
    *,
    key: str = "diet",
    subject: str = "user",
    predicate: tuple[str, ...] = ("diet",),
    at: float = TODAY,
    confidence: float = 1.0,
) -> MemoryRecord:
    """A profile record saying one thing about one subject."""
    return MemoryRecord(
        id=f"{key}:{value}",
        kind=MemoryKind.PROFILE,
        scope=SCOPE,
        key=key,
        value=value,
        source="turn",
        subject=subject,
        predicate=predicate,
        valid_from=at,
        confidence=confidence,
    )


def store(
    *,
    now: float = TODAY,
    clock: FakeClock | None = None,
    capabilities: MemoryCapabilities | None = None,
    contradictions: ContradictionPolicy | None = None,
    decay: DecayPolicy | None = None,
) -> InMemoryMemoryStore:
    """A store whose clock a test can move."""
    return InMemoryMemoryStore(
        clock=clock or FakeClock(start=now),
        capabilities=capabilities,
        contradictions=contradictions,
        decay=decay,
    )


class TestANewFactSupersedesRatherThanDestroys:
    async def test_the_old_record_is_kept_and_closed(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, fact("vegetarian", at=LAST_MONTH))
        written = await kept.supersede(SCOPE, fact("eats fish"))

        assert written.resolution is Resolution.SUPERSEDE
        assert written.superseded is not None
        assert written.superseded.value == "vegetarian"
        assert written.superseded.valid_to == TODAY
        assert written.superseded.superseded_by == written.record.id

    async def test_recall_today_returns_only_the_new_fact(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.supersede(SCOPE, fact("eats fish"))

        live = await kept.profile(SCOPE, "diet")
        assert live is not None
        assert live.value == "eats fish"

    async def test_recall_as_of_last_month_returns_the_original(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.supersede(SCOPE, fact("eats fish"))

        then = await kept.profile(SCOPE, "diet", as_of=LAST_MONTH + 1)
        assert then is not None
        assert then.value == "vegetarian"

    async def test_a_first_write_supersedes_nothing(self) -> None:
        kept = store()
        written = await kept.supersede(SCOPE, fact("vegetarian"))

        assert written.superseded is None
        assert written.record.version == 1

    async def test_each_supersession_takes_the_next_version(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))
        second = await kept.supersede(SCOPE, fact("pescatarian"))

        assert second.record.version == 2

    async def test_the_write_is_recorded_at_the_time_it_arrived(self) -> None:
        kept = store()
        written = await kept.supersede(SCOPE, fact("vegetarian", at=LAST_MONTH))

        assert written.record.valid_from == LAST_MONTH
        assert written.record.recorded_at == TODAY

    async def test_superseding_something_erased_writes_it_fresh(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.erase(SCOPE)

        written = await kept.supersede(SCOPE, fact("eats fish"))
        assert written.superseded is None
        assert await kept.history(SCOPE, "diet") == (written.record,)

    async def test_a_record_from_another_scope_is_refused(self) -> None:
        kept = store()
        other = MemoryScope(tenant_id="other", user_id="u1")
        with pytest.raises(Exception, match="belongs to"):
            await kept.supersede(other, fact("vegetarian"))

    async def test_a_store_that_does_not_do_this_says_so(self) -> None:
        kept = store(capabilities=MemoryCapabilities(supports_supersession=False))
        with pytest.raises(Exception, match="supersede"):
            await kept.supersede(SCOPE, fact("vegetarian"))


class TestOnlyOneWriterWins:
    async def test_the_loser_of_a_race_is_told_rather_than_overwritten(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))

        async def write(value: str) -> object:
            try:
                return await kept.supersede(SCOPE, fact(value), expected_version=1)
            except MemoryConflictError as clash:
                return clash

        first, second = await asyncio.gather(write("eats fish"), write("vegan"))
        outcomes = [first, second]
        clashes = [one for one in outcomes if isinstance(one, MemoryConflictError)]
        assert len(clashes) == 1
        assert clashes[0].expected_version == 1
        assert clashes[0].actual_version == 2

    async def test_the_loser_did_not_land(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))
        await kept.supersede(SCOPE, fact("eats fish"), expected_version=1)

        with pytest.raises(MemoryConflictError):
            await kept.supersede(SCOPE, fact("vegan"), expected_version=1)

        assert len(await kept.history(SCOPE, "diet")) == 2

    async def test_expecting_a_version_of_nothing_is_a_conflict(self) -> None:
        kept = store()
        with pytest.raises(MemoryConflictError):
            await kept.supersede(SCOPE, fact("vegetarian"), expected_version=1)

    async def test_no_expectation_takes_whatever_is_live(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))
        await kept.supersede(SCOPE, fact("eats fish"))

        live = await kept.profile(SCOPE, "diet")
        assert live is not None
        assert live.value == "eats fish"


class TestPartialOverlapIsNotResolvedForYou:
    async def test_both_stay_live_and_the_conflict_is_named(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", predicate=("diet",)))
        written = await kept.supersede(SCOPE, fact("no shellfish", predicate=("diet", "allergies")))

        assert written.resolution is Resolution.BRANCH
        assert written.contradiction is not None
        assert {held.value for held in written.contradiction.holds} == {
            "vegetarian",
            "no shellfish",
        }

    async def test_recall_refuses_to_pick_a_side(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", predicate=("diet",)))
        await kept.supersede(SCOPE, fact("no shellfish", predicate=("diet", "allergies")))

        with pytest.raises(MemoryContradictionError) as raised:
            await kept.profile(SCOPE, "diet")
        assert raised.value.key == "diet"

    async def test_belief_returns_the_marker_instead_of_raising(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", predicate=("diet",)))
        await kept.supersede(SCOPE, fact("no shellfish", predicate=("diet", "allergies")))

        held = await kept.belief(SCOPE, "diet")
        assert held.record is None
        assert isinstance(held.contradiction, Contradiction)
        assert held.contradiction.subject == "user"

    async def test_naming_what_a_write_settles_resolves_the_branch(self) -> None:
        kept = store()
        first = await kept.supersede(SCOPE, fact("vegetarian", predicate=("diet",)))
        second = await kept.supersede(SCOPE, fact("no shellfish", predicate=("diet", "allergies")))
        settled = await kept.supersede(
            SCOPE,
            fact("eats fish, no shellfish", predicate=("diet", "allergies")),
            resolves=(first.record.id, second.record.id),
        )

        assert settled.resolution is Resolution.SUPERSEDE
        held = await kept.belief(SCOPE, "diet")
        assert held.contradiction is None
        assert held.record is not None
        assert held.record.value == "eats fish, no shellfish"

    async def test_settling_something_that_is_not_live_is_refused(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))
        with pytest.raises(ValueError, match="nothing live"):
            await kept.supersede(SCOPE, fact("eats fish"), resolves=("diet:invented",))

    async def test_a_different_subject_under_one_key_still_branches(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", subject="user"))
        written = await kept.supersede(SCOPE, fact("vegan", subject="partner"))

        assert written.resolution is Resolution.BRANCH
        assert written.superseded is None

    async def test_disjoint_predicates_under_one_key_branch(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", predicate=("diet",)))
        written = await kept.supersede(SCOPE, fact("window", predicate=("seating",)))

        assert written.resolution is Resolution.BRANCH
        assert written.contradiction is not None


class TestThePolicyDecides:
    async def test_a_rejecting_policy_refuses_the_write(self) -> None:
        class Refuse:
            def resolve(
                self,
                existing: MemoryRecord,  # noqa: ARG002 — the point is that it looks at neither
                incoming: MemoryRecord,  # noqa: ARG002 — the point is that it looks at neither
            ) -> Resolution:
                """Never accept a change of mind."""
                return Resolution.REJECT

        kept = store(contradictions=Refuse())
        await kept.supersede(SCOPE, fact("vegetarian"))
        with pytest.raises(MemoryContradictionError):
            await kept.supersede(SCOPE, fact("eats fish"))

        live = await kept.profile(SCOPE, "diet")
        assert live is not None
        assert live.value == "vegetarian"

    async def test_the_default_policy_supersedes_a_matching_pair(self) -> None:
        policy = SupersedeMatching()
        assert policy.resolve(fact("a"), fact("b")) is Resolution.SUPERSEDE

    async def test_the_default_policy_branches_on_partial_overlap(self) -> None:
        policy = SupersedeMatching()
        overlapping = fact("b", predicate=("diet", "allergies"))
        assert policy.resolve(fact("a"), overlapping) is Resolution.BRANCH

    async def test_a_record_with_no_predicate_falls_back_to_its_key(self) -> None:
        kept = store()
        bare = fact("vegetarian", predicate=())
        await kept.supersede(SCOPE, bare)
        written = await kept.supersede(SCOPE, fact("eats fish", predicate=()))

        assert written.resolution is Resolution.SUPERSEDE


class TestAChainResolvesToOneRecordPerInstant:
    async def test_every_instant_has_exactly_one_live_record(self) -> None:
        moving = FakeClock(start=10.0)
        kept = store(clock=moving)
        for step, value in enumerate(("a", "b", "c", "d"), start=1):
            moving.advance(10.0)
            await kept.supersede(SCOPE, fact(value, at=step * 10.0))

        for instant, expected in ((15.0, "a"), (25.0, "b"), (35.0, "c"), (100.0, "d")):
            live = await kept.profile(SCOPE, "diet", as_of=instant)
            assert live is not None, instant
            assert live.value == expected

    async def test_the_trail_is_queryable_oldest_first(self) -> None:
        kept = store()
        for value in ("a", "b", "c"):
            await kept.supersede(SCOPE, fact(value))

        trail = await kept.history(SCOPE, "diet")
        assert [record.value for record in trail] == ["a", "b", "c"]
        assert [record.version for record in trail] == [1, 2, 3]
        assert trail[0].superseded_by == trail[1].id

    async def test_the_trail_covers_the_whole_scope(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", key="diet"))
        await kept.supersede(SCOPE, fact("window", key="seat", predicate=("seating",)))

        assert {record.key for record in await kept.history(SCOPE)} == {"diet", "seat"}

    async def test_another_scope_sees_none_of_it(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian"))
        assert await kept.history(MemoryScope(tenant_id="other", user_id="u1")) == ()

    async def test_nothing_written_is_an_empty_trail(self) -> None:
        assert await store().history(SCOPE, "diet") == ()


class TestAFactCanStartLater:
    async def test_a_future_fact_is_not_believed_yet(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.supersede(SCOPE, fact("eats fish", at=TODAY * 2))

        live = await kept.profile(SCOPE, "diet")
        assert live is not None
        assert live.value == "vegetarian"

    async def test_it_is_believed_once_the_time_comes(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.supersede(SCOPE, fact("eats fish", at=TODAY * 2))

        live = await kept.profile(SCOPE, "diet", as_of=TODAY * 3)
        assert live is not None
        assert live.value == "eats fish"

    async def test_the_old_fact_closes_when_the_new_one_starts(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", at=LAST_MONTH))
        written = await kept.supersede(SCOPE, fact("eats fish", at=TODAY * 2))

        assert written.superseded is not None
        assert written.superseded.valid_to == TODAY * 2

    async def test_it_is_on_the_trail_all_along(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("eats fish", at=TODAY * 2))
        assert len(await kept.history(SCOPE, "diet")) == 1


class TestDecayIsVisibleRatherThanSilent:
    async def test_an_old_fact_stops_being_recalled(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=10.0, floor=0.5))
        await kept.supersede(SCOPE, fact("vegetarian", at=TODAY - 100.0))

        assert await kept.profile(SCOPE, "diet") is None

    async def test_the_store_says_it_was_decayed_not_missing(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=10.0, floor=0.5))
        await kept.supersede(SCOPE, fact("vegetarian", at=TODAY - 100.0))

        held = await kept.belief(SCOPE, "diet")
        assert held.record is None
        assert [record.value for record in held.decayed] == ["vegetarian"]
        assert held.weight == pytest.approx(0.0, abs=1e-3)

    async def test_a_recent_fact_is_still_recalled(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=10.0, floor=0.5))
        await kept.supersede(SCOPE, fact("vegetarian", at=TODAY))

        held = await kept.belief(SCOPE, "diet")
        assert held.record is not None
        assert held.weight == pytest.approx(1.0)
        assert held.decayed == ()

    async def test_a_whole_scope_going_quiet_is_countable(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=1.0, floor=0.9))
        await kept.supersede(SCOPE, fact("vegetarian", at=TODAY - 100.0))
        await kept.supersede(SCOPE, fact("window", key="seat", predicate=("seating",), at=1.0))

        beliefs = [await kept.belief(SCOPE, key) for key in ("diet", "seat")]
        assert all(held.record is None for held in beliefs)
        assert all(held.decayed for held in beliefs)

    async def test_decay_never_deletes(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=1.0, floor=0.9))
        await kept.supersede(SCOPE, fact("vegetarian", at=TODAY - 100.0))

        assert len(await kept.history(SCOPE, "diet")) == 1

    async def test_low_confidence_falls_below_the_floor(self) -> None:
        kept = store(decay=ConfidenceFloor(minimum=0.6))
        await kept.supersede(SCOPE, fact("vegetarian", confidence=0.4))

        assert await kept.profile(SCOPE, "diet") is None

    async def test_confident_enough_is_recalled_unweighted(self) -> None:
        kept = store(decay=ConfidenceFloor(minimum=0.6))
        await kept.supersede(SCOPE, fact("vegetarian", confidence=0.9))

        held = await kept.belief(SCOPE, "diet")
        assert held.record is not None
        assert held.weight == pytest.approx(0.9)

    async def test_a_decayed_episode_ranks_below_a_fresh_one(self) -> None:
        kept = store(decay=HalfLife(half_life_seconds=100.0))
        for value, at in (("old", TODAY - 1_000.0), ("new", TODAY)):
            await kept.log(
                SCOPE,
                MemoryRecord(
                    id=value,
                    kind=MemoryKind.EPISODIC,
                    scope=SCOPE,
                    key=value,
                    value=value,
                    source="turn",
                    valid_from=at,
                ),
            )
        found = await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
        assert [hit.record.value for hit in found] == ["new", "old"]
        assert found[0].score > found[1].score

    async def test_without_a_policy_nothing_decays(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", at=0.0))

        held = await kept.belief(SCOPE, "diet")
        assert held.record is not None
        assert held.weight == pytest.approx(1.0)

    async def test_a_fact_with_no_start_time_does_not_age(self) -> None:
        policy = HalfLife(half_life_seconds=10.0)
        assert policy.weigh(
            fact("vegetarian", at=0.0).model_copy(update={"valid_from": None}), now=TODAY
        ) == pytest.approx(1.0)


class TestBeliefsRead:
    async def test_an_unknown_key_holds_nothing(self) -> None:
        held = await store().belief(SCOPE, "diet")
        assert held.record is None
        assert held.contradiction is None
        assert held.decayed == ()

    async def test_belief_reads_as_of_a_past_instant(self) -> None:
        kept = store()
        await kept.supersede(SCOPE, fact("vegetarian", at=LAST_MONTH))
        await kept.supersede(SCOPE, fact("eats fish"))

        held = await kept.belief(SCOPE, "diet", as_of=LAST_MONTH + 1)
        assert held.record is not None
        assert held.record.value == "vegetarian"

    async def test_a_store_without_as_of_refuses_the_question(self) -> None:
        kept = store(capabilities=MemoryCapabilities(supports_as_of=False))
        await kept.supersede(SCOPE, fact("vegetarian"))
        with pytest.raises(Exception, match="as of"):
            await kept.profile(SCOPE, "diet", as_of=LAST_MONTH)
