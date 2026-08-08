"""What every memory adapter has to be, before any of them exist.

Four kinds of memory behind one protocol, every operation scoped, and a capability a
store does not have refused when the store is bound rather than silently returning
nothing on every run for a month.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from tesserix_adk.core import (
    CapabilityError,
    EmbeddingDimensionError,
    MemoryCorruptionError,
    MemoryLimitError,
    MemoryScopeError,
    SchemaViolationError,
    validated,
    verify_conformance,
)
from tesserix_adk.memory import (
    MemoryCapabilities,
    MemoryHit,
    MemoryKind,
    MemoryNeeds,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    require_memory,
)
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1", agent="planner")


def store(**capabilities: Any) -> InMemoryMemoryStore:
    """A fake store whose capabilities the test chooses."""
    return InMemoryMemoryStore(clock=FakeClock(), capabilities=MemoryCapabilities(**capabilities))


def record(
    kind: MemoryKind = MemoryKind.WORKING,
    *,
    key: str = "k",
    value: object = "v",
    scope: MemoryScope = SCOPE,
    **rest: object,
) -> MemoryRecord:
    """A record with the fields a test is not asserting on already filled in."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=scope,
        key=key,
        value=value,  # type: ignore[arg-type]
        source="test",
        **rest,  # type: ignore[arg-type]
    )


class TestScopeIsNotOptional:
    def test_a_scope_without_a_tenant_is_refused_rather_than_defaulted(self) -> None:
        """A default tenant is one typo away from being every tenant."""
        with pytest.raises(ValidationError):
            MemoryScope(user_id="u1")  # type: ignore[call-arg]

    def test_a_payload_missing_the_tenant_names_the_field(self) -> None:
        with pytest.raises(SchemaViolationError) as failure:
            validated(MemoryScope, {"user_id": "u1"})

        assert "tenant_id" in str(failure.value)

    def test_a_tenant_alone_is_a_scope(self) -> None:
        """Not everything remembered belongs to a user; a tenant-wide fact does not."""
        assert MemoryScope(tenant_id="acme").user_id is None

    def test_an_empty_tenant_is_not_a_tenant(self) -> None:
        with pytest.raises(ValidationError):
            MemoryScope(tenant_id="  ")

    def test_two_scopes_with_the_same_parts_are_the_same_scope(self) -> None:
        assert MemoryScope(tenant_id="acme", user_id="u1") == MemoryScope(
            tenant_id="acme", user_id="u1"
        )

    def test_a_scope_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            SCOPE.tenant_id = "other"

    async def test_every_operation_takes_one(self) -> None:
        """No unscoped overload exists, so a call site cannot forget."""
        from inspect import signature

        scoped = [
            "write",
            "read",
            "append",
            "expire",
            "upsert",
            "profile",
            "log",
            "episodes",
            "index",
            "search",
            "erase",
        ]
        for name in scoped:
            assert "scope" in signature(getattr(MemoryStore, name)).parameters


class TestTheFourKinds:
    async def test_working_memory_round_trips(self) -> None:
        kept = store()
        await kept.write(SCOPE, record(value={"trip": "BOM-DEL"}))

        found = await kept.read(SCOPE, "k")

        assert found is not None
        assert found.value == {"trip": "BOM-DEL"}

    async def test_reading_an_absent_key_is_none_rather_than_an_error(self) -> None:
        assert await store().read(SCOPE, "never-written") is None

    async def test_writing_replaces_rather_than_merges(self) -> None:
        kept = store()
        await kept.write(SCOPE, record(value={"a": 1}))
        await kept.write(SCOPE, record(value={"b": 2}))

        found = await kept.read(SCOPE, "k")

        assert found is not None
        assert found.value == {"b": 2}

    async def test_appending_builds_a_sequence(self) -> None:
        kept = store()
        await kept.append(SCOPE, "turns", "one")
        await kept.append(SCOPE, "turns", "two")

        found = await kept.read(SCOPE, "turns")

        assert found is not None
        assert found.value == ["one", "two"]

    async def test_an_expired_key_reads_as_absent(self) -> None:
        clock = FakeClock()
        kept = InMemoryMemoryStore(clock=clock, capabilities=MemoryCapabilities())
        await kept.write(SCOPE, record())
        await kept.expire(SCOPE, "k", ttl_seconds=60)

        clock.advance(61)

        assert await kept.read(SCOPE, "k") is None

    async def test_a_key_inside_its_window_still_reads(self) -> None:
        clock = FakeClock()
        kept = InMemoryMemoryStore(clock=clock, capabilities=MemoryCapabilities())
        await kept.write(SCOPE, record())
        await kept.expire(SCOPE, "k", ttl_seconds=60)

        clock.advance(59)

        assert await kept.read(SCOPE, "k") is not None

    async def test_a_profile_upsert_is_read_back_typed(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="seat", value="aisle"))

        found = await kept.profile(SCOPE, "seat")

        assert found is not None
        assert found.kind is MemoryKind.PROFILE
        assert found.value == "aisle"

    async def test_profile_and_working_do_not_share_a_key_space(self) -> None:
        """One namespace across kinds is a working key quietly overwriting a preference."""
        kept = store()
        await kept.write(SCOPE, record(key="seat", value="working"))
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="seat", value="profile"))

        working = await kept.read(SCOPE, "seat")

        assert working is not None
        assert working.value == "working"

    async def test_episodes_come_back_inside_the_window_only(self) -> None:
        kept = store()
        for at in (10.0, 20.0, 30.0):
            await kept.log(SCOPE, record(MemoryKind.EPISODIC, key=f"e{at}", valid_from=at))

        found = await kept.episodes(
            SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, since=15.0, until=25.0)
        )

        assert [hit.record.key for hit in found] == ["e20.0"]

    async def test_episodes_come_back_newest_first(self) -> None:
        kept = store()
        for at in (10.0, 30.0, 20.0):
            await kept.log(SCOPE, record(MemoryKind.EPISODIC, key=f"e{at}", valid_from=at))

        found = await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))

        assert [hit.record.key for hit in found] == ["e30.0", "e20.0", "e10.0"]

    async def test_a_query_limit_is_honoured(self) -> None:
        kept = store()
        for at in range(5):
            await kept.log(SCOPE, record(MemoryKind.EPISODIC, key=f"e{at}", valid_from=float(at)))

        found = await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=2))

        assert len(found) == 2

    async def test_semantic_search_ranks_by_distance(self) -> None:
        kept = store()
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="near", embedding=(1.0, 0.0)))
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="far", embedding=(0.0, 1.0)))

        found = await kept.search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(0.9, 0.1))
        )

        assert [hit.record.key for hit in found] == ["near", "far"]
        assert found[0].score > found[1].score

    async def test_a_hit_carries_the_record_it_scored(self) -> None:
        kept = store()
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="near", embedding=(1.0, 0.0)))

        found = await kept.search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0))
        )

        assert isinstance(found[0], MemoryHit)
        assert found[0].record.source == "test"


class TestOneTenantCannotReadAnother:
    async def test_a_write_under_one_tenant_is_invisible_to_the_next(self) -> None:
        kept = store()
        await kept.write(SCOPE, record())

        other = MemoryScope(tenant_id="other", user_id="u1", session_id="s1", agent="planner")

        assert await kept.read(other, "k") is None

    async def test_one_user_s_profile_is_not_another_s(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="seat", value="aisle"))

        other = SCOPE.model_copy(update={"user_id": "u2"})

        assert await kept.profile(other, "seat") is None

    async def test_a_record_written_into_a_scope_it_does_not_belong_to_is_refused(self) -> None:
        """The record says whose it is; disagreeing with the call is a bug, not a merge."""
        kept = store()

        with pytest.raises(MemoryScopeError) as failure:
            await kept.write(SCOPE.model_copy(update={"user_id": "u2"}), record())

        assert "u2" in str(failure.value)

    async def test_search_does_not_reach_across_tenants(self) -> None:
        kept = store()
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="near", embedding=(1.0, 0.0)))

        found = await kept.search(
            MemoryScope(tenant_id="other"),
            MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0)),
        )

        assert list(found) == []


class TestACapabilityIsCheckedWhenTheStoreIsBound:
    def test_a_store_that_cannot_do_semantic_recall_says_so_at_bind_time(self) -> None:
        """Empty recalls on every run for a month is the alternative."""
        with pytest.raises(CapabilityError) as failure:
            require_memory(store(supports_semantic=False), MemoryNeeds(semantic=True))

        assert "semantic" in str(failure.value)
        assert "InMemoryMemoryStore" in str(failure.value)

    def test_a_store_that_can_is_bound_without_complaint(self) -> None:
        require_memory(store(), MemoryNeeds(semantic=True, as_of=True, erasure=True))

    def test_every_missing_capability_is_named_at_once(self) -> None:
        """One deploy per missing capability is three deploys."""
        with pytest.raises(CapabilityError) as failure:
            require_memory(
                store(supports_semantic=False, supports_as_of=False, supports_erasure=False),
                MemoryNeeds(semantic=True, as_of=True, erasure=True),
            )

        assert {"semantic", "as_of", "erasure"} <= set(str(failure.value).split())

    def test_needing_nothing_binds_to_anything(self) -> None:
        require_memory(
            store(supports_semantic=False, supports_as_of=False, supports_erasure=False),
            MemoryNeeds(),
        )

    async def test_an_unsupported_operation_still_refuses_at_run_time(self) -> None:
        """A consumer who skipped the bind check gets an error, not an empty list."""
        with pytest.raises(CapabilityError):
            await store(supports_semantic=False).search(
                SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0,))
            )

    async def test_a_store_without_as_of_refuses_a_query_that_asks_for_it(self) -> None:
        with pytest.raises(CapabilityError):
            await store(supports_as_of=False).episodes(
                SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, as_of=5.0)
            )

    async def test_a_store_without_erasure_refuses_rather_than_reporting_nothing_erased(
        self,
    ) -> None:
        """Zero rows erased and cannot erase are the same number and opposite facts."""
        with pytest.raises(CapabilityError):
            await store(supports_erasure=False).erase(SCOPE)


class TestWhatGoesWrongIsTyped:
    async def test_a_record_that_no_longer_deserialises_names_itself(self) -> None:
        kept = store()
        kept.plant(SCOPE, MemoryKind.WORKING, "k", {"id": "k", "not": "a record"})

        with pytest.raises(MemoryCorruptionError) as failure:
            await kept.read(SCOPE, "k")

        assert failure.value.record_id == "k"
        assert failure.value.payload == {"id": "k", "not": "a record"}

    async def test_a_corrupt_record_is_not_quietly_dropped_from_a_search(self) -> None:
        """A prompt assembled from what survived is a prompt nobody can explain."""
        kept = store()
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="good", embedding=(1.0, 0.0)))
        kept.plant(SCOPE, MemoryKind.SEMANTIC, "bad", {"id": "bad"})

        with pytest.raises(MemoryCorruptionError):
            await kept.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0)))

    async def test_a_value_over_the_adapter_limit_is_refused_at_the_write(self) -> None:
        kept = store(max_value_bytes=16)

        with pytest.raises(MemoryLimitError) as failure:
            await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="bio", value="x" * 64))

        assert failure.value.limit == 16

    async def test_a_value_inside_the_limit_is_written(self) -> None:
        kept = store(max_value_bytes=1024)
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="bio", value="x" * 64))

        assert await kept.profile(SCOPE, "bio") is not None

    async def test_an_embedding_of_the_wrong_width_is_refused_rather_than_scored(self) -> None:
        kept = store(embedding_dimensions=3)

        with pytest.raises(EmbeddingDimensionError) as failure:
            await kept.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0)))

        assert (failure.value.expected, failure.value.received) == (3, 2)

    async def test_indexing_the_wrong_width_is_refused_too(self) -> None:
        kept = store(embedding_dimensions=3)

        with pytest.raises(EmbeddingDimensionError):
            await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="k", embedding=(1.0, 0.0)))

    async def test_a_semantic_record_with_no_embedding_is_refused(self) -> None:
        with pytest.raises(EmbeddingDimensionError):
            await store().index(SCOPE, record(MemoryKind.SEMANTIC, key="k"))

    async def test_a_search_with_no_embedding_is_refused(self) -> None:
        with pytest.raises(EmbeddingDimensionError):
            await store().search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC))

    async def test_a_record_filed_under_the_wrong_kind_is_refused(self) -> None:
        with pytest.raises(MemoryScopeError):
            await store().upsert(SCOPE, record(MemoryKind.WORKING, key="seat"))


class TestTheEdgesTwoProductsAlreadyHitByHand:
    async def test_concurrent_appends_are_ordered_and_none_is_lost(self) -> None:
        kept = store()

        await asyncio.gather(*(kept.append(SCOPE, "turns", n) for n in range(50)))

        found = await kept.read(SCOPE, "turns")
        assert found is not None
        assert sorted(found.value) == list(range(50))  # type: ignore[type-var, arg-type]

    async def test_every_append_gets_its_own_position(self) -> None:
        kept = store()

        positions = await asyncio.gather(*(kept.append(SCOPE, "turns", n) for n in range(50)))

        assert sorted(positions) == list(range(1, 51))

    async def test_a_read_during_an_erasure_sees_all_of_it_or_none_of_it(self) -> None:
        """A half-erased scope in a prompt is the worst of both."""
        kept = store()
        for n in range(20):
            await kept.log(SCOPE, record(MemoryKind.EPISODIC, key=f"e{n}", valid_from=float(n)))

        erased, found = await asyncio.gather(
            kept.erase(SCOPE), kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=50))
        )

        assert erased == 20
        assert len(found) in (0, 20)

    async def test_erasure_reaches_every_kind(self) -> None:
        kept = store()
        await kept.write(SCOPE, record())
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, key="seat", value="aisle"))
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="s", embedding=(1.0, 0.0)))

        await kept.erase(SCOPE)

        assert await kept.read(SCOPE, "k") is None
        assert await kept.profile(SCOPE, "seat") is None

    async def test_erasure_stops_at_the_scope_it_was_given(self) -> None:
        kept = store()
        await kept.write(SCOPE, record())
        other = SCOPE.model_copy(update={"user_id": "u2"})
        await kept.write(other, record(scope=other))

        await kept.erase(SCOPE)

        assert await kept.read(other, "k") is not None

    async def test_as_of_reads_what_was_true_then(self) -> None:
        kept = store()
        await kept.log(
            SCOPE,
            record(MemoryKind.EPISODIC, key="old", valid_from=0.0, valid_to=10.0),
        )
        await kept.log(SCOPE, record(MemoryKind.EPISODIC, key="new", valid_from=10.0))

        found = await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, as_of=5.0))

        assert [hit.record.key for hit in found] == ["old"]

    def test_confidence_outside_zero_to_one_is_not_a_confidence(self) -> None:
        with pytest.raises(ValidationError):
            record(confidence=1.5)


class TestTheFakeIsTheContract:
    def test_it_satisfies_the_protocol(self) -> None:
        verify_conformance(store(), MemoryStore)

    def test_it_declares_what_it_can_do(self) -> None:
        assert store().capabilities.supports_semantic is True

    def test_a_store_is_not_shared_between_tests(self) -> None:
        assert store() is not store()

    async def test_a_vector_of_another_width_resembles_nothing(self) -> None:
        kept = store()
        await kept.index(SCOPE, record(MemoryKind.SEMANTIC, key="s", embedding=(1.0, 0.0)))

        found = await kept.search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )

        assert [hit.score for hit in found] == [0.0]
