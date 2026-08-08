"""What never reaches memory, and what leaves it when somebody asks."""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import MASK, PartialErasureError
from tesserix_adk.memory import (
    Derivation,
    ErasureReceipt,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    PatternRedactor,
)
from tesserix_adk.testing import FakeClock, FakeTracer, InMemoryMemoryStore, RecordedEvent

SCOPE = MemoryScope(tenant_id="acme", user_id="u1")
OTHER = MemoryScope(tenant_id="globex", user_id="u9")
NOW = 1_000.0


def record(
    kind: MemoryKind,
    key: str,
    value: object = "v",
    *,
    scope: MemoryScope = SCOPE,
    embedding: tuple[float, ...] | None = None,
) -> MemoryRecord:
    """A record of any kind, filled in enough to be stored."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=scope,
        key=key,
        value=value,  # type: ignore[arg-type]
        source="turn",
        valid_from=NOW,
        embedding=embedding,
    )


class Index:
    """A derived-artefact index that says what it was asked to purge.

    Args:
        name: What erasure matches `Derivation.adapter` against.
        reachable: Whether it can be spoken to at all, so a test can take it away.
    """

    def __init__(self, name: str = "vectors", *, reachable: bool = True) -> None:
        self.name = name
        self.reachable = reachable
        self.held: dict[str, str] = {}
        self.purged: list[str] = []

    async def purge(self, artefact_ids: tuple[str, ...]) -> int:
        """Drop each artefact, or fail the way a call to an unreachable service fails."""
        if not self.reachable:
            raise ConnectionError("vector index unreachable")
        self.purged.extend(artefact_ids)
        gone = [one for one in artefact_ids if self.held.pop(one, None) is not None]
        return len(gone)


def store(**kwargs: object) -> InMemoryMemoryStore:
    """A store with a clock a test can read back off the receipt."""
    return InMemoryMemoryStore(clock=FakeClock(start=NOW), **kwargs)  # type: ignore[arg-type]


class TestNothingSensitiveReachesTheStore:
    async def test_a_card_number_is_masked_before_it_is_written(self) -> None:
        kept = store()
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k", "card 4111 1111 1111 1111"))

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert "4111" not in str(held.value)
        assert MASK in str(held.value)

    async def test_the_record_names_what_was_masked(self) -> None:
        kept = store()
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k", {"who": "ada@example.com"}))

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert held.redacted == ("who",)

    async def test_an_ordinary_value_passes_through_untouched(self) -> None:
        kept = store()
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k", {"seat": "aisle"}))

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert held.value == {"seat": "aisle"}
        assert held.redacted == ()

    async def test_it_reaches_inside_nested_values(self) -> None:
        kept = store()
        await kept.write(
            SCOPE,
            record(MemoryKind.WORKING, "k", {"trip": {"contact": ["ada@example.com", "ok"]}}),
        )

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert held.value == {"trip": {"contact": [MASK, "ok"]}}
        assert held.redacted == ("trip.contact.0",)

    async def test_every_write_path_is_covered(self) -> None:
        kept = store()
        await kept.upsert(SCOPE, record(MemoryKind.PROFILE, "p", "ada@example.com"))
        await kept.log(SCOPE, record(MemoryKind.EPISODIC, "e", "ada@example.com"))
        await kept.append(SCOPE, "turns", "ada@example.com")
        await kept.index(
            SCOPE, record(MemoryKind.SEMANTIC, "s", "ada@example.com", embedding=(1.0,))
        )

        profile = await kept.profile(SCOPE, "p")
        episodes = await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
        appended = await kept.read(SCOPE, "turns")
        found = await kept.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0,)))
        assert profile is not None
        assert profile.value == MASK
        assert episodes[0].record.value == MASK
        assert appended is not None
        assert appended.value == [MASK]
        assert found[0].record.value == MASK

    async def test_a_supersession_is_redacted_too(self) -> None:
        kept = store()
        written = await kept.supersede(SCOPE, record(MemoryKind.PROFILE, "p", "ada@example.com"))
        assert written.record.value == MASK

    async def test_a_deployment_can_add_its_own_shape(self) -> None:
        kept = store(redactor=PatternRedactor(extra_patterns=(r"CASE-\d+",)))
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k", "filed under CASE-4471"))

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert held.value == "filed under [redacted]"

    async def test_turning_it_off_is_deliberate_and_visible(self) -> None:
        kept = store(redactor=None)
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k", "ada@example.com"))

        held = await kept.read(SCOPE, "k")
        assert held is not None
        assert held.value == "ada@example.com"


class TestTheReceiptSaysWhatWent:
    async def test_it_counts_every_kind_it_removed(self) -> None:
        kept = await filled()
        receipt = await kept.erase(SCOPE)

        assert isinstance(receipt, ErasureReceipt)
        assert receipt.counts[MemoryKind.PROFILE.value] == 1
        assert receipt.counts[MemoryKind.EPISODIC.value] == 1
        assert receipt.counts[MemoryKind.SEMANTIC.value] == 1
        assert receipt.complete
        assert receipt.completed_at == NOW

    async def test_recall_of_every_kind_comes_back_empty(self) -> None:
        kept = await filled()
        await kept.erase(SCOPE)

        assert await kept.profile(SCOPE, "p") is None
        assert await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC)) == []
        found = await kept.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0,)))
        assert found == []

    async def test_it_names_the_adapters_it_spoke_to(self) -> None:
        index = Index()
        kept = await filled(indices=(index,))
        receipt = await kept.erase(SCOPE)

        assert "vectors" in receipt.adapters

    async def test_erasing_one_kind_leaves_the_others(self) -> None:
        kept = await filled()
        receipt = await kept.erase(SCOPE, kinds=(MemoryKind.EPISODIC,))

        assert set(receipt.counts) == {MemoryKind.EPISODIC.value}
        assert await kept.profile(SCOPE, "p") is not None

    async def test_another_scope_is_untouched(self) -> None:
        kept = await filled()
        await kept.upsert(OTHER, record(MemoryKind.PROFILE, "p", scope=OTHER))
        await kept.erase(SCOPE)

        assert await kept.profile(OTHER, "p") is not None


class TestDerivedArtefactsGoWithTheirSource:
    async def test_the_vector_is_purged_not_merely_unlinked(self) -> None:
        index = Index()
        kept = await filled(indices=(index,))
        index.held["vec-1"] = "the sensitive sentence"
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )

        await kept.erase(SCOPE)
        assert index.held == {}
        assert index.purged == ["vec-1"]

    async def test_a_semantic_search_for_the_erased_text_finds_nothing(self) -> None:
        kept = await filled()
        await kept.erase(SCOPE)

        found = await kept.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0,)))
        assert found == []
        assert await kept.derivations(SCOPE) == ()

    async def test_a_derivation_can_be_read_back_before_erasure(self) -> None:
        kept = await filled()
        made = Derivation(artefact_id="sum-1", source_id="episodic:e", adapter="summaries")
        await kept.derived(SCOPE, made)

        assert await kept.derivations(SCOPE, source_id="episodic:e") == (made,)

    async def test_an_artefact_two_tenants_share_is_never_purged_cross_tenant(self) -> None:
        index = Index("cache")
        kept = await filled(indices=(index,))
        index.held["emb-shared"] = "hello"
        shared = Derivation(artefact_id="emb-shared", source_id="episodic:e", adapter="cache")
        await kept.derived(SCOPE, shared)
        await kept.derived(OTHER, shared)

        await kept.erase(SCOPE)
        assert index.held == {"emb-shared": "hello"}
        assert await kept.derivations(OTHER) == (shared,)
        assert await kept.derivations(SCOPE) == ()

    async def test_every_version_of_a_supersession_chain_goes(self) -> None:
        kept = store()
        for value in ("a", "b", "c"):
            await kept.supersede(SCOPE, record(MemoryKind.PROFILE, "p", value))

        receipt = await kept.erase(SCOPE)
        assert receipt.counts[MemoryKind.PROFILE.value] == 3
        assert await kept.history(SCOPE) == ()


class TestWhenAnIndexCannotBeReached:
    async def test_erasure_says_so_rather_than_reporting_success(self) -> None:
        kept = await filled(indices=(Index(reachable=False),))
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )

        with pytest.raises(PartialErasureError) as failed:
            await kept.erase(SCOPE)
        assert failed.value.adapter == "vectors"
        assert not failed.value.receipt.complete
        assert failed.value.receipt.outstanding == ("vectors",)

    async def test_the_records_are_already_out_of_reach(self) -> None:
        kept = await filled(indices=(Index(reachable=False),))
        await kept.write(SCOPE, record(MemoryKind.WORKING, "k"))
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        with pytest.raises(PartialErasureError):
            await kept.erase(SCOPE)

        assert await kept.read(SCOPE, "k") is None
        assert await kept.profile(SCOPE, "p") is None
        assert await kept.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC)) == []

    async def test_it_is_resumable_once_the_index_is_back(self) -> None:
        index = Index(reachable=False)
        kept = await filled(indices=(index,))
        index.held["vec-1"] = "sentence"
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        with pytest.raises(PartialErasureError):
            await kept.erase(SCOPE)

        index.reachable = True
        receipt = await kept.erase(SCOPE)
        assert receipt.complete
        assert index.held == {}

    async def test_resuming_does_not_count_what_it_already_removed_twice(self) -> None:
        index = Index(reachable=False)
        kept = await filled(indices=(index,))
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        with pytest.raises(PartialErasureError) as failed:
            await kept.erase(SCOPE)
        first = failed.value.receipt

        index.reachable = True
        second = await kept.erase(SCOPE)
        assert first.counts[MemoryKind.PROFILE.value] == 1
        assert second.counts.get(MemoryKind.PROFILE.value, 0) == 0


class TestErasureIsHonestAboutDoingNothing:
    async def test_a_dry_run_changes_nothing(self) -> None:
        index = Index()
        kept = await filled(indices=(index,))
        index.held["vec-1"] = "sentence"
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )

        receipt = await kept.erase(SCOPE, dry_run=True)
        assert receipt.dry_run
        assert receipt.counts[MemoryKind.PROFILE.value] == 1
        assert await kept.profile(SCOPE, "p") is not None
        assert index.held == {"vec-1": "sentence"}
        assert index.purged == []

    async def test_a_dry_run_is_not_a_completed_erasure(self) -> None:
        kept = await filled()
        receipt = await kept.erase(SCOPE, dry_run=True)
        assert not receipt.complete
        assert receipt.completed_at is None

    async def test_running_it_again_removes_nothing_and_says_so(self) -> None:
        kept = await filled()
        await kept.erase(SCOPE)
        again = await kept.erase(SCOPE)

        assert again.complete
        assert sum(again.counts.values()) == 0

    async def test_erasing_a_scope_that_was_never_written_to_is_not_an_error(self) -> None:
        receipt = await store().erase(OTHER)
        assert receipt.complete
        assert sum(receipt.counts.values()) == 0


class TestAWriteThatArrivesDuringAnErasure:
    async def test_it_is_not_swept_up_by_a_snapshot_taken_before_it(self) -> None:
        held = asyncio.Event()

        class Slow(Index):
            async def purge(self, artefact_ids: tuple[str, ...]) -> int:
                """Wait long enough for a writer to get in."""
                await held.wait()
                return await super().purge(artefact_ids)

        kept = await filled(indices=(Slow(),))
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        erasing = asyncio.create_task(kept.erase(SCOPE))
        await asyncio.sleep(0)
        await kept.write(SCOPE, record(MemoryKind.WORKING, "later", "after the snapshot"))
        held.set()
        receipt = await erasing

        assert receipt.complete
        assert await kept.read(SCOPE, "later") is not None
        assert await kept.profile(SCOPE, "p") is None


class TestErasureIsAudited:
    async def test_it_publishes_ids_and_counts(self) -> None:
        tracer = FakeTracer()
        kept = await filled(tracer=tracer)
        await kept.erase(SCOPE)

        published = _erasures(tracer)
        assert len(published) == 1
        assert published[0].attributes["adk.memory.records"] == "3"

    async def test_it_publishes_nothing_that_was_erased(self) -> None:
        tracer = FakeTracer()
        kept = await filled(tracer=tracer)
        await kept.erase(SCOPE)

        said = str(tracer.recorded)
        assert "ada@example.com" not in said
        assert "the sensitive sentence" not in said

    async def test_a_partial_erasure_is_published_as_incomplete(self) -> None:
        tracer = FakeTracer()
        kept = await filled(indices=(Index(reachable=False),), tracer=tracer)
        await kept.derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        with pytest.raises(PartialErasureError):
            await kept.erase(SCOPE)

        published = _erasures(tracer)
        assert published[0].attributes["adk.memory.complete"] == "false"

    async def test_a_dry_run_is_not_published_as_an_erasure(self) -> None:
        tracer = FakeTracer()
        kept = await filled(tracer=tracer)
        await kept.erase(SCOPE, dry_run=True)

        assert _erasures(tracer) == []


def _erasures(tracer: FakeTracer) -> list[RecordedEvent]:
    """Every erasure the tracer was told about."""
    return [event for event in tracer.recorded if event.name == "adk.memory.erased"]


async def filled(**kwargs: object) -> InMemoryMemoryStore:
    """A store holding one record of each durable kind, under `SCOPE`."""
    kept = store(**kwargs)
    await kept.upsert(SCOPE, record(MemoryKind.PROFILE, "p", "ada@example.com"))
    await kept.log(SCOPE, record(MemoryKind.EPISODIC, "e", "the sensitive sentence"))
    await kept.index(SCOPE, record(MemoryKind.SEMANTIC, "s", "known", embedding=(1.0,)))
    return kept
