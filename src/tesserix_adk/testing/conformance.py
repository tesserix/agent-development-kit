"""Shared conformance suites for the core protocols.

An implementation is substitutable only if it behaves the way the runtime assumes,
and structural typing cannot express "deleting an absent key is not an error". These
suites carry those assumptions. First- and third-party implementations subclass one,
supply the implementation under test, and inherit the whole suite:

```python
from tesserix_adk.testing import KeyValueStoreConformance


class TestRedisStore(KeyValueStoreConformance):
    def make_store(self):
        return RedisKeyValueStore(url="redis://localhost")
```

Adding a member to a protocol means adding its case here in the same change, so
every implementation learns about it by failing rather than by drifting.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core.capabilities import Capability, ModelCapabilities
from tesserix_adk.core.checkpoint import Checkpoint, CheckpointStore, PendingCall
from tesserix_adk.core.errors import (
    BudgetExceededError,
    BudgetUnavailableError,
    CapabilityError,
    LeaseLostError,
    MemoryConflictError,
    MemoryScopeError,
    RunLeaseError,
    StateConflictError,
    StateInUseError,
    StateNotFoundError,
    WorkItemNotFoundError,
)
from tesserix_adk.core.idempotency import IdempotencyStore
from tesserix_adk.core.lease import LeaseStore, RunLease
from tesserix_adk.core.ledger import LedgerKey, SpendLedger, Window, WindowKind
from tesserix_adk.core.primitives import Message, TextPart, ToolCall, Usage
from tesserix_adk.core.propagation import HEADER, arriving, carried, restored
from tesserix_adk.core.protocols import (
    BudgetPolicy,
    Clock,
    KeyValueStore,
    ModelProvider,
    Tracer,
    verify_conformance,
)
from tesserix_adk.core.provider import ModelRequest, ModelResponse
from tesserix_adk.core.queue import WorkItem, WorkQueue, WorkState
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import (
    RunRecord,
    SessionRecord,
    StateDelta,
    StateKey,
    StateQuery,
    StateStore,
)
from tesserix_adk.core.tenancy import TenantContext, tenant_here
from tesserix_adk.memory import (
    Derivation,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
)
from tesserix_adk.rag.retrieval import Branch, IndexQuery, SearchIndex
from tesserix_adk.testing.retrieval import Indexed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

__all__ = [
    "CONFORMANCE_CORPUS",
    "BudgetPolicyConformance",
    "CheckpointStoreConformance",
    "ClockConformance",
    "IdempotencyStoreConformance",
    "KeyValueStoreConformance",
    "LeaseStoreConformance",
    "MemoryStoreConformance",
    "ModelProviderConformance",
    "SearchIndexConformance",
    "StateStoreConformance",
    "TenantPropagationConformance",
    "TracerConformance",
    "WorkQueueConformance",
]


def _request(*texts: str) -> ModelRequest:
    """The smallest request a provider can be asked to answer."""
    return ModelRequest(
        model="conformance",
        messages=tuple(Message(role="user", content=[TextPart(text=t)]) for t in texts),
    )


class ModelProviderConformance(ABC):
    """Behaviour every `ModelProvider` implementation must exhibit.

    A provider is substitutable only if the kit can read its limits before it calls, so
    most of this suite is about the capability record rather than the completion.
    """

    @abstractmethod
    def make_provider(self) -> ModelProvider:
        """Return a provider under test, able to answer several requests."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_provider(), ModelProvider)

    def test_it_is_named(self) -> None:
        name = self.make_provider().name
        assert isinstance(name, str)
        assert name

    def test_it_declares_what_it_can_do(self) -> None:
        assert isinstance(self.make_provider().capabilities, ModelCapabilities)

    def test_the_declaration_does_not_change_between_reads(self) -> None:
        """A record that varies is a record nothing can be checked against."""
        provider = self.make_provider()
        assert provider.capabilities == provider.capabilities

    async def test_it_answers_with_a_response(self) -> None:
        assert isinstance(await self.make_provider().complete(_request("hello")), ModelResponse)

    def test_it_counts_tokens_without_going_negative(self) -> None:
        assert self.make_provider().count_tokens(_request("hello").messages) >= 0

    def test_a_longer_prompt_does_not_count_for_less(self) -> None:
        provider = self.make_provider()
        short = provider.count_tokens(_request("hello").messages)
        long = provider.count_tokens(_request("hello", "hello again at some length").messages)
        assert long >= short

    async def test_streaming_it_never_declared_is_refused(self) -> None:
        """A provider that buffers one chunk and calls it a stream is worse than a refusal."""
        provider = self.make_provider()
        if provider.capabilities.supports(Capability.STREAMING):
            return
        with pytest.raises(CapabilityError):
            await provider.stream(_request("hello"))


class KeyValueStoreConformance(ABC):
    """Behaviour every `KeyValueStore` implementation must exhibit."""

    @abstractmethod
    def make_store(self) -> KeyValueStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), KeyValueStore)

    async def test_get_returns_none_for_an_absent_key(self) -> None:
        assert await self.make_store().get("absent") is None

    async def test_put_then_get_round_trips(self) -> None:
        store = self.make_store()
        await store.put("k", {"v": 1})
        assert await store.get("k") == {"v": 1}

    async def test_put_replaces_rather_than_merges(self) -> None:
        store = self.make_store()
        await store.put("k", {"a": 1})
        await store.put("k", {"b": 2})
        assert await store.get("k") == {"b": 2}

    async def test_delete_removes_the_key(self) -> None:
        store = self.make_store()
        await store.put("k", "v")
        await store.delete("k")
        assert await store.get("k") is None

    async def test_deleting_an_absent_key_is_not_an_error(self) -> None:
        await self.make_store().delete("never-existed")

    async def test_keys_do_not_collide_across_distinct_values(self) -> None:
        store = self.make_store()
        await store.put("a", 1)
        await store.put("b", 2)
        assert (await store.get("a"), await store.get("b")) == (1, 2)


class ClockConformance(ABC):
    """Behaviour every `Clock` implementation must exhibit."""

    @abstractmethod
    def make_clock(self) -> Clock:
        """Return a fresh clock under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_clock(), Clock)

    def test_now_does_not_go_backwards(self) -> None:
        clock = self.make_clock()
        assert clock.now() <= clock.now()

    async def test_sleep_does_not_move_time_backwards(self) -> None:
        clock = self.make_clock()
        before = clock.now()
        await clock.sleep(0)
        assert clock.now() >= before


class BudgetPolicyConformance(ABC):
    """Behaviour every `BudgetPolicy` implementation must exhibit."""

    @abstractmethod
    def make_policy(self) -> BudgetPolicy:
        """Return a fresh policy under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_policy(), BudgetPolicy)

    async def test_a_reservation_within_budget_is_permitted(self) -> None:
        await self.make_policy().reserve(1)

    async def test_recording_after_reserving_does_not_raise(self) -> None:
        policy = self.make_policy()
        await policy.reserve(10)
        await policy.record(Usage(input_tokens=8, output_tokens=0))

    def test_a_ceiling_is_readable_off_the_policy(self) -> None:
        """A limit nobody can read afterwards is a limit nobody can audit."""
        assert self.make_policy().resolved.limits is not None

    def test_a_child_spends_what_the_parent_has_left(self) -> None:
        """A sub-agent handed a fresh allowance is a way to spend one ceiling twice."""
        policy = self.make_policy()
        assert policy.child().limits() == policy.limits()


class TracerConformance(ABC):
    """Behaviour every `Tracer` implementation must exhibit.

    The defining requirement is that tracing fails open: a collector outage must
    degrade observability, never stop a run.
    """

    @abstractmethod
    def make_tracer(self) -> Tracer:
        """Return a fresh tracer under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_tracer(), Tracer)

    def test_event_never_raises(self) -> None:
        self.make_tracer().event("anything", attribute=object())

    def test_span_yields_and_closes(self) -> None:
        with self.make_tracer().span("work", attribute=1):
            pass

    def test_span_does_not_swallow_the_bodys_exception(self) -> None:
        tracer = self.make_tracer()
        raised = False
        try:
            with tracer.span("work"):
                raise ValueError("from the body")
        except ValueError:
            raised = True
        assert raised, "a span must not swallow the exception its body raised"


class SpendLedgerConformance(ABC):
    """Behaviour every `SpendLedger` implementation must exhibit.

    A store that gets any of these wrong lets a tenant past its ceiling, which is the one
    thing the ledger exists to prevent. Structural typing cannot express "the ceiling check
    and the hold happen together", so it is asserted here and every implementation —
    in-process, Redis, PostgreSQL, or a deployment's own — inherits the suite.
    """

    ceiling = Decimal("10.00")

    @abstractmethod
    def make_ledger(self) -> SpendLedger:
        """Return a fresh, empty ledger under test."""

    def a_key(self, tenant: str = "conformance") -> LedgerKey:
        """A window to spend against. Rolling by default, because that is the harder one."""
        return LedgerKey(
            tenant=tenant, agent=None, window=Window(kind=WindowKind.ROLLING, seconds=3_600)
        )

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_ledger(), SpendLedger)

    async def test_settled_spend_is_visible_to_the_next_reader(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.settle(
            await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling), Decimal("1.00")
        )
        assert (await ledger.read_window(key)).settled == Decimal("1.00")

    async def test_a_hold_counts_before_it_settles(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.reserve(key, Decimal("4.00"), ceiling=self.ceiling)
        assert (await ledger.read_window(key)).committed == Decimal("4.00")

    async def test_the_ceiling_refuses_rather_than_overshoots(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.reserve(key, Decimal("9.50"), ceiling=self.ceiling)
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling)

    async def test_concurrent_holds_do_not_pass_the_ceiling(self) -> None:
        """The reserve-and-check has to be one operation, whatever the store."""
        ledger, key = self.make_ledger(), self.a_key()
        granted = await asyncio.gather(
            *(self._try(ledger, key, Decimal("1.00")) for _ in range(40))
        )
        assert sum(granted) == 10

    async def test_a_release_gives_the_allowance_back(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.release(await ledger.reserve(key, Decimal("9.00"), ceiling=self.ceiling))
        assert (await ledger.read_window(key)).reserved == Decimal(0)

    async def test_settling_twice_is_refused(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        held = await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling)
        await ledger.settle(held, Decimal("1.00"))
        with pytest.raises(BudgetUnavailableError):
            await ledger.settle(held, Decimal("1.00"))

    async def test_one_tenant_cannot_read_another_s_window(self) -> None:
        ledger = self.make_ledger()
        await ledger.settle(
            await ledger.reserve(self.a_key("one"), Decimal("2.00"), ceiling=self.ceiling),
            Decimal("2.00"),
        )
        assert (await ledger.read_window(self.a_key("two"))).settled == Decimal(0)

    async def test_erasure_leaves_nothing_of_the_tenant_behind(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.settle(
            await ledger.reserve(key, Decimal("2.00"), ceiling=self.ceiling), Decimal("2.00")
        )
        assert (await ledger.forget(key.tenant)).settled == Decimal("2.00")
        assert (await ledger.read_window(key)).settled == Decimal(0)

    async def _try(self, ledger: SpendLedger, key: LedgerKey, amount: Decimal) -> bool:
        try:
            held = await ledger.reserve(key, amount, ceiling=self.ceiling)
        except BudgetExceededError:
            return False
        await ledger.settle(held, amount)
        return True


class IdempotencyStoreConformance(ABC):
    """Behaviour every `IdempotencyStore` implementation must exhibit.

    A store that gets any of these wrong books a second seat, which is the one thing it
    exists to prevent. The guarantee is at-most-once per key within the retention window,
    and it holds only if a claim is exclusive: two callers asking together get one winner.
    """

    @abstractmethod
    def make_store(self) -> IdempotencyStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), IdempotencyStore)

    async def test_an_unclaimed_key_is_free(self) -> None:
        claimed = await self.make_store().begin("k", tenant="conformance", ttl_seconds=900)
        assert claimed.outcome is None
        assert claimed.in_flight is False

    async def test_a_claimed_key_is_in_flight_until_it_is_recorded(self) -> None:
        store = self.make_store()
        await store.begin("k", tenant="conformance", ttl_seconds=900)
        assert (await store.begin("k", tenant="conformance", ttl_seconds=900)).in_flight is True

    async def test_a_recorded_outcome_answers_the_next_caller(self) -> None:
        store = self.make_store()
        await store.begin("k", tenant="conformance", ttl_seconds=900)
        await store.record("k", tenant="conformance", outcome="done", ttl_seconds=900)
        assert (await store.begin("k", tenant="conformance", ttl_seconds=900)).outcome == "done"

    async def test_only_one_of_many_concurrent_callers_wins_the_claim(self) -> None:
        """The claim-and-check has to be one operation, whatever the store."""
        store = self.make_store()
        claims = await asyncio.gather(
            *(store.begin("k", tenant="conformance", ttl_seconds=900) for _ in range(20))
        )
        assert sum(1 for claim in claims if not claim.in_flight and claim.outcome is None) == 1

    async def test_an_abandoned_claim_frees_the_key(self) -> None:
        store = self.make_store()
        await store.begin("k", tenant="conformance", ttl_seconds=900)
        await store.abandon("k", tenant="conformance")
        assert (await store.begin("k", tenant="conformance", ttl_seconds=900)).in_flight is False

    async def test_abandoning_a_key_nobody_holds_is_not_an_error(self) -> None:
        await self.make_store().abandon("never-claimed", tenant="conformance")

    async def test_one_tenant_cannot_see_another_s_record(self) -> None:
        store = self.make_store()
        await store.record("k", tenant="one", outcome="done", ttl_seconds=900)
        assert (await store.begin("k", tenant="two", ttl_seconds=900)).outcome is None

    async def test_erasure_leaves_nothing_of_the_tenant_behind(self) -> None:
        store = self.make_store()
        await store.record("k", tenant="one", outcome="done", ttl_seconds=900)
        assert await store.forget(tenant="one") == 1
        assert (await store.begin("k", tenant="one", ttl_seconds=900)).outcome is None


SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1", agent="planner")


def _record(
    kind: MemoryKind,
    key: str,
    value: JsonValue = "v",
    *,
    valid_from: float | None = None,
    embedding: tuple[float, ...] | None = None,
) -> MemoryRecord:
    """A record with the fields the suite is not asserting on already filled in."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=SCOPE,
        key=key,
        value=value,
        source="conformance",
        valid_from=valid_from,
        embedding=embedding,
    )


class MemoryStoreConformance(ABC):
    """Behaviour every `MemoryStore` implementation must exhibit.

    Capability-gated cases skip themselves against a store that declares it cannot do
    the thing, so an adapter is held to what it claims and not to what it does not.
    """

    @abstractmethod
    def make_store(self) -> MemoryStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), MemoryStore)

    async def test_working_memory_round_trips(self) -> None:
        store = self.make_store()
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k", {"a": 1}))
        found = await store.read(SCOPE, "k")
        assert found is not None
        assert found.value == {"a": 1}

    async def test_reading_an_absent_key_is_none(self) -> None:
        assert await self.make_store().read(SCOPE, "absent") is None

    async def test_writing_replaces_rather_than_merges(self) -> None:
        store = self.make_store()
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k", {"a": 1}))
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k", {"b": 2}))
        found = await store.read(SCOPE, "k")
        assert found is not None
        assert found.value == {"b": 2}

    async def test_appends_are_ordered_and_none_is_lost(self) -> None:
        store = self.make_store()
        positions = await asyncio.gather(*(store.append(SCOPE, "turns", n) for n in range(20)))
        assert sorted(positions) == list(range(1, 21))

    async def test_a_profile_upsert_is_read_back(self) -> None:
        store = self.make_store()
        await store.upsert(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))
        found = await store.profile(SCOPE, "seat")
        assert found is not None
        assert found.value == "aisle"

    async def test_kinds_do_not_share_a_key_space(self) -> None:
        store = self.make_store()
        await store.write(SCOPE, _record(MemoryKind.WORKING, "seat", "working"))
        await store.upsert(SCOPE, _record(MemoryKind.PROFILE, "seat", "profile"))
        found = await store.read(SCOPE, "seat")
        assert found is not None
        assert found.value == "working"

    async def test_episodes_come_back_newest_first(self) -> None:
        store = self.make_store()
        for at in (10.0, 30.0, 20.0):
            await store.log(SCOPE, _record(MemoryKind.EPISODIC, f"e{at}", valid_from=at))
        found = await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
        assert [hit.record.key for hit in found] == ["e30.0", "e20.0", "e10.0"]

    async def test_a_window_excludes_what_falls_outside_it(self) -> None:
        store = self.make_store()
        for at in (10.0, 20.0, 30.0):
            await store.log(SCOPE, _record(MemoryKind.EPISODIC, f"e{at}", valid_from=at))
        found = await store.episodes(
            SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, since=15.0, until=25.0)
        )
        assert [hit.record.key for hit in found] == ["e20.0"]

    async def test_a_scope_cannot_read_another_s_records(self) -> None:
        store = self.make_store()
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k"))
        other = MemoryScope(tenant_id="other", user_id="u1", session_id="s1", agent="planner")
        assert await store.read(other, "k") is None

    async def test_a_record_from_another_scope_is_refused(self) -> None:
        store = self.make_store()
        with pytest.raises(MemoryScopeError):
            await store.write(
                SCOPE.model_copy(update={"user_id": "u2"}), _record(MemoryKind.WORKING, "k")
            )

    async def test_semantic_search_ranks_the_closer_record_higher(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_semantic:
            return
        await store.index(SCOPE, _record(MemoryKind.SEMANTIC, "near", embedding=(1.0, 0.0)))
        await store.index(SCOPE, _record(MemoryKind.SEMANTIC, "far", embedding=(0.0, 1.0)))
        found = await store.search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(0.9, 0.1))
        )
        assert next(hit.record.key for hit in found) == "near"

    async def test_semantic_recall_it_never_declared_is_refused(self) -> None:
        """Returning nothing forever is the alternative, and nobody notices it."""
        store = self.make_store()
        if store.capabilities.supports_semantic:
            return
        with pytest.raises(CapabilityError):
            await store.search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0,)))

    async def test_a_superseded_record_is_closed_and_kept(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_supersession:
            return
        first = await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))
        second = await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "window"))
        assert second.superseded is not None
        assert second.superseded.superseded_by == second.record.id
        assert [held.value for held in await store.history(SCOPE, "seat")] == ["aisle", "window"]
        assert first.record.version < second.record.version

    async def test_only_one_record_is_live_after_a_supersession(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_supersession:
            return
        await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))
        await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "window"))
        live = await store.profile(SCOPE, "seat")
        assert live is not None
        assert live.value == "window"

    async def test_a_stale_expected_version_is_refused(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_supersession:
            return
        first = await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))
        await store.supersede(
            SCOPE,
            _record(MemoryKind.PROFILE, "seat", "window"),
            expected_version=first.record.version,
        )
        with pytest.raises(MemoryConflictError):
            await store.supersede(
                SCOPE,
                _record(MemoryKind.PROFILE, "seat", "middle"),
                expected_version=first.record.version,
            )

    async def test_belief_says_nothing_rather_than_guessing(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_supersession:
            return
        held = await store.belief(SCOPE, "never-written")
        assert held.record is None
        assert held.contradiction is None

    async def test_supersession_it_never_declared_is_refused(self) -> None:
        store = self.make_store()
        if store.capabilities.supports_supersession:
            return
        with pytest.raises(CapabilityError):
            await store.supersede(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))

    async def test_erasure_removes_every_kind_under_the_scope(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_erasure:
            return
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k"))
        await store.upsert(SCOPE, _record(MemoryKind.PROFILE, "seat", "aisle"))
        receipt = await store.erase(SCOPE)
        assert receipt.complete
        assert receipt.records >= 2
        assert await store.read(SCOPE, "k") is None
        assert await store.profile(SCOPE, "seat") is None

    async def test_a_dry_run_reports_without_removing(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_erasure:
            return
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k"))
        receipt = await store.erase(SCOPE, dry_run=True)
        assert receipt.dry_run
        assert not receipt.complete
        assert receipt.records >= 1
        assert await store.read(SCOPE, "k") is not None

    async def test_erasing_a_second_time_removes_nothing(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_erasure:
            return
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k"))
        await store.erase(SCOPE)
        again = await store.erase(SCOPE)
        assert again.complete
        assert again.records == 0

    async def test_a_sensitive_value_is_masked_on_the_way_in(self) -> None:
        store = self.make_store()
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k", "write to ada@example.com"))
        held = await store.read(SCOPE, "k")
        assert held is not None
        assert "ada@example.com" not in str(held.value)

    async def test_what_was_derived_is_reported_before_it_is_erased(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_erasure:
            return
        made = Derivation(artefact_id="a1", source_id="working:k", adapter="vectors")
        await store.derived(SCOPE, made)
        assert list(await store.derivations(SCOPE, source_id="working:k")) == [made]
        await store.erase(SCOPE)
        assert list(await store.derivations(SCOPE)) == []

    async def test_erasure_stops_at_the_scope_it_was_given(self) -> None:
        store = self.make_store()
        if not store.capabilities.supports_erasure:
            return
        other = SCOPE.model_copy(update={"user_id": "u2"})
        await store.write(SCOPE, _record(MemoryKind.WORKING, "k"))
        await store.write(
            other, _record(MemoryKind.WORKING, "k").model_copy(update={"scope": other})
        )
        await store.erase(SCOPE)
        assert await store.read(other, "k") is not None

    async def test_erasure_it_never_declared_is_refused(self) -> None:
        store = self.make_store()
        if store.capabilities.supports_erasure:
            return
        with pytest.raises(CapabilityError):
            await store.erase(SCOPE)


def _run(
    run_id: str,
    *,
    tenant: str = "conformance",
    session_id: str | None = None,
    state: RunState = RunState.PENDING,
    message_cursor: int = 0,
    iterations: int = 0,
    cost_micros: int = 0,
    pending_tool_calls: tuple[ToolCall, ...] = (),
) -> RunRecord:
    """The smallest run record a store can be asked to hold."""
    return RunRecord(
        run_id=run_id,
        tenant=tenant,
        agent_name="planner",
        session_id=session_id,
        state=state,
        message_cursor=message_cursor,
        iterations=iterations,
        cost_micros=cost_micros,
        pending_tool_calls=pending_tool_calls,
    )


class StateStoreConformance(ABC):
    """Behaviour every `StateStore` implementation must exhibit.

    A store that gets any of these wrong loses an update between two workers, and a lost
    update is invisible: the run continues with spend nobody recorded and a cursor that
    replays a turn. The guarantees are that a write states the version it read, that
    patches accumulate rather than overwrite, and that a tenant sees only its own.
    """

    @abstractmethod
    def make_store(self) -> StateStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), StateStore)

    async def test_a_run_that_was_never_written_is_none(self) -> None:
        key = StateKey(tenant="conformance", id="absent")
        assert await self.make_store().get_run(key) is None

    async def test_a_written_run_comes_back(self) -> None:
        store = self.make_store()
        written = await store.put_run(_run("r1", message_cursor=3))
        read = await store.get_run(written.key)
        assert read is not None
        assert read.message_cursor == 3

    async def test_a_first_write_lands_at_version_one(self) -> None:
        assert (await self.make_store().put_run(_run("r1"))).version == 1

    async def test_a_second_write_of_a_stale_version_is_refused(self) -> None:
        """Two workers holding the same run: the second must not silently win."""
        store = self.make_store()
        stored = await store.put_run(_run("r1"))
        await store.put_run(stored.model_copy(update={"iterations": 1}))
        with pytest.raises(StateConflictError) as refused:
            await store.put_run(stored.model_copy(update={"iterations": 9}))
        assert refused.value.expected_version == 1
        assert refused.value.actual_version == 2

    async def test_a_create_that_finds_something_there_is_refused(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1"))
        with pytest.raises(StateConflictError):
            await store.put_run(_run("r1"))

    async def test_only_one_of_many_concurrent_creates_wins(self) -> None:
        store = self.make_store()
        written = await asyncio.gather(
            *(store.put_run(_run("r1")) for _ in range(20)), return_exceptions=True
        )
        assert sum(1 for outcome in written if isinstance(outcome, RunRecord)) == 1

    async def test_a_patch_adds_rather_than_sets(self) -> None:
        store = self.make_store()
        stored = await store.put_run(_run("r1", iterations=2, cost_micros=10))
        patched = await store.patch_run(stored.key, StateDelta(iterations=1, cost_micros=5))
        assert patched.iterations == 3
        assert patched.cost_micros == 15

    async def test_concurrent_patches_all_land(self) -> None:
        """Two workers each adding what they spent commute; two setting a total do not."""
        store = self.make_store()
        stored = await store.put_run(_run("r1"))
        await asyncio.gather(
            *(store.patch_run(stored.key, StateDelta(iterations=1)) for _ in range(10))
        )
        read = await store.get_run(stored.key)
        assert read is not None
        assert read.iterations == 10

    async def test_a_patch_of_a_run_that_is_not_there_is_refused(self) -> None:
        key = StateKey(tenant="conformance", id="absent")
        with pytest.raises(StateNotFoundError):
            await self.make_store().patch_run(key, StateDelta(iterations=1))

    async def test_a_pending_tool_call_survives_a_round_trip(self) -> None:
        """A run abandoned mid tool call is a state to resume, not a corruption."""
        store = self.make_store()
        call = ToolCall(id="c1", name="search", arguments={"query": "kyoto"})
        stored = await store.put_run(_run("r1", pending_tool_calls=(call,)))
        read = await store.get_run(stored.key)
        assert read is not None
        assert read.mid_tool_call is True
        assert read.pending_tool_calls[0].name == "search"

    async def test_a_secret_in_a_tool_argument_is_masked_before_it_is_stored(self) -> None:
        store = self.make_store()
        secret = "sk-live-0123456789abcd"  # noqa: S105 — a fixture, not a credential; gitleaks:allow
        call = ToolCall(id="c1", name="pay", arguments={"token": secret})
        stored = await store.put_run(_run("r1", pending_tool_calls=(call,)))
        read = await store.get_run(stored.key)
        assert read is not None
        assert "sk-live" not in str(read.pending_tool_calls[0].arguments)

    async def test_deleting_a_run_removes_it(self) -> None:
        store = self.make_store()
        stored = await store.put_run(_run("r1"))
        await store.delete_run(stored.key)
        assert await store.get_run(stored.key) is None

    async def test_deleting_a_run_that_is_not_there_is_not_an_error(self) -> None:
        await self.make_store().delete_run(StateKey(tenant="conformance", id="absent"))

    async def test_a_session_round_trips(self) -> None:
        store = self.make_store()
        written = await store.put_session(
            SessionRecord(session_id="s1", tenant="conformance", runs=("r1",))
        )
        read = await store.get_session(written.key)
        assert read is not None
        assert read.runs == ("r1",)

    async def test_a_stale_session_write_is_refused(self) -> None:
        store = self.make_store()
        stored = await store.put_session(SessionRecord(session_id="s1", tenant="conformance"))
        await store.put_session(stored)
        with pytest.raises(StateConflictError):
            await store.put_session(stored)

    async def test_a_session_with_a_live_run_is_not_deleted(self) -> None:
        """A live run whose session has gone is work nothing will ever find again."""
        store = self.make_store()
        session = await store.put_session(SessionRecord(session_id="s1", tenant="conformance"))
        await store.put_run(_run("r1", session_id="s1", state=RunState.RUNNING))
        with pytest.raises(StateInUseError) as refused:
            await store.delete_session(session.key)
        assert refused.value.live_runs == ("r1",)

    async def test_a_cascade_takes_the_runs_with_it(self) -> None:
        store = self.make_store()
        session = await store.put_session(SessionRecord(session_id="s1", tenant="conformance"))
        run = await store.put_run(_run("r1", session_id="s1", state=RunState.RUNNING))
        await store.delete_session(session.key, cascade=True)
        assert await store.get_session(session.key) is None
        assert await store.get_run(run.key) is None

    async def test_a_finished_run_does_not_hold_its_session_open(self) -> None:
        store = self.make_store()
        session = await store.put_session(SessionRecord(session_id="s1", tenant="conformance"))
        await store.put_run(_run("r1", session_id="s1", state=RunState.COMPLETED))
        await store.delete_session(session.key)
        assert await store.get_session(session.key) is None

    async def test_deleting_a_session_that_is_not_there_is_not_an_error(self) -> None:
        await self.make_store().delete_session(StateKey(tenant="conformance", id="absent"))

    async def test_a_listing_returns_what_was_written(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1"))
        await store.put_run(_run("r2"))
        page = await store.list_runs(StateQuery(tenant="conformance"))
        assert {record.run_id for record in page.records} == {"r1", "r2"}
        assert page.cursor is None

    async def test_a_listing_can_be_narrowed_to_one_state(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1", state=RunState.RUNNING))
        await store.put_run(_run("r2", state=RunState.COMPLETED))
        page = await store.list_runs(StateQuery(tenant="conformance", state=RunState.RUNNING))
        assert [record.run_id for record in page.records] == ["r1"]

    async def test_a_listing_can_be_narrowed_by_age(self) -> None:
        """How abandoned work is found: everything last written before some moment."""
        store = self.make_store()
        await store.put_run(_run("r1"))
        stale = await store.list_runs(StateQuery(tenant="conformance", updated_before=float("inf")))
        fresh = await store.list_runs(
            StateQuery(tenant="conformance", updated_before=float("-inf"))
        )
        assert [record.run_id for record in stale.records] == ["r1"]
        assert fresh.records == ()

    async def test_a_cursor_walks_every_run_exactly_once(self) -> None:
        store = self.make_store()
        for index in range(5):
            await store.put_run(_run(f"r{index}"))
        seen: list[str] = []
        cursor: str | None = None
        while True:
            page = await store.list_runs(StateQuery(tenant="conformance", limit=2, cursor=cursor))
            seen.extend(record.run_id for record in page.records)
            cursor = page.cursor
            if cursor is None:
                break
        assert sorted(seen) == [f"r{index}" for index in range(5)]

    async def test_an_exhausted_listing_carries_no_cursor(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1"))
        page = await store.list_runs(StateQuery(tenant="conformance", limit=1))
        assert page.cursor is None

    async def test_one_tenant_cannot_read_another_s_run(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1", tenant="one"))
        assert await store.get_run(StateKey(tenant="two", id="r1")) is None

    async def test_one_tenant_cannot_list_another_s_runs(self) -> None:
        store = self.make_store()
        await store.put_run(_run("r1", tenant="one"))
        assert (await store.list_runs(StateQuery(tenant="two"))).records == ()


def _checkpoint(
    run_id: str = "r1",
    *,
    tenant: str = "conformance",
    iterations: int = 0,
    messages: tuple[Message, ...] = (),
    pending: tuple[PendingCall, ...] = (),
) -> Checkpoint:
    """A checkpoint with the fields a store has to round-trip."""
    return Checkpoint(
        run_id=run_id,
        tenant=tenant,
        agent_name="planner",
        iterations=iterations,
        messages=messages,
        pending=pending,
    )


class CheckpointStoreConformance(ABC):
    """Behaviour every `CheckpointStore` implementation must exhibit.

    A store that keeps an older frontier alongside the newest one resumes a run into work
    it already did; one that leaks across tenants hands a run's conversation to somebody
    else. The guarantees are one checkpoint per run, last write wins, and tenant isolation.
    """

    @abstractmethod
    def make_store(self) -> CheckpointStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), CheckpointStore)

    async def test_a_run_that_was_never_checkpointed_is_none(self) -> None:
        assert await self.make_store().latest("absent", tenant="conformance") is None

    async def test_a_written_checkpoint_comes_back(self) -> None:
        store = self.make_store()
        await store.put(_checkpoint(iterations=3))
        read = await store.latest("r1", tenant="conformance")
        assert read is not None
        assert read.iterations == 3

    async def test_only_the_newest_frontier_is_kept(self) -> None:
        """An older one is work already done, and resuming from it repeats it."""
        store = self.make_store()
        await store.put(_checkpoint(iterations=1))
        await store.put(_checkpoint(iterations=2))
        read = await store.latest("r1", tenant="conformance")
        assert read is not None
        assert read.iterations == 2

    async def test_the_conversation_survives_a_round_trip(self) -> None:
        store = self.make_store()
        messages = (Message(role="user", content=[TextPart(text="book it")]),)
        await store.put(_checkpoint(messages=messages))
        read = await store.latest("r1", tenant="conformance")
        assert read is not None
        assert read.messages == messages

    async def test_an_outstanding_call_survives_a_round_trip(self) -> None:
        """Without it a resume cannot tell which calls it still has to account for."""
        store = self.make_store()
        call = ToolCall(id="c1", name="book", arguments={"flight": "BA117"})
        pending = (PendingCall(call=call, idempotency_key="k1", dispatched=True),)
        await store.put(_checkpoint(pending=pending))
        read = await store.latest("r1", tenant="conformance")
        assert read is not None
        assert read.pending[0].idempotency_key == "k1"
        assert read.pending[0].dispatched is True

    async def test_forgetting_removes_the_frontier(self) -> None:
        store = self.make_store()
        await store.put(_checkpoint())
        await store.forget("r1", tenant="conformance")
        assert await store.latest("r1", tenant="conformance") is None

    async def test_forgetting_a_run_that_was_never_checkpointed_is_not_an_error(self) -> None:
        await self.make_store().forget("absent", tenant="conformance")

    async def test_two_runs_do_not_share_a_frontier(self) -> None:
        store = self.make_store()
        await store.put(_checkpoint("r1", iterations=1))
        await store.put(_checkpoint("r2", iterations=2))
        read = await store.latest("r1", tenant="conformance")
        assert read is not None
        assert read.iterations == 1

    async def test_one_tenant_cannot_read_another_s_checkpoint(self) -> None:
        store = self.make_store()
        await store.put(_checkpoint("r1", tenant="one"))
        assert await store.latest("r1", tenant="two") is None

    async def test_forgetting_one_tenant_s_run_leaves_another_s(self) -> None:
        store = self.make_store()
        await store.put(_checkpoint("r1", tenant="one"))
        await store.forget("r1", tenant="two")
        assert await store.latest("r1", tenant="one") is not None


class LeaseStoreConformance(ABC):
    """Behaviour every `LeaseStore` implementation must exhibit.

    The guarantees are that one holder at a time may advance a run, that a lease nobody
    released lapses on the store's clock rather than stranding the run, that every
    acquisition raises the fence so a superseded holder is refused on the number, and that
    a lease never spans a tenant.
    """

    @abstractmethod
    def make_store(self) -> LeaseStore:
        """Return a fresh store with no lease in it."""

    @abstractmethod
    async def advance(self, store: LeaseStore, seconds: float) -> None:
        """Move the store's clock on, so a lease can lapse without a test sleeping."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), LeaseStore)

    async def test_a_run_nobody_has_taken_has_no_lease(self) -> None:
        assert await self.make_store().held("r1", tenant="conformance") is None

    async def test_the_first_worker_takes_the_run(self) -> None:
        store = self.make_store()
        lease = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        assert lease.holder == "w1"
        assert lease.fence >= 1

    async def test_the_second_worker_is_refused_and_told_who_holds_it(self) -> None:
        store = self.make_store()
        await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        with pytest.raises(RunLeaseError) as refused:
            await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        assert refused.value.holder == "w1"

    async def test_a_lapsed_lease_is_takeable(self) -> None:
        store = self.make_store()
        await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await self.advance(store, 61.0)
        taken = await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        assert taken.holder == "w2"

    async def test_taking_a_lapsed_lease_raises_the_fence(self) -> None:
        store = self.make_store()
        first = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await self.advance(store, 61.0)
        second = await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        assert first.superseded_by(second)

    async def test_a_superseded_holder_cannot_renew(self) -> None:
        store = self.make_store()
        first = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await self.advance(store, 61.0)
        await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        with pytest.raises(RunLeaseError):
            await store.renew(first, ttl_seconds=60.0)

    async def test_renewing_keeps_the_fence_and_moves_the_expiry(self) -> None:
        store = self.make_store()
        held = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await self.advance(store, 30.0)
        renewed = await store.renew(held, ttl_seconds=60.0)
        assert renewed.fence == held.fence
        assert renewed.expires_at > held.expires_at

    async def test_renewing_a_lease_nobody_holds_is_refused(self) -> None:
        store = self.make_store()
        lease = RunLease(run_id="r1", tenant="conformance", holder="w1", expires_at=60.0)
        with pytest.raises(RunLeaseError):
            await store.renew(lease, ttl_seconds=60.0)

    async def test_releasing_frees_the_run_immediately(self) -> None:
        store = self.make_store()
        held = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await store.release(held)
        taken = await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        assert taken.holder == "w2"

    async def test_releasing_a_lease_that_has_moved_on_leaves_the_new_one_alone(self) -> None:
        store = self.make_store()
        first = await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        await self.advance(store, 61.0)
        await store.acquire("r1", tenant="conformance", holder="w2", ttl_seconds=60.0)
        await store.release(first)
        current = await store.held("r1", tenant="conformance")
        assert current is not None
        assert current.holder == "w2"

    async def test_a_lease_never_spans_a_tenant(self) -> None:
        store = self.make_store()
        await store.acquire("r1", tenant="one", holder="w1", ttl_seconds=60.0)
        other = await store.acquire("r1", tenant="two", holder="w2", ttl_seconds=60.0)
        assert other.holder == "w2"

    async def test_two_runs_do_not_block_each_other(self) -> None:
        store = self.make_store()
        await store.acquire("r1", tenant="conformance", holder="w1", ttl_seconds=60.0)
        other = await store.acquire("r2", tenant="conformance", holder="w2", ttl_seconds=60.0)
        assert other.holder == "w2"


def _work(
    item_id: str = "i1", *, tenant: str = "conformance", dedupe_key: str | None = None
) -> WorkItem:
    """An item with the fields a queue has to round-trip."""
    return WorkItem(id=item_id, tenant=tenant, dedupe_key=dedupe_key, payload={"run": item_id})


class WorkQueueConformance(ABC):
    """Behaviour every `WorkQueue` implementation must exhibit.

    The guarantees are that a claim is exclusive, that a lapsed claim comes back with its
    attempt counted, that an item which has run out of attempts is in the dead letter
    rather than in a loop, and that no tenant can be starved or read by another.
    """

    @abstractmethod
    def make_queue(self) -> WorkQueue:
        """Return a fresh, empty queue whose policy allows at least three attempts."""

    @abstractmethod
    async def advance(self, queue: WorkQueue, seconds: float) -> None:
        """Move the store's clock on, so a lease can lapse without a test sleeping."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_queue(), WorkQueue)

    async def test_an_empty_queue_hands_out_nothing(self) -> None:
        assert await self.make_queue().claim(worker="w1") is None

    async def test_an_enqueued_item_is_claimable(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        claimed = await queue.claim(worker="w1")
        assert claimed is not None
        assert claimed.id == "i1"
        assert claimed.worker == "w1"

    async def test_a_claimed_item_is_not_handed_to_a_second_worker(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1")
        assert await queue.claim(worker="w2") is None

    async def test_a_completed_item_never_comes_back(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1")
        await queue.complete("i1", tenant="conformance", worker="w1")
        assert await queue.claim(worker="w2") is None

    async def test_a_lapsed_claim_is_requeued_with_its_attempt_counted(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1", lease_seconds=1.0)
        await self.advance(queue, 2.0)
        reaped = await queue.reap()
        assert [item.id for item in reaped] == ["i1"]
        assert reaped[0].attempts == 1

    async def test_a_live_worker_keeps_its_claim(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1", lease_seconds=10.0)
        await self.advance(queue, 5.0)
        await queue.heartbeat("i1", tenant="conformance", worker="w1")
        await self.advance(queue, 5.0)
        assert await queue.reap() == ()

    async def test_a_worker_that_lost_its_claim_is_refused(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1", lease_seconds=1.0)
        await self.advance(queue, 2.0)
        await queue.reap()
        with pytest.raises(LeaseLostError):
            await queue.complete("i1", tenant="conformance", worker="w1")

    async def test_an_item_the_queue_does_not_hold_is_a_refusal(self) -> None:
        with pytest.raises(WorkItemNotFoundError):
            await self.make_queue().complete("absent", tenant="conformance", worker="w1")

    async def test_a_failure_comes_back_for_another_attempt(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1")
        failed = await queue.fail("i1", tenant="conformance", worker="w1", error="boom")
        assert failed.attempts == 1
        assert failed.failures == ("boom",)

    async def test_an_item_that_cannot_succeed_skips_its_remaining_attempts(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1")
        failed = await queue.fail(
            "i1", tenant="conformance", worker="w1", error="malformed", retryable=False
        )
        assert failed.state is WorkState.DEAD_LETTERED
        assert [item.id for item in await queue.dead_letters(tenant="conformance")] == ["i1"]

    async def test_a_duplicate_job_is_collapsed_into_the_first(self) -> None:
        queue = self.make_queue()
        first = await queue.enqueue(_work("i1", dedupe_key="nightly"))
        second = await queue.enqueue(_work("i2", dedupe_key="nightly"))
        assert second.id == first.id

    async def test_a_restarted_worker_gives_back_what_it_held(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.claim(worker="w1")
        adopted = await queue.adopt(worker="w1")
        assert [item.id for item in adopted] == ["i1"]
        claimed = await queue.claim(worker="w2")
        assert claimed is not None
        assert claimed.id == "i1"

    async def test_one_tenant_cannot_starve_another(self) -> None:
        queue = self.make_queue()
        for index in range(4):
            await queue.enqueue(_work(f"loud{index}", tenant="loud"))
        await queue.enqueue(_work("quiet1", tenant="quiet"))
        claimed = [await queue.claim(worker=f"w{index}") for index in range(4)]
        assert any(item is not None and item.tenant == "quiet" for item in claimed)

    async def test_one_tenant_cannot_act_on_another_s_item(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1", tenant="one"))
        await queue.claim(worker="w1")
        with pytest.raises(WorkItemNotFoundError):
            await queue.complete("i1", tenant="two", worker="w1")

    async def test_the_depth_counts_what_is_waiting(self) -> None:
        queue = self.make_queue()
        await queue.enqueue(_work("i1"))
        await queue.enqueue(_work("i2"))
        await queue.claim(worker="w1")
        stats = await queue.stats()
        assert (stats.depth, stats.claimed) == (1, 1)


class TenantPropagationConformance(ABC):
    """Behaviour every transport that carries the tenant across a hop must exhibit.

    Subclassed by an integration — a broker, an MCP server, a peer client — to prove the
    context survives its particular idea of a message. The guarantees are that the whole
    context arrives, that header case is not load-bearing, that a transport with a header
    ceiling still delivers an intact tenant, and that consecutive messages on one
    connection do not bleed into each other.
    """

    @abstractmethod
    def round_trip(self, headers: Mapping[str, str]) -> Mapping[str, str]:
        """Return `headers` as the far side of this transport would see them."""

    def test_the_tenant_survives_the_hop(self) -> None:
        sent = carried(TenantContext(tenant="conformance"))
        assert restored(self.round_trip(sent)).tenant == "conformance"

    def test_the_rest_of_the_context_survives_with_it(self) -> None:
        """A tenant that arrives without its principal cannot be audited on the far side."""
        sent = carried(TenantContext(tenant="conformance", user="ada", locale="en-GB"))
        arrived = restored(self.round_trip(sent))
        assert (arrived.user, arrived.locale) == ("ada", "en-GB")

    def test_header_case_is_not_load_bearing(self) -> None:
        sent = {HEADER.upper(): carried(TenantContext(tenant="conformance"))[HEADER]}
        assert restored(self.round_trip(sent)).tenant == "conformance"

    def test_a_context_too_large_for_the_transport_still_delivers_the_tenant(self) -> None:
        """Shedding an optional field is acceptable; losing the tenant is not."""
        wordy = TenantContext(tenant="conformance", user="ada", correlation_id="c" * 4096)
        arrived = restored(self.round_trip(carried(wordy)))
        assert (arrived.tenant, arrived.user, arrived.partial) == ("conformance", "ada", True)

    def test_consecutive_messages_do_not_bleed(self) -> None:
        """One connection carrying two tenants is two scopes, never one over the pair."""
        seen = []
        for tenant in ("one", "two"):
            with arriving(self.round_trip(carried(TenantContext(tenant=tenant)))) as here:
                seen.append(here.tenant)
        assert seen == ["one", "two"]

    def test_the_binding_does_not_outlive_the_message(self) -> None:
        with arriving(self.round_trip(carried(TenantContext(tenant="conformance")))):
            pass
        assert tenant_here() is None


CONFORMANCE_CORPUS = (
    Indexed(
        "refunds",
        "Refunds are paid within fourteen days of the request.",
        vector=(1.0, 0.0, 0.0),
        document_id="handbook",
        tenant="conformance",
        metadata={"section": "refunds"},
    ),
    Indexed(
        "berths",
        "Berths are allocated by seniority at the start of a voyage.",
        vector=(0.0, 1.0, 0.0),
        document_id="handbook",
        tenant="conformance",
        metadata={"section": "berths"},
    ),
    Indexed(
        "somebody-elses",
        "Refunds are paid within fourteen days of the request.",
        vector=(1.0, 0.0, 0.0),
        document_id="handbook",
        tenant="other",
        metadata={"section": "refunds"},
    ),
)
"""The corpus every `SearchIndex` suite is run against. Two tenants, on purpose."""


class SearchIndexConformance(ABC):
    """Behaviour every `SearchIndex` implementation must exhibit.

    The guarantee that matters is that the tenant and the predicates are applied inside
    the store's own query, not to its answer: a store that fetches `k` neighbours and
    filters them afterwards returns fewer than `k` of the caller's chunks, and none at all
    where the nearest `k` all belong to somebody else. The corpus holds one passage of a
    second tenant identical to the first tenant's, so a store that leaks fails here rather
    than in production.
    """

    @abstractmethod
    async def make_index(self, passages: Sequence[Indexed]) -> SearchIndex:
        """Return a store holding exactly `passages` and nothing else."""

    @abstractmethod
    def branch(self) -> Branch:
        """Which branch the store under test is being verified for."""

    async def test_satisfies_the_protocol(self) -> None:
        verify_conformance(await self.make_index(()), SearchIndex)

    async def test_it_supports_the_branch_it_is_verified_for(self) -> None:
        assert (await self.make_index(())).supports(self.branch())

    async def test_an_empty_store_returns_nothing(self) -> None:
        index = await self.make_index(())
        assert list(await index.search(self._asking("refunds"))) == []

    async def test_it_finds_the_passage_the_query_is_about(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        found = await index.search(self._asking("refunds"))
        assert [hit.chunk_id for hit in found][:1] == ["refunds"]

    async def test_another_tenants_passage_is_never_returned(self) -> None:
        """Identical text under a second tenant: only a predicate inside the query stops it."""
        index = await self.make_index(CONFORMANCE_CORPUS)
        found = await index.search(self._asking("refunds", k=10))
        assert "somebody-elses" not in [hit.chunk_id for hit in found]

    async def test_k_caps_what_comes_back(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        assert len(await index.search(self._asking("refunds", k=1))) <= 1

    async def test_hits_come_back_best_first(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        scores = [hit.score for hit in await index.search(self._asking("refunds", k=10))]
        assert scores == sorted(scores, reverse=True)

    async def test_a_hit_carries_what_a_citation_needs(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        found = await index.search(self._asking("refunds"))
        assert found
        assert (found[0].chunk_id, found[0].document_id) == ("refunds", "handbook")
        assert found[0].text

    async def test_a_predicate_excludes_what_does_not_satisfy_it(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        found = await index.search(
            self._asking("refunds", k=10, predicates={"section": ("berths",)})
        )
        assert "refunds" not in [hit.chunk_id for hit in found]

    async def test_a_predicate_nothing_satisfies_returns_nothing(self) -> None:
        index = await self.make_index(CONFORMANCE_CORPUS)
        found = await index.search(
            self._asking("refunds", k=10, predicates={"section": ("galley",)})
        )
        assert list(found) == []

    def _asking(
        self,
        chunk_id: str,
        *,
        k: int = 5,
        predicates: Mapping[str, tuple[str, ...]] | None = None,
    ) -> IndexQuery:
        """A query for the corpus passage `chunk_id`, in the branch under test."""
        wanted = next(passage for passage in CONFORMANCE_CORPUS if passage.chunk_id == chunk_id)
        return IndexQuery(
            tenant="conformance",
            collection="handbook",
            branch=self.branch(),
            k=k,
            text=wanted.text,
            vector=wanted.vector,
            predicates=dict(predicates or {}),
        )
