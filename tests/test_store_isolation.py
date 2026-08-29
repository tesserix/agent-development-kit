"""Isolation the calling code cannot forget, because it never states the tenant."""

from __future__ import annotations

import pytest

from tesserix_adk.adapters import (
    ADAPTER_GUARANTEES,
    ErasureSweep,
    IsolationGuarantee,
    Isolator,
    Partition,
    PartitionMechanism,
    partitioned,
)
from tesserix_adk.core import (
    MissingTenantContextError,
    TenantIsolationError,
    tenant_scope,
)

pytestmark = pytest.mark.anyio

ACME = "acme"
RIVAL = "rival"


class TestDerivingThePartition:
    def test_the_partition_comes_from_the_bound_context(self) -> None:
        with tenant_scope(ACME):
            assert Partition.current().tenant == ACME

    def test_no_context_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(MissingTenantContextError, match="partition"):
            Partition.current()

    def test_a_nested_scope_partitions_where_it_is_read(self) -> None:
        with tenant_scope(ACME):
            with tenant_scope(RIVAL, crossing="support ticket 41"):
                assert Partition.current().tenant == RIVAL
            assert Partition.current().tenant == ACME

    def test_a_tenant_cannot_be_handed_in_positionally_on_the_way_past(self) -> None:
        """The documented way in is `current()`; a bare string is not a partition."""
        with tenant_scope(ACME), pytest.raises(TypeError):
            Partition(RIVAL)  # type: ignore[call-arg]


class TestKeys:
    def test_a_key_carries_the_tenant_ahead_of_everything_else(self) -> None:
        with tenant_scope(ACME):
            assert Partition.current().key("adk:cache", "sha256") == "adk:cache:acme:sha256"

    def test_two_tenants_never_share_a_key(self) -> None:
        with tenant_scope(ACME):
            mine = Partition.current().key("adk:cache", "same-digest")
        with tenant_scope(RIVAL):
            theirs = Partition.current().key("adk:cache", "same-digest")
        assert mine != theirs

    def test_a_glob_in_a_tenant_name_cannot_reach_another_tenants_keys(self) -> None:
        with tenant_scope("acme*"), pytest.raises(ValueError, match="pattern"):
            Partition.current().pattern("adk:cache")

    def test_a_purge_pattern_is_bounded_to_the_tenant(self) -> None:
        with tenant_scope(ACME):
            assert Partition.current().pattern("adk:cache") == "adk:cache:acme:*"


class TestPredicates:
    def test_a_read_predicate_binds_the_tenant_rather_than_inlining_it(self) -> None:
        with tenant_scope(ACME):
            clause, value = Partition.current().predicate(column="tenant", position=1)
        assert clause == "tenant = $1"
        assert value == ACME

    def test_the_predicate_position_follows_the_arguments_already_bound(self) -> None:
        with tenant_scope(ACME):
            clause, _ = Partition.current().predicate(column="tenant", position=3)
        assert clause == "tenant = $3"


class TestIntegrityOnRead:
    def test_a_record_from_this_tenant_passes(self) -> None:
        with tenant_scope(ACME):
            Partition.current().verify(ACME, where="pgvector.search")

    def test_a_record_from_another_tenant_is_withheld_and_signalled(self) -> None:
        with tenant_scope(ACME), pytest.raises(TenantIsolationError) as leaked:
            Partition.current().verify(RIVAL, where="pgvector.search")
        assert leaked.value.found == RIVAL
        assert leaked.value.where == "pgvector.search"
        assert leaked.value.tenant == ACME

    def test_a_record_with_no_tenant_at_all_is_withheld(self) -> None:
        """An unlabelled row predates the guarantee; it is not assumed to be ours."""
        with tenant_scope(ACME), pytest.raises(TenantIsolationError):
            Partition.current().verify("", where="postgres.read")

    def test_a_page_of_rows_is_checked_before_any_of_it_is_returned(self) -> None:
        rows = [{"tenant": ACME, "id": "1"}, {"tenant": RIVAL, "id": "2"}]
        with tenant_scope(ACME), pytest.raises(TenantIsolationError, match="rival"):
            Partition.current().only(rows, tenant_of=lambda row: str(row["tenant"]), where="page")

    def test_a_clean_page_comes_back_unchanged(self) -> None:
        rows = [{"tenant": ACME, "id": "1"}, {"tenant": ACME, "id": "2"}]
        with tenant_scope(ACME):
            kept = Partition.current().only(
                rows, tenant_of=lambda row: str(row["tenant"]), where="page"
            )
        assert kept == rows


class TestBatches:
    def test_a_batch_spanning_tenants_is_partitioned_rather_than_run(self) -> None:
        items = [(ACME, "a"), (RIVAL, "b"), (ACME, "c")]
        assert partitioned(items, tenant_of=lambda item: item[0]) == {
            ACME: [(ACME, "a"), (ACME, "c")],
            RIVAL: [(RIVAL, "b")],
        }

    def test_a_write_batch_may_not_span_the_bound_tenant(self) -> None:
        items = [(ACME, "a"), (RIVAL, "b")]
        with tenant_scope(ACME), pytest.raises(TenantIsolationError, match="batch"):
            Partition.current().only(items, tenant_of=lambda item: item[0], where="batch")

    def test_an_empty_batch_is_not_an_isolation_problem(self) -> None:
        assert partitioned([], tenant_of=lambda item: str(item)) == {}


class TestTheGuarantee:
    def test_an_adapter_states_its_mechanism_and_its_limits(self) -> None:
        guarantee = IsolationGuarantee(
            adapter="RedisCacheStore",
            mechanism=PartitionMechanism.KEY_PREFIX,
            partitions="every key is <namespace>:<tenant>:...",
            limits=("a client with the raw connection can read any key",),
        )
        statement = guarantee.statement()
        assert "RedisCacheStore" in statement
        assert "key prefix" in statement
        assert "a client with the raw connection" in statement

    def test_a_guarantee_with_no_stated_limit_is_refused(self) -> None:
        """Every mechanism here has a limit; a guarantee claiming none is unread."""
        with pytest.raises(ValueError, match="limits"):
            IsolationGuarantee(
                adapter="PgvectorIndex",
                mechanism=PartitionMechanism.PRE_FILTERED_ANN,
                partitions="tenant in the WHERE",
                limits=(),
            )

    def test_a_post_filtered_search_is_not_a_mechanism_on_offer(self) -> None:
        assert "POST_FILTERED" not in {member.name for member in PartitionMechanism}

    def test_every_tenant_partitioned_store_states_one(self) -> None:
        """A store shipping without a stated guarantee is a guarantee nobody can check."""
        partitioned_stores = {
            "GraphMemoryStore",
            "PgvectorIndex",
            "PgvectorMemoryStore",
            "PostgresMemoryStore",
            "RedisCacheStore",
            "RedisMemoryStore",
            "RoutedMemoryStore",
        }
        assert partitioned_stores <= set(ADAPTER_GUARANTEES)

    def test_a_vector_store_promises_the_filter_runs_before_the_ranking(self) -> None:
        for adapter in ("PgvectorIndex", "PgvectorMemoryStore"):
            assert ADAPTER_GUARANTEES[adapter].mechanism is PartitionMechanism.PRE_FILTERED_ANN

    def test_the_cache_guarantee_covers_the_error_cache_too(self) -> None:
        assert "cached error" in ADAPTER_GUARANTEES["RedisCacheStore"].partitions


class TestErasure:
    def test_an_erasure_that_left_something_behind_is_not_complete(self) -> None:
        sweep = ErasureSweep(tenant=ACME, remaining={"rows": 0, "embeddings": 3, "keys": 0})
        assert sweep.complete is False
        assert sweep.outstanding == ("embeddings",)

    def test_an_erasure_reaching_every_derived_artefact_is_complete(self) -> None:
        sweep = ErasureSweep(tenant=ACME, remaining={"rows": 0, "embeddings": 0, "keys": 0})
        assert sweep.complete is True
        assert sweep.outstanding == ()

    def test_an_erasure_that_verified_nothing_is_not_evidence(self) -> None:
        with pytest.raises(ValueError, match="verif"):
            ErasureSweep(tenant=ACME, remaining={})


class TestTheIsolator:
    async def test_a_read_is_scoped_and_checked_without_the_caller_saying_so(self) -> None:
        rows = {ACME: [{"tenant": ACME, "body": "ours"}], RIVAL: [{"tenant": RIVAL, "body": "no"}]}
        isolator: Isolator[dict[str, str]] = Isolator(
            tenant_of=lambda row: row["tenant"], where="fake.read"
        )

        async def read() -> list[dict[str, str]]:
            return rows[Partition.current().tenant]

        with tenant_scope(ACME):
            assert await isolator.read(read) == [{"tenant": ACME, "body": "ours"}]

    async def test_a_store_returning_a_neighbours_row_is_caught_at_the_adapter(self) -> None:
        isolator: Isolator[dict[str, str]] = Isolator(
            tenant_of=lambda row: row["tenant"], where="fake.read"
        )

        async def leaky() -> list[dict[str, str]]:
            return [{"tenant": RIVAL, "body": "somebody else's best match"}]

        with tenant_scope(ACME), pytest.raises(TenantIsolationError, match=r"fake\.read"):
            await isolator.read(leaky)

    async def test_a_read_outside_a_scope_never_reaches_the_store(self) -> None:
        reached = False

        async def read() -> list[dict[str, str]]:
            nonlocal reached
            reached = True
            return []

        isolator: Isolator[dict[str, str]] = Isolator(
            tenant_of=lambda row: row["tenant"], where="fake.read"
        )
        with pytest.raises(MissingTenantContextError):
            await isolator.read(read)
        assert reached is False
