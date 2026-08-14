"""The pgvector index, whose whole job is to put the tenant in the WHERE.

These tests read the statement the adapter sends. That is unusual and deliberate: the
difference between a store that is safe under multi-tenancy and one that leaks is not
visible in its results against a single-tenant fixture, only in whether the predicate is
in the query or applied to the answer. Nothing here opens a socket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.pgvector import (
    DEFAULT_CHUNK_TABLES,
    ChunkTables,
    PgvectorIndex,
    PgvectorSettings,
)
from tesserix_adk.core.errors import ConfigurationError, ProviderUnavailableError
from tesserix_adk.rag import Branch, IndexQuery, SearchIndex
from tesserix_adk.testing import FakeIndex, Indexed
from tesserix_adk.testing.conformance import SearchIndexConformance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SETTINGS = PgvectorSettings(dsn=SecretStr("postgresql://localhost/adk"))


class FakeSql:
    """Answers with what the test says the database returned, and records what was asked."""

    def __init__(self, *replies: Any, fails: Sequence[Exception] = ()) -> None:
        self.replies = list(replies)
        self.fails = list(fails)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Return the next scripted reply, or raise the next scripted failure."""
        self.calls.append((statement, args))
        if self.fails:
            raise self.fails.pop(0)
        return self.replies.pop(0) if self.replies else []

    @property
    def sent(self) -> str:
        """The last statement."""
        return self.calls[-1][0]

    @property
    def bound(self) -> tuple[Any, ...]:
        """What was bound to it."""
        return self.calls[-1][1]


def asking(
    branch: Branch = Branch.SEMANTIC,
    *,
    k: int = 5,
    predicates: Mapping[str, tuple[str, ...]] | None = None,
) -> IndexQuery:
    """One query in either branch, with the tenant a retriever would have set."""
    return IndexQuery(
        tenant="acme",
        collection="handbook",
        branch=branch,
        k=k,
        text="refunds",
        vector=(0.5, 0.25),
        predicates=dict(predicates or {}),
    )


def row(chunk_id: str = "c1", metadata: Any = None) -> list[Any]:
    """One chunk as the columns come back."""
    return [chunk_id, "handbook", "Refunds are paid in fourteen days.", 0.75, metadata]


class TestTheTenantIsInTheStatement:
    async def test_the_semantic_query_binds_the_tenant_and_collection(self) -> None:
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(asking())

        assert "WHERE tenant = $2 AND collection = $3" in sql.sent
        assert sql.bound[1:3] == ("acme", "handbook")

    async def test_the_keyword_query_binds_them_too(self) -> None:
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(asking(Branch.KEYWORD))

        assert "WHERE tenant = $3 AND collection = $4" in sql.sent
        assert sql.bound[2:4] == ("acme", "handbook")

    async def test_the_limit_is_in_the_statement_not_applied_afterwards(self) -> None:
        """Trimming k rows in Python would already have read another tenant's chunks."""
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(asking(k=3))

        assert "LIMIT $4" in sql.sent
        assert sql.bound[-1] == 3

    async def test_the_nearest_neighbours_are_ordered_by_the_index_operator(self) -> None:
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(asking())

        assert "ORDER BY embedding <=> $1::vector" in sql.sent


class TestPredicatesArePushedDownAndBound:
    async def test_a_predicate_becomes_a_bound_clause(self) -> None:
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(
            asking(predicates={"section": ("refunds", "berths")})
        )

        assert "AND metadata ->> $4 = ANY($5)" in sql.sent
        assert sql.bound[3:5] == ("section", ["refunds", "berths"])

    async def test_the_key_is_bound_as_well_as_the_values(self) -> None:
        """A metadata key is caller input; interpolating it would put SQL in the WHERE."""
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(
            asking(predicates={"section' OR true --": ("x",)})
        )

        assert "OR true" not in sql.sent
        assert "section' OR true --" in sql.bound

    async def test_the_limit_follows_however_many_predicates_there_were(self) -> None:
        sql = FakeSql([row()])
        await PgvectorIndex(sql, settings=SETTINGS).search(
            asking(predicates={"section": ("refunds",), "year": ("2026",)})
        )

        assert "LIMIT $8" in sql.sent
        assert sql.bound[-1] == 5


class TestTheKeywordBranch:
    async def test_it_parses_with_the_configured_text_search_config(self) -> None:
        sql = FakeSql([row()])
        settings = PgvectorSettings(
            dsn=SecretStr("postgresql://localhost/adk"), text_search_config="simple"
        )
        await PgvectorIndex(sql, settings=settings).search(asking(Branch.KEYWORD))

        assert "websearch_to_tsquery($1::regconfig, $2)" in sql.sent
        assert sql.bound[0:2] == ("simple", "refunds")


class TestWhatComesBack:
    async def test_a_row_becomes_a_hit(self) -> None:
        sql = FakeSql([row(metadata={"section": "refunds"})])
        found = await PgvectorIndex(sql, settings=SETTINGS).search(asking())

        assert [(hit.chunk_id, hit.document_id, hit.score) for hit in found] == [
            ("c1", "handbook", 0.75)
        ]
        assert found[0].metadata == {"section": "refunds"}

    async def test_metadata_that_arrives_as_text_is_read_as_json(self) -> None:
        """Drivers differ over whether jsonb comes back decoded; neither is an error."""
        sql = FakeSql([row(metadata='{"section": "refunds"}')])
        found = await PgvectorIndex(sql, settings=SETTINGS).search(asking())

        assert found[0].metadata == {"section": "refunds"}

    async def test_a_row_with_no_metadata_is_a_hit_with_none(self) -> None:
        sql = FakeSql([row(metadata=None)])
        found = await PgvectorIndex(sql, settings=SETTINGS).search(asking())

        assert found[0].metadata == {}

    async def test_no_rows_is_no_hits_rather_than_an_error(self) -> None:
        assert list(await PgvectorIndex(FakeSql(), settings=SETTINGS).search(asking())) == []


class TestWhenTheDatabaseWillNotAnswer:
    async def test_a_driver_failure_reads_as_the_store_being_unavailable(self) -> None:
        """A retriever needs to tell a branch that is down from a branch that found nothing."""
        sql = FakeSql(fails=[RuntimeError("connection reset")])

        with pytest.raises(ProviderUnavailableError, match="pgvector"):
            await PgvectorIndex(sql, settings=SETTINGS).search(asking())

    async def test_a_branch_it_does_not_back_is_refused_rather_than_attempted(self) -> None:
        index = PgvectorIndex(FakeSql(), settings=SETTINGS, branches=(Branch.SEMANTIC,))

        assert not index.supports(Branch.KEYWORD)
        with pytest.raises(ConfigurationError, match="keyword"):
            await index.search(asking(Branch.KEYWORD))


class TestRefusingADatabaseOfTheWrongShape:
    async def test_it_opens_against_the_schema_it_was_written_for(self) -> None:
        sql = FakeSql([[1]], [["5000ms"]])
        index = await PgvectorIndex.open(sql, settings=SETTINGS)

        assert index.name == "pgvector"

    async def test_a_schema_that_has_moved_is_refused(self) -> None:
        sql = FakeSql([[2]], [["5000ms"]])

        with pytest.raises(ConfigurationError, match="version 2"):
            await PgvectorIndex.open(sql, settings=SETTINGS)

    async def test_a_schema_nobody_has_applied_is_refused(self) -> None:
        sql = FakeSql([], [["5000ms"]])

        with pytest.raises(ConfigurationError, match="version 0"):
            await PgvectorIndex.open(sql, settings=SETTINGS)

    async def test_a_connection_with_no_statement_timeout_is_refused(self) -> None:
        """One retrieval that never returns holds a pooled connection until a restart."""
        sql = FakeSql([[1]], [["0"]])

        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await PgvectorIndex.open(sql, settings=SETTINGS)

    def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain table identifier"):
            ChunkTables(chunks="adk_chunks; DROP TABLE adk_chunks")

    async def test_the_table_it_reads_is_the_one_it_was_given(self) -> None:
        sql = FakeSql([row()])
        tables = ChunkTables(chunks="tenant_chunks")
        await PgvectorIndex(sql, settings=SETTINGS, tables=tables).search(asking())

        assert "FROM tenant_chunks" in sql.sent
        assert DEFAULT_CHUNK_TABLES.chunks == "adk_chunks"

    def test_it_calls_itself_what_it_was_named(self) -> None:
        assert PgvectorIndex(FakeSql(), settings=SETTINGS, name="chunks").name == "chunks"


class TestTheAdapterIsASearchIndex(SearchIndexConformance):
    """The protocol suite, run against the adapter with the database scripted."""

    def branch(self) -> Branch:
        """The semantic branch is the one pgvector exists for."""
        return Branch.SEMANTIC

    async def make_index(self, passages: Sequence[Indexed]) -> SearchIndex:
        """The adapter over a session that answers as the statement would."""
        return PgvectorIndex(_Scripted(passages), settings=SETTINGS)


class _Scripted:
    """A session that answers a pgvector statement the way the database would.

    It reads the bound tenant and predicates rather than the SQL, which is the point: an
    adapter that left the tenant out of its arguments gets everybody's chunks back here,
    and the conformance suite fails.
    """

    def __init__(self, passages: Sequence[Indexed]) -> None:
        self._passages = list(passages)

    async def fetch(self, _statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """The rows the WHERE and the LIMIT would have selected, read off the arguments."""
        tenant, collection = args[1], args[2]
        wanted = [float(value) for value in str(args[0]).strip("[]").split(",")]
        predicates = dict(zip(args[3:-1:2], args[4:-1:2], strict=True))
        rows: list[list[Any]] = [
            [
                passage.chunk_id,
                passage.document_id,
                passage.text,
                _near(passage.vector, wanted),
                passage.metadata,
            ]
            for passage in self._passages
            if passage.tenant == tenant
            and all(passage.metadata.get(key) in values for key, values in predicates.items())
        ]
        rows.sort(key=lambda held: (-float(held[3]), str(held[0])))
        assert collection == "handbook"
        return rows[: args[-1]]


def _near(left: Sequence[float], right: Sequence[float]) -> float:
    """What the distance operator would score, near enough for a scripted row."""
    return sum(a * b for a, b in zip(left, right, strict=True))


class TestTheFakeIndexIsASearchIndexToo(SearchIndexConformance):
    """The suite over the in-process fake, so a consumer copying it starts from green."""

    def branch(self) -> Branch:
        """The keyword branch, so the fake's other half is covered as well."""
        return Branch.KEYWORD

    async def make_index(self, passages: Sequence[Indexed]) -> SearchIndex:
        """The fake holding exactly those passages."""
        return FakeIndex(*passages)
