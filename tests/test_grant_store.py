"""Grants in PostgreSQL, read by the runtime and written by nothing inside a run.

A grant is a record of what a person decided, so the store never rewrites one: re-granting
mints a new id, and the row an old decision was recorded against stays readable as what it
permitted. Every test here is scripted against a fake session — nothing opens a socket.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.grants import (
    DEFAULT_GRANT_TABLES,
    EXPECTED_GRANT_SCHEMA,
    GRANT_SCHEMA_VERSION,
    GrantTables,
    PostgresGrantSettings,
    PostgresGrantStore,
)
from tesserix_adk.core.autonomy import (
    AutonomyGrant,
    AutonomyLevel,
    Ceiling,
    GrantIssuer,
    GrantReader,
    Revocation,
)
from tesserix_adk.core.errors import ConfigurationError, StatePersistenceError
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence

NOW = 1_000.0
DAY = 86_400.0
SETTINGS = PostgresGrantSettings(dsn=SecretStr("postgresql://localhost/adk"))


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


class DriverError(Exception):
    """A driver error carrying the sqlstate a real one would."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


def held(**fields: object) -> AutonomyGrant:
    """One grant, filled in enough to be issued."""
    defaults: dict[str, object] = {
        "id": "g1",
        "tenant": "acme",
        "action_class": "booking.change",
        "level": AutonomyLevel.ACT_WITHIN_LIMITS,
        "granted_by": "ops@acme.example",
        "issued_at": NOW,
        "expires_at": NOW + DAY,
        "ceiling": Ceiling(amount=Decimal("5000"), currency="INR", window_seconds=DAY),
    }
    return AutonomyGrant.model_validate(defaults | fields)


def row(grant: AutonomyGrant | None = None) -> list[Any]:
    """A stored grant as the store reads it back."""
    return [(grant or held()).model_dump_json()]


def store(sql: FakeSql, *, tables: GrantTables = DEFAULT_GRANT_TABLES) -> PostgresGrantStore:
    """A store over a scripted session."""
    return PostgresGrantStore(sql, clock=FakeClock(start=NOW), settings=SETTINGS, tables=tables)


class TestRefusingADatabaseThatIsNotTheRightShape:
    """Startup is where a moved column is caught, not the first grant it silently drops."""

    async def test_a_schema_the_adapter_was_not_written_for_is_refused(self) -> None:
        sql = FakeSql([[GRANT_SCHEMA_VERSION + 1]])
        with pytest.raises(ConfigurationError, match="schema is version"):
            await store(sql).verify()

    async def test_a_database_with_no_schema_row_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="schema is version 0"):
            await store(FakeSql([])).verify()

    async def test_a_connection_that_lets_a_statement_run_forever_is_refused(self) -> None:
        sql = FakeSql([[GRANT_SCHEMA_VERSION]], [["0"]])
        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await store(sql).verify()

    async def test_opening_verifies_before_it_hands_the_store_over(self) -> None:
        sql = FakeSql([[GRANT_SCHEMA_VERSION]], [["5000ms"]])
        opened = await PostgresGrantStore.open(sql, clock=FakeClock(start=NOW), settings=SETTINGS)
        assert isinstance(opened, PostgresGrantStore)

    def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain table identifier"):
            GrantTables(grants="adk_grants; DROP TABLE adk_runs")

    def test_the_schema_is_the_deployment_s_to_apply(self) -> None:
        assert "CREATE TABLE adk_grants" in EXPECTED_GRANT_SCHEMA
        assert "adk_grants_live" in EXPECTED_GRANT_SCHEMA


class TestReadingWhatIsLive:
    """The runtime sees unexpired grants for the tenant and the ones above it, and no more."""

    async def test_a_grant_comes_back_as_it_was_written(self) -> None:
        sql = FakeSql([row()])
        [found] = await store(sql).grants_for(tenant="acme", action_class="booking.change")
        assert found == held()

    async def test_the_read_asks_for_one_tenant_and_one_class_at_one_moment(self) -> None:
        sql = FakeSql([row()])
        await store(sql).grants_for(tenant="acme/eu", action_class="booking.change")
        assert sql.bound == ("acme/eu", "booking.change", NOW)
        assert "expires_at >" in sql.sent

    async def test_a_tenant_with_nothing_granted_reads_as_nothing(self) -> None:
        assert await store(FakeSql([])).grants_for(tenant="acme", action_class="x") == []

    async def test_the_store_is_a_reader_by_shape(self) -> None:
        assert isinstance(store(FakeSql()), GrantReader)

    async def test_tables_can_be_moved_without_touching_the_statements(self) -> None:
        sql = FakeSql([row()])
        await store(sql, tables=GrantTables(grants="tenant_grants")).grants_for(
            tenant="acme", action_class="booking.change"
        )
        assert "FROM tenant_grants" in sql.sent


class TestIssuing:
    """Writing a grant is somebody's decision being recorded, and it happens once."""

    async def test_an_issued_grant_is_written_whole(self) -> None:
        sql = FakeSql([[1]])
        await store(sql).issue(held())
        written = json.loads(sql.bound[-1])
        assert written["granted_by"] == "ops@acme.example"
        assert sql.bound[0] == "g1"

    async def test_the_stored_grant_is_what_comes_back(self) -> None:
        assert await store(FakeSql([[1]])).issue(held()) == held()

    async def test_an_id_that_is_already_in_use_is_refused_rather_than_overwritten(self) -> None:
        with pytest.raises(ConfigurationError, match="already exists"):
            await store(FakeSql([])).issue(held())

    async def test_a_unique_violation_from_the_database_says_the_same_thing(self) -> None:
        sql = FakeSql(fails=[DriverError("23505")])
        with pytest.raises(ConfigurationError, match="already exists"):
            await store(sql).issue(held())

    async def test_the_insert_never_updates_on_conflict(self) -> None:
        sql = FakeSql([[1]])
        await store(sql).issue(held())
        assert "ON CONFLICT" not in sql.sent.upper()


class TestWhenTheDatabaseSaysNo:
    """A store that cannot answer says so; it never answers with an empty set of grants."""

    async def test_a_read_that_fails_is_raised_rather_than_read_as_no_grants(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError, match="attempts"):
            await store(sql).grants_for(tenant="acme", action_class="booking.change")

    async def test_a_write_that_fails_is_raised(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")])
        with pytest.raises(StatePersistenceError):
            await store(sql).issue(held())

    async def test_the_failure_names_the_store_that_could_not_answer(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError) as raised:
            await store(sql).grants_for(tenant="acme", action_class="booking.change")
        assert "PostgresGrantStore" in str(raised.value)


class TestWithdrawing:
    """A withdrawal is another append, so nothing here can put an old authority back."""

    def _taken_back(self, **fields: object) -> Revocation:
        """One withdrawal, filled in enough to be recorded."""
        defaults: dict[str, object] = {
            "grant_id": "g1",
            "revoked_by": "ops@acme.example",
            "revoked_at": NOW + 1,
        }
        return Revocation.model_validate(defaults | fields)

    async def test_a_withdrawal_is_written_whole(self) -> None:
        sql = FakeSql([])
        await store(sql).revoke(self._taken_back(reason="the card was reported stolen"))
        assert json.loads(sql.bound[-1])["reason"] == "the card was reported stolen"
        assert sql.bound[0] == "g1"

    async def test_a_withdrawal_of_a_whole_class_names_no_grant(self) -> None:
        sql = FakeSql([])
        await store(sql).revoke(
            self._taken_back(grant_id=None, tenant="acme", action_class="payment.refund")
        )
        assert sql.bound[:3] == (None, "acme", "payment.refund")

    async def test_repeating_a_withdrawal_is_the_same_withdrawal(self) -> None:
        sql = FakeSql(fails=[DriverError("23505")])
        await store(sql).revoke(self._taken_back())

    async def test_a_withdrawal_the_database_would_not_take_is_raised(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")])
        with pytest.raises(StatePersistenceError):
            await store(sql).revoke(self._taken_back())

    async def test_nothing_here_removes_a_withdrawal(self) -> None:
        assert "DELETE" not in EXPECTED_GRANT_SCHEMA.replace("REVOKE UPDATE, DELETE", "")
        assert "adk_grant_revocations" in EXPECTED_GRANT_SCHEMA

    async def test_a_withdrawn_grant_is_not_read_back_as_live(self) -> None:
        sql = FakeSql([row()])
        await store(sql).grants_for(tenant="acme", action_class="booking.change")
        assert "NOT EXISTS" in sql.sent

    async def test_the_store_is_an_issuer_by_shape(self) -> None:
        assert isinstance(store(FakeSql()), GrantIssuer)
