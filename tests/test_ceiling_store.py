"""The ceiling ledger in PostgreSQL, where the limit is enforced by the statement.

The point of every test here is that no decision is taken in Python: the reserve either
lands a row under the limit or lands nothing, and a retry meets a unique index rather than
a lookup somebody could race. Everything is scripted against a fake session — nothing
opens a socket.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.ceiling import (
    CEILING_SCHEMA_VERSION,
    DEFAULT_CEILING_TABLES,
    EXPECTED_CEILING_SCHEMA,
    CeilingTables,
    PostgresCeilingLedger,
    PostgresCeilingSettings,
)
from tesserix_adk.core.autonomy import Ceiling
from tesserix_adk.core.ceiling import CeilingLedger, HoldState
from tesserix_adk.core.errors import (
    CeilingExceededError,
    ConfigurationError,
    InexactAmountError,
    StatePersistenceError,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence

NOW = 1_000.0
DAY = 86_400.0
LIMIT = Ceiling(amount=Decimal("10000"), currency="INR", window_seconds=DAY)
SETTINGS = PostgresCeilingSettings(dsn=SecretStr("postgresql://localhost/adk"))


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


def row(state: str = "held", amount: str = "4000") -> list[Any]:
    """One reservation as the columns come back."""
    return [
        7,
        "acme",
        "booking.change",
        "INR",
        Decimal(amount),
        "run_1:call-1",
        NOW,
        NOW + 300,
        state,
    ]


def ledger(sql: FakeSql, tables: CeilingTables = DEFAULT_CEILING_TABLES) -> PostgresCeilingLedger:
    """A ledger over the scripted session, on a clock a test can move."""
    return PostgresCeilingLedger(
        sql, clock=FakeClock(start=NOW), settings=SETTINGS, tables=tables, entropy=lambda: 0.0
    )


async def reserving(
    held: PostgresCeilingLedger, amount: str = "4000", key: str = "run_1:call-1"
) -> Any:
    """One reservation for the usual tenant and class."""
    return await held.reserve(
        tenant="acme",
        action_class="booking.change",
        ceiling=LIMIT,
        amount=Decimal(amount),
        idempotency_key=key,
    )


class TestTakingHeadroomInOneStatement:
    """The ceiling test is the statement's `WHERE`, so nothing decides between two reads."""

    async def test_a_reservation_that_landed_comes_back_as_held(self) -> None:
        sql = FakeSql([row()])
        taken = await reserving(ledger(sql))
        assert taken.state is HoldState.HELD
        assert taken.amount == Decimal("4000")

    async def test_the_ceiling_is_bound_to_the_statement_not_compared_in_python(self) -> None:
        sql = FakeSql([row()])
        await reserving(ledger(sql))
        assert "INSERT INTO adk_ceiling_holds" in sql.sent
        assert sql.bound[4] == Decimal("4000")
        assert sql.bound[-1] == LIMIT.amount

    async def test_a_row_that_did_not_land_is_a_refusal_naming_the_headroom(self) -> None:
        sql = FakeSql([], [[Decimal("9000")]])
        with pytest.raises(CeilingExceededError, match="over the 1000 INR"):
            await reserving(ledger(sql))

    async def test_an_amount_that_cannot_be_exact_never_reaches_the_database(self) -> None:
        sql = FakeSql()
        with pytest.raises(InexactAmountError):
            await ledger(sql).reserve(
                tenant="acme",
                action_class="booking.change",
                ceiling=LIMIT,
                amount=0.1,  # type: ignore[arg-type]
                idempotency_key="run_1:call-1",
            )
        assert sql.calls == []

    async def test_a_database_that_could_not_answer_is_not_a_reserve_of_nothing(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")])
        with pytest.raises(StatePersistenceError):
            await reserving(ledger(sql))


class TestARetryOfTheSameCall:
    """A key already held is the same action, and it takes no second headroom."""

    async def test_a_key_in_use_comes_back_as_what_it_already_took(self) -> None:
        sql = FakeSql([row()], fails=[DriverError("23505")])
        taken = await reserving(ledger(sql))
        assert taken.idempotency_key == "run_1:call-1"
        assert "SELECT" in sql.sent

    async def test_a_key_the_index_refused_and_the_read_cannot_find_is_a_failure(self) -> None:
        sql = FakeSql([], fails=[DriverError("23505")])
        with pytest.raises(StatePersistenceError):
            await reserving(ledger(sql))


class TestSettlingWhatWasHeld:
    """A settle only ever matches a row still held, so nothing settles twice."""

    async def test_committing_returns_the_row_it_settled(self) -> None:
        sql = FakeSql([row(state="committed")])
        settled = await ledger(sql).commit("run_1:call-1")
        assert settled is not None
        assert settled.state is HoldState.COMMITTED
        assert sql.bound[1] == "committed"

    async def test_committing_a_key_nobody_holds_settles_to_nothing(self) -> None:
        assert await ledger(FakeSql([])).commit("run_1:call-1") is None

    async def test_releasing_only_matches_a_row_still_held(self) -> None:
        sql = FakeSql([row(state="released")])
        await ledger(sql).release("run_1:call-1")
        assert "state = 'held'" in sql.sent
        assert sql.bound[1] == "released"


class TestWhatTheWindowCounts:
    """Held and committed both count; released and expired do not."""

    async def test_the_sum_comes_back_exactly(self) -> None:
        sql = FakeSql([[Decimal("9000.25")]])
        spent = await ledger(sql).committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        )
        assert spent == Decimal("9000.25")

    async def test_a_window_nobody_spent_in_is_nothing_rather_than_none(self) -> None:
        sql = FakeSql([[None]])
        assert await ledger(sql).committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")

    async def test_an_empty_answer_is_nothing_too(self) -> None:
        assert await ledger(FakeSql([])).committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")

    async def test_reaping_returns_how_many_it_released(self) -> None:
        sql = FakeSql([[1], [2]])
        assert await ledger(sql).reap() == 2
        assert "state = 'held'" in sql.sent


class TestRefusingADatabaseThatIsNotTheShapeExpected:
    """Checked once at startup, because a column that moved is a write into the wrong shape."""

    async def test_a_version_that_moved_is_refused(self) -> None:
        sql = FakeSql([[CEILING_SCHEMA_VERSION + 1]])
        with pytest.raises(ConfigurationError, match="schema is version"):
            await PostgresCeilingLedger.open(
                sql, clock=FakeClock(start=NOW), settings=SETTINGS, entropy=lambda: 0.0
            )

    async def test_a_database_with_no_schema_row_is_refused(self) -> None:
        sql = FakeSql([])
        with pytest.raises(ConfigurationError, match="schema is version 0"):
            await ledger(sql).verify()

    async def test_a_connection_with_no_statement_timeout_is_refused(self) -> None:
        sql = FakeSql([[CEILING_SCHEMA_VERSION]], [["0"]])
        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await ledger(sql).verify()

    async def test_a_database_that_is_the_right_shape_opens(self) -> None:
        sql = FakeSql([[CEILING_SCHEMA_VERSION]], [["30s"]])
        opened = await PostgresCeilingLedger.open(
            sql, clock=FakeClock(start=NOW), settings=SETTINGS, entropy=lambda: 0.0
        )
        assert isinstance(opened, CeilingLedger)

    def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain table identifier"):
            CeilingTables(holds="holds; DROP TABLE adk_grants")

    def test_the_expected_schema_says_what_the_adapter_relies_on(self) -> None:
        assert "idempotency_key text NOT NULL UNIQUE" in EXPECTED_CEILING_SCHEMA
        assert "numeric(20, 4)" in EXPECTED_CEILING_SCHEMA
        assert f"('ceiling', {CEILING_SCHEMA_VERSION})" in EXPECTED_CEILING_SCHEMA

    def test_the_ledger_is_recognised_by_shape(self) -> None:
        assert isinstance(ledger(FakeSql()), CeilingLedger)
