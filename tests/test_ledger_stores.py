"""What the shared stores send, and what they make of what comes back.

The Lua and the SQL are verified against a real server by `SpendLedgerConformance`, which
is what `tests/integration/` runs. These are the translation: the right script, the right
keys, and a reply turned into the right answer or the right refusal.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from tesserix_adk.adapters import PostgresLedger, RedisLedger
from tesserix_adk.core.errors import BudgetExceededError, BudgetUnavailableError
from tesserix_adk.core.ledger import LedgerKey, Reservation, Window, WindowKind
from tesserix_adk.testing import FakeClock

HOUR = Window(kind=WindowKind.ROLLING, seconds=3_600)
KEY = LedgerKey(tenant="acme", agent=None, window=HOUR)
CEILING = Decimal("10.00")


class FakeRedis:
    """Records what was evaluated and answers with whatever the test says the server said."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.fail: Exception | None = None

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        if self.fail is not None:
            raise self.fail
        self.calls.append((script, numkeys, args))
        return self.replies.pop(0) if self.replies else [1, "9.00"]

    @property
    def keys(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[:numkeys]

    @property
    def argv(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[numkeys:]


class FakeSql:
    """The same, for a database."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.fail: Exception | None = None

    async def fetch(self, statement: str, *args: Any) -> list[list[Any]]:
        if self.fail is not None:
            raise self.fail
        self.statements.append((statement, args))
        return self.replies.pop(0) if self.replies else [[True, Decimal("9.00")]]

    @property
    def args(self) -> tuple[Any, ...]:
        return self.statements[-1][1]


def redis_ledger(client: FakeRedis, **kwargs: Any) -> RedisLedger:
    return RedisLedger(client, clock=FakeClock(start=7_200), **kwargs)


def postgres_ledger(executor: FakeSql, **kwargs: Any) -> PostgresLedger:
    return PostgresLedger(executor, clock=FakeClock(start=7_200), **kwargs)


class TestTheRedisLedgerSpeaksOneScriptPerOperation:
    async def test_a_granted_reservation_carries_the_lease_the_server_was_told(self) -> None:
        client = FakeRedis([1, "9.00"])
        held = await redis_ledger(client).reserve(
            KEY, Decimal("1.00"), ceiling=CEILING, lease_seconds=60
        )
        assert held.expires_at == 7_260
        assert held.amount == Decimal("1.00")

    async def test_the_window_and_the_hold_are_separate_keys(self) -> None:
        """One is pruned by time, the other by lease, and they expire differently."""
        client = FakeRedis()
        await redis_ledger(client).reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        settled, held = client.keys
        assert settled.startswith("adk:ledger:acme::rolling:3600")
        assert held.endswith(":held")
        assert settled != held

    async def test_the_ceiling_and_the_window_edge_are_the_server_s_to_apply(self) -> None:
        client = FakeRedis()
        await redis_ledger(client).reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        assert "10.00" in client.argv
        assert "3600.0" in client.argv

    async def test_a_refusal_from_the_server_is_a_refusal_here(self) -> None:
        client = FakeRedis([0, "0.25"])
        with pytest.raises(BudgetExceededError, match=r"0\.25"):
            await redis_ledger(client).reserve(KEY, Decimal("1.00"), ceiling=CEILING)

    async def test_settling_names_the_reservation_and_what_it_spent(self) -> None:
        client = FakeRedis(1)
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await redis_ledger(client).settle(held, Decimal("0.40"))
        assert client.argv[0] == "r1"
        assert client.argv[1] == "0.40"

    async def test_a_hold_the_server_no_longer_has_is_not_settled_silently(self) -> None:
        client = FakeRedis(0)
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        with pytest.raises(BudgetUnavailableError, match="not open"):
            await redis_ledger(client).settle(held, Decimal("0.40"))

    async def test_releasing_spends_nothing(self) -> None:
        client = FakeRedis(1)
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await redis_ledger(client).release(held)
        assert client.argv[1] == "0"

    async def test_progress_is_recorded_against_the_hold(self) -> None:
        client = FakeRedis(1)
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await redis_ledger(client).record_progress(held, Decimal("0.30"))
        assert client.argv[:2] == ("r1", "0.30")

    async def test_progress_on_a_closed_hold_is_refused(self) -> None:
        client = FakeRedis(0)
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        with pytest.raises(BudgetUnavailableError):
            await redis_ledger(client).record_progress(held, Decimal("0.30"))

    async def test_a_window_read_comes_back_as_money(self) -> None:
        client = FakeRedis(["3.50", "1.25"])
        window = await redis_ledger(client).read_window(KEY)
        assert (window.settled, window.reserved) == (Decimal("3.50"), Decimal("1.25"))
        assert window.resets_at == 10_800

    async def test_reconciliation_returns_what_the_sweep_closed(self) -> None:
        assert await redis_ledger(FakeRedis(3)).reconcile() == 3

    async def test_forgetting_a_tenant_returns_the_aggregate_that_went(self) -> None:
        forgotten = await redis_ledger(FakeRedis(["6.00", "1.00"])).forget("acme")
        assert forgotten.settled == Decimal("6.00")

    async def test_a_tenant_is_erased_by_prefix_and_only_its_own(self) -> None:
        client = FakeRedis(["0", "0"])
        await redis_ledger(client).forget("acme")
        assert client.argv[0] == "adk:ledger:acme:*"


class TestTheRedisLedgerFailsClosed:
    async def test_an_unreachable_server_refuses_rather_than_permits(self) -> None:
        client = FakeRedis()
        client.fail = OSError("connection reset")
        with pytest.raises(BudgetUnavailableError, match="connection reset"):
            await redis_ledger(client).reserve(KEY, Decimal("1.00"), ceiling=CEILING)

    async def test_a_read_that_cannot_reach_the_server_says_so(self) -> None:
        client = FakeRedis()
        client.fail = OSError("connection reset")
        with pytest.raises(BudgetUnavailableError):
            await redis_ledger(client).read_window(KEY)

    async def test_degraded_mode_is_configured_and_recorded_never_inferred(self) -> None:
        client = FakeRedis()
        client.fail = OSError("connection reset")
        ledger = redis_ledger(client, degraded_allowed=True)
        held = await ledger.reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        assert held.degraded is True
        assert ledger.degradations == 1

    async def test_a_degraded_settlement_does_not_pretend_to_have_been_recorded(self) -> None:
        client = FakeRedis()
        client.fail = OSError("connection reset")
        ledger = redis_ledger(client, degraded_allowed=True)
        held = await ledger.reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("1.00"))
        assert ledger.degradations == 2


class TestShardedWrites:
    async def test_a_busy_tenant_s_writes_spread_over_shards(self) -> None:
        client = FakeRedis()
        ledger = redis_ledger(client, shards=8)
        seen = set()
        for _ in range(20):
            await ledger.reserve(KEY, Decimal("0.01"), ceiling=CEILING)
            seen.add(client.keys[0])
        assert len(seen) > 1

    async def test_a_read_sums_every_shard(self) -> None:
        client = FakeRedis(["1.00", "0"])
        await redis_ledger(client, shards=8).read_window(KEY)
        assert client.argv[-1] == "8"


class TestThePostgresLedger:
    async def test_a_reservation_is_one_statement_so_it_cannot_race(self) -> None:
        """Two statements are two chances for another replica to slip between them."""
        sql = FakeSql([[True, Decimal("9.00")]])
        await postgres_ledger(sql).reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        assert len(sql.statements) == 1

    async def test_a_refusal_carries_what_was_left(self) -> None:
        sql = FakeSql([[False, Decimal("0.25")]])
        with pytest.raises(BudgetExceededError, match=r"0\.25"):
            await postgres_ledger(sql).reserve(KEY, Decimal("1.00"), ceiling=CEILING)

    async def test_the_window_edge_is_passed_not_assumed(self) -> None:
        sql = FakeSql()
        await postgres_ledger(sql).reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        assert 3_600.0 in sql.args

    async def test_settling_records_what_was_spent(self) -> None:
        sql = FakeSql([[True]])
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await postgres_ledger(sql).settle(held, Decimal("0.40"))
        assert Decimal("0.40") in sql.args

    async def test_a_hold_the_database_no_longer_has_is_not_settled_silently(self) -> None:
        sql = FakeSql([])
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        with pytest.raises(BudgetUnavailableError, match="not open"):
            await postgres_ledger(sql).settle(held, Decimal("0.40"))

    async def test_releasing_records_nothing_spent(self) -> None:
        sql = FakeSql([[True]])
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await postgres_ledger(sql).release(held)
        assert Decimal(0) in sql.args

    async def test_progress_is_recorded_against_the_hold(self) -> None:
        sql = FakeSql([[True]])
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        await postgres_ledger(sql).record_progress(held, Decimal("0.30"))
        assert Decimal("0.30") in sql.args

    async def test_progress_on_a_closed_hold_is_refused(self) -> None:
        sql = FakeSql([])
        held = Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)
        with pytest.raises(BudgetUnavailableError):
            await postgres_ledger(sql).record_progress(held, Decimal("0.30"))

    async def test_a_window_read_comes_back_as_money(self) -> None:
        sql = FakeSql([[Decimal("3.50"), Decimal("1.25")]])
        window = await postgres_ledger(sql).read_window(KEY)
        assert (window.settled, window.reserved) == (Decimal("3.50"), Decimal("1.25"))

    async def test_an_empty_window_reads_as_nothing_rather_than_failing(self) -> None:
        window = await postgres_ledger(FakeSql([[None, None]])).read_window(KEY)
        assert window.committed == Decimal(0)

    async def test_reconciliation_returns_what_it_closed(self) -> None:
        assert await postgres_ledger(FakeSql([[2]])).reconcile() == 2

    async def test_forgetting_a_tenant_returns_the_aggregate_that_went(self) -> None:
        forgotten = await postgres_ledger(FakeSql([[Decimal("6.00"), Decimal("1.00")]])).forget(
            "acme"
        )
        assert (forgotten.settled, forgotten.reserved) == (Decimal("6.00"), Decimal("1.00"))

    async def test_forgetting_a_tenant_with_no_records_is_not_an_error(self) -> None:
        assert (await postgres_ledger(FakeSql([[None, None]])).forget("nobody")).settled == 0

    async def test_the_schema_is_created_on_request_never_on_the_hot_path(self) -> None:
        """A ledger that runs DDL while serving is one that fails at the worst moment."""
        sql = FakeSql([])
        await postgres_ledger(sql).ensure_schema()
        assert "create table" in sql.statements[0][0].lower()

    async def test_an_unreachable_database_refuses_rather_than_permits(self) -> None:
        sql = FakeSql()
        sql.fail = OSError("the connection is closed")
        with pytest.raises(BudgetUnavailableError):
            await postgres_ledger(sql).reserve(KEY, Decimal("1.00"), ceiling=CEILING)

    async def test_degraded_mode_is_configured_and_recorded(self) -> None:
        sql = FakeSql()
        sql.fail = OSError("the connection is closed")
        ledger = postgres_ledger(sql, degraded_allowed=True)
        assert (await ledger.reserve(KEY, Decimal("1.00"), ceiling=CEILING)).degraded is True


class TestEveryOperationFailsClosed:
    """A store that cannot be reached must refuse on every call, not only on reserve."""

    def a_hold(self) -> Reservation:
        return Reservation(id="r1", key=KEY.at(0), amount=Decimal("1.00"), expires_at=9_000)

    @pytest.mark.parametrize(
        "call",
        [
            lambda ledger, held: ledger.settle(held, Decimal("0.40")),
            lambda ledger, held: ledger.record_progress(held, Decimal("0.40")),
            lambda ledger, _: ledger.reconcile(),
            lambda ledger, _: ledger.forget("acme"),
        ],
        ids=["settle", "progress", "reconcile", "forget"],
    )
    async def test_an_unreachable_redis_refuses_on_every_operation(self, call: Any) -> None:
        client = FakeRedis()
        client.fail = OSError("connection reset")
        ledger = redis_ledger(client)
        with pytest.raises(BudgetUnavailableError, match="connection reset"):
            await call(ledger, self.a_hold())

    @pytest.mark.parametrize(
        "call",
        [
            lambda ledger, held: ledger.settle(held, Decimal("0.40")),
            lambda ledger, _: ledger.read_window(KEY),
            lambda ledger, _: ledger.reconcile(),
            lambda ledger, _: ledger.forget("acme"),
        ],
        ids=["settle", "read", "reconcile", "forget"],
    )
    async def test_an_unreachable_database_refuses_on_every_operation(self, call: Any) -> None:
        sql = FakeSql()
        sql.fail = OSError("the connection is closed")
        ledger = postgres_ledger(sql)
        with pytest.raises(BudgetUnavailableError, match="connection is closed"):
            await call(ledger, self.a_hold())

    async def test_a_degraded_settlement_is_recorded_by_the_database_ledger_too(self) -> None:
        sql = FakeSql()
        sql.fail = OSError("the connection is closed")
        ledger = postgres_ledger(sql, degraded_allowed=True)
        held = await ledger.reserve(KEY, Decimal("1.00"), ceiling=CEILING)
        await ledger.settle(held, Decimal("1.00"))
        assert ledger.degradations == 2
