"""What the outbox writes, and what the relay refuses to say it delivered.

No server here: the fake answers with whatever the test says the database returned, so
these are about the statements and the decisions around them. That the statements are
valid SQL doing what the protocol says is `tests/integration`, against a real PostgreSQL.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.outbox import (
    EXPECTED_OUTBOX_SCHEMA,
    OUTBOX_SCHEMA_VERSION,
    OutboxRelay,
    OutboxTables,
    PostgresOutbox,
    PostgresOutboxSettings,
)
from tesserix_adk.core import (
    ConfigurationError,
    Delivery,
    EventEnvelope,
    Eventing,
    RunCompleted,
    tenant_scope,
)
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

TENANT = "acme"
NOW = 1_000.0
SETTINGS = PostgresOutboxSettings(dsn=SecretStr("postgresql://adk:s3cret@db:5432/adk"))
HEALTHY = [[[OUTBOX_SCHEMA_VERSION]], [["5s"]]]


class FakeSql:
    """Answers with what the test says the database returned, and records what was asked."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.rolled_back = False
        self.committed = 0

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Answer one statement."""
        self.calls.append((statement, args))
        return self.replies.pop(0) if self.replies else []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[FakeSql]:
        """One transaction, which a test can ask whether it committed."""
        try:
            yield self
        except Exception:
            self.rolled_back = True
            raise
        self.committed += 1

    @property
    def sent(self) -> str:
        """The last statement, tables and all."""
        return self.calls[-1][0]

    @property
    def bound(self) -> tuple[Any, ...]:
        """What the last statement was given."""
        return self.calls[-1][1]


@dataclass(slots=True)
class _DeadLetter:
    letters: list[tuple[bytes, str, tuple[str, ...]]] = field(default_factory=list)

    async def bury(self, payload: bytes, *, reason: str, history: tuple[str, ...] = ()) -> None:
        self.letters.append((payload, reason, history))


class _Broken:
    """A transport that is not there."""

    async def publish(self, event: EventEnvelope) -> None:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("the broker is not there")

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("the broker is not there")


async def _event(run_id: str = "run_1") -> EventEnvelope:
    eventing = Eventing(clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope(TENANT, user="ada"):
        event = await eventing.emit(RunCompleted(run_id=run_id, iterations=2))
    assert event is not None
    return event


def _outbox(sql: FakeSql, **options: Any) -> PostgresOutbox:
    return PostgresOutbox(sql, clock=FakeClock(NOW), settings=SETTINGS, **options)


def _relay(sql: FakeSql, publisher: Any, **options: Any) -> OutboxRelay:
    return OutboxRelay(sql, publisher, clock=FakeClock(NOW), worker="relay-1", **options)


class TestWritingTheEventWhereTheStateGoes:
    async def test_the_event_is_inserted_rather_than_published(self) -> None:
        sql = FakeSql([[1]])
        await _outbox(sql).publish(await _event())
        assert "INSERT INTO adk_outbox" in sql.sent

    async def test_it_writes_on_the_session_the_caller_gives_it(self) -> None:
        sql, caller = FakeSql(), FakeSql([[1]])
        await _outbox(sql).bound(caller).publish(await _event())
        assert (caller.calls, sql.calls) != ([], [])
        assert sql.calls == []

    async def test_the_row_carries_the_event_id_the_transport_deduplicates_on(self) -> None:
        sql = FakeSql([[1]])
        event = await _event()
        await _outbox(sql).publish(event)
        assert event.event_id in sql.bound

    async def test_the_row_carries_the_run_so_ordering_has_something_to_key_on(self) -> None:
        sql = FakeSql([[1]])
        await _outbox(sql).publish(await _event("run_7"))
        assert "run_7" in sql.bound

    async def test_a_second_insert_of_the_same_event_does_not_duplicate_the_row(self) -> None:
        sql = FakeSql([[1]])
        await _outbox(sql).publish(await _event())
        assert "ON CONFLICT" in sql.sent

    async def test_a_batch_is_one_statement_per_event_in_order(self) -> None:
        sql = FakeSql([[1]], [[2]])
        first, second = await _event("run_1"), await _event("run_2")
        await _outbox(sql).publish_batch((first, second))
        assert [call[1][0] for call in sql.calls] == [first.event_id, second.event_id]

    async def test_what_is_stored_is_the_redacted_envelope_and_nothing_else(self) -> None:
        sql = FakeSql([[1]])
        event = await _event()
        await _outbox(sql).publish(event)
        assert event.to_json() in sql.bound


class TestTheDatabaseItRefusesToStartAgainst:
    async def test_a_schema_at_another_version_is_refused(self) -> None:
        sql = FakeSql([[0]], [["5s"]])
        with pytest.raises(ConfigurationError, match="outbox"):
            await PostgresOutbox.open(sql, clock=FakeClock(NOW), settings=SETTINGS)

    async def test_a_connection_that_could_run_forever_is_refused(self) -> None:
        sql = FakeSql([[OUTBOX_SCHEMA_VERSION]], [["0"]])
        with pytest.raises(ConfigurationError, match="timeout"):
            await PostgresOutbox.open(sql, clock=FakeClock(NOW), settings=SETTINGS)

    async def test_a_healthy_database_opens(self) -> None:
        sql = FakeSql(*HEALTHY)
        outbox = await PostgresOutbox.open(sql, clock=FakeClock(NOW), settings=SETTINGS)
        assert isinstance(outbox, PostgresOutbox)

    async def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="identifier"):
            OutboxTables(rows="adk_outbox; DROP TABLE adk_runs")

    async def test_the_documented_schema_names_the_columns_the_relay_reads(self) -> None:
        for column in ("event_id", "run_id", "payload", "published_at", "claimed_by"):
            assert column in EXPECTED_OUTBOX_SCHEMA


class TestTheRelay:
    async def test_it_publishes_what_it_claimed(self) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]], [[1]])
        published = InMemoryEventPublisher()
        delivered = await _relay(sql, published).deliver()
        assert (delivered, len(published.events)) == (1, 1)

    async def test_it_claims_before_it_publishes_so_two_replicas_cannot_both_send(
        self,
    ) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]], [[1]])
        await _relay(sql, InMemoryEventPublisher()).deliver()
        assert "claimed_by" in sql.calls[0][0]
        assert "relay-1" in sql.calls[0][1]

    async def test_a_run_is_claimed_whole_so_its_events_cannot_overtake_each_other(
        self,
    ) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]], [[1]])
        await _relay(sql, InMemoryEventPublisher()).deliver()
        assert "pg_try_advisory_xact_lock" in sql.calls[0][0]

    async def test_it_marks_them_published_only_after_the_transport_acknowledged(
        self,
    ) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]], [[1]])
        await _relay(sql, InMemoryEventPublisher()).deliver()
        assert "published_at" in sql.sent
        assert sql.committed == 1

    async def test_a_transport_that_is_down_marks_nothing(self) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]])
        relay = _relay(sql, _Broken())
        with pytest.raises(ConnectionError):
            await relay.deliver()
        assert sql.rolled_back
        assert len(sql.calls) == 1

    async def test_a_transport_that_is_down_is_counted(self) -> None:
        event = await _event()
        sql = FakeSql([[1, "run_1", event.to_json()]])
        relay = _relay(sql, _Broken())
        with pytest.raises(ConnectionError):
            await relay.deliver()
        assert relay.failed == 1

    async def test_an_empty_outbox_publishes_nothing(self) -> None:
        sql = FakeSql([])
        published = InMemoryEventPublisher()
        assert await _relay(sql, published).deliver() == 0
        assert published.events == ()

    async def test_a_row_that_is_not_an_envelope_is_buried_rather_than_retried(self) -> None:
        letters = _DeadLetter()
        sql = FakeSql([[1, "run_1", "{not json"]], [[1]])
        relay = _relay(sql, InMemoryEventPublisher(), dead_letter=letters)
        await relay.deliver()
        assert letters.letters[0][1] == "undecodable"
        assert relay.buried == 1

    async def test_an_oversized_event_does_not_block_the_head_of_the_queue(self) -> None:
        letters = _DeadLetter()
        first, second = await _event("run_1"), await _event("run_2")
        sql = FakeSql(
            [[1, "run_1", first.to_json()], [2, "run_2", second.to_json()]],
            [[1]],
        )
        published = InMemoryEventPublisher()
        relay = _relay(
            sql, published, dead_letter=letters, max_event_bytes=len(first.to_json()) - 1
        )
        await relay.deliver()
        assert letters.letters[0][1] == "too_large"
        assert relay.buried == 2


class TestLagAndPruning:
    async def test_the_lag_is_what_is_unpublished_and_how_old_the_oldest_is(self) -> None:
        sql = FakeSql([[12, 45.0]])
        lag = await _relay(sql, InMemoryEventPublisher()).lag()
        assert (lag.unpublished, lag.oldest_seconds) == (12, 45.0)

    async def test_an_empty_outbox_has_no_lag(self) -> None:
        sql = FakeSql([])
        lag = await _relay(sql, InMemoryEventPublisher()).lag()
        assert (lag.unpublished, lag.oldest_seconds) == (0, 0.0)

    async def test_pruning_removes_published_rows_past_their_retention(self) -> None:
        sql = FakeSql([[7]])
        removed = await _relay(sql, InMemoryEventPublisher(), retention_seconds=3_600.0).prune()
        assert removed == 7
        assert NOW - 3_600.0 in sql.bound

    async def test_pruning_never_removes_a_row_that_has_not_been_published(self) -> None:
        sql = FakeSql([[0]])
        await _relay(sql, InMemoryEventPublisher()).prune()
        assert "published_at IS NOT NULL" in sql.sent

    async def test_pruning_an_empty_outbox_removes_nothing(self) -> None:
        sql = FakeSql([])
        assert await _relay(sql, InMemoryEventPublisher()).prune() == 0

    async def test_a_relay_without_a_dead_letter_still_stops_a_poison_row(self) -> None:
        sql = FakeSql([[1, "run_1", "{not json"]], [[1]])
        relay = _relay(sql, InMemoryEventPublisher())
        await relay.deliver()
        assert relay.buried == 1
