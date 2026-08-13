"""Where the record of a decision is kept, and what the store will not let anyone do to it.

The store's job is narrow and unforgiving: take every decision, give none of them back
changed, and answer one question well — what did this tenant's agents do in this period,
declines included. Every test here is scripted against a fake session; nothing opens a
socket.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.audit import (
    AUDIT_SCHEMA_VERSION,
    DEFAULT_AUDIT_TABLES,
    EXPECTED_AUDIT_SCHEMA,
    AuditTables,
    JetStreamAudit,
    JetStreamPublisher,
    PostgresAuditSettings,
    PostgresAuditSink,
)
from tesserix_adk.core.audit import AuditDecision, AuditEvent, AuditSink, pseudonym
from tesserix_adk.core.autonomy import AutonomyLevel
from tesserix_adk.core.errors import ConfigurationError, StatePersistenceError
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence

NOW = 1_000.0
SETTINGS = PostgresAuditSettings(dsn=SecretStr("postgresql://localhost/adk"))


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


class FakeStream:
    """A JetStream client that keeps what it was asked to publish."""

    def __init__(self, *, fails: Exception | None = None) -> None:
        self.published: list[tuple[str, bytes]] = []
        self.fails = fails

    async def publish(self, subject: str, payload: bytes) -> None:
        """Take the message, or fail the way an unacknowledged publish does."""
        if self.fails is not None:
            raise self.fails
        self.published.append((subject, payload))


def decided(**fields: object) -> AuditEvent:
    """One decision, filled in enough to be recorded."""
    defaults: dict[str, object] = {
        "run_id": "run_1",
        "sequence": 0,
        "tenant": "acme",
        "user": "ops@acme.example",
        "agent_name": "concierge",
        "agent_version": "3",
        "tool": "booking.change",
        "action_class": "booking.change",
        "level": AutonomyLevel.ACT_WITHIN_LIMITS,
        "decision": AuditDecision.EXECUTED,
        "headroom_before": Decimal("5000"),
        "headroom_after": Decimal("4100"),
        "arguments_digest": "d0",
        "idempotency_key": "booking.change:d0",
        "recorded_at": NOW,
    }
    return AuditEvent.model_validate(defaults | fields)


def sink(sql: FakeSql, *, tables: AuditTables = DEFAULT_AUDIT_TABLES) -> PostgresAuditSink:
    """A sink over a scripted session."""
    return PostgresAuditSink(sql, clock=FakeClock(start=NOW), settings=SETTINGS, tables=tables)


class TestRefusingADatabaseThatIsNotTheRightShape:
    """A column that moved is a decision recorded into the wrong shape, caught at startup."""

    async def test_a_schema_the_adapter_was_not_written_for_is_refused(self) -> None:
        sql = FakeSql([[AUDIT_SCHEMA_VERSION + 1]])
        with pytest.raises(ConfigurationError, match="schema is version"):
            await sink(sql).verify()

    async def test_a_database_with_no_schema_row_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="version 0"):
            await sink(FakeSql([])).verify()

    async def test_a_connection_that_lets_a_statement_run_forever_is_refused(self) -> None:
        sql = FakeSql([[AUDIT_SCHEMA_VERSION]], [["0"]])
        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await sink(sql).verify()

    async def test_opening_verifies_before_it_hands_the_sink_over(self) -> None:
        sql = FakeSql([[AUDIT_SCHEMA_VERSION]], [["5000ms"]])
        opened = await PostgresAuditSink.open(sql, clock=FakeClock(start=NOW), settings=SETTINGS)
        assert isinstance(opened, PostgresAuditSink)

    def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain table identifier"):
            AuditTables(events="adk_audit; DROP TABLE adk_runs")

    def test_the_sink_is_one_by_shape(self) -> None:
        assert isinstance(sink(FakeSql()), AuditSink)


class TestAppendingADecision:
    """One decision, one row, whatever the retries did."""

    async def test_a_decision_is_written_whole(self) -> None:
        sql = FakeSql([[1]])
        await sink(sql).append(decided())
        assert json.loads(sql.bound[-1])["agent_name"] == "concierge"

    async def test_the_row_carries_what_the_question_is_asked_of(self) -> None:
        sql = FakeSql([[1]])
        await sink(sql).append(decided())
        assert sql.bound[:6] == ("run_1", "booking.change:d0", "executed", "acme", 0, NOW)

    async def test_what_was_written_is_what_comes_back(self) -> None:
        assert await sink(FakeSql([[1]])).append(decided()) == decided()

    async def test_recording_the_same_decision_twice_reads_the_first_one_back(self) -> None:
        first = decided(reason="within the ceiling")
        sql = FakeSql([], [[first.model_dump_json()]])
        assert await sink(sql).append(decided()) == first

    async def test_a_repeat_the_database_no_longer_has_is_still_the_decision(self) -> None:
        sql = FakeSql([], [])
        assert await sink(sql).append(decided()) == decided()

    async def test_an_existing_row_is_never_rewritten(self) -> None:
        sql = FakeSql([[1]])
        await sink(sql).append(decided())
        assert "DO NOTHING" in sql.calls[0][0]

    async def test_tables_can_be_moved_without_touching_the_statements(self) -> None:
        sql = FakeSql([[1]])
        await sink(sql, tables=AuditTables(events="court_audit")).append(decided())
        assert "INSERT INTO court_audit" in sql.sent


class TestAskingWhatWasDone:
    """One tenant, one period, declines included — the question the store exists for."""

    async def test_the_period_and_the_tenant_are_what_is_asked(self) -> None:
        sql = FakeSql([])
        await sink(sql).records(tenant="acme", since=NOW, until=NOW + 60)
        assert sql.bound == ("acme", NOW, NOW + 60, None)

    async def test_an_open_ended_period_binds_no_end(self) -> None:
        sql = FakeSql([])
        await sink(sql).records(tenant="acme")
        assert sql.bound == ("acme", 0.0, None, None)

    async def test_one_kind_of_decision_can_be_asked_for_alone(self) -> None:
        sql = FakeSql([])
        await sink(sql).records(tenant="acme", decision=AuditDecision.REFUSED)
        assert sql.bound[-1] == "refused"

    async def test_the_records_come_back_as_they_were_written(self) -> None:
        sql = FakeSql([[decided().model_dump_json()], [decided(sequence=1).model_dump_json()]])
        [first, second] = await sink(sql).records(tenant="acme")
        assert (first.sequence, second.sequence) == (0, 1)

    async def test_they_are_read_in_the_order_they_were_taken(self) -> None:
        sql = FakeSql([])
        await sink(sql).records(tenant="acme")
        assert "ORDER BY recorded_at, run_id, sequence" in sql.sent

    async def test_a_tenant_with_nothing_recorded_reads_as_nothing(self) -> None:
        assert await sink(FakeSql([])).records(tenant="acme") == ()


class TestAnErasureRequest:
    """The person goes, the decision stays — a deletion would take the defence with it."""

    async def test_the_person_is_replaced_by_a_stand_in(self) -> None:
        sql = FakeSql([[1], [1]])
        assert await sink(sql).pseudonymise(tenant="acme", subject="ops@acme.example") == 2
        assert sql.bound == ("acme", "ops@acme.example", pseudonym("ops@acme.example"))

    async def test_two_deployments_cannot_join_on_the_same_person(self) -> None:
        sql = FakeSql([[1]])
        salted = PostgresAuditSink(
            sql, clock=FakeClock(start=NOW), settings=SETTINGS, pseudonym_salt="court"
        )
        await salted.pseudonymise(tenant="acme", subject="ops@acme.example")
        assert sql.bound[-1] == pseudonym("ops@acme.example", salt="court")

    async def test_a_subject_nobody_recorded_changes_nothing(self) -> None:
        assert await sink(FakeSql([])).pseudonymise(tenant="acme", subject="nobody") == 0

    async def test_the_erasure_is_the_only_statement_that_is_not_an_insert(self) -> None:
        sql = FakeSql([])
        await sink(sql).pseudonymise(tenant="acme", subject="ops@acme.example")
        assert sql.sent.strip().startswith("UPDATE")


class TestWhenTheDatabaseSaysNo:
    """A store that cannot answer says so; it never answers with no decisions."""

    async def test_a_write_that_fails_is_raised_so_the_call_does_not_go_out(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError):
            await sink(sql).append(decided())

    async def test_a_read_that_fails_is_raised_rather_than_read_as_nothing_done(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError, match="attempts"):
            await sink(sql).records(tenant="acme")

    async def test_an_erasure_that_fails_is_raised(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError):
            await sink(sql).pseudonymise(tenant="acme", subject="ops@acme.example")

    async def test_the_failure_names_the_store_that_could_not_answer(self) -> None:
        sql = FakeSql(fails=[DriverError("08006")] * 3)
        with pytest.raises(StatePersistenceError) as raised:
            await sink(sql).records(tenant="acme")
        assert "PostgresAuditSink" in str(raised.value)


class TestWhatTheSchemaWillNotAllow:
    """The shape is the deployment's to apply, and it is written to be unhelpful."""

    def test_nothing_deletes_a_decision(self) -> None:
        assert "REVOKE UPDATE, DELETE ON adk_audit FROM PUBLIC" in EXPECTED_AUDIT_SCHEMA

    def test_the_erasure_statement_is_granted_to_a_role_of_its_own(self) -> None:
        assert "GRANT UPDATE (payload) ON adk_audit TO adk_erasure" in EXPECTED_AUDIT_SCHEMA

    def test_one_decision_about_one_call_can_only_be_recorded_once(self) -> None:
        assert "UNIQUE (run_id, idempotency_key, decision)" in EXPECTED_AUDIT_SCHEMA

    def test_the_question_it_is_asked_has_an_index(self) -> None:
        assert "adk_audit_asked ON adk_audit (tenant, recorded_at)" in EXPECTED_AUDIT_SCHEMA

    def test_how_long_it_is_kept_is_written_down(self) -> None:
        assert "Retention" in EXPECTED_AUDIT_SCHEMA


class TestOnAStream:
    """A second administrative domain, where database access is not enough to undo a record."""

    async def test_a_decision_is_published_whole(self) -> None:
        stream = FakeStream()
        await JetStreamAudit(stream).append(decided())
        [(_, payload)] = stream.published
        assert json.loads(payload)["decision"] == "executed"

    async def test_the_tenant_is_in_the_subject_so_a_consumer_can_be_scoped_to_it(self) -> None:
        stream = FakeStream()
        await JetStreamAudit(stream).append(decided(tenant="acme"))
        assert stream.published[0][0] == "adk.audit.acme"

    async def test_the_subject_root_is_the_deployment_s_to_choose(self) -> None:
        stream = FakeStream()
        await JetStreamAudit(stream, subject="court.decisions").append(decided())
        assert stream.published[0][0] == "court.decisions.acme"

    async def test_what_was_published_is_what_comes_back(self) -> None:
        assert await JetStreamAudit(FakeStream()).append(decided()) == decided()

    async def test_a_tenant_that_would_widen_who_hears_it_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain subject token"):
            await JetStreamAudit(FakeStream()).append(decided(tenant="acme.*"))

    def test_a_subject_root_that_is_a_wildcard_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain subject token"):
            JetStreamAudit(FakeStream(), subject=">")

    async def test_a_publish_the_stream_did_not_acknowledge_is_raised(self) -> None:
        stream = FakeStream(fails=TimeoutError("no ack"))
        with pytest.raises(TimeoutError):
            await JetStreamAudit(stream).append(decided())

    def test_the_publisher_is_one_by_shape(self) -> None:
        assert isinstance(FakeStream(), JetStreamPublisher)
