"""Persisted state outlives the code that wrote it, and has to say what wrote it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tesserix_adk.core import (
    CHECKPOINT_FORMAT,
    CURRENT_VERSIONS,
    SUPPORTED_WINDOW,
    AdkModel,
    Checkpoint,
    ConfigurationError,
    Envelope,
    RunRecord,
    RunState,
    SessionRecord,
    StateKind,
    StateMigration,
    StateMigrationError,
    StateRegistry,
    UnsupportedStateVersionError,
    WorkItem,
    canonical_json,
    packed,
    revived,
    unpacked,
)

ORDERS = "orders"
_AT = datetime(2026, 8, 21, 9, 30, tzinfo=UTC)


class _Priced(AdkModel):
    """A consumer's own record, carrying the two types JSON has no place for."""

    eur: Decimal
    legs: tuple[Decimal, ...] = ()
    at: datetime | None = None


FIXTURES = Path(__file__).parent / "fixtures" / "state"


def _record(**overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {"run_id": "run_1", "tenant": "acme", "agent_name": "planner"}
    return RunRecord(**{**fields, **overrides})


def _registry(
    *migrations: StateMigration, current: int = 1, window: int = SUPPORTED_WINDOW
) -> StateRegistry:
    registry = StateRegistry(current={ORDERS: current}, window=window)
    for migration in migrations:
        registry.register(migration)
    return registry


def _renaming(from_version: int, to_version: int) -> StateMigration:
    def rename(payload: dict[str, Any]) -> dict[str, Any]:
        moved = {name: value for name, value in payload.items() if name != f"v{from_version}"}
        return {**moved, f"v{to_version}": payload.get(f"v{from_version}", "")}

    return StateMigration(
        kind=ORDERS,
        from_version=from_version,
        to_version=to_version,
        migrate=rename,
        note=f"v{from_version} became v{to_version}",
    )


def _enveloped(payload: dict[str, Any], *, version: int, kind: str = ORDERS) -> Envelope:
    return Envelope(kind=kind, schema_version=version, payload=payload)


class TestSerialisationThatHashesTheSameTwice:
    def test_key_order_does_not_change_the_bytes(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_a_moment_survives_the_round_trip_exactly(self) -> None:
        moment = datetime(2026, 8, 21, 9, 30, tzinfo=UTC)
        assert revived(json.loads(canonical_json({"at": moment})))["at"] == moment

    def test_a_monetary_amount_never_goes_through_a_float(self) -> None:
        amount = Decimal("12.345678901234567890")
        assert revived(json.loads(canonical_json({"eur": amount})))["eur"] == amount

    def test_a_number_json_cannot_represent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Out of range"):
            canonical_json({"ratio": float("nan")})

    def test_something_that_is_not_state_is_refused_rather_than_stringified(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({"connection": object()})


class TestTaggedValuesComeBackAsThemselves:
    def test_a_tagged_moment_opens_as_a_datetime(self) -> None:
        envelope = _enveloped(
            {"eur": "1", "at": {"$datetime": _AT.isoformat()}}, version=1, kind=ORDERS
        )
        assert envelope.opened(_Priced).at == _AT

    def test_a_tagged_amount_opens_without_going_through_a_float(self) -> None:
        envelope = _enveloped({"eur": {"$decimal": "412.35"}}, version=1, kind=ORDERS)
        assert envelope.opened(_Priced).eur == Decimal("412.35")

    def test_a_tag_nested_in_the_payload_is_revived_too(self) -> None:
        envelope = _enveloped({"eur": "1", "legs": [{"$decimal": "0.10"}]}, version=1, kind=ORDERS)
        assert envelope.opened(_Priced).legs == (Decimal("0.10"),)


class TestTheEnvelope:
    def test_it_says_what_wrote_it_and_at_which_version(self) -> None:
        envelope = Envelope.around(_record(), kind=StateKind.RUN)
        assert (envelope.kind, envelope.schema_version) == (
            StateKind.RUN,
            CURRENT_VERSIONS[StateKind.RUN],
        )

    def test_it_survives_a_round_trip_through_text(self) -> None:
        envelope = Envelope.around(_record(cost_micros=7), kind=StateKind.RUN)
        assert Envelope.from_json(envelope.to_json()) == envelope

    def test_two_equal_payloads_hash_alike(self) -> None:
        first = _enveloped({"a": 1, "b": 2}, version=1)
        second = _enveloped({"b": 2, "a": 1}, version=1)
        assert first.digest() == second.digest()

    def test_a_different_payload_hashes_differently(self) -> None:
        assert _enveloped({"a": 1}, version=1).digest() != _enveloped({"a": 2}, version=1).digest()

    def test_the_record_comes_back_as_itself(self) -> None:
        record = _record(cost_micros=9, iterations=3)
        assert unpacked(packed(record, kind=StateKind.RUN), RunRecord, kind=StateKind.RUN) == record


class TestAWriterThatKnewLessThanThisReader:
    def test_a_field_it_never_wrote_is_absent_rather_than_invented(self) -> None:
        envelope = _enveloped(
            {"run_id": "run_1", "tenant": "acme", "agent_name": "planner"},
            version=1,
            kind=StateKind.RUN,
        )
        assert envelope.opened(RunRecord).session_id is None

    def test_counters_survive_at_full_precision(self) -> None:
        record = _record(cost_micros=9_007_199_254_740_993, iterations=2)
        restored = unpacked(packed(record, kind=StateKind.RUN), RunRecord, kind=StateKind.RUN)
        assert restored.cost_micros == 9_007_199_254_740_993


class TestAWriterThatKnewMoreThanThisReader:
    def test_a_field_this_reader_has_no_place_for_is_kept(self) -> None:
        envelope = _enveloped(
            {"run_id": "run_1", "tenant": "acme", "agent_name": "planner", "budget_id": "b-1"},
            version=1,
            kind=StateKind.RUN,
        )
        assert envelope.preserved(RunRecord) == {"budget_id": "b-1"}

    def test_the_record_still_opens(self) -> None:
        envelope = _enveloped(
            {"run_id": "run_1", "tenant": "acme", "agent_name": "planner", "budget_id": "b-1"},
            version=1,
            kind=StateKind.RUN,
        )
        assert envelope.opened(RunRecord).run_id == "run_1"

    def test_it_is_still_there_after_this_reader_writes_it_back(self) -> None:
        envelope = _enveloped(
            {"run_id": "run_1", "tenant": "acme", "agent_name": "planner", "budget_id": "b-1"},
            version=1,
            kind=StateKind.RUN,
        )
        record = envelope.opened(RunRecord)
        written = Envelope.around(
            record, kind=StateKind.RUN, preserved=envelope.preserved(RunRecord)
        )
        assert written.payload["budget_id"] == "b-1"

    def test_a_state_this_reader_has_never_heard_of_is_not_quietly_remapped(self) -> None:
        envelope = _enveloped(
            {
                "run_id": "run_1",
                "tenant": "acme",
                "agent_name": "planner",
                "state": "compensating",
            },
            version=1,
            kind=StateKind.RUN,
        )
        assert envelope.preserved(RunRecord) == {"state": "compensating"}
        assert envelope.opened(RunRecord).state is RunState.PENDING

    def test_and_it_comes_back_when_the_record_is_written_again(self) -> None:
        envelope = _enveloped(
            {
                "run_id": "run_1",
                "tenant": "acme",
                "agent_name": "planner",
                "state": "compensating",
            },
            version=1,
            kind=StateKind.RUN,
        )
        written = Envelope.around(
            envelope.opened(RunRecord),
            kind=StateKind.RUN,
            preserved=envelope.preserved(RunRecord),
        )
        assert written.payload["state"] == "compensating"


class TestMigrationsAppliedOnRead:
    def test_a_registered_step_runs(self) -> None:
        registry = _registry(_renaming(1, 2), current=2)
        upgraded = registry.upgraded(_enveloped({"v1": "kept"}, version=1))
        assert (upgraded.schema_version, upgraded.payload) == (2, {"v2": "kept"})

    def test_several_steps_run_in_order(self) -> None:
        registry = _registry(_renaming(1, 2), _renaming(2, 3), current=3, window=3)
        upgraded = registry.upgraded(_enveloped({"v1": "kept"}, version=1))
        assert (upgraded.schema_version, upgraded.payload) == (3, {"v3": "kept"})

    def test_a_payload_already_current_is_returned_untouched(self) -> None:
        registry = _registry(_renaming(1, 2), current=2)
        envelope = _enveloped({"v2": "kept"}, version=2)
        assert registry.upgraded(envelope) is envelope

    def test_a_step_that_fails_leaves_the_stored_payload_alone(self) -> None:
        def explode(payload: dict[str, Any]) -> dict[str, Any]:
            del payload
            raise RuntimeError("half way")

        registry = _registry(
            StateMigration(kind=ORDERS, from_version=1, to_version=2, migrate=explode), current=2
        )
        envelope = _enveloped({"v1": "kept"}, version=1)
        with pytest.raises(StateMigrationError) as failed:
            registry.upgraded(envelope)
        assert (failed.value.from_version, failed.value.to_version) == (1, 2)
        assert envelope.payload == {"v1": "kept"}

    def test_a_gap_in_the_ladder_is_refused_when_it_is_registered(self) -> None:
        with pytest.raises(ConfigurationError, match="one version at a time"):
            _registry(_renaming(1, 3), current=3)

    def test_two_migrations_for_the_same_step_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="already"):
            _registry(_renaming(1, 2), _renaming(1, 2), current=2)

    def test_a_step_the_reader_has_no_migration_for_is_refused(self) -> None:
        registry = _registry(current=2)
        with pytest.raises(UnsupportedStateVersionError):
            registry.upgraded(_enveloped({"v1": "kept"}, version=1))


class TestVersionsOutsideTheWindow:
    def test_a_newer_payload_names_both_versions_and_is_left_alone(self) -> None:
        registry = _registry(current=1)
        envelope = _enveloped({"v2": "kept"}, version=2)
        with pytest.raises(UnsupportedStateVersionError) as refused:
            registry.upgraded(envelope)
        assert (refused.value.found, refused.value.supported) == (2, 1)
        assert envelope.payload == {"v2": "kept"}

    def test_a_newer_payload_is_not_this_worker_s_to_repair(self) -> None:
        registry = _registry(current=1)
        with pytest.raises(UnsupportedStateVersionError, match="newer"):
            registry.upgraded(_enveloped({}, version=2))

    def test_a_payload_older_than_the_window_says_what_to_do_about_it(self) -> None:
        registry = _registry(*(_renaming(step, step + 1) for step in range(1, 5)), current=5)
        with pytest.raises(UnsupportedStateVersionError, match="drain"):
            registry.upgraded(_enveloped({"v1": "kept"}, version=1))

    def test_the_window_leaves_room_for_one_minor_of_dual_read(self) -> None:
        assert SUPPORTED_WINDOW >= 2


class TestWhatTheKitItselfVersions:
    def test_every_persisted_kind_has_a_current_version(self) -> None:
        assert set(CURRENT_VERSIONS) == set(StateKind)

    def test_a_checkpoint_has_one_version_and_not_two(self) -> None:
        assert CURRENT_VERSIONS[StateKind.CHECKPOINT] == CHECKPOINT_FORMAT

    def test_each_kind_round_trips_through_its_own_envelope(self) -> None:
        records = (
            (StateKind.SESSION, SessionRecord(session_id="s-1", tenant="acme"), SessionRecord),
            (StateKind.RUN, _record(), RunRecord),
            (
                StateKind.CHECKPOINT,
                Checkpoint(run_id="run_1", tenant="acme", agent_name="p"),
                Checkpoint,
            ),
            (StateKind.WORK_ITEM, WorkItem(id="w-1", tenant="acme"), WorkItem),
        )
        for kind, record, model in records:
            assert unpacked(packed(record, kind=kind), model, kind=kind) == record


class TestPayloadsWrittenBeforeEnvelopesExisted:
    def test_a_bare_record_is_read_as_the_first_version(self) -> None:
        legacy = _record().model_dump_json()
        assert unpacked(legacy, RunRecord, kind=StateKind.RUN) == _record()

    def test_text_that_is_not_a_payload_at_all_is_refused(self) -> None:
        with pytest.raises(StateMigrationError, match="not"):
            unpacked("{not json", RunRecord, kind=StateKind.RUN)


class TestFixturesFromEveryReleaseSoFar:
    def test_every_shipped_fixture_still_opens(self) -> None:
        models: dict[str, type[AdkModel]] = {
            StateKind.SESSION: SessionRecord,
            StateKind.RUN: RunRecord,
            StateKind.CHECKPOINT: Checkpoint,
            StateKind.WORK_ITEM: WorkItem,
        }
        for fixture in sorted(FIXTURES.glob("*.json")):
            kind = StateKind(fixture.stem.rsplit("-v", 1)[0])
            assert unpacked(fixture.read_text(), models[kind], kind=kind) is not None

    def test_there_is_a_fixture_for_every_kind_at_every_supported_version(self) -> None:
        shipped = {fixture.stem for fixture in FIXTURES.glob("*.json")}
        wanted = {
            f"{kind}-v{version}"
            for kind, current in CURRENT_VERSIONS.items()
            for version in range(max(1, current - SUPPORTED_WINDOW + 1), current + 1)
        }
        assert wanted <= shipped
