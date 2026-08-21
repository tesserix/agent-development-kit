"""The event contract: what a consumer may rely on, and what a release may not change.

An emitted event is another team's contract the moment it is consumed. These tests hold the
registry, the upcasters and the compatibility check to that, so a rename cannot reach a
consumer's dashboard through a kit minor.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import pytest
from tools.event_schemas import SNAPSHOT, main, render

from tesserix_adk.core.errors import (
    ConfigurationError,
    UnknownEventTypeError,
    UnsupportedEventVersionError,
)
from tesserix_adk.core.event_schema import (
    EVENT_SCHEMAS,
    EventSchemaRegistry,
    compatibility_breaks,
    event_schemas,
    read_envelope,
)
from tesserix_adk.core.events import (
    Delivery,
    EventEnvelope,
    Eventing,
    EventPayload,
    EventType,
    RunCompleted,
    ToolCallCompleted,
)
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher

if TYPE_CHECKING:
    from pathlib import Path

TENANT = "acme"


class ToolCallCompletedV2(ToolCallCompleted):
    """The same event with one more optional attribute, which is the only additive shape."""

    version: ClassVar[int] = 2
    attempt: int = 0


class Unregistered(EventPayload):
    """A payload whose type nothing published a schema for."""

    type: ClassVar[EventType] = "invented"  # type: ignore[assignment]
    run_id: str = ""


def _registry() -> EventSchemaRegistry:
    registry = EventSchemaRegistry()
    registry.register(RunCompleted)
    registry.register(ToolCallCompleted)
    return registry


async def _emitted(payload: EventPayload) -> EventEnvelope:
    eventing = Eventing(InMemoryEventPublisher(), clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope(TENANT, user="ada"):
        event = await eventing.emit(payload)
    assert event is not None
    return event


class TestTheRegistry:
    def test_every_event_the_kit_emits_has_a_registered_schema(self) -> None:
        for event_type in EventType:
            assert EVENT_SCHEMAS.current_version(event_type) >= 1

    def test_a_version_maps_to_the_model_that_validates_it(self) -> None:
        registry = _registry()
        assert registry.model_for(EventType.RUN_COMPLETED, 1) is RunCompleted

    def test_registering_the_same_type_and_version_twice_is_refused(self) -> None:
        registry = _registry()
        with pytest.raises(ConfigurationError, match="already registered"):
            registry.register(RunCompleted)

    def test_a_second_version_lives_beside_the_first(self) -> None:
        registry = _registry()
        registry.register(ToolCallCompletedV2)
        assert registry.versions(EventType.TOOL_CALL_COMPLETED) == (1, 2)
        assert registry.current_version(EventType.TOOL_CALL_COMPLETED) == 2

    def test_a_version_nobody_registered_is_not_a_crash(self) -> None:
        registry = _registry()
        with pytest.raises(UnsupportedEventVersionError, match="run_completed"):
            registry.model_for(EventType.RUN_COMPLETED, 9)

    def test_an_event_type_nobody_registered_is_not_a_crash(self) -> None:
        registry = EventSchemaRegistry()
        with pytest.raises(UnsupportedEventVersionError):
            registry.model_for(EventType.RUN_COMPLETED, 1)


class TestPublishing:
    async def test_an_event_type_with_no_registered_schema_is_refused(self) -> None:
        with pytest.raises(UnknownEventTypeError, match="invented"):
            await _emitted(Unregistered(run_id="run_1"))

    async def test_the_envelope_carries_the_version_of_its_type(self) -> None:
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))
        assert event.schema_version == EVENT_SCHEMAS.current_version(EventType.RUN_COMPLETED)


class TestReadingWhatAnotherVersionPublished:
    async def test_an_older_consumer_tolerates_a_field_it_has_never_heard_of(self) -> None:
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))
        forward = json.loads(event.to_json()) | {"invented_field": "from a later kit"}

        assert read_envelope(json.dumps(forward)).run_id == "run_1"

    async def test_an_unknown_attribute_is_kept_rather_than_dropped(self) -> None:
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))
        forward = json.loads(event.to_json())
        forward["attributes"]["invented"] = "5"

        assert read_envelope(json.dumps(forward)).attributes["invented"] == "5"

    async def test_a_version_above_the_window_is_parked_not_crashed(self) -> None:
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))
        ahead = json.loads(event.to_json()) | {"schema_version": 99}

        with pytest.raises(UnsupportedEventVersionError) as parked:
            read_envelope(json.dumps(ahead))
        assert parked.value.details["event_type"] == "run_completed"

    async def test_an_event_type_this_kit_has_never_heard_of_is_parked(self) -> None:
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))
        unknown = json.loads(event.to_json()) | {"type": "quantum_completed"}

        with pytest.raises(UnknownEventTypeError, match="quantum_completed"):
            read_envelope(json.dumps(unknown))

    async def test_an_older_event_is_upcast_to_the_current_model(self) -> None:
        registry = _registry()
        registry.register(ToolCallCompletedV2)
        registry.register_upcaster(
            EventType.TOOL_CALL_COMPLETED,
            from_version=1,
            upcast=lambda attributes: attributes | {"attempt": "1"},
        )
        event = await _emitted(
            ToolCallCompleted(run_id="run_1", tool="search", tool_call_id="c1", state="ok")
        )

        current = registry.upcast(event)

        assert (current.schema_version, current.attributes["attempt"]) == (2, "1")

    async def test_upcasting_walks_every_step_between_the_versions(self) -> None:
        registry = _registry()
        registry.register(ToolCallCompletedV2)
        registry.register(_third_version())
        registry.register_upcaster(
            EventType.TOOL_CALL_COMPLETED,
            from_version=1,
            upcast=lambda attributes: attributes | {"attempt": "1"},
        )
        registry.register_upcaster(
            EventType.TOOL_CALL_COMPLETED,
            from_version=2,
            upcast=lambda attributes: attributes | {"cost_micros": "0"},
        )
        event = await _emitted(
            ToolCallCompleted(run_id="run_1", tool="search", tool_call_id="c1", state="ok")
        )

        current = registry.upcast(event)

        assert current.schema_version == 3
        assert (current.attributes["attempt"], current.attributes["cost_micros"]) == ("1", "0")

    async def test_a_missing_step_is_refused_rather_than_guessed(self) -> None:
        registry = _registry()
        registry.register(ToolCallCompletedV2)
        event = await _emitted(
            ToolCallCompleted(run_id="run_1", tool="search", tool_call_id="c1", state="ok")
        )

        with pytest.raises(UnsupportedEventVersionError, match="upcast"):
            registry.upcast(event)

    async def test_an_event_already_at_the_current_version_is_left_alone(self) -> None:
        registry = _registry()
        event = await _emitted(RunCompleted(run_id="run_1", iterations=2))

        assert registry.upcast(event) == event

    async def test_the_upcast_event_still_validates_against_its_model(self) -> None:
        registry = _registry()
        registry.register(ToolCallCompletedV2)
        registry.register_upcaster(
            EventType.TOOL_CALL_COMPLETED,
            from_version=1,
            upcast=lambda attributes: attributes | {"attempt": "1"},
        )
        event = await _emitted(
            ToolCallCompleted(run_id="run_1", tool="search", tool_call_id="c1", state="ok")
        )

        payload = registry.payload_of(registry.upcast(event))

        assert isinstance(payload, ToolCallCompletedV2)
        assert payload.attempt == 1


class TestTheGeneratedSchemas:
    def test_there_is_one_schema_per_type_and_version(self) -> None:
        schemas = event_schemas()
        assert "run_completed@1" in schemas
        assert schemas["run_completed@1"]["properties"]["run_id"]

    def test_the_schema_names_the_envelope_fields_a_consumer_may_rely_on(self) -> None:
        envelope = event_schemas()["envelope"]
        for field in ("event_id", "type", "schema_version", "occurred_at", "tenant"):
            assert field in envelope["properties"]


class TestTheCompatibilityCheck:
    def test_an_added_optional_field_is_compatible(self) -> None:
        previous = {"run_completed@1": _schema({"run_id": {"type": "string"}}, required=[])}
        current = {
            "run_completed@1": _schema(
                {"run_id": {"type": "string"}, "attempt": {"type": "integer"}}, required=[]
            )
        }
        assert compatibility_breaks(previous, current) == ()

    def test_a_renamed_field_fails_the_build_naming_it(self) -> None:
        previous = {"run_completed@1": _schema({"run_id": {"type": "string"}}, required=[])}
        current = {"run_completed@1": _schema({"the_run": {"type": "string"}}, required=[])}

        breaks = compatibility_breaks(previous, current)

        assert len(breaks) == 1
        assert "run_completed@1" in breaks[0]
        assert "run_id" in breaks[0]

    def test_a_removed_field_is_a_break(self) -> None:
        previous = {
            "run_completed@1": _schema(
                {"run_id": {"type": "string"}, "iterations": {"type": "integer"}}, required=[]
            )
        }
        current = {"run_completed@1": _schema({"run_id": {"type": "string"}}, required=[])}

        assert "iterations" in compatibility_breaks(previous, current)[0]

    def test_a_changed_field_type_is_a_break(self) -> None:
        previous = {"run_completed@1": _schema({"iterations": {"type": "integer"}}, required=[])}
        current = {"run_completed@1": _schema({"iterations": {"type": "string"}}, required=[])}

        assert "iterations" in compatibility_breaks(previous, current)[0]

    def test_a_newly_required_field_is_a_break(self) -> None:
        previous = {"run_completed@1": _schema({"run_id": {"type": "string"}}, required=[])}
        current = {"run_completed@1": _schema({"run_id": {"type": "string"}}, required=["run_id"])}

        assert "run_id" in compatibility_breaks(previous, current)[0]

    def test_a_removed_enum_member_is_a_break(self) -> None:
        previous = {"run_completed@1": _schema({"state": {"enum": ["ok", "failed"]}}, required=[])}
        current = {"run_completed@1": _schema({"state": {"enum": ["ok"]}}, required=[])}

        assert "failed" in compatibility_breaks(previous, current)[0]

    def test_an_added_enum_member_is_compatible(self) -> None:
        previous = {"run_completed@1": _schema({"state": {"enum": ["ok"]}}, required=[])}
        current = {"run_completed@1": _schema({"state": {"enum": ["ok", "held"]}}, required=[])}

        assert compatibility_breaks(previous, current) == ()

    def test_a_removed_event_version_is_a_break(self) -> None:
        previous = {"run_completed@1": _schema({}, required=[])}

        assert "run_completed@1" in compatibility_breaks(previous, {})[0]

    def test_a_new_event_version_is_compatible(self) -> None:
        current = {"run_completed@1": _schema({}, required=[]), "run_completed@2": _schema({})}

        assert compatibility_breaks({"run_completed@1": _schema({}, required=[])}, current) == ()

    def test_the_current_schemas_are_compatible_with_themselves(self) -> None:
        assert compatibility_breaks(event_schemas(), event_schemas()) == ()


def _schema(properties: dict[str, object], required: list[str] | None = None) -> dict[str, object]:
    return {"properties": properties, "required": required or []}


def _third_version() -> type[EventPayload]:
    class ToolCallCompletedV3(ToolCallCompletedV2):
        """A third version, so upcasting has more than one step to walk."""

        version: ClassVar[int] = 3
        cost_micros: int = 0

    return ToolCallCompletedV3


class TestTheCommittedContracts:
    """The snapshot is the artefact consumers pin, so it is diffed like one."""

    def test_the_committed_file_matches_the_current_contracts(self) -> None:
        assert SNAPSHOT.read_text(encoding="utf-8") == render(event_schemas()), (
            "event contracts changed. Run `make event-schemas`, review the diff, and record "
            "the compatibility decision in the same pull request."
        )

    def test_the_gate_passes_on_the_committed_file(self) -> None:
        assert main([]) == 0

    def test_the_gate_fails_on_a_renamed_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        properties = broken["run_completed@1"]["properties"]
        properties["iterations_count"] = properties.pop("iterations")
        snapshot = tmp_path / "event-schemas.json"
        snapshot.write_text(json.dumps(broken), encoding="utf-8")
        monkeypatch.setattr("tools.event_schemas.SNAPSHOT", snapshot)

        assert main([]) == 1

    def test_the_gate_fails_on_an_additive_change_that_was_not_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        del recorded["run_completed@1"]["properties"]["iterations"]
        snapshot = tmp_path / "event-schemas.json"
        snapshot.write_text(render(recorded), encoding="utf-8")
        monkeypatch.setattr("tools.event_schemas.SNAPSHOT", snapshot)

        assert main([]) == 1

    def test_writing_regenerates_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "event-schemas.json"
        monkeypatch.setattr("tools.event_schemas.SNAPSHOT", snapshot)

        assert main(["--write"]) == 0
        assert json.loads(snapshot.read_text(encoding="utf-8"))["run_completed@1"]
