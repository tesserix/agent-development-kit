"""What a consumer may rely on, and what a release may not change under it.

An emitted event becomes another team's contract the moment somebody subscribes. A field
renamed in a kit minor breaks every dashboard reading it, and the break is invisible here:
the publisher's tests all pass. So the contract is registered, generated as JSON Schema,
committed, and diffed in CI — an incompatible change fails the build naming the event, the
version and the field.

The rules the diff enforces, which are also what a consumer is promised:

* within a major, fields may only be **added**, and only as optional;
* a field name is never reused for another meaning, and a removed field is a major;
* an enum may gain members, never lose them, so a consumer treats an unknown member as
  unknown rather than as an error;
* unknown fields are tolerated on the way in, so a newer publisher does not break an older
  consumer sharing the stream.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.errors import (
    ConfigurationError,
    UnknownEventTypeError,
    UnsupportedEventVersionError,
)
from tesserix_adk.core.events import EventEnvelope, EventPayload, EventType

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "EVENT_SCHEMAS",
    "EventSchemaRegistry",
    "Upcaster",
    "compatibility_breaks",
    "event_schemas",
    "read_envelope",
]

type Upcaster = Callable[[dict[str, str]], dict[str, str]]
"""Rewrites one version's attributes into the next one's. Registered per step, never skipped."""


class EventSchemaRegistry:
    """Which model validates which event type at which version, and how to move between them.

    The kit's own registry is `EVENT_SCHEMAS`, populated at import with every built-in
    payload. A consumer builds its own only to test a contract change in isolation.
    """

    __slots__ = ("_models", "_upcasters")

    def __init__(self) -> None:
        self._models: dict[tuple[EventType, int], type[EventPayload]] = {}
        self._upcasters: dict[tuple[EventType, int], Upcaster] = {}

    def register(self, model: type[EventPayload]) -> None:
        """Publish `model` as the schema for its type at its declared version.

        Raises:
            ConfigurationError: That type and version already has a model. Two models for
                one version is two contracts under one name.
        """
        key = (model.type, model.version)
        if key in self._models:
            raise ConfigurationError(
                f"{model.type.value} version {model.version} is already registered to "
                f"{self._models[key].__name__}; a version is a contract, not a name"
            )
        self._models[key] = model

    def register_upcaster(
        self,
        event_type: EventType,
        *,
        from_version: int,
        upcast: Upcaster,
    ) -> None:
        """Teach the registry how to read `from_version` as the version after it."""
        self._upcasters[(event_type, from_version)] = upcast

    def versions(self, event_type: EventType) -> tuple[int, ...]:
        """Every version of `event_type` this kit can read, oldest first."""
        return tuple(sorted(version for kind, version in self._models if kind is event_type))

    def current_version(self, event_type: EventType) -> int:
        """The version this kit publishes.

        Raises:
            UnknownEventTypeError: Nothing registered a schema for this type.
        """
        found = self.versions(event_type)
        if not found:
            raise UnknownEventTypeError(
                f"{event_type} has no registered schema, so no consumer has been told "
                f"what it means; register a payload model before publishing it",
                event_type=str(event_type),
            )
        return found[-1]

    def model_for(self, event_type: EventType, version: int) -> type[EventPayload]:
        """The model that validates this type at this version.

        Raises:
            UnsupportedEventVersionError: This kit cannot read that version.
        """
        model = self._models.get((event_type, version))
        if model is None:
            raise UnsupportedEventVersionError(
                f"{event_type} version {version} is outside the versions this kit "
                f"reads; park the message rather than guessing what it meant",
                event_type=str(event_type),
                version=version,
                supported=self.versions(event_type),
            )
        return model

    def upcast(self, envelope: EventEnvelope) -> EventEnvelope:
        """The same event read as the current version, one registered step at a time.

        Raises:
            UnsupportedEventVersionError: A step has no upcaster, or the event is newer
                than this kit. Guessing across a missing step invents data.
        """
        current = self.current_version(envelope.type)
        if envelope.schema_version > current:
            raise UnsupportedEventVersionError(
                f"{envelope.type.value} version {envelope.schema_version} is newer than the "
                f"{current} this kit reads; park it for a consumer that has been upgraded",
                event_type=envelope.type.value,
                version=envelope.schema_version,
                supported=self.versions(envelope.type),
                tenant=envelope.tenant,
            )
        attributes = dict(envelope.attributes)
        for version in range(envelope.schema_version, current):
            step = self._upcasters.get((envelope.type, version))
            if step is None:
                raise UnsupportedEventVersionError(
                    f"nothing can upcast {envelope.type.value} from version {version} to "
                    f"{version + 1}; register the step or drop the version at a major",
                    event_type=envelope.type.value,
                    version=version,
                    supported=self.versions(envelope.type),
                    tenant=envelope.tenant,
                )
            attributes = step(attributes)
        if envelope.schema_version == current:
            return envelope
        return envelope.model_copy(update={"schema_version": current, "attributes": attributes})

    def payload_of(self, envelope: EventEnvelope) -> EventPayload:
        """The envelope's body as the model for its version, validated.

        Raises:
            UnsupportedEventVersionError: This kit cannot read that version.
            ValidationError: The attributes are not what that version declared.
        """
        model = self.model_for(envelope.type, envelope.schema_version)
        # The envelope carries every attribute as a string, so reading one back is a lax parse.
        return model.model_validate(dict(envelope.attributes), strict=False)

    def read(self, raw: str | bytes) -> EventEnvelope:
        """Decode what another publisher wrote, tolerating what this kit has not heard of.

        Envelope fields this version does not know are dropped rather than refused, because
        a newer publisher on the same stream must not stop an older consumer. Unknown
        attributes are kept: they are the body, and the consumer may know what they mean.

        Raises:
            UnknownEventTypeError: The event names a type this kit has no schema for.
            UnsupportedEventVersionError: Its version is outside the readable window.
        """
        decoded = json.loads(raw)
        kind = str(decoded.get("type", ""))
        if kind not in set(EventType):
            raise UnknownEventTypeError(
                f"{kind!r} is not an event type this kit knows; park it rather than acting "
                f"on a contract nothing here has read",
                event_type=kind,
            )
        known = {name: decoded[name] for name in EventEnvelope.model_fields if name in decoded}
        return self.upcast(EventEnvelope.model_validate_json(json.dumps(known)))

    def requires(self, event_type: EventType) -> int:
        """The version to stamp on an event of this type, refusing an unregistered one.

        Raises:
            UnknownEventTypeError: Nothing registered a schema for this type.
        """
        return self.current_version(event_type)

    def schemas(self) -> dict[str, dict[str, Any]]:
        """Every registered contract as JSON Schema, keyed `type@version`, plus the envelope."""
        rendered: dict[str, dict[str, Any]] = {
            "envelope": EventEnvelope.model_json_schema(mode="serialization")
        }
        for (event_type, version), model in self._models.items():
            rendered[f"{event_type.value}@{version}"] = model.model_json_schema(
                mode="serialization"
            )
        return dict(sorted(rendered.items()))


EVENT_SCHEMAS = EventSchemaRegistry()
"""The kit's own registry, holding every event it publishes."""


def event_schemas() -> dict[str, dict[str, Any]]:
    """The kit's contracts as JSON Schema. Written to `docs/event-schemas.json` and released."""
    return EVENT_SCHEMAS.schemas()


def read_envelope(raw: str | bytes) -> EventEnvelope:
    """Decode an event another publisher wrote, upcast to the version this kit reads.

    Raises:
        UnknownEventTypeError: The event names a type this kit has no schema for.
        UnsupportedEventVersionError: Its version is outside the readable window.
    """
    return EVENT_SCHEMAS.read(raw)


def compatibility_breaks(
    previous: Mapping[str, Mapping[str, Any]], current: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    """Every way `current` breaks a consumer written against `previous`, one message each.

    Additive changes — a new optional field, a new enum member, a new version of a type —
    return nothing. Everything else names the schema and the field, because a build that
    fails without saying which field gets worked around rather than fixed.
    """
    breaks: list[str] = []
    for name, before in previous.items():
        after = current.get(name)
        if after is None:
            breaks.append(f"{name} was removed; a consumer subscribed to it stops working")
            continue
        breaks.extend(_field_breaks(name, before, after))
    return tuple(breaks)


def _field_breaks(name: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """What changed inside one schema that a consumer cannot absorb."""
    breaks: list[str] = []
    was = dict(before.get("properties", {}))
    now = dict(after.get("properties", {}))
    for field, shape in was.items():
        if field not in now:
            breaks.append(
                f"{name}.{field} was removed or renamed; within a major a field name keeps "
                f"its meaning, and a rename is both a removal and a reuse"
            )
            continue
        breaks.extend(_shape_breaks(f"{name}.{field}", shape, now[field]))
    added_required = set(after.get("required", ())) - set(before.get("required", ()))
    breaks.extend(
        f"{name}.{field} became required; an older publisher does not send it"
        for field in sorted(added_required)
    )
    return breaks


def _shape_breaks(name: str, was: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """A field that kept its name but not its meaning."""
    breaks: list[str] = []
    if was.get("type") != now.get("type"):
        breaks.append(
            f"{name} changed type from {was.get('type')!r} to {now.get('type')!r}; a "
            f"consumer parsing it fails on the first event"
        )
    lost = [member for member in was.get("enum", ()) if member not in now.get("enum", ())]
    if lost:
        breaks.append(
            f"{name} no longer accepts {lost}; a consumer may gain members it does not "
            f"recognise, never lose ones it handles"
        )
    return breaks


def _register_builtins() -> None:
    """Register every payload the kit ships, so nothing is publishable without a contract."""
    seen: list[type[EventPayload]] = []
    pending = list(EventPayload.__subclasses__())
    while pending:
        model = pending.pop()
        seen.append(model)
        pending.extend(model.__subclasses__())
    for model in seen:
        EVENT_SCHEMAS.register(model)


_register_builtins()
