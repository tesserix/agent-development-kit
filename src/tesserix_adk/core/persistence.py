"""Persisted state outlives the code that wrote it, so it has to say what wrote it.

An upgrade that renames a field makes every in-flight checkpoint unreadable. The failure
does not look like a failure: the runs are simply never picked up again, and a rolling
deploy that was meant to be routine becomes a coordinated drain. So every persisted payload
the kit writes carries the version it was written at, migrations are registered per version
step and applied on read, and a payload from a version this reader postdates is left exactly
where it is for a worker that understands it.

Two rules keep two kit versions writing the same store during a rolling deploy: within a
major, a field may only be added and only as optional, and a field name is never reused for
a different meaning. What follows from those is that a payload carrying a field this reader
has no place for is preserved rather than dropped — an older worker that writes a record
back must not silently delete what a newer one put there.

The serialisation is canonical for the same reason a checkpoint has a digest: the same
state must produce the same bytes on any machine, and money must never round-trip through
a float, which is where a ledger stops adding up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import Field, ValidationError

from tesserix_adk.core.checkpoint import CHECKPOINT_FORMAT
from tesserix_adk.core.errors import (
    ConfigurationError,
    StateMigrationError,
    UnsupportedStateVersionError,
)
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "CURRENT_VERSIONS",
    "STATE_REGISTRY",
    "SUPPORTED_WINDOW",
    "Envelope",
    "StateKind",
    "StateMigration",
    "StateRegistry",
    "canonical_json",
    "packed",
    "revived",
    "unpacked",
]

ModelT = TypeVar("ModelT", bound=AdkModel)

_DATETIME = "$datetime"
_DECIMAL = "$decimal"


class StateKind(StrEnum):
    """A thing the kit persists. Migrations are registered per kind, not per store."""

    SESSION = "session"
    RUN = "run"
    CHECKPOINT = "checkpoint"
    WORK_ITEM = "work_item"


SUPPORTED_WINDOW = 2
"""How many versions back a reader accepts: this one, and the one it is replacing.

Two is the smallest number that lets a rolling deploy finish. A format change ships with
its migration and one minor of dual-read support, and anything older is drained rather than
guessed at.
"""

CURRENT_VERSIONS: Mapping[str, int] = {
    StateKind.SESSION: 1,
    StateKind.RUN: 1,
    StateKind.CHECKPOINT: CHECKPOINT_FORMAT,
    StateKind.WORK_ITEM: 1,
}
"""The version this build writes, per kind. A checkpoint's is the checkpoint format itself."""


def canonical_json(value: object) -> str:
    """Serialise state to the one form it hashes and compares by.

    Keys are ordered, whitespace is fixed, moments are tagged as moments and decimals as
    decimals. A decimal never becomes a float on the way through: an amount that lost a
    ulp in the store is a ledger that no longer adds up.

    Args:
        value: The state, as JSON-safe data plus datetimes and decimals.

    Returns:
        The canonical text.

    Raises:
        ValueError: A float JSON cannot represent — `nan` or an infinity.
        TypeError: Something that is not state at all, rather than its `repr`.

    Example:
        >>> canonical_json({"b": 1, "a": 2})
        '{"a":2,"b":1}'
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_tagged,
    )


def revived(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The counterpart of `canonical_json`: tagged moments and decimals back as themselves.

    Example:
        >>> revived(json.loads(canonical_json({"eur": Decimal("1.10")})))["eur"]
        Decimal('1.10')
    """
    return {name: _revived(value) for name, value in payload.items()}


@dataclass(frozen=True, slots=True)
class StateMigration:
    """One version step, applied on read.

    Args:
        kind: What it migrates.
        from_version: The version it reads.
        to_version: The version it produces. Always one more than `from_version`: a
            registry of single steps composes, and a registry of leaps has holes in it.
        migrate: The payload transformation. Pure, and free to fail — a migration that
            raises leaves the stored item untouched.
        note: What changed, for whoever reads the deprecation window later.
    """

    kind: str
    from_version: int
    to_version: int
    migrate: Callable[[dict[str, Any]], dict[str, Any]]
    note: str = ""


class Envelope(AdkModel):
    """A persisted payload and the version it was written at.

    Args:
        kind: What this is. The kit's own are `StateKind`; a consumer versioning its own
            records uses its own names in its own registry.
        schema_version: The version the writer wrote it at.
        payload: The record as data, including anything the writer knew and this reader
            does not.
        written_by: The kit version that wrote it, where the writer recorded one. For
            diagnosis only: nothing branches on it.
    """

    kind: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    written_by: str = ""

    @classmethod
    def around(
        cls,
        record: AdkModel,
        *,
        kind: str,
        preserved: Mapping[str, Any] | None = None,
        registry: StateRegistry | None = None,
    ) -> Envelope:
        """Wrap a record for storage at the version this build writes.

        Args:
            record: What to persist.
            kind: Which kind it is.
            preserved: Fields a newer writer put there that this reader has no place for,
                from `preserved()`. Passed back so an older worker writing a record does
                not silently delete what a newer one wrote. Omit to drop them deliberately.
            registry: Which registry decides the current version. The kit's by default.

        Returns:
            The envelope, ready for `to_json()`.
        """
        payload = json.loads(canonical_json(record.model_dump(mode="json")))
        return cls(
            kind=kind,
            schema_version=(registry or STATE_REGISTRY).version(kind),
            payload={**payload, **dict(preserved or {})},
        )

    def opened(self, model: type[ModelT]) -> ModelT:
        """The record this reader can make of the payload.

        Fields it has no place for are left out rather than forced in; `preserved()` is
        where they are.

        Raises:
            StateMigrationError: The payload will not make a record of this type at all.
        """
        unreadable = set(self.preserved(model))
        try:
            return model.model_validate_json(
                canonical_json(
                    {
                        name: _untagged(value)
                        for name, value in self.payload.items()
                        if name not in unreadable
                    }
                )
            )
        except ValidationError as invalid:
            raise StateMigrationError(
                f"stored {self.kind} at version {self.schema_version} does not make a "
                f"{model.__name__}",
                kind=self.kind,
                from_version=self.schema_version,
                to_version=self.schema_version,
            ) from invalid

    def preserved(self, model: type[ModelT]) -> dict[str, Any]:
        """Everything in the payload this reader has no place for.

        Both halves of that: a field the model does not declare, and a declared field
        whose value is outside what this build knows the field can be — a state a newer
        kit added, say. Neither is dropped and neither is remapped onto something this
        reader does understand, which would be a rolling deploy quietly rewriting state.
        """
        declared = model.model_fields
        return {
            name: value
            for name, value in self.payload.items()
            if name not in declared or _unrecognised(declared[name].annotation, value)
        }

    def to_json(self) -> str:
        """The canonical text this is stored as."""
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, text: str) -> Envelope:
        """An envelope back from what was stored.

        Raises:
            StateMigrationError: The text is not an envelope.
        """
        return cls.model_validate(_parsed(text, kind=""))

    def digest(self) -> str:
        """A stable digest of the whole envelope, equal for equal state on any machine."""
        return sha256(self.to_json().encode()).hexdigest()


class StateRegistry:
    """Which version each kind is written at, and how to read the versions before it.

    Args:
        current: The version this build writes, per kind.
        window: How many versions back it will read. See `SUPPORTED_WINDOW`.
    """

    def __init__(self, current: Mapping[str, int], *, window: int = SUPPORTED_WINDOW) -> None:
        self._current = dict(current)
        self._window = window
        self._steps: dict[tuple[str, int], StateMigration] = {}

    def version(self, kind: str) -> int:
        """The version this build writes for `kind`."""
        return self._current.get(kind, 1)

    def oldest(self, kind: str) -> int:
        """The oldest version this build will read. Anything older is drained, not guessed."""
        return max(1, self.version(kind) - self._window + 1)

    def register(self, migration: StateMigration) -> None:
        """Add one version step.

        Raises:
            ConfigurationError: The step spans more than one version, or one is registered
                for that step already. Two migrations for the same step means the payload a
                run resumes from depends on import order.
        """
        if migration.to_version != migration.from_version + 1:
            raise ConfigurationError(
                f"a migration moves one version at a time, and this one moves {migration.kind} "
                f"from {migration.from_version} to {migration.to_version}"
            )
        step = (migration.kind, migration.from_version)
        if step in self._steps:
            raise ConfigurationError(
                f"a migration for {migration.kind} version {migration.from_version} is already "
                f"registered"
            )
        self._steps[step] = migration

    def upgraded(self, envelope: Envelope) -> Envelope:
        """The payload at the version this build reads.

        Args:
            envelope: What was stored.

        Returns:
            The same envelope where it is already current, or a new one with every step
            applied in order.

        Raises:
            UnsupportedStateVersionError: It postdates this build, predates the supported
                window, or needs a step nothing registered. The stored item is untouched
                in every case, for a worker that understands it to claim.
            StateMigrationError: A step raised. The stored item is untouched.
        """
        current = self.version(envelope.kind)
        if envelope.schema_version == current:
            return envelope
        self._within_the_window(envelope, current)
        payload = dict(envelope.payload)
        for version in range(envelope.schema_version, current):
            payload = self._stepped(envelope, payload, version)
        return envelope.model_copy(update={"schema_version": current, "payload": payload})

    def _within_the_window(self, envelope: Envelope, current: int) -> None:
        """Refuse a version this build has no business rewriting."""
        if envelope.schema_version > current:
            raise UnsupportedStateVersionError(
                f"stored {envelope.kind} is at version {envelope.schema_version}, which is "
                f"newer than the {current} this build reads; leaving it for a worker that "
                f"understands it",
                kind=envelope.kind,
                found=envelope.schema_version,
                supported=current,
                oldest=self.oldest(envelope.kind),
            )
        if envelope.schema_version < self.oldest(envelope.kind):
            raise UnsupportedStateVersionError(
                f"stored {envelope.kind} is at version {envelope.schema_version}, older than "
                f"the {self.oldest(envelope.kind)} this build supports; drain work at that "
                f"version on the last build that read it, then upgrade",
                kind=envelope.kind,
                found=envelope.schema_version,
                supported=current,
                oldest=self.oldest(envelope.kind),
            )

    def _stepped(self, envelope: Envelope, payload: dict[str, Any], version: int) -> dict[str, Any]:
        """One version step, or a refusal that leaves the stored item exactly as it was."""
        migration = self._steps.get((envelope.kind, version))
        if migration is None:
            raise UnsupportedStateVersionError(
                f"nothing knows how to read {envelope.kind} version {version} into {version + 1}",
                kind=envelope.kind,
                found=envelope.schema_version,
                supported=self.version(envelope.kind),
                oldest=self.oldest(envelope.kind),
            )
        try:
            return migration.migrate(payload)
        except Exception as failure:
            raise StateMigrationError(
                f"migrating {envelope.kind} from {version} to {version + 1} failed, and the "
                f"stored item is untouched",
                kind=envelope.kind,
                from_version=version,
                to_version=version + 1,
            ) from failure


STATE_REGISTRY = StateRegistry(CURRENT_VERSIONS)
"""The kit's own registry. A consumer versioning its own records builds its own."""


def packed(
    record: AdkModel,
    *,
    kind: str,
    preserved: Mapping[str, Any] | None = None,
    registry: StateRegistry | None = None,
) -> str:
    """A record as the text a store holds, versioned and canonical."""
    return Envelope.around(record, kind=kind, preserved=preserved, registry=registry).to_json()


def unpacked[ModelT: AdkModel](
    text: str, model: type[ModelT], *, kind: str, registry: StateRegistry | None = None
) -> ModelT:
    """A record back from what a store held, migrated on the way.

    Text written before envelopes existed is read as version one, so a store that predates
    this does not have to be drained to be readable.

    Raises:
        UnsupportedStateVersionError: The payload is outside what this build reads.
        StateMigrationError: The text is not a payload, or will not make this record.
    """
    stored = _parsed(text, kind=kind)
    envelope = (
        Envelope.model_validate(stored)
        if _is_envelope(stored)
        else Envelope(kind=kind, schema_version=1, payload=stored)
    )
    return (registry or STATE_REGISTRY).upgraded(envelope).opened(model)


def _is_envelope(stored: Mapping[str, Any]) -> bool:
    """Whether this was written by a build that versioned what it wrote."""
    return {"kind", "schema_version", "payload"} <= set(stored)


def _parsed(text: str, *, kind: str) -> dict[str, Any]:
    """What was stored, as data, or a refusal naming what it is not."""
    try:
        stored = json.loads(text)
    except json.JSONDecodeError as invalid:
        raise StateMigrationError(
            f"stored {kind or 'state'} is not JSON, so nothing can be read out of it",
            kind=kind,
        ) from invalid
    if not isinstance(stored, dict):
        raise StateMigrationError(
            f"stored {kind or 'state'} is a JSON {type(stored).__name__} rather than a record",
            kind=kind,
        )
    return stored


def _unrecognised(annotation: object, value: object) -> bool:
    """Whether a declared field's value is outside what this build knows it can be."""
    if not isinstance(annotation, type) or not issubclass(annotation, Enum):
        return False
    return all(value != member.value for member in annotation)


def _tagged(value: object) -> dict[str, str]:
    """A moment or an amount, in a form that comes back as itself."""
    if isinstance(value, datetime):
        return {_DATETIME: value.isoformat()}
    if isinstance(value, Decimal):
        return {_DECIMAL: str(value)}
    raise TypeError(f"{type(value).__name__} is not state the kit knows how to persist")


def _untagged(value: object) -> Any:  # noqa: ANN401 — whatever was persisted
    """A tagged moment or amount as the JSON scalar the model parses it from."""
    if isinstance(value, dict):
        if _DATETIME in value:
            return str(value[_DATETIME])
        if _DECIMAL in value:
            return str(value[_DECIMAL])
        return {name: _untagged(nested) for name, nested in value.items()}
    if isinstance(value, list):
        return [_untagged(nested) for nested in value]
    return value


def _revived(value: object) -> Any:  # noqa: ANN401 — whatever was persisted
    """One value back from its canonical form."""
    if isinstance(value, dict):
        if _DATETIME in value:
            return datetime.fromisoformat(str(value[_DATETIME]))
        if _DECIMAL in value:
            return Decimal(str(value[_DECIMAL]))
        return {name: _revived(nested) for name, nested in value.items()}
    if isinstance(value, list):
        return [_revived(nested) for nested in value]
    return value
