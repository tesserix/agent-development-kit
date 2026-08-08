"""What repeating a tool call does, said precisely enough for the runtime to act on it.

A timeout is not a failure: it is the absence of an answer, and the side effect may well
have landed. Without a declaration and a key, the runtime's only options are to retry and
risk a second booking or to give up and strand the run. Both are wrong often enough to
matter, so a tool says which of the three it is and the kit derives a key it can hold.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Claim",
    "Idempotency",
    "IdempotencyPolicy",
    "IdempotencyStore",
    "idempotency_key",
]


class Idempotency(StrEnum):
    """What happens when the same call runs twice.

    `READ_ONLY` changes nothing, so repeating it is free. `IDEMPOTENT` changes something,
    but the second call lands on the same state as the first. `EFFECTFUL` is the one that
    needs a key: repeating it books a second seat.
    """

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    EFFECTFUL = "effectful"


@dataclass(frozen=True, slots=True)
class IdempotencyPolicy:
    """A tool's declaration about repeating it, and what identifies one call.

    Args:
        kind: Which of the three this tool is.
        key_arguments: Which arguments identify the call, where only some do. Empty means
            all of them, which is right until a payload carries a fresh request id on
            every attempt — then the retry looks like a different call and the key stops
            being one.
    """

    kind: Idempotency = Idempotency.EFFECTFUL
    key_arguments: tuple[str, ...] = ()

    @property
    def deduplicated(self) -> bool:
        """Whether the dispatcher holds a record for this tool at all."""
        return self.kind is not Idempotency.READ_ONLY


@dataclass(frozen=True, slots=True)
class Claim:
    """What the store said when a call asked to run.

    Args:
        outcome: What a previous call recorded, where one did. Present means do not run.
        in_flight: Whether another caller holds the key right now and has not finished.
    """

    outcome: str | None = None
    in_flight: bool = False


@runtime_checkable
class IdempotencyStore(Protocol):
    """Where the record of an executed side effect lives.

    Implementations are tenant-scoped and hold the key, not the arguments. The guarantee
    is at-most-once per key within the retention window: a record that has expired is
    treated as unknown, not as permission.
    """

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:
        """Claim `key` for this caller, or say who already has it.

        Returns a reservation carrying the recorded outcome where the call has already
        run, `in_flight` where another caller is running it now, and neither where this
        caller may proceed — in which case the key is held until `record` or `abandon`.
        """
        ...

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:
        """Record what the call returned, so a repeat returns it rather than running."""
        ...

    async def abandon(self, key: str, *, tenant: str) -> None:
        """Release `key` without a record, because the call did not complete."""
        ...

    async def forget(self, *, tenant: str) -> int:
        """Remove every record for `tenant`, and return how many. Erasure reaches here."""
        ...


def idempotency_key(
    *,
    tenant: str,
    run_id: str,
    tool: str,
    arguments: Mapping[str, Any],
    key_arguments: tuple[str, ...] = (),
) -> str | None:
    """Derive the key identifying one side effect, or `None` where it cannot be derived.

    The key is a digest, so nothing recoverable from the arguments is written to a store
    or read by an operator. Two payloads that mean the same thing derive the same key:
    keys are ordered, text is NFC-normalised, and a float is written the one way that
    round-trips.

    The call id is deliberately not part of it. Two concurrent identical calls carry
    different ids and are the duplicate this exists to collapse.

    Returns:
        The key, or `None` where a named key argument is absent — which the caller must
        treat as no guarantee rather than as a fresh call.
    """
    if key_arguments:
        missing = [name for name in key_arguments if name not in arguments]
        if missing:
            return None
        arguments = {name: arguments[name] for name in key_arguments}
    payload = json.dumps(
        {"tenant": tenant, "run": run_id, "tool": tool, "arguments": _canonical(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _canonical(value: Any) -> Any:
    """The one spelling of a value, so two payloads that mean the same thing digest alike."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return repr(value) if value != int(value) else f"{int(value)}.0"
    if isinstance(value, dict):
        return {_canonical(name): _canonical(held) for name, held in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(held) for held in value]
    return value
