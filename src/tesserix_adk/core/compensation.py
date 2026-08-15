"""What a run applied to the world, and what it takes to put it back.

A multi-step agent gets halfway: the hotel is booked and then the flight call fails. Without
a reversal path the consumer is left holding real-world state the run never accounted for,
and the tempting shortcut is to let the model choose what to undo. That is the boundary this
module holds — unwinding applied work is a transaction, not a judgement call, so the reversal
is declared next to the forward action and driven by the runtime in reverse order.

The ledger keeps the key and a digest of the arguments, never the arguments. A card number
that was safe to send to a payment provider is not safe to keep in a durable list of things
to undo, and the reversal needs the key rather than the payload.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

__all__ = [
    "Applied",
    "CompensatedFailure",
    "Compensating",
    "CompensationIncomplete",
    "CompensationLedger",
    "Reversal",
    "ReversalOutcome",
    "arguments_digest",
]


def arguments_digest(arguments: Mapping[str, JsonValue]) -> str:
    """A stable fingerprint of what a call was given, safe to keep next to its key.

    Argument order never changes it, so the same call digests the same way whichever
    provider ordered the keys.

    Example:
        >>> arguments_digest({"b": 2, "a": 1}) == arguments_digest({"a": 1, "b": 2})
        True
    """
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Applied(AdkModel):
    """One forward action that really happened, and enough to take it back.

    Args:
        run_id: The run that applied it.
        tenant: The isolation boundary. A reversal never spans one.
        branch: Which fan-out branch applied it. Branches unwind independently, so an
            action from one is never reversed while walking another. Empty is the run's
            own sequence.
        step: Where in the branch it happened. Unwinding walks these downwards.
        tool: What was called.
        idempotency_key: The key the forward call ran under. The reversal is issued
            against this rather than against a fresh one, so a repeated unwind cancels
            the same booking rather than a second thing.
        arguments_digest: A fingerprint of the arguments, never the arguments.
        result_ref: What the call returned, as a reference a reversal can quote — a
            booking reference, a payment id. Empty means the result never came back, and
            an action in that state is queried before anything is reversed or retried.
        irreversible: Whether the world will not take this back. Money movement is the
            usual case. An irreversible action is never auto-compensated.
        applied_at: Unix seconds, on the recorder's clock, for the reconciliation report.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    branch: str = ""
    step: int = Field(default=0, ge=0)
    tool: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    arguments_digest: str = ""
    result_ref: str = ""
    irreversible: bool = False
    applied_at: float = 0.0

    @property
    def outcome_known(self) -> bool:
        """Whether anything can say what this call did. False is the case to query first."""
        return bool(self.result_ref)


class ReversalOutcome(StrEnum):
    """What happened when one applied action was taken back."""

    REVERSED = "reversed"
    ALREADY_REVERSED = "already_reversed"
    REFUSED_IRREVERSIBLE = "refused_irreversible"
    FAILED = "failed"
    UNKNOWN = "unknown"


_SETTLED = frozenset({ReversalOutcome.REVERSED, ReversalOutcome.ALREADY_REVERSED})


class Reversal(AdkModel):
    """One attempt at taking one applied action back.

    Args:
        applied: What was being reversed.
        outcome: How it went. Only `reversed` and `already_reversed` leave nothing behind.
        detail: What a person reconciling this needs to read — the supplier's refusal, the
            reason nothing could decide. Never the arguments.
        attempts: How many times the reversal was tried before it settled.
    """

    applied: Applied
    outcome: ReversalOutcome
    detail: str = ""
    attempts: int = Field(default=1, ge=0)

    @property
    def settled(self) -> bool:
        """Whether the world no longer holds this action."""
        return self.outcome in _SETTLED


class CompensatedFailure(AdkModel):
    """A run that failed and left nothing applied behind it.

    Args:
        run_id: The run that failed.
        tenant: Its isolation boundary.
        cause: What failed first, as the consumer's own error rendered it. The reversal is
            the response to this, so reporting one without the other explains nothing.
        reversals: Every action taken back, in the order they were taken back.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    cause: str
    reversals: tuple[Reversal, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Always false. A compensated run is a failed run whose mess was cleaned up."""
        return False


class CompensationIncomplete(AdkModel):
    """A run that failed and could not put everything back.

    The alert-grade state. It carries every action still applied so a person can reconcile
    by hand, and it is a different type from `CompensatedFailure` so no consumer can
    mistake it for a clean rollback.

    Args:
        run_id: The run that failed.
        tenant: Its isolation boundary.
        cause: What failed first.
        outstanding: Every action the world still holds, newest first.
        reversals: Every attempt made, including the ones that failed.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    cause: str
    outstanding: tuple[Applied, ...]
    reversals: tuple[Reversal, ...] = ()

    @model_validator(mode="after")
    def _something_is_actually_outstanding(self) -> Self:
        """Nothing outstanding is a completed compensation, and must be reported as one."""
        if not self.outstanding:
            raise ValueError("an incomplete compensation with nothing outstanding is complete")
        return self

    @property
    def succeeded(self) -> bool:
        """Always false, and more urgently so than a compensated failure."""
        return False


@runtime_checkable
class CompensationLedger(Protocol):
    """Where the record of applied, un-reversed work lives.

    Implementations are tenant-scoped. The guarantee is that an action recorded here is
    visible to whatever unwinds the run, including a different process after a restart —
    a ledger that loses a record loses the only reason anyone knows to cancel the hotel.
    """

    async def record(self, action: Applied) -> None:
        """Write down that `action` was applied. Recording the same one twice records one."""
        ...

    async def mark(self, reversal: Reversal) -> None:
        """Record an attempt at a reversal, and settle the action where it succeeded."""
        ...

    async def outstanding(
        self, run_id: str, *, tenant: str, branch: str | None = None
    ) -> Sequence[Applied]:
        """Every action the world still holds for this run, newest first.

        A branch narrows it to that branch alone, because two fan-out branches unwind
        independently and one must never reverse the other's work.
        """
        ...

    async def forget(self, *, tenant: str) -> int:
        """Remove every record for `tenant`, and return how many. Erasure reaches here."""
        ...


@runtime_checkable
class Compensating(Protocol):
    """What the runtime needs of a reversal, satisfied by `@compensating_activity`.

    `irreversible` is read before anything is applied, not after it has failed: an action
    nothing can take back has to be sequenced and gated rather than unwound.
    """

    irreversible: bool

    async def compensate(self, action: Applied) -> str:
        """Take `action` back and return a reference to the reversal.

        Raises:
            Exception: Whatever the reversal raised. The runtime records the attempt and
                carries on with the rest of the unwind rather than stopping at the first
                supplier that is down.
        """
        ...
