"""The record of what an agent did unattended, and of what it was not allowed to do.

Telemetry cannot answer the question asked after an autonomy incident: spans are sampled,
dropped under load and stripped of the context that made the decision. So this is a
separate store on a separate path, and a refusal is written with exactly the same weight as
an action — a ceiling nobody recorded holding is a ceiling nobody can show held.

The vocabulary lives in `core` because the runtime writes it before an adapter exists, the
same reason redaction does. What is written is a digest of the payload and never the
payload: an audit store outlives the run, is read by compliance rather than by the party to
the case, and is the last place a card number should turn up.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal  # noqa: TC003 — pydantic resolves field types at class creation
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.autonomy import AutonomyLevel
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.redaction import scrub

__all__ = [
    "AuditDecision",
    "AuditEvent",
    "AuditSink",
    "digest_of_arguments",
    "pseudonym",
]


class AuditDecision(StrEnum):
    """What was decided about one attempted action.

    `EXECUTED` is written before the call goes out, not after it returns: a record written
    afterwards is the record an outage eats. It says the call was cleared and by what — the
    tool's own outcome is on the run.
    """

    EXECUTED = "executed"
    ESCALATED = "escalated"
    REFUSED = "refused"
    REVOKED = "revoked"


class AuditEvent(AdkModel):
    """One decision about one attempted action, as it is stored.

    Args:
        run_id: The run the decision was taken in.
        sequence: Where this sits in that run's order. Monotonic per run, so concurrent
            branches are reconstructable; not gapless, because a retried write keeps the
            number the first attempt was stored under.
        tenant: The isolation boundary the action was taken in.
        user: On whose behalf, where a run has one. Pseudonymised by an erasure request.
        agent_name: Which agent attempted it.
        agent_version: At which version, so a decision reads against the agent that made it.
        tool: The call that was attempted.
        action_class: What that call does in the world, per the action registry.
        level: The autonomy level that applied.
        decision: Acted, escalated to a human, refused, or withdrawn under the run.
        reason: Why, in a form an audit reader can act on.
        grant_id: Which grant answered, where one did. Recorded on escalations too: the
            question of which grant was not enough is the one an operator asks first.
        headroom_before: What was left under the ceiling when the decision was taken.
        headroom_after: What this action would leave. Absent where no ceiling applied.
        approver: Who decided, where a human did. Absent means unattended, which is the
            whole point of the record. Pseudonymised by an erasure request.
        arguments_digest: The payload's digest, taken after redaction. Never the payload.
        idempotency_key: What makes the write idempotent under retry: one decision about
            one call yields one record however many times the activity runs.
        recorded_at: When the decision was taken.
    """

    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    tenant: str = Field(min_length=1)
    user: str | None = None
    agent_name: str = ""
    agent_version: str = ""
    tool: str = Field(min_length=1)
    action_class: str = Field(min_length=1)
    level: AutonomyLevel = AutonomyLevel.ASK_ALWAYS
    decision: AuditDecision
    reason: str = ""
    grant_id: str | None = None
    headroom_before: Decimal | None = None
    headroom_after: Decimal | None = None
    approver: str | None = None
    arguments_digest: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    recorded_at: float

    @property
    def unattended(self) -> bool:
        """Whether this action went out with nobody deciding it."""
        return self.decision is AuditDecision.EXECUTED and self.approver is None


@runtime_checkable
class AuditSink(Protocol):
    """Where decisions are appended, and read back by whoever is accountable for them.

    Deliberately not the telemetry pipeline. Sampling that is correct for spans is a
    missing record here, and a missing record is the case the store exists for.
    """

    async def append(self, event: AuditEvent) -> AuditEvent:
        """Record `event`, and return what is now stored for that decision.

        Idempotent on `(run_id, idempotency_key, decision)`: a retried write returns the
        record already there rather than adding a second one.

        Raises:
            Exception: Whatever the store raises. The caller fails the action closed.
        """
        ...

    async def records(
        self,
        *,
        tenant: str,
        since: float = 0.0,
        until: float | None = None,
        decision: AuditDecision | None = None,
    ) -> tuple[AuditEvent, ...]:
        """Every decision for `tenant` in the period, oldest first, declines included."""
        ...

    async def pseudonymise(self, *, tenant: str, subject: str) -> int:
        """Replace `subject` wherever it names a person, and return how many rows changed.

        The decision survives; the person does not. An erasure that removed the record
        would take the evidence that the action was permitted with it.
        """
        ...


def digest_of_arguments(arguments: Mapping[str, Any], *, extra_patterns: Sequence[str] = ()) -> str:
    """The digest an audit record carries in place of the payload.

    Redaction happens before the digest, not instead of it: the value never reaches this
    process's memory in a form that could be logged, and two payments of different amounts
    still digest differently because only the sensitive runs are masked.

    Args:
        arguments: What the call was to be made with.
        extra_patterns: Shapes this deployment knows about, such as a local case reference.

    Example:
        >>> digest_of_arguments({"amount": 900}) == digest_of_arguments({"amount": 900})
        True
    """
    canonical = json.dumps(_scrubbed(arguments, extra_patterns), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def pseudonym(subject: str, *, salt: str = "") -> str:
    """A stable stand-in for a person, for an audit record that outlives their consent.

    Stable so that a series of decisions by one person stays one series, salted so that two
    deployments cannot join their audit stores on it.

    Example:
        >>> pseudonym("ada@example.com") == pseudonym("ada@example.com")
        True
    """
    return f"anon:{hashlib.sha256(f'{salt}:{subject}'.encode()).hexdigest()[:16]}"


def _scrubbed(value: Any, patterns: Sequence[str]) -> Any:  # noqa: ANN401 — tool arguments are whatever the tool declared
    """The same structure with every sensitive-looking run masked, at any depth."""
    if isinstance(value, str):
        return scrub(value, patterns)
    if isinstance(value, Mapping):
        return {key: _scrubbed(held, patterns) for key, held in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrubbed(held, patterns) for held in value]
    return value
