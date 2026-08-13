"""What a run has to have written down to stop for three days and carry on afterwards.

An approval that takes a person a working week is the normal case, and a run that holds a
worker open across it is a run that dies with the next deploy. So the loop stops instead:
it writes its frontier, hands out a token, and returns. Nothing is held — no task, no
connection, no queue slot — and the decision, whenever it arrives, resumes the original run
rather than starting a second one wearing its name.

The token is what binds a decision to a run. It is single-use, tenant-bound, expiring, and
carries the digest of the arguments the approver was shown, so an answer cannot be replayed
into a second execution or moved onto a different payload.

The decisions behind these types are in `docs/suspension.md`.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.hooks import ApprovalRecord  # noqa: TC001 — pydantic resolves this
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import ToolCall  # noqa: TC001 — pydantic resolves this

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "DEFAULT_SUSPENSION_SECONDS",
    "ApprovalToken",
    "PendingDecision",
    "SuspendedRun",
    "SuspensionStore",
    "TokenAttempt",
    "TokenRedeemer",
    "digest_of_token",
    "mint_token",
]

DEFAULT_SUSPENSION_SECONDS = 259_200.0
"""Three days. Long enough for a weekend and a Monday, short enough to be a deadline."""


def digest_of_token(value: str) -> str:
    """Return the SHA-256 of a token value, which is what a store may keep."""
    return hashlib.sha256(value.encode()).hexdigest()


class ApprovalToken(AdkModel):
    """A bearer secret that resolves to one suspended run and one held action.

    The value is a credential: whoever holds it can answer for the run it names, so it is
    handed to the approver and never written to a log. Stores keep `digest` instead.

    Args:
        value: The secret itself.
        record_id: The approval request it answers.
        run_id: The run it resumes.
        tenant: The isolation boundary it is bound to.
        arguments_digest: The payload the approver was shown.
        issued_at: Unix seconds.
        expires_at: Unix seconds. Past it the token buys a denial, never a grant.
    """

    value: str = Field(min_length=32)
    record_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    arguments_digest: str = Field(min_length=64, max_length=64)
    issued_at: float = 0.0
    expires_at: float = 0.0

    @property
    def digest(self) -> str:
        """What a store keeps in place of the value."""
        return digest_of_token(self.value)

    def expired_by(self, now: float) -> bool:
        """Whether `now` is past the window this token was good for."""
        return now >= self.expires_at


def mint_token(
    record: ApprovalRecord,
    *,
    issued_at: float = 0.0,
    ttl_seconds: float = DEFAULT_SUSPENSION_SECONDS,
) -> ApprovalToken:
    """Issue a token for `record`, good for `ttl_seconds` from `issued_at`."""
    return ApprovalToken(
        value=secrets.token_urlsafe(32),
        record_id=record.id,
        run_id=record.run_id,
        tenant=record.tenant,
        arguments_digest=record.arguments_digest,
        issued_at=issued_at,
        expires_at=issued_at + ttl_seconds,
    )


class SuspendedRun(AdkModel):
    """A run stopped on a question, and everything needed to carry it on.

    Held apart from the checkpoint because the two answer different questions: the
    checkpoint is where the conversation got to, and this is what is being asked of whom,
    under which token, until when.

    Args:
        run_id: The stopped run.
        tenant: The isolation boundary.
        agent_name: Which agent asked.
        record: What was asked, in the form the approver sees.
        call: The held call, so a resume dispatches exactly what was approved.
        token_digest: The token that resolves to this, by digest rather than by value.
        suspended_at: Unix seconds.
        expires_at: Unix seconds. Past it the run is closed as denied.
        model: The model in use when it stopped.
        prompt_version: The prompt version in use when it stopped.
        iterations: How far round the loop it had got.
        spent: Whether the decision has already been taken. A second one changes nothing.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    record: ApprovalRecord
    call: ToolCall
    token_digest: str = Field(min_length=64, max_length=64)
    suspended_at: float = 0.0
    expires_at: float = 0.0
    model: str = ""
    prompt_version: str = ""
    iterations: int = Field(default=0, ge=0)
    spent: bool = False

    def expired_by(self, now: float) -> bool:
        """Whether nobody answered in time, which is a denial rather than a wait."""
        return now >= self.expires_at

    def held_for(self, now: float) -> float:
        """How long the run has been stopped, in seconds, as at `now`."""
        return max(0.0, now - self.suspended_at)


class PendingDecision(AdkModel):
    """One line of somebody's 'waiting on you', carrying no argument values.

    What a queue outlives is the run, and who reads it is whoever is on rota; the digest
    and the summary are enough to decide with, and the account number is not theirs.

    Args:
        record_id: What to answer.
        run_id: The run it stops.
        tenant: The isolation boundary.
        agent_name: Who asked.
        tool_name: What they want to call.
        summary: What the approver is shown.
        reason: Why it needs a person.
        arguments_digest: What they are deciding about, by digest.
        asked_at: Unix seconds.
        expires_at: Unix seconds.
    """

    record_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    summary: str = ""
    reason: str = ""
    arguments_digest: str = Field(min_length=64, max_length=64)
    asked_at: float = 0.0
    expires_at: float = 0.0

    @classmethod
    def of(cls, suspended: SuspendedRun) -> Self:
        """Project what a person may see out of a suspended run."""
        return cls(
            record_id=suspended.record.id,
            run_id=suspended.run_id,
            tenant=suspended.tenant,
            agent_name=suspended.agent_name,
            tool_name=suspended.record.tool_name,
            summary=suspended.record.summary,
            reason=suspended.record.reason,
            arguments_digest=suspended.record.arguments_digest,
            asked_at=suspended.record.requested_at,
            expires_at=suspended.expires_at,
        )


class TokenAttempt(AdkModel):
    """Somebody presenting a token, and what came of it.

    Refusals are the point: a token presented twice, or by the wrong tenant, is the shape
    of an approval being replayed, and it is worth more than a raised exception.

    Args:
        run_id: The run named by the token, where it named a real one.
        tenant: Who presented it as.
        presented_by: The identity that presented it.
        at: Unix seconds.
        accepted: Whether it resolved to a live suspension.
        reason: Why not, where it did not.
    """

    run_id: str = ""
    tenant: str = Field(min_length=1)
    presented_by: str = Field(min_length=1)
    at: float = 0.0
    accepted: bool = False
    reason: str = ""


@runtime_checkable
class SuspensionStore(Protocol):
    """Where stopped runs wait, keyed by run and reachable by token digest."""

    async def put(self, suspended: SuspendedRun) -> None:
        """Store the suspension, replacing any earlier one for the run."""
        ...

    async def get(self, run_id: str, *, tenant: str) -> SuspendedRun | None:
        """Return the suspension of `run_id`, or `None` where the run is not stopped."""
        ...

    async def by_token(self, token_digest: str, *, tenant: str) -> SuspendedRun | None:
        """Return the suspension a token resolves to, within `tenant` and nowhere else."""
        ...

    async def spend(self, run_id: str, *, tenant: str) -> bool:
        """Mark the decision taken, returning `False` where somebody already took it."""
        ...

    async def pending(self, *, tenant: str) -> tuple[SuspendedRun, ...]:
        """Every run in `tenant` waiting on somebody, oldest first."""
        ...

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the suspension, because the run is going again."""
        ...

    async def attempted(self, attempt: TokenAttempt) -> None:
        """Record somebody presenting a token, accepted or not."""
        ...


@runtime_checkable
class TokenRedeemer(Protocol):
    """A gate that can turn a token back into the run it stopped."""

    async def redeem(self, token: str, *, tenant: str, presented_by: str) -> SuspendedRun:
        """Return the suspension `token` resolves to, once.

        Raises:
            ApprovalTokenError: If it is unknown to `tenant`, or already spent.
        """
        ...
