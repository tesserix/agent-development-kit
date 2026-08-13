"""What a run has been permitted, bound to exactly the payload a human was shown.

An approval that is not bound to its arguments is a licence: the repair loop that fixes a
malformed amount, the retry that re-sends the call, and the tool result that suggests a
larger refund all execute under a decision nobody made about them. The ledger holds a grant
to one payload, once, for one run.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import ApprovalBindingError
from tesserix_adk.core.hooks import ApprovalDecision, digest_of

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from tesserix_adk.core.hooks import ApprovalRecord, ApprovalTransport
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_APPROVAL_WAIT_SECONDS",
    "TIMEOUT_IDENTITY",
    "ApprovalLedger",
    "TransportGate",
    "self_granted",
]

DEFAULT_APPROVAL_WAIT_SECONDS = 900.0

# Not a person, and it must not read as one: nobody decided this call.
TIMEOUT_IDENTITY = "system:timeout"

# What a machine identity is written as, so an agent approving itself is recognisable
# whichever convention the caller's directory uses.
_MACHINE_PREFIXES = ("agent:", "agent/", "service:", "service/", "bot:", "sa:")


class ApprovalLedger:
    """The grants one run is holding, and what each of them covers."""

    def __init__(self) -> None:
        self._granted: dict[str, str] = {}
        self._spent: set[str] = set()
        self._void = False

    def bind(self, record: ApprovalRecord) -> None:
        """Record that `record` was granted, for the payload it was raised over."""
        self._granted[record.id] = record.arguments_digest

    def spend(self, record: ApprovalRecord, arguments: Mapping[str, Any]) -> None:
        """Use the grant for `record` to execute `arguments`, exactly once.

        Raises:
            ApprovalBindingError: If the run was cancelled, if nothing granted this record,
                if the grant has already been used, or if the arguments are not the ones it
                was raised over. Every one of them fails closed and needs a fresh decision.
        """
        if self._void:
            raise ApprovalBindingError(
                f"approval for {record.tool_name!r} belongs to a run that is over; "
                f"a decision nobody is waiting on executes nothing"
            )
        granted = self._granted.get(record.id)
        if granted is None:
            raise ApprovalBindingError(f"approval for {record.tool_name!r} was never granted")
        if record.id in self._spent:
            raise ApprovalBindingError(
                f"approval for {record.tool_name!r} was already used; one decision is one "
                f"execution, so a replayed answer buys nothing"
            )
        if digest_of(arguments) != granted:
            raise ApprovalBindingError(
                f"the arguments for {record.tool_name!r} are not the ones that were "
                f"approved; permission is for one payload, not for the tool"
            )
        self._spent.add(record.id)

    def void(self) -> None:
        """Invalidate every grant, because the run they belong to has ended."""
        self._void = True


class _Wall:
    """Wall-clock time, for a gate nobody injected one into."""

    def now(self) -> float:
        """Return Unix seconds."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Suspend for `seconds`."""
        await asyncio.sleep(seconds)


def self_granted(record: ApprovalRecord, decision: ApprovalDecision) -> bool:
    """Whether the answer names the agent that asked the question.

    An agent's own service identity approving its own payment is not a second pair of
    eyes, and it is the shape an over-broad token takes in practice.
    """
    who = decision.decided_by.strip().casefold()
    for prefix in _MACHINE_PREFIXES:
        if who.startswith(prefix):
            who = who[len(prefix) :]
            break
    return who.strip() == record.agent_name.strip().casefold()


class TransportGate:
    """An `ApprovalGate` that asks over a transport and holds the call for the answer.

    Delivery is substitutable and waiting is not: whichever transport the question goes
    out on, a run waits here, one answer settles one request, and silence past
    `wait_seconds` is a denial rather than a grant.

    Args:
        transport: Where the question goes.
        wait_seconds: How long a call is held before the gate answers for the absent
            approver. Denial, always.
        clock: Time source. Injected so a test does not sleep.
    """

    def __init__(
        self,
        transport: ApprovalTransport,
        *,
        wait_seconds: float = DEFAULT_APPROVAL_WAIT_SECONDS,
        clock: Clock | None = None,
    ) -> None:
        self._transport = transport
        self._wait = wait_seconds
        self._clock: Clock = clock or _Wall()
        self._waiting: dict[str, asyncio.Future[ApprovalDecision]] = {}

    @property
    def waiting(self) -> tuple[str, ...]:
        """The requests currently held, for a UI poll and for a test."""
        return tuple(self._waiting)

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Deliver `record` and return the decision about it.

        Raises:
            Exception: Whatever the transport raised. A question nobody received is a
                failure rather than a decision, so it never becomes a grant.
        """
        pending: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._waiting[record.id] = pending
        try:
            answered = await self._transport.deliver(record)
            return answered if answered is not None else await self._held(record, pending)
        finally:
            self._waiting.pop(record.id, None)

    def decide(self, decision: ApprovalDecision) -> bool:
        """Answer a held call, returning whether anything was still waiting for it.

        A repeated answer, and an answer to a run that has ended, are both stale: they
        settle nothing, because the decision they would settle is already spent.
        """
        pending = self._waiting.get(decision.record_id)
        if pending is None or pending.done():
            return False
        pending.set_result(decision)
        return True

    def abandon(self, record_id: str, *, why: str) -> None:
        """Stop holding a call, because the run waiting on it has ended."""
        pending = self._waiting.pop(record_id, None)
        if pending is not None and not pending.done():
            pending.cancel(why)

    async def _held(
        self, record: ApprovalRecord, pending: asyncio.Future[ApprovalDecision]
    ) -> ApprovalDecision:
        """Wait for an answer, and treat running out of patience as a refusal."""
        expiry = asyncio.ensure_future(self._clock.sleep(self._wait))
        racing: set[asyncio.Future[Any]] = {pending, expiry}
        try:
            await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
        finally:
            expiry.cancel()
        if pending.done():
            return pending.result()
        return ApprovalDecision(
            record_id=record.id,
            granted=False,
            decided_by=TIMEOUT_IDENTITY,
            decided_at=self._clock.now(),
            reason=f"nobody decided within {self._wait:g}s, and silence is not permission",
        )
