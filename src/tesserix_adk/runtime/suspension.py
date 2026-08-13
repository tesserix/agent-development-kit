"""Stopping a run on a question, and resolving the answer back to it days later.

`DeferringGate` is an approval gate that does not wait. It puts the question on a transport,
issues a token, and tells the loop to stop — so a decision that takes a weekend costs a row
in a store rather than a worker and a connection. `redeem` is the other half: it turns a
token back into the run it stopped, once, for the tenant it was issued to.

The failure this exists to prevent is not a slow approval. It is the same decision arriving
twice and being executed twice.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import ApprovalTokenError
from tesserix_adk.core.suspension import (
    DEFAULT_SUSPENSION_SECONDS,
    SuspendedRun,
    TokenAttempt,
    digest_of_token,
    mint_token,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tesserix_adk.core.hooks import ApprovalDecision, ApprovalRecord, ApprovalTransport
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.core.suspension import ApprovalToken, SuspensionStore

__all__ = [
    "ApprovalDeferred",
    "DeferringGate",
    "MemorySuspensionStore",
]


class ApprovalDeferred(Exception):  # noqa: N818 — a signal to stop, not a failure
    """Raised by a gate to say the answer will arrive long after this process is gone.

    The loop reads it as an instruction to write the frontier down and return, which is
    why it is not an `AdkError`: nothing has gone wrong.

    Args:
        token: What resolves a later decision back to this run.
        store: Where the stopped run is to wait. Carried on the signal so that a gate is
            free to keep its suspensions wherever it likes without the loop being told.
    """

    def __init__(self, token: ApprovalToken, store: SuspensionStore) -> None:
        super().__init__(f"{token.record_id} will be decided out of band")
        self.token = token
        self.store = store


class MemorySuspensionStore:
    """A `SuspensionStore` in a dict, keyed by tenant and run.

    Nothing here outlives the process, which is what surviving a three-day approval is not.
    It exists so suspension can be exercised without a database.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[str, str], SuspendedRun] = {}
        self._attempts: list[TokenAttempt] = []

    async def put(self, suspended: SuspendedRun) -> None:
        """Store the suspension, replacing any earlier one for the run."""
        self._held[suspended.tenant, suspended.run_id] = suspended

    async def get(self, run_id: str, *, tenant: str) -> SuspendedRun | None:
        """Return the suspension of `run_id`, or `None` where the run is not stopped."""
        return self._held.get((tenant, run_id))

    async def by_token(self, token_digest: str, *, tenant: str) -> SuspendedRun | None:
        """Return the suspension a token resolves to, within `tenant` and nowhere else."""
        for (held_tenant, _), suspended in self._held.items():
            if held_tenant == tenant and suspended.token_digest == token_digest:
                return suspended
        return None

    async def spend(self, run_id: str, *, tenant: str) -> bool:
        """Mark the decision taken, returning `False` where somebody already took it."""
        suspended = self._held.get((tenant, run_id))
        if suspended is None or suspended.spent:
            return False
        self._held[tenant, run_id] = suspended.model_copy(update={"spent": True})
        return True

    async def pending(self, *, tenant: str) -> tuple[SuspendedRun, ...]:
        """Every run in `tenant` waiting on somebody, oldest first."""
        waiting = [
            suspended
            for (held_tenant, _), suspended in self._held.items()
            if held_tenant == tenant and not suspended.spent
        ]
        return tuple(sorted(waiting, key=lambda one: one.suspended_at))

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the suspension, because the run is going again."""
        self._held.pop((tenant, run_id), None)

    async def attempted(self, attempt: TokenAttempt) -> None:
        """Record somebody presenting a token, accepted or not."""
        self._attempts.append(attempt)

    @property
    def attempts(self) -> tuple[TokenAttempt, ...]:
        """Every token presented at this store, in order."""
        return tuple(self._attempts)


class _Wall:
    """Wall-clock time, for a gate nobody injected one into."""

    def now(self) -> float:
        """Return Unix seconds."""
        return time.time()


class DeferringGate:
    """An approval gate that stops the run instead of waiting on it.

    A transport that answers inline is honoured as the answer, because a console operator
    typing `y` should not cost a database round trip and a resume. Anything else defers.

    Args:
        transport: How the question reaches whoever decides.
        store: Where the stopped run waits.
        hand_to: How the token itself reaches whoever decides. Kept apart from the
            transport because the question is for a queue and the token is a credential:
            the two usually go to different places and are worth different care.
        ttl_seconds: How long the token is good for. Past it the run is closed as denied,
            because a question nobody answered in three days is not a question still open.
        clock: What stamps the suspension. Absent, wall time.
    """

    def __init__(
        self,
        transport: ApprovalTransport,
        store: SuspensionStore,
        *,
        hand_to: Callable[[ApprovalToken], Awaitable[None]] | None = None,
        ttl_seconds: float = DEFAULT_SUSPENSION_SECONDS,
        clock: Clock | None = None,
    ) -> None:
        self._transport = transport
        self._store = store
        self._hand_to = hand_to
        self._ttl = ttl_seconds
        self._clock: Clock | _Wall = clock or _Wall()

    @property
    def store(self) -> SuspensionStore:
        """Where this gate's stopped runs wait, for whoever resumes them."""
        return self._store

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Deliver the question, and stop the run unless the answer came back with it.

        Raises:
            ApprovalDeferred: Where nobody answered inline, carrying the token that
                resolves a later decision back to this run.
        """
        answered = await self._transport.deliver(record)
        if answered is not None:
            return answered
        token = mint_token(record, issued_at=self._clock.now(), ttl_seconds=self._ttl)
        if self._hand_to is not None:
            await self._hand_to(token)
        raise ApprovalDeferred(token, self._store)

    async def pending(self, *, tenant: str) -> tuple[SuspendedRun, ...]:
        """Every run in `tenant` waiting on somebody."""
        return await self._store.pending(tenant=tenant)

    async def redeem(self, token: str, *, tenant: str, presented_by: str) -> SuspendedRun:
        """Turn a token back into the run it stopped, once.

        Every presentation is recorded, accepted or not: a token presented twice is what a
        replayed approval looks like from here, and it is worth more than an exception.

        Raises:
            ApprovalTokenError: If the token is unknown to `tenant`, or the decision it
                names has already been taken.
        """
        digest = digest_of_token(token)
        suspended = await self._store.by_token(digest, tenant=tenant)
        if suspended is None:
            await self._refuse(tenant, presented_by, "no run in this tenant is waiting on it")
            raise ApprovalTokenError(
                "the token presented resolves to no suspended run in this tenant",
                presented_by=presented_by,
            )
        if not await self._store.spend(suspended.run_id, tenant=tenant):
            await self._refuse(
                tenant, presented_by, "the decision it names was already taken", suspended.run_id
            )
            raise ApprovalTokenError(
                f"the decision for {suspended.run_id} was already taken, "
                f"so this token buys nothing",
                run_id=suspended.run_id,
                presented_by=presented_by,
            )
        await self._store.attempted(
            TokenAttempt(
                run_id=suspended.run_id,
                tenant=tenant,
                presented_by=presented_by,
                at=self._clock.now(),
                accepted=True,
            )
        )
        return suspended

    async def _refuse(self, tenant: str, presented_by: str, why: str, run_id: str = "") -> None:
        """Record a presentation that bought nothing."""
        await self._store.attempted(
            TokenAttempt(
                run_id=run_id,
                tenant=tenant,
                presented_by=presented_by,
                at=self._clock.now(),
                accepted=False,
                reason=why,
            )
        )
