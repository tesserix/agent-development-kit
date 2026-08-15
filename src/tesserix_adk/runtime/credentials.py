"""Keeping a long run's authority current without restarting the run.

Short-lived credentials make long runs fragile: a run that outlives its token fails
partway through, and the usual recoveries — a longer ttl, or retrying the whole run —
trade away either the short lifetime or the side effects already committed. So the run
refreshes in place, and re-derives its authority when it does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from pydantic import Field

from tesserix_adk.core import (
    AdkModel,
    AgentIdentity,
    AuthorisationError,
    AuthorityRevokedError,
    CredentialExpiredError,
    McpAuthError,
    McpAuthReason,
    RetryConfig,
    ScopeSet,
)
from tesserix_adk.runtime.retry import RetryPlan

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from random import Random

    from tesserix_adk.core import Clock, ExpiringCredential, ExpiringCredentialSource

__all__ = ["DEFAULT_SKEW_SECONDS", "Reauthoriser", "RefreshPolicy", "RunCredentials"]

DEFAULT_SKEW_SECONDS = 30.0
"""How far ahead of expiry a credential is replaced, absent a configured window."""

T = TypeVar("T")


@runtime_checkable
class Reauthoriser(Protocol):
    """Whatever answers "does this caller still hold what they held" mid-run."""

    async def reauthorise(self, identity: AgentIdentity, /) -> AgentIdentity:
        """The caller's current authority.

        Raises:
            AuthorisationError: If the caller no longer has any.
        """


class RefreshPolicy(AdkModel):
    """When to refresh, and how hard to try.

    Args:
        skew_seconds: How far ahead of expiry to replace a credential. Wide enough to
            cover the clock difference between here and the far side, since a credential
            that is fresh by this clock and stale by theirs is rejected downstream.
        retry: How a failing mint is retried. The delay is full-jittered, so a fleet of
            runs sharing one principal does not refresh in unison.
    """

    skew_seconds: float = Field(default=DEFAULT_SKEW_SECONDS, ge=0.0)
    retry: RetryConfig = RetryConfig(max_attempts=3, base_delay_seconds=0.1)


class RunCredentials:
    """One run's credentials: refreshed ahead of expiry, re-derived when they are.

    Args:
        source: Whatever mints credentials.
        identity: Who the run acts for, as resolved when it started.
        clock: What decides whether a credential is still good.
        policy: When to refresh and how hard to try.
        reauthorise: What re-derives the caller's authority on each mint. Absent, the
            authority the run started with is used, still checked for expiry.
        random: The source of backoff jitter, injected so a test can seed it.
    """

    def __init__(
        self,
        source: ExpiringCredentialSource,
        *,
        identity: AgentIdentity,
        clock: Clock,
        policy: RefreshPolicy | None = None,
        reauthorise: Reauthoriser | None = None,
        random: Random | None = None,
    ) -> None:
        self._source = source
        self._identity = identity
        self._clock = clock
        self._policy = policy or RefreshPolicy()
        self._reauthorise = reauthorise
        self._plan = RetryPlan(self._policy.retry, random=random)
        self._held: dict[str, ExpiringCredential] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._halted = ""

    @property
    def identity(self) -> AgentIdentity:
        """The authority the run currently holds, which only ever narrows."""
        return self._identity

    @property
    def halted(self) -> bool:
        """Whether the run's authority is gone and no further call may be dispatched."""
        return bool(self._halted)

    @property
    def held(self) -> int:
        """How many credentials are cached."""
        return len(self._held)

    def invalidate(self, audience: str) -> None:
        """Drop what is held for `audience`, so the next call mints."""
        self._held.pop(audience, None)

    def suspend(self) -> None:
        """Forget every credential, because a suspended run may resume days later.

        A token carried across a suspension is a token that outlived the window it was
        minted for. The authority is re-derived on resume instead.
        """
        self._held.clear()

    async def for_call(
        self,
        *,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> ExpiringCredential:
        """The credential for one call, minted or refreshed if the held one is near expiry.

        Raises:
            AuthorityRevokedError: If the run's authority is gone. The run stops here.
            AuthorisationError: If the call asks for more than the caller holds.
            CredentialExpiredError: If the mint could not be reached. Transient, and the
                same call is worth making again.
        """
        wanted = tuple(needs)
        held = self._held.get(audience)
        if held is not None and self._usable(held, wanted):
            return held
        return await self._refresh(
            audience=audience, needs=wanted, run_id=run_id, agent_version=agent_version
        )

    async def call(
        self,
        operation: Callable[[ExpiringCredential, str], Awaitable[T]],
        *,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        idempotency_key: str | None = None,
    ) -> T:
        """Run `operation` with a live credential, refreshing once if the far side refuses.

        Args:
            operation: What to do with the credential and the idempotency key.
            audience: What the credential is for.
            needs: What the call needs to be allowed to do.
            run_id: The run making the call.
            idempotency_key: The key the call is made under. The retry after a refresh
                reuses it, so a refreshed call cannot duplicate a side effect. Without
                one there is no retry at all.

        Raises:
            CredentialExpiredError: With `outcome="unknown"` where a call carrying no
                idempotency key was already in flight. It may have landed, so it is
                reported as unknown rather than repeated or assumed failed.
        """
        credential = await self.for_call(audience=audience, needs=needs, run_id=run_id)
        try:
            return await operation(credential, idempotency_key or "")
        except (CredentialExpiredError, McpAuthError) as rejected:
            if not _is_expiry(rejected):
                raise
            if idempotency_key is None:
                raise CredentialExpiredError(
                    f"the credential for {audience} lapsed with a call in flight and no "
                    f"idempotency key, so whether it landed is unknown",
                    audience=audience,
                    outcome="unknown",
                ) from rejected
            self.invalidate(audience)
            fresh = await self.for_call(audience=audience, needs=needs, run_id=run_id)
            return await operation(fresh, idempotency_key)

    async def _refresh(
        self, *, audience: str, needs: Iterable[str], run_id: str, agent_version: str
    ) -> ExpiringCredential:
        """Mint once for this audience, however many callers are waiting on it."""
        wanted = tuple(needs)
        lock = self._locks.setdefault(audience, asyncio.Lock())
        async with lock:
            held = self._held.get(audience)
            if held is not None and self._usable(held, wanted):
                return held
            identity = await self._current()
            identity.check(wanted, where=audience)
            minted = await self._mint(
                identity=identity,
                audience=audience,
                needs=wanted,
                run_id=run_id,
                agent_version=agent_version,
            )
            self._held[audience] = minted
            return minted

    def _usable(self, held: ExpiringCredential, wanted: tuple[str, ...]) -> bool:
        """Whether what is held is both live and wide enough for the call being made."""
        if held.expired(self._clock.now(), skew=self._policy.skew_seconds):
            return False
        return ScopeSet.of(*wanted).names <= held.scopes

    async def _current(self) -> AgentIdentity:
        """The caller's authority now, re-derived where something can answer that.

        Raises:
            AuthorityRevokedError: If the authority has lapsed or been withdrawn.
        """
        if self._halted:
            raise AuthorityRevokedError(
                f"the run is halted: {self._halted}",
                agent=self._identity.agent,
                subject=self._identity.principal.subject,
            )
        try:
            self._identity.check_live(self._clock.now(), where="refresh")
            if self._reauthorise is not None:
                current = await self._reauthorise.reauthorise(self._identity)
                self._identity = current.narrowed(
                    agent=current.agent, declared=self._identity.effective
                )
        except AuthorisationError as withdrawn:
            raise self._halt(withdrawn) from withdrawn
        return self._identity

    async def _mint(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: tuple[str, ...],
        run_id: str,
        agent_version: str,
    ) -> ExpiringCredential:
        """Ask the mint, retrying a transient failure with a jittered backoff."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._source.for_tool(
                    identity=identity,
                    audience=audience,
                    needs=sorted(ScopeSet.of(*needs)),
                    run_id=run_id,
                    agent_version=agent_version,
                )
            except AuthorisationError as refused:
                raise self._halt(refused) from refused
            except Exception as failure:
                delay = self._plan.delay_for(attempt)
                if delay is None:
                    raise CredentialExpiredError(
                        f"no credential for {audience} could be minted in "
                        f"{attempt} attempts, so the call was not made",
                        audience=audience,
                        scopes=needs,
                    ) from failure
                await self._clock.sleep(delay)

    def _halt(self, withdrawn: AuthorisationError) -> AuthorityRevokedError:
        """Stop the run, and say what stopped it."""
        self._halted = str(withdrawn)
        self._held.clear()
        return AuthorityRevokedError(
            str(withdrawn),
            agent=self._identity.agent,
            subject=self._identity.principal.subject,
            where="refresh",
        )


def _is_expiry(rejected: CredentialExpiredError | McpAuthError) -> bool:
    """Whether a refusal from the far side is one a fresh credential would fix."""
    if isinstance(rejected, McpAuthError):
        return rejected.reason is McpAuthReason.EXPIRED
    return rejected.outcome == "not_started"
