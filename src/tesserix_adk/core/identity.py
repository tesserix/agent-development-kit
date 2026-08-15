"""Whose authority an agent runs on, and why it can never be more than the caller's.

The service account an agent is given in production holds the union of everything any user
might need, because that is the account somebody could get provisioned. A low-privilege
customer conversation then executes against it, and nothing in the code says the agent is
acting for that customer — so nothing refuses the tool call that reads every tenant's
bookings. The escalation is not a bug in a tool; it is the absence of a statement.

Here the statement is a value. A `Principal` is who the run acts for and what they hold.
An agent declares the scopes it needs, and `AgentIdentity.resolve` intersects the two once,
at run start. The result is frozen: what a run may do cannot change while it is in flight,
and there is no operation on this module's types that widens a scope set. `narrowed` is the
only way to build an identity from another, so a peer agent with a broader declaration
cannot proxy around the caller who was refused.

Absence fails closed. An agent that declares scopes and finds no principal bound does not
fall back to the process identity to keep the run going, because that fallback is exactly
the escalation. An agent that declares none is unchanged: it holds no authority to spend.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import field_validator

from tesserix_adk.core.errors import AuthorisationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.tool_access import canonical

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "AgentIdentity",
    "AuthMethod",
    "Principal",
    "ScopeSet",
    "current_principal",
    "principal_here",
    "principal_scope",
]

_CURRENT: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
    "tesserix_adk_principal", default=None
)


class AuthMethod(StrEnum):
    """How the principal was established, which is what its expiry means.

    A system-initiated run is modelled as a principal like any other. "No user" is not an
    absence of authority — it is a scheduled job holding exactly the scopes somebody gave
    that job, which is a thing a reviewer can read and trim.
    """

    USER_SESSION = "user_session"
    SERVICE_ACCOUNT = "service_account"
    SCHEDULED_JOB = "scheduled_job"
    DELEGATED = "delegated"


@dataclass(frozen=True, slots=True)
class ScopeSet:
    """A set of scopes that can be narrowed and never widened.

    Names are compared in one normal form, because `Bookings:Read` and `bookings:read`
    arriving from two products is an intersection that silently empties and an agent that
    is refused everything for a reason nobody can see in a diff.

    Args:
        names: The scopes held, already normalised. Build with `of` rather than by hand.

    Example:
        >>> ScopeSet.of("bookings:read", "bookings:write").narrowed_to(
        ...     ScopeSet.of("Bookings:Read")
        ... )
        ScopeSet.of('bookings:read')
    """

    names: frozenset[str] = frozenset()

    @classmethod
    def of(cls, *names: str) -> Self:
        """A set from scope names in any spelling, normalised on the way in."""
        return cls(frozenset(canonical(name) for name in names if name.strip()))

    def narrowed_to(self, other: ScopeSet) -> ScopeSet:
        """What both sets hold. There is no operation here that produces more."""
        return ScopeSet(self.names & other.names)

    def permits(self, scope: str) -> bool:
        """Whether `scope` is held, compared in the same normal form."""
        return canonical(scope) in self.names

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        """Which of `required` is not held, in the order asked for."""
        return tuple(scope for scope in required if not self.permits(scope))

    def __contains__(self, scope: object) -> bool:
        """Whether a scope name is held; anything that is not a string is not."""
        return isinstance(scope, str) and self.permits(scope)

    def __iter__(self) -> Iterator[str]:
        """The scopes in a stable order, so a refusal reads the same twice."""
        return iter(sorted(self.names))

    def __len__(self) -> int:
        """How many scopes are held."""
        return len(self.names)

    def __bool__(self) -> bool:
        """Whether anything is held at all."""
        return bool(self.names)

    def __repr__(self) -> str:
        """Stable across runs: an unordered frozenset repr is a diff nobody can read."""
        return f"ScopeSet.of({', '.join(repr(scope) for scope in self)})"


class Principal(AdkModel):
    """Who a run acts for, and what they were granted.

    Public on the run entry point, so fields added later are optional. Frozen, like every
    boundary model here.

    Args:
        subject: Who is acting — a user id, a service account name, a job name.
        tenant: The isolation boundary they act within.
        scopes: What they hold. Empty is valid and means they hold nothing.
        method: How the authority was established.
        expires_at: When the authority lapses, on the same clock the runner reads —
            seconds, not a wall-clock date, because that is what a run compares against.
            A session that ends mid-run must not leave the run holding what it granted.

    Example:
        >>> Principal(subject="ada", tenant="acme", scopes=frozenset({"bookings:read"})).granted
        ScopeSet.of('bookings:read')
    """

    subject: str
    tenant: str
    scopes: frozenset[str] = frozenset()
    method: AuthMethod = AuthMethod.USER_SESSION
    expires_at: float | None = None

    @field_validator("subject", "tenant")
    @classmethod
    def _named(cls, value: str) -> str:
        """Blank is not a principal; it is a grant that matches whoever asks."""
        if not value.strip():
            raise ValueError("a principal names a subject and a tenant")
        return value

    @property
    def granted(self) -> ScopeSet:
        """What this principal holds, normalised."""
        return ScopeSet.of(*self.scopes)

    def expired(self, now: float) -> bool:
        """Whether the authority has lapsed by `now`."""
        return self.expires_at is not None and now >= self.expires_at

    def delegating(self, *, until: float, scopes: Iterable[str] | None = None) -> Self:
        """The same subject, recorded as a delegation with its own expiry.

        A durable workflow that resumes hours later cannot run on the session that started
        it, and it must not run on the process identity either. It runs on a delegation
        somebody recorded, which is why `until` is required rather than inherited.

        Args:
            until: When the delegation lapses, in clock seconds.
            scopes: A narrower grant, where the continuation needs less than the caller
                held. Never wider: anything not already held is dropped.

        Returns:
            A new principal. Nothing is mutated.
        """
        held = self.granted
        kept = held.narrowed_to(ScopeSet.of(*scopes)) if scopes is not None else held
        return self.model_copy(
            update={
                "scopes": frozenset(kept.names),
                "method": AuthMethod.DELEGATED,
                "expires_at": until,
            }
        )


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """What one agent may do on one run: its declaration, cut down to the caller's grant.

    Built by `resolve` or `narrowed`, never by widening. Frozen for the run, so the answer
    to "what could this run reach" is one value rather than a history of settings.

    Args:
        agent: Whose identity it is.
        principal: Who the run acts for.
        declared: What the agent said it needs.
        effective: What it actually holds — the intersection, computed once.
        chain: The agents this identity was narrowed through, outermost first.
    """

    agent: str
    principal: Principal
    declared: ScopeSet = field(default_factory=ScopeSet)
    effective: ScopeSet = field(default_factory=ScopeSet)
    chain: tuple[str, ...] = ()

    @classmethod
    def resolve(
        cls,
        *,
        agent: str,
        declared: Iterable[str],
        principal: Principal,
        now: float | None = None,
    ) -> Self:
        """The effective identity for a run, or a refusal before it starts.

        Args:
            agent: Whose identity this is.
            declared: The scopes the agent declares it needs.
            principal: Who the run acts for.
            now: The clock reading to judge expiry against. Skipped
                where none is given, for a caller that has already checked.

        Returns:
            The identity, whose effective scopes are the declaration intersected with
            what the principal holds.

        Raises:
            AuthorisationError: If the principal's authority has already lapsed. The run
                fails closed here rather than falling back to the process identity.

        Example:
            >>> caller = Principal(subject="ada", tenant="acme", scopes=frozenset({"b:read"}))
            >>> AgentIdentity.resolve(
            ...     agent="desk", declared=("b:read", "b:write"), principal=caller
            ... ).effective
            ScopeSet.of('b:read')
        """
        if now is not None and principal.expired(now):
            raise AuthorisationError(
                f"the authority {agent!r} would run on expired at {principal.expires_at}",
                agent=agent,
                subject=principal.subject,
                where="run",
            )
        asked = ScopeSet.of(*declared)
        return cls(
            agent=agent,
            principal=principal,
            declared=asked,
            effective=asked.narrowed_to(principal.granted),
        )

    def permits(self, scope: str) -> bool:
        """Whether the effective set carries `scope`."""
        return self.effective.permits(scope)

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        """Which of `required` the effective set does not carry."""
        return self.effective.missing(required)

    def check(self, required: Iterable[str], *, where: str = "") -> None:
        """Refuse unless every scope in `required` is held.

        Args:
            required: What the operation needs.
            where: What was about to happen, named in the refusal so an operator sees the
                egress point rather than only the accessor.

        Raises:
            AuthorisationError: Naming the first missing scope. A generic permission
                failure sends somebody to the wrong system to fix it.
        """
        absent = self.missing(required)
        if not absent:
            return
        held = ", ".join(self.effective) or "nothing"
        raise AuthorisationError(
            f"{self.agent!r} acting for {self.principal.subject!r} does not hold "
            f"{absent[0]!r}" + (f" ({where})" if where else "") + f"; it holds {held}",
            scope=absent[0],
            agent=self.agent,
            subject=self.principal.subject,
            where=where,
        )

    def check_live(self, now: float, *, where: str = "") -> None:
        """Refuse once the caller's authority has lapsed, however long the run has left.

        Raises:
            AuthorisationError: If the principal expired since the run started. A run
                that outlives the session that authorised it is acting on nobody's behalf.
        """
        if not self.principal.expired(now):
            return
        raise AuthorisationError(
            f"the authority {self.agent!r} runs on lapsed at {self.principal.expires_at}"
            + (f" ({where})" if where else ""),
            agent=self.agent,
            subject=self.principal.subject,
            where=where,
        )

    def narrowed(self, *, agent: str, declared: Iterable[str]) -> Self:
        """The identity a peer agent runs on, which this one cannot widen.

        The peer's own declaration is intersected with what is effective here, so an agent
        holding more than its caller cannot be invoked as a way around the refusal.

        Args:
            agent: The peer.
            declared: What the peer declares it needs.

        Returns:
            A new identity carrying the call path it was narrowed through.
        """
        asked = ScopeSet.of(*declared)
        return type(self)(
            agent=agent,
            principal=self.principal,
            declared=asked,
            effective=asked.narrowed_to(self.effective),
            chain=(*self.chain, self.agent),
        )

    def unused(self, used: Iterable[str]) -> tuple[str, ...]:
        """Declared scopes that nothing reached for, so a declaration can be trimmed.

        Args:
            used: The scopes actually required over the runs being reviewed.

        Returns:
            The declared scopes none of them needed, in a stable order.
        """
        reached = ScopeSet.of(*used)
        return tuple(scope for scope in self.declared if not reached.permits(scope))


def principal_here() -> Principal | None:
    """The principal bound here, or None outside a scope.

    For code that legitimately runs with no caller — a health check, a corpus job — and
    branches on that. Anything spending authority uses `current_principal` instead.

    Example:
        >>> principal_here() is None
        True
    """
    return _CURRENT.get()


def current_principal(*, where: str = "current_principal") -> Principal:
    """The principal bound here, or a refusal.

    Args:
        where: What was about to happen, so the refusal names the egress point.

    Returns:
        The principal established by the nearest enclosing `principal_scope`.

    Raises:
        AuthorisationError: Where nothing is bound. Never the process identity, which is
            the escalation this module exists to prevent.

    Example:
        >>> caller = Principal(subject="ada", tenant="acme")
        >>> with principal_scope(caller):
        ...     current_principal().subject
        'ada'
    """
    here = _CURRENT.get()
    if here is None:
        raise AuthorisationError(
            f"{where} needs a principal and none is bound; enter principal_scope(...) at "
            f"the boundary that knows who this run acts for",
            where=where,
        )
    return here


@contextmanager
def principal_scope(principal: Principal) -> Iterator[Principal]:
    """Bind who a run acts for, restoring what was there before.

    Args:
        principal: Who is acting.

    Yields:
        The bound principal, which is what `current_principal` returns inside the block.

    Example:
        >>> with principal_scope(Principal(subject="ada", tenant="acme")) as caller:
        ...     caller.tenant
        'acme'
    """
    token = _CURRENT.set(principal)
    try:
        yield principal
    finally:
        _CURRENT.reset(token)
