"""What a tool authenticates with downstream: narrow, short-lived and attributable.

A static key shared by every agent and every tenant is one value that grants everything,
forever, to whoever holds it — and a downstream audit log full of that key's requests
cannot say which run made any of them. The leak is unbounded in scope, in time and in
blame at once.

So a tool call carries a credential minted for one audience, holding only the scopes that
tool needs intersected with what the run actually holds, expiring in minutes, and carrying
the run id, agent and agent version as attribution. A leaked one is narrow, dated and
traceable to the run that leaked it.

Credentials are `SecretStr` throughout. They are revealed in exactly one place — the
headers that make the call — and never logged, never put on a span, and never returned to
the model. Failure is a typed `CredentialError`: a tool that cannot mint what it needs
fails, rather than falling back to something broader.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import Field, SecretStr, field_validator, model_validator

from tesserix_adk.core import (
    AdkModel,
    AgentIdentity,
    AuthorisationError,
    CredentialError,
    ScopeSet,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tesserix_adk.core import SecretRef
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.core.secrets import SecretResolver

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "EXPIRY_SKEW_SECONDS",
    "MAX_TTL_SECONDS",
    "Attribution",
    "CachingCredentials",
    "Credential",
    "CredentialBroker",
    "CredentialProvider",
    "CredentialRequest",
    "ExchangedCredentials",
    "StaticCredentials",
    "TokenExchange",
]

# Long enough that a normal tool call does not spend its life at the token endpoint, short
# enough that a leaked credential is stale by the time anybody notices it was leaked.
DEFAULT_TTL_SECONDS = 300.0

# The hard ceiling. A consumer may ask for less and cannot configure past this: a credential
# that outlives the run it was minted for is a static key with extra steps.
MAX_TTL_SECONDS = 3600.0

# Spend a credential slightly before it expires, so a clock a few seconds fast downstream
# does not reject one this process still considers live.
EXPIRY_SKEW_SECONDS = 30.0


class Attribution(AdkModel):
    """Who a downstream request is for, so its audit log can answer that question.

    Args:
        run_id: Which run made the call.
        agent: Which agent made it.
        agent_version: Which revision of that agent, so a change in downstream behaviour
            is attributable to a declaration rather than to a deployment date.
        tenant: Whose work it was.
        subject: Who the run acts for.
    """

    run_id: str
    agent: str
    agent_version: str = "1.0.0"
    tenant: str = ""
    subject: str = ""

    @field_validator("run_id", "agent")
    @classmethod
    def _named(cls, value: str) -> str:
        """Attribution to nobody in particular is the thing this exists to prevent."""
        if not value.strip():
            raise ValueError("attribution names the run and the agent that made the call")
        return value

    def headers(self) -> dict[str, str]:
        """The attribution as request headers. Nothing secret is in it.

        Example:
            >>> Attribution(run_id="run_1", agent="desk").headers()["X-Tesserix-Run"]
            'run_1'
        """
        stated = {
            "X-Tesserix-Run": self.run_id,
            "X-Tesserix-Agent": self.agent,
            "X-Tesserix-Agent-Version": self.agent_version,
            "X-Tesserix-Tenant": self.tenant,
            "X-Tesserix-Subject": self.subject,
        }
        return {name: value for name, value in stated.items() if value}


class CredentialRequest(AdkModel):
    """What a tool needs to authenticate one call.

    Args:
        audience: Which downstream this is for. A credential minted for one audience is
            not usable at another, which is what bounds a leak to one service.
        scopes: What it may do there, already narrowed to what the run holds.
        attribution: Who is asking.
        ttl_seconds: How long it should live. Capped at `MAX_TTL_SECONDS`.
    """

    audience: str
    scopes: frozenset[str] = frozenset()
    attribution: Attribution
    ttl_seconds: float = Field(default=DEFAULT_TTL_SECONDS, gt=0)

    @field_validator("audience")
    @classmethod
    def _addressed(cls, value: str) -> str:
        """A credential for no audience in particular is usable everywhere."""
        if not value.strip():
            raise ValueError("a credential request names the audience it is for")
        return value

    @field_validator("ttl_seconds")
    @classmethod
    def _inside_the_ceiling(cls, value: float) -> float:
        if value > MAX_TTL_SECONDS:
            raise ValueError(
                f"{value}s is past the {MAX_TTL_SECONDS}s ceiling; a credential outliving "
                f"its run is a static key with extra steps"
            )
        return value

    def cache_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        """What makes two requests the same credential. Tenant and subject are part of it."""
        return (
            self.attribution.tenant,
            self.attribution.subject,
            self.audience,
            tuple(sorted(self.scopes)),
        )


class Credential(AdkModel):
    """A minted credential, redacting itself everywhere but the call it authenticates.

    Args:
        token: The value. A `SecretStr`, so no rendering of this object reveals it.
        audience: What it is usable at.
        scopes: What it permits there.
        expires_at: When it stops being usable, in the kit's monotonic seconds.
        attribution: What the downstream audit log will record.
        long_lived: That this came from a static key rather than an exchange — the
            documented exception, flagged so it can be counted and driven to zero.
    """

    token: SecretStr
    audience: str
    scopes: frozenset[str] = frozenset()
    expires_at: float
    attribution: Attribution
    long_lived: bool = False

    @model_validator(mode="after")
    def _usable_when_minted(self) -> Self:
        if self.expires_at < 0:
            raise ValueError("a credential that is already expired authenticates nothing")
        return self

    def expired(self, now: float, *, skew: float = EXPIRY_SKEW_SECONDS) -> bool:
        """Whether this is too close to expiry to spend.

        Args:
            now: The current monotonic time.
            skew: How early to give up on it, for a downstream clock running fast.
        """
        return now >= self.expires_at - skew

    def headers(self) -> dict[str, str]:
        """The headers that make the call. This is the one place the value is revealed.

        Returns:
            The bearer token and the attribution. Never log this dictionary.
        """
        return {
            "Authorization": f"Bearer {self.token.get_secret_value()}",
            **self.attribution.headers(),
        }


@runtime_checkable
class CredentialProvider(Protocol):
    """Where a `CredentialRequest` becomes a `Credential`."""

    async def issue(self, request: CredentialRequest) -> Credential:
        """Mint a credential for the request.

        Raises:
            CredentialError: If the credential could not be minted, or the requested
                scopes were refused. Never falls back to something broader.
        """
        ...


@runtime_checkable
class TokenExchange(Protocol):
    """A token endpoint, as the one call the kit needs of it.

    Kept as a protocol rather than an HTTP client so a deployment's own exchange — RFC 8693
    token exchange, workload identity, an internal minting service — is a class with one
    method and no dependency of the kit's.
    """

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Return the minted token and the lifetime actually granted, in seconds."""
        ...


@dataclass(frozen=True, slots=True)
class ExchangedCredentials:
    """Credentials minted by an exchange, which is the path this whole module is for.

    The granted lifetime is what expires, not the one asked for: an endpoint that hands
    back an hour when asked for five minutes is honoured only up to `MAX_TTL_SECONDS`.

    Args:
        exchange: The token endpoint.
        clock: Where expiry is measured.
    """

    exchange: TokenExchange
    clock: Clock

    async def issue(self, request: CredentialRequest) -> Credential:
        """Mint a credential by exchanging for one.

        Raises:
            CredentialError: If the endpoint refused or could not be reached. A tool
                without the credential it needs fails; it does not proceed with a broader
                one, which is the fallback this refusal exists to prevent.
        """
        try:
            token, granted = await self.exchange.exchange(request)
        except CredentialError:
            raise
        except Exception as failure:
            raise CredentialError(
                f"no credential could be minted for {request.audience}",
                audience=request.audience,
            ) from failure
        lifetime = min(granted, request.ttl_seconds, MAX_TTL_SECONDS)
        return Credential(
            token=SecretStr(token),
            audience=request.audience,
            scopes=request.scopes,
            expires_at=self.clock.now() + lifetime,
            attribution=request.attribution,
        )


@dataclass(frozen=True, slots=True)
class StaticCredentials:
    """The documented exception: a third party that only accepts a long-lived key.

    Every credential this mints is flagged `long_lived`, so the exception is countable
    rather than invisible. The key itself is still a `SecretRef` resolved at the point of
    use, and still per-audience — one third party's key is not another's.

    Args:
        secrets: Where the key is resolved from.
        keys: Which reference holds each audience's key. An audience not listed here is
            refused, so the exception cannot spread by omission.
        clock: Where expiry is measured.
        ttl_seconds: How long a minted credential is treated as usable. The key outlives
            it; this bounds how long one is held and reused in memory.
    """

    secrets: SecretResolver
    keys: Mapping[str, SecretRef]
    clock: Clock
    ttl_seconds: float = DEFAULT_TTL_SECONDS

    async def issue(self, request: CredentialRequest) -> Credential:
        """Read the static key documented for this audience.

        Raises:
            CredentialError: If no key is documented for the audience, or the reference
                could not be resolved.
        """
        ref = self.keys.get(request.audience)
        if ref is None:
            raise CredentialError(
                f"no static key is documented for {request.audience}",
                audience=request.audience,
            )
        tenant = request.attribution.tenant
        value = await self.secrets.resolve(ref.for_tenant(tenant) if tenant else ref)
        return Credential(
            token=value,
            audience=request.audience,
            scopes=request.scopes,
            expires_at=self.clock.now() + min(self.ttl_seconds, MAX_TTL_SECONDS),
            attribution=request.attribution,
            long_lived=True,
        )


class CachingCredentials:
    """Credentials reused while they are live, minted once per key when they are not.

    The key is tenant, subject, audience and scopes — so nothing is ever shared across a
    tenant or a caller — and a fan-out asking for the same one shares a single mint rather
    than stampeding the endpoint. A retry after a partial failure gets the credential the
    first attempt used, which is what keeps a non-idempotent downstream call from being
    authenticated twice as two different callers.

    Args:
        inner: What actually mints.
        clock: Where expiry is measured.
    """

    def __init__(self, inner: CredentialProvider, *, clock: Clock) -> None:
        self._inner = inner
        self._clock = clock
        self._held: dict[tuple[str, str, str, tuple[str, ...]], Credential] = {}
        self._locks: dict[tuple[str, str, str, tuple[str, ...]], asyncio.Lock] = {}

    async def issue(self, request: CredentialRequest) -> Credential:
        """Return a live credential for the request, minting one only where needed."""
        key = request.cache_key()
        live = self._live(key)
        if live is not None:
            return live
        async with self._locks.setdefault(key, asyncio.Lock()):
            live = self._live(key)
            if live is not None:
                return live
            minted = await self._inner.issue(request)
            self._held[key] = minted
            return minted

    def invalidate_all(self) -> None:
        """Drop everything held, for a revocation nobody wants to wait out."""
        self._held.clear()

    @property
    def cached(self) -> int:
        """How many credentials are held, so a test can assert nothing was left behind."""
        return len(self._held)

    def _live(self, key: tuple[str, str, str, tuple[str, ...]]) -> Credential | None:
        held = self._held.get(key)
        if held is None:
            return None
        if held.expired(self._clock.now()):
            del self._held[key]
            return None
        return held


class CredentialBroker:
    """What a tool asks, so a tool never states its own authority.

    The scopes minted are what the tool needs intersected with what the run holds, and a
    tool asking for anything the run does not hold is refused here rather than downstream:
    the request is never made, so there is nothing to read back from an audit log and
    nothing to explain.

    Args:
        provider: Where credentials come from. Wrap it in `CachingCredentials` unless the
            endpoint is free.
        clock: Where expiry is measured.
        ttl_seconds: How long to ask for. Refused above `MAX_TTL_SECONDS`.
    """

    def __init__(
        self,
        provider: CredentialProvider,
        *,
        clock: Clock,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds > MAX_TTL_SECONDS:
            raise ValueError(
                f"{ttl_seconds}s is past the {MAX_TTL_SECONDS}s ceiling and cannot be configured"
            )
        self._provider = provider
        self._clock = clock
        self._ttl = ttl_seconds

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> Credential:
        """Mint the credential one tool call needs.

        Args:
            identity: What the run holds, resolved once at its start.
            audience: Which downstream the call is to.
            needs: What the tool needs there.
            run_id: Which run is calling, for the downstream audit log.
            agent_version: Which revision of the agent is calling.

        Raises:
            AuthorisationError: If the tool asks for a scope the run does not hold, or
                names no scope at all.
            CredentialError: If the credential could not be minted.
        """
        asked = ScopeSet.of(*needs)
        if not asked:
            raise AuthorisationError(
                f"a credential for {audience} names no scope, which is a request for "
                f"whatever the audience will give",
                agent=identity.agent,
                where=audience,
            )
        identity.check(tuple(asked), where=audience)
        request = CredentialRequest(
            audience=audience,
            scopes=frozenset(asked),
            attribution=Attribution(
                run_id=run_id,
                agent=identity.agent,
                agent_version=agent_version,
                tenant=identity.principal.tenant,
                subject=identity.principal.subject,
            ),
            ttl_seconds=self._ttl,
        )
        return await self._provider.issue(request)
