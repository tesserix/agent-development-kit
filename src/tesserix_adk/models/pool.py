"""Who owns a provider connection, and for how long.

A client built per run pays a DNS lookup and a TLS handshake for every agent turn, and
under load the sockets outlive the runs that opened them until the process runs out of
descriptors. So clients are keyed and shared: same provider, endpoint, credential and
transport settings means the same warm pool, and anything else means a different one.

The key is what makes the sharing safe. It carries a digest of the credential rather than
the credential, so two tenants against one endpoint can never be handed each other's
connection, and a rotated key opens a new pool while the old one finishes what is already
in flight. Lifetime is explicit — the registry is an async context manager and closing it
closes everything it opened — because a pool nobody closes is the leak this exists to fix.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self

import httpx

from tesserix_adk.core.errors import PoolExhaustedError
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from tesserix_adk.core.protocols import Clock

__all__ = ["ClientKey", "ClientPool", "PoolConfig", "PoolMetrics"]

_DIGEST_CHARACTERS = 16
_DIGEST_KEY = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """How many connections a provider is allowed, and how long an idle one is kept.

    Defaults suit a service handling ordinary agent traffic: enough connections that a
    burst does not queue, few enough that one process cannot exhaust a partner's
    per-client limit on its own. A vendor with a tighter limit gets an entry in
    `per_provider` rather than a lower ceiling for everyone.

    Attributes:
        max_connections: The ceiling on connections to one endpoint. The pool waits at
            this number rather than growing past it, because a pool that grows on demand
            turns a downstream slowdown into a descriptor exhaustion.
        max_keepalive: How many of those are kept warm when idle.
        keepalive_seconds: How long an idle connection is kept before it is dropped. Long
            keep-alive behind a load balancer pins traffic to one backend, so this is
            short enough that the fleet is re-balanced without anybody restarting.
        acquire_seconds: How long a caller waits for a free connection before failing with
            `PoolExhaustedError`. Bounded, so a saturated pool is reported inside the
            run's own deadline rather than queueing past it.
        per_provider: Overrides by provider name. An entry replaces the defaults for that
            provider entirely.

    Example:
        >>> tighter = PoolConfig(max_connections=8, max_keepalive=4)
        >>> shared = PoolConfig(per_provider={"anthropic": tighter})
        >>> shared.for_provider("anthropic").max_connections
        8
    """

    max_connections: int = 100
    max_keepalive: int = 20
    keepalive_seconds: float = 30.0
    acquire_seconds: float = 10.0
    per_provider: Mapping[str, PoolConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a shape the transport would accept and then behave strangely under."""
        if self.max_keepalive > self.max_connections:
            raise ValueError(
                f"keepalive ceiling {self.max_keepalive} is above the connection ceiling "
                f"{self.max_connections}: a connection cannot be kept warm after a pool "
                f"that never opened it"
            )
        if min(self.max_connections, self.max_keepalive) < 0:
            raise ValueError("a connection ceiling cannot be negative")
        if min(self.keepalive_seconds, self.acquire_seconds) <= 0:
            raise ValueError("a keepalive and an acquisition wait must both be positive")

    def for_provider(self, provider: str) -> PoolConfig:
        """The configuration this provider runs under, which may be the default."""
        return self.per_provider.get(provider, self)

    def _limits(self) -> httpx.Limits:
        """The declared ceilings, as the transport expresses them."""
        return httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive,
            keepalive_expiry=self.keepalive_seconds,
        )


@dataclass(frozen=True, slots=True)
class ClientKey:
    """What makes two requests eligible to share a connection.

    The credential appears as a digest, never as itself: a key is logged, compared and
    put in a metric label, and a secret that reaches any of those has leaked.

    Attributes:
        provider: Who is being called.
        base_url: Which endpoint. A regional host is a different pool.
        credential_digest: A truncated digest of the credential in use.
        transport: The transport settings that must match — timeouts and limits — since a
            client shared across two of them honours only one.
    """

    provider: str
    base_url: str
    credential_digest: str
    transport: str

    @classmethod
    def of(cls, *, provider: str, base_url: str, credential: str, transport: str) -> ClientKey:
        """Build a key, digesting the credential on the way in."""
        return cls(
            provider=provider,
            base_url=base_url,
            credential_digest=digest_of(credential),
            transport=transport,
        )


def digest_of(credential: str) -> str:
    """A short process-scoped credential fingerprint, safe to log and compare.

    Example:
        >>> digest_of("sk-abc") == digest_of("sk-abc")
        True
    """
    return hmac.new(_DIGEST_KEY, credential.encode(), hashlib.sha256).hexdigest()[
        :_DIGEST_CHARACTERS
    ]


@dataclass(frozen=True, slots=True)
class PoolMetrics:
    """A snapshot of what the registry has done, taken rather than watched.

    A snapshot, so a reader that prints four counters prints four counters from one
    moment rather than four moments.

    Attributes:
        opened: Clients created.
        reused: Times an existing client was handed back instead.
        retired: Clients replaced by rotation, whose in-flight work was left to finish.
        inherited: Clients discarded because they came from another process.
        exhaustions: Requests that failed because no connection came free in time.
        open_now: Clients currently held.
        waited_seconds: Total time spent inside provider requests waiting on the pool.
    """

    opened: int = 0
    reused: int = 0
    retired: int = 0
    inherited: int = 0
    exhaustions: int = 0
    open_now: int = 0
    waited_seconds: float = 0.0


class ClientPool:
    """The registry that owns provider clients and their lifetime.

    Args:
        config: The declared ceilings and waits, with any per-provider overrides.
        transport: An injected transport, which is how tests run without a socket and how
            a consumer supplies its own proxy layer.
        clock: Where elapsed time comes from, injected so a test does not sleep.
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or PoolConfig()
        self._transport = transport
        self._clock = clock or SystemClock()
        self._clients: dict[ClientKey, httpx.AsyncClient] = {}
        self._owners: dict[ClientKey, int] = {}
        self._borrowed: dict[int, int] = {}
        self._retiring: list[httpx.AsyncClient] = []
        self._metrics = PoolMetrics()
        self._closed = False

    @property
    def metrics(self) -> PoolMetrics:
        """A snapshot of the counters, taken now."""
        return replace(self._metrics, open_now=len(self._clients))

    @property
    def keys(self) -> tuple[ClientKey, ...]:
        """The keys currently held, for a metric label or an operator's inspection."""
        return tuple(self._clients)

    def config_for(self, provider: str) -> PoolConfig:
        """The ceilings this provider runs under, which may be the defaults."""
        return self._config.for_provider(provider)

    def _client(
        self,
        *,
        provider: str,
        base_url: str,
        credential: str = "",
        timeout_seconds: float | None = None,
        connect_seconds: float | None = None,
    ) -> httpx.AsyncClient:
        """Return the client for this key, opening one only where none is held.

        Raises:
            RuntimeError: If the registry has been closed. Handing out a client from a
                closed registry hands out a connection nobody will close.
        """
        if self._closed:
            raise RuntimeError(
                "this client pool is closed; open a new one rather than reusing connections "
                "whose owner has already gone"
            )
        key = self._key(provider, base_url, credential, timeout_seconds, connect_seconds)
        held = self._clients.get(key)
        if held is not None and self._owners.get(key) != os.getpid():
            # A pool inherited across a fork is half-open whatever the bookkeeping says:
            # the descriptors belong to the parent's event loop.
            self._forget(key)
            self._metrics = replace(self._metrics, inherited=self._metrics.inherited + 1)
            held = None
        if held is not None:
            self._metrics = replace(self._metrics, reused=self._metrics.reused + 1)
            return held
        return self._opened(key, provider, base_url, timeout_seconds, connect_seconds)

    @asynccontextmanager
    async def _borrowed_client(
        self,
        *,
        provider: str,
        base_url: str,
        credential: str = "",
        timeout_seconds: float | None = None,
        connect_seconds: float | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        """Hold a client for the length of a block, so rotation cannot close it underneath.

        A rotation during the block retires the client rather than closing it; the last
        borrower out closes it.
        """
        client = self._client(
            provider=provider,
            base_url=base_url,
            credential=credential,
            timeout_seconds=timeout_seconds,
            connect_seconds=connect_seconds,
        )
        self._borrowed[id(client)] = self._borrowed.get(id(client), 0) + 1
        try:
            yield client
        finally:
            remaining = self._borrowed[id(client)] - 1
            self._borrowed[id(client)] = remaining
            if remaining == 0:
                del self._borrowed[id(client)]
                if client in self._retiring:
                    self._retiring.remove(client)
                    await client.aclose()

    def _key_for(
        self,
        *,
        provider: str,
        base_url: str,
        credential: str = "",
        timeout_seconds: float | None = None,
        connect_seconds: float | None = None,
    ) -> ClientKey:
        """The key these settings resolve to, so a caller can hold one without the secret."""
        return self._key(provider, base_url, credential, timeout_seconds, connect_seconds)

    async def retire(
        self,
        *,
        provider: str,
        base_url: str,
        credential: str = "",
        timeout_seconds: float | None = None,
        connect_seconds: float | None = None,
    ) -> None:
        """Stop handing this client out, and close it once nothing is still using it.

        This is what a credential rotation does: the next request opens a pool on the new
        key while the requests already on the old one finish on it.
        """
        await self._retire_key(
            self._key(provider, base_url, credential, timeout_seconds, connect_seconds)
        )

    async def _retire_key(self, key: ClientKey) -> None:
        """Retire by key, for a caller that kept the key rather than the credential."""
        client = self._clients.get(key)
        if client is None:
            return
        self._forget(key)
        self._metrics = replace(self._metrics, retired=self._metrics.retired + 1)
        if self._borrowed.get(id(client)):
            self._retiring.append(client)
            return
        await client.aclose()

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        provider: str,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Send one request, translating a saturated pool into the kit's own error.

        Raises:
            PoolExhaustedError: If no connection came free inside `acquire_seconds`.
        """
        started = self._now()
        try:
            return await client.request(method, url, json=json, headers=dict(headers or {}))
        except httpx.PoolTimeout as saturated:
            raise self._exhausted(provider, self._now() - started) from saturated
        finally:
            self._metrics = replace(
                self._metrics, waited_seconds=self._metrics.waited_seconds + self._now() - started
            )

    def _exhausted(self, provider: str, waited: float | None = None) -> PoolExhaustedError:
        """Count a saturated pool and build the error that reports it."""
        self._metrics = replace(self._metrics, exhaustions=self._metrics.exhaustions + 1)
        details = {"provider": provider}
        if waited is not None:
            details["waited"] = f"{waited:.3f}"
        return PoolExhaustedError(
            f"every connection to {provider} was in use and none came free in "
            f"{self._config.for_provider(provider).acquire_seconds:g}s",
            provider=provider,
            details=details,
        )

    async def aclose(self) -> None:
        """Close every client this registry opened, including the retiring ones."""
        self._closed = True
        closing = [*self._clients.values(), *self._retiring]
        self._clients.clear()
        self._owners.clear()
        self._retiring.clear()
        for client in closing:
            await client.aclose()

    async def __aenter__(self) -> Self:
        """Return self, so a process can own its pools for the length of a block."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close everything opened inside the block."""
        await self.aclose()

    def _key(
        self,
        provider: str,
        base_url: str,
        credential: str,
        timeout_seconds: float | None,
        connect_seconds: float | None,
    ) -> ClientKey:
        limits = self._config.for_provider(provider)
        return ClientKey.of(
            provider=provider,
            base_url=base_url,
            credential=credential,
            transport=(
                f"{timeout_seconds}/{connect_seconds}/{limits.max_connections}/"
                f"{limits.max_keepalive}/{limits.keepalive_seconds}/{limits.acquire_seconds}"
            ),
        )

    def _opened(
        self,
        key: ClientKey,
        provider: str,
        base_url: str,
        timeout_seconds: float | None,
        connect_seconds: float | None,
    ) -> httpx.AsyncClient:
        limits = self._config.for_provider(provider)
        client = httpx.AsyncClient(
            base_url=base_url,
            limits=limits._limits(),
            timeout=httpx.Timeout(
                connect=connect_seconds if connect_seconds is not None else 10.0,
                read=timeout_seconds if timeout_seconds is not None else 60.0,
                write=30.0,
                pool=limits.acquire_seconds,
            ),
            transport=self._transport,
        )
        self._clients[key] = client
        self._owners[key] = os.getpid()
        self._metrics = replace(self._metrics, opened=self._metrics.opened + 1)
        return client

    def _forget(self, key: ClientKey) -> None:
        self._clients.pop(key, None)
        self._owners.pop(key, None)

    def _now(self) -> float:
        return self._clock.now()
