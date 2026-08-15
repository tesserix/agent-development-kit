"""Configuration holds a reference to a credential; only the call site holds the value.

A key read straight into a config field is a key in whatever that config is later handed
to: a log line, a span attribute, an error payload, a `repr` in a traceback frame. The
leak is never the line somebody wrote on purpose — it is the sixth-hand print of an object
that happened to contain it.

So a config field holds a `SecretRef`: a name and a version, which are safe to print, to
commit and to put in a diff. The value is fetched at the point of use through a
`SecretResolver`, arrives as a pydantic `SecretStr` that redacts itself in every rendering
the kit has, and is revealed explicitly on the line that needs it.

Resolution is cached with a ttl and can be invalidated, so a rotated secret is picked up
without a restart and a stale one is never served past its window. Failure is typed and
names the reference, never the value: the kit does not fall back to an unauthenticated
call, and it does not serve an expired entry to stay up.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import SecretStr

from tesserix_adk.core.errors import SecretResolutionError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.tenant_config import SecretRef  # noqa: TC001 — a runtime type here

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tesserix_adk.core.protocols import Clock, SecretProvider

__all__ = [
    "CachingSecrets",
    "ChainedSecrets",
    "EnvironmentSecrets",
    "ProvidedSecrets",
    "SecretResolver",
    "literal_credentials",
]

# Field names that hold a credential often enough that a literal in one is a mistake worth
# refusing. Matched as substrings, lowercased.
_CREDENTIAL_NAMES = ("password", "secret", "token", "api_key", "apikey", "dsn", "credential")


@runtime_checkable
class SecretResolver(Protocol):
    """Where a `SecretRef` is turned into a value, asked at the point of use.

    Async because the answer usually comes over a network. Implementations raise
    `SecretResolutionError` rather than returning None: a caller that cannot tell "absent"
    from "backend down" fails open, and failing open on a credential means an
    unauthenticated call somebody has to notice in a dashboard.
    """

    async def resolve(self, ref: SecretRef) -> SecretStr:
        """Return the value behind `ref`.

        Raises:
            SecretResolutionError: If the reference names nothing, or the backend could
                not be reached. The refusal names the reference, never the value.
        """
        ...


@dataclass(frozen=True, slots=True)
class EnvironmentSecrets:
    """Resolution from the process environment, which is the local development path.

    A name is upper-cased and its punctuation folded to underscores, so `openai-key` reads
    `OPENAI_KEY`. Versions are ignored: the environment has one value at a time, and that
    is what makes it a development resolver rather than a production one.

    Args:
        prefix: Prepended to every variable name, for a process holding more than one
            kit's secrets.
        environ: Where to read from. Injected so a test never mutates the real one.

    Example:
        >>> import asyncio
        >>> secrets = EnvironmentSecrets(environ={"OPENAI_KEY": "sk-test-1"})
        >>> asyncio.run(secrets.resolve(SecretRef(name="openai-key"))).get_secret_value()
        'sk-test-1'
    """

    prefix: str = ""
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)

    def variable_for(self, ref: SecretRef) -> str:
        """The environment variable this reference reads."""
        folded = "".join(letter if letter.isalnum() else "_" for letter in ref.name).upper()
        return f"{self.prefix}{folded}"

    async def resolve(self, ref: SecretRef) -> SecretStr:
        """Return the value from the environment.

        Raises:
            SecretResolutionError: If the variable is unset or empty. An empty credential
                is a failed authentication somewhere further away.
        """
        name = self.variable_for(ref)
        value = self.environ.get(name)
        if not value:
            raise SecretResolutionError(
                f"{ref.describe()} is not set in the environment (looked for {name})",
                ref=ref.describe(),
            )
        return SecretStr(value)


@dataclass(frozen=True, slots=True)
class ProvidedSecrets:
    """A `SecretProvider` — the kit's synchronous lookup — read as a `SecretResolver`.

    Args:
        provider: What holds the values.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.testing import FakeSecrets
        >>> secrets = ProvidedSecrets(FakeSecrets({"openai-key": "sk-test-1"}))
        >>> asyncio.run(secrets.resolve(SecretRef(name="openai-key"))).get_secret_value()
        'sk-test-1'
    """

    provider: SecretProvider

    async def resolve(self, ref: SecretRef) -> SecretStr:
        """Return what the provider holds for the reference's name.

        Raises:
            SecretResolutionError: If the provider holds nothing for it.
        """
        value = self.provider.secret(ref.name)
        if not value:
            raise SecretResolutionError(
                f"{ref.describe()} is not held by {type(self.provider).__name__}",
                ref=ref.describe(),
            )
        return SecretStr(value)


@dataclass(frozen=True, slots=True)
class ChainedSecrets:
    """Resolvers tried in order, the first that holds the secret answering.

    The order is the kit's config precedence: what a deployment injects wins over what a
    developer's environment happens to hold, so a laptop's stale key cannot shadow the one
    a service is meant to run on. A backend that is down is not "does not hold it" — but
    the chain cannot tell the two apart from outside, so a refusal from the last resolver
    is what surfaces, with the whole chain named.

    Args:
        resolvers: In precedence order. An empty chain resolves nothing, loudly.

    Example:
        >>> import asyncio
        >>> chain = ChainedSecrets(
        ...     (EnvironmentSecrets(environ={}), EnvironmentSecrets(environ={"K": "sk-test-1"}))
        ... )
        >>> asyncio.run(chain.resolve(SecretRef(name="k"))).get_secret_value()
        'sk-test-1'
    """

    resolvers: tuple[SecretResolver, ...] = ()

    async def resolve(self, ref: SecretRef) -> SecretStr:
        """Return the first value any resolver holds.

        Raises:
            SecretResolutionError: If none of them holds it, naming how many were asked.
        """
        for resolver in self.resolvers:
            try:
                return await resolver.resolve(ref)
            except SecretResolutionError:
                continue
        raise SecretResolutionError(
            f"{ref.describe()} was not held by any of the {len(self.resolvers)} resolvers "
            f"in the chain",
            ref=ref.describe(),
        )


@dataclass(frozen=True, slots=True)
class _Entry:
    """One cached value and when it stops being usable."""

    value: SecretStr
    until: float


class CachingSecrets:
    """A resolver with a ttl, so a rotated secret is picked up without a restart.

    Nothing is served past its ttl: an expired entry is refetched, and a backend that is
    down produces a refusal rather than the last value that worked. Concurrent callers
    asking for the same reference share one fetch, so a cold start does not turn every
    provider construction into its own round trip; different references still resolve in
    parallel.

    A rotation mid-process affects the next resolution. A call already holding a revealed
    value completes with it, which is what the backend's own overlap window is for.

    Args:
        inner: What actually fetches.
        clock: Where the ttl is measured, injected so a test does not sleep.
        ttl_seconds: How long a value may be reused.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.testing import FakeClock
        >>> cache = CachingSecrets(
        ...     EnvironmentSecrets(environ={"K": "sk-test-1"}), clock=FakeClock(), ttl_seconds=60
        ... )
        >>> asyncio.run(cache.resolve(SecretRef(name="k"))).get_secret_value()
        'sk-test-1'
    """

    def __init__(self, inner: SecretResolver, *, clock: Clock, ttl_seconds: float = 300.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive; a zero ttl caches nothing")
        self._inner = inner
        self._clock = clock
        self._ttl = ttl_seconds
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def resolve(self, ref: SecretRef) -> SecretStr:
        """Return the cached value where it is still live, and fetch where it is not.

        Raises:
            SecretResolutionError: Whatever the inner resolver raises. An expired entry is
                not served to paper over a backend that is down.
        """
        key = self._key(ref)
        held = self._live(key)
        if held is not None:
            return held
        async with self._locks.setdefault(key, asyncio.Lock()):
            # Another caller may have fetched it while this one waited for the lock.
            held = self._live(key)
            if held is not None:
                return held
            value = await self._inner.resolve(ref)
            self._entries[key] = _Entry(value=value, until=self._clock.now() + self._ttl)
            return value

    def invalidate(self, ref: SecretRef) -> None:
        """Drop one reference, so the next resolution fetches. Absent is not an error."""
        self._entries.pop(self._key(ref), None)

    def invalidate_all(self) -> None:
        """Drop everything cached, for a rotation nobody wants to wait a ttl for."""
        self._entries.clear()

    @property
    def cached(self) -> int:
        """How many references are held, so a test can assert a fetch did not repeat."""
        return len(self._entries)

    def _live(self, key: str) -> SecretStr | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._clock.now() >= entry.until:
            del self._entries[key]
            return None
        return entry.value

    @staticmethod
    def _key(ref: SecretRef) -> str:
        return f"{ref.tenant or ''}/{ref.describe()}"


def literal_credentials(config: AdkModel) -> tuple[str, ...]:
    """Fields that look like credentials and hold a literal rather than a reference.

    A config file is committed, copied into a ticket and pasted into a chat. A field named
    `api_key` holding a string is a credential in all three, so the linter names it before
    it gets there. `SecretStr` and `SecretRef` both pass: one redacts itself, the other is
    not a value at all.

    Args:
        config: The model to check, recursively through nested models.

    Returns:
        The dotted paths of the offending fields, in a stable order.

    Example:
        >>> from tesserix_adk.core.config import ProviderConfig
        >>> literal_credentials(ProviderConfig(endpoint="https://example.invalid"))
        ()
    """
    return tuple(sorted(_literals(config, prefix="")))


def _literals(config: AdkModel, *, prefix: str) -> Iterable[str]:
    for name, value in config:
        path = f"{prefix}{name}"
        if isinstance(value, AdkModel):
            yield from _literals(value, prefix=f"{path}.")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                if isinstance(item, AdkModel):
                    yield from _literals(item, prefix=f"{path}[{index}].")
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, AdkModel):
                    yield from _literals(item, prefix=f"{path}[{key!r}].")
                elif isinstance(item, str) and item and _looks_like_a_credential(str(key)):
                    yield f"{path}[{key!r}]"
        elif isinstance(value, str) and value and _looks_like_a_credential(name):
            yield path


def _looks_like_a_credential(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _CREDENTIAL_NAMES)
