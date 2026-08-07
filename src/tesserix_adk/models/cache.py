"""Serving a model call from a previous one, with the rules written down.

Every ad-hoc cache added to a product goes wrong the same two ways: it keys on the user's
prompt alone, so a tool-schema change serves an answer shaped for the old schema, and it
leaves the tenant out of the key, so one customer is served another's answer. Both are
key-design mistakes, so the key here carries every determinant of the answer and the
tenant is in it whether or not the caller remembered to think about it.

The second half of the problem is what may be cached at all. A cache that stores a
sampled answer is a cache that pretends a random draw was a fact. `CachePolicy` refuses
those rather than storing them, and `not_cacheable` lets any code path that read
personalised memory or ran a side-effecting tool say so for the call it is inside.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.primitives import Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

    from tesserix_adk.core.capabilities import ModelCapabilities
    from tesserix_adk.core.primitives import Message
    from tesserix_adk.core.protocols import Clock, ModelProvider
    from tesserix_adk.core.provider import ModelRequest, ModelResponse
    from tesserix_adk.core.streaming import StreamEvent
    from tesserix_adk.models.embeddings import EmbeddingProvider, Vector

__all__ = [
    "CacheEntry",
    "CacheKey",
    "CacheMetrics",
    "CacheOutcome",
    "CachePolicy",
    "CacheStatus",
    "CacheStore",
    "Cacheability",
    "CachingProvider",
    "MemoryCacheStore",
    "MemorySemanticIndex",
    "SemanticConfig",
    "SemanticIndex",
    "not_cacheable",
]

_MARK: ContextVar[str | None] = ContextVar("tesserix_adk_not_cacheable", default=None)


class CacheStatus(StrEnum):
    """What the cache did for one call, as the word an operator reads in a trace."""

    HIT = "hit"
    SEMANTIC_HIT = "semantic_hit"
    MISS = "miss"
    REFUSED = "refused"
    STORE_UNAVAILABLE = "store_unavailable"


@dataclass(frozen=True, slots=True)
class Cacheability:
    """Whether a call may be cached, and the rule that decided it.

    Args:
        allowed: Whether the answer may be stored and later served.
        reason: Why not, where it may not. Empty when it may.
    """

    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """The written-down rules about what may be cached, and for how long.

    Sampling settings are the ones the caller declares in `parameters`. A call that samples
    must say so there: the kit cannot see a provider's defaults, and treating an
    undeclared setting as non-deterministic would refuse every call anyone makes.

    Args:
        ttl_seconds: How long an entry may be served for.
        deterministic_only: Refuse to cache a call whose declared sampling admits variety.

    Raises:
        ValueError: If the TTL is negative.
    """

    ttl_seconds: float = 3600.0
    deterministic_only: bool = True

    def __post_init__(self) -> None:
        """Refuse a lifetime that would make every entry stale on arrival."""
        if self.ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must not be negative, got {self.ttl_seconds}")

    def verdict(self, parameters: Mapping[str, object]) -> Cacheability:
        """Decide whether a call with these declared parameters may be cached."""
        marked = _MARK.get()
        if marked is not None:
            return Cacheability(allowed=False, reason=marked)
        if not self.deterministic_only:
            return Cacheability(allowed=True)
        temperature = parameters.get("temperature")
        if isinstance(temperature, (int, float)) and temperature > 0:
            return Cacheability(
                allowed=False, reason=f"temperature {temperature} samples rather than decides"
            )
        drawn = parameters.get("n")
        if isinstance(drawn, int) and drawn > 1:
            return Cacheability(allowed=False, reason=f"n={drawn} asks for more than one draw")
        return Cacheability(allowed=True)


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Every determinant of an answer, so that changing one cannot serve the old answer.

    Args:
        tenant: The isolation boundary. In the key as well as in the wrapper that built
            it, because a shared store holds more than one tenant's entries.
        model: Which model answered.
        prompt: A digest of the assembled prompt as it goes on the wire.
        tools: A digest of the tool schemas the model was told about.
        output_schema: The hash of the output contract, or an empty string for none.
        parameters: A digest of the declared call parameters.
        prompt_version: The prompt design's version, so retiring one stops serving it.
        model_version: The model build, so a vendor's silent upgrade is a miss.
    """

    tenant: str
    model: str
    prompt: str
    tools: str
    output_schema: str
    parameters: str
    prompt_version: str
    model_version: str

    @property
    def digest(self) -> str:
        """The key as one string, which is what a store is keyed by."""
        parts = (
            self.tenant,
            self.model,
            self.prompt,
            self.tools,
            self.output_schema,
            self.parameters,
            self.prompt_version,
            self.model_version,
        )
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One stored answer and everything needed to decide whether to serve it.

    Args:
        key: What was asked, in full.
        response: What came back.
        stored_at: When it was stored, on the cache's own clock.
        expires_at: When it stops being servable.
        embedding_model: Which embedding model indexed it, where a semantic tier is on.
        threshold: The similarity a semantic match had to reach, recorded with the entry
            so that reading it back does not depend on today's configuration.
    """

    key: CacheKey
    response: ModelResponse
    stored_at: float
    expires_at: float
    embedding_model: str | None = None
    threshold: float | None = None

    def expired(self, now: float) -> bool:
        """Whether this entry has outlived its TTL."""
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class CacheOutcome:
    """What the cache did for one call, handed to an observer for the run's trace.

    Args:
        status: Hit, miss, refusal or outage.
        digest: The key consulted.
        saved: The usage the caller did not spend, which is zero on anything but a hit.
        reason: Why a refusal was refused, or why an outage degraded.
        similarity: How close a semantic match was, where one was served.
    """

    status: CacheStatus
    digest: str
    saved: Usage
    reason: str = ""
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class CacheMetrics:
    """Counters an operator reads to decide whether the cache earns its risk.

    Args:
        hits: Answers served from an exact key.
        semantic_hits: Answers served from a near match.
        misses: Cacheable calls the cache could not answer.
        refusals: Calls the rules said may not be cached.
        stores: Entries written.
        coalesced: Callers that waited on somebody else's call rather than making one.
        store_failures: Store operations that failed and were degraded past.
        saved: Total usage not spent because an answer was served.
    """

    hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    refusals: int = 0
    stores: int = 0
    coalesced: int = 0
    store_failures: int = 0
    saved: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))


@runtime_checkable
class CacheStore(Protocol):
    """Where entries live. Narrow, so no store is hardwired into the kit."""

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry stored under `digest`, or `None`.

        Raises:
            Exception: Any store failure. The caller degrades to a live call.
        """
        ...

    async def put(self, entry: CacheEntry) -> None:
        """Store `entry` under its key's digest.

        Raises:
            Exception: Any store failure. The caller degrades to not caching.
        """
        ...

    async def purge(self, **by: str | None) -> int:
        """Remove entries matching every criterion given, and return how many.

        Criteria are `tenant`, `prompt_version`, `model_version` and `digest`. Called with
        none, it removes everything.

        Raises:
            Exception: Any store failure.
        """
        ...


@runtime_checkable
class SemanticIndex(Protocol):
    """Where the vectors behind an optional near-match tier live."""

    async def add(self, digest: str, vector: Vector, *, tenant: str, embedding_model: str) -> None:
        """Record `vector` as the embedding of the call stored under `digest`."""
        ...

    async def nearest(
        self, vector: Vector, *, tenant: str, embedding_model: str
    ) -> tuple[str, float] | None:
        """Return the closest digest and its similarity, within the tenant and model."""
        ...

    async def purge(self, **by: str | None) -> int:
        """Remove indexed vectors matching every criterion given, and return how many."""
        ...


class MemoryCacheStore:
    """An in-process store, for a single replica and for tests.

    Holds entries in a dict and nowhere else, so nothing is written to disk and nothing
    outlives the process. A deployment with more than one replica wants a shared store.
    """

    def __init__(self) -> None:
        self._held: dict[str, CacheEntry] = {}

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry stored under `digest`, or `None`."""
        return self._held.get(digest)

    async def put(self, entry: CacheEntry) -> None:
        """Store `entry`, replacing anything under the same digest."""
        self._held[entry.key.digest] = entry

    async def purge(self, **by: str | None) -> int:
        """Remove entries matching every criterion given, and return how many."""
        doomed = [digest for digest, entry in self._held.items() if _matches(entry, by)]
        for digest in doomed:
            del self._held[digest]
        return len(doomed)

    async def every(self) -> list[CacheEntry]:
        """Every entry held, for inspection and for tests."""
        return list(self._held.values())


class MemorySemanticIndex:
    """An in-process vector index, exact over few entries and honest about being so.

    Scans every vector for the tenant, which is fine for a replica-local cache and is not
    a vector database. A deployment needing one points `SemanticIndex` at it.
    """

    def __init__(self) -> None:
        self._held: dict[str, tuple[Vector, str, str]] = {}

    async def add(self, digest: str, vector: Vector, *, tenant: str, embedding_model: str) -> None:
        """Record `vector` under `digest` for this tenant and embedding model."""
        self._held[digest] = (vector, tenant, embedding_model)

    async def nearest(
        self, vector: Vector, *, tenant: str, embedding_model: str
    ) -> tuple[str, float] | None:
        """Return the closest digest and its cosine similarity, or `None` if empty."""
        candidates = [
            (digest, _cosine(vector, held))
            for digest, (held, whose, model) in self._held.items()
            if whose == tenant and model == embedding_model
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[1])

    async def purge(self, **by: str | None) -> int:
        """Remove vectors for the tenant or digest given, and return how many."""
        wanted_tenant = by.get("tenant")
        wanted_digest = by.get("digest")
        doomed = [
            digest
            for digest, (_, whose, _) in self._held.items()
            if (wanted_tenant is None or whose == wanted_tenant)
            and (wanted_digest is None or digest == wanted_digest)
        ]
        for digest in doomed:
            del self._held[digest]
        return len(doomed)


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    """The optional near-match tier, which is off unless one of these is passed.

    Args:
        embedder: What turns a prompt into a vector.
        index: Where those vectors live.
        model: Which embedding model `embedder` runs. Recorded on every entry, and
            compared on every lookup, so upgrading the model invalidates the old vectors
            rather than comparing across two vector spaces.
        threshold: The cosine similarity a match must reach. Below it, the call is a miss.

    Raises:
        ValueError: If the threshold is outside 0 to 1.
    """

    embedder: EmbeddingProvider
    index: SemanticIndex
    model: str
    threshold: float = 0.95

    def __post_init__(self) -> None:
        """Refuse a threshold that is not a similarity."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")


class CachingProvider:
    """A `ModelProvider` that answers from a previous call where the rules allow it.

    Wraps another provider, so a consumer gets caching by changing where the provider is
    constructed and nothing else. Bound to one tenant: a provider built for one customer
    has no way to ask the store for another's entry, whatever the key says.

    Streaming is not cached. A cached stream is a replay of tokens that already arrived,
    which is a different feature wearing this one's name, so `stream` passes straight
    through.
    """

    def __init__(
        self,
        inner: ModelProvider,
        store: CacheStore,
        *,
        tenant: str,
        policy: CachePolicy | None = None,
        parameters: Mapping[str, object] | None = None,
        prompt_version: str = "",
        model_version: str = "",
        semantic: SemanticConfig | None = None,
        clock: Clock | None = None,
        observer: Callable[[CacheOutcome], None] | None = None,
    ) -> None:
        """Wrap `inner`, caching its answers in `store` for `tenant`.

        Raises:
            ValueError: If the tenant is empty. A run with no tenant cannot be isolated.
        """
        if not tenant:
            raise ValueError("tenant must not be empty; a cache with no tenant cannot isolate one")
        self._inner = inner
        self._store = store
        self._tenant = tenant
        self._policy = policy or CachePolicy()
        self._parameters = dict(parameters or {})
        self._prompt_version = prompt_version
        self._model_version = model_version
        self._semantic = semantic
        self._clock = clock
        self._observer = observer
        self._metrics = CacheMetrics()
        self._inflight: dict[str, asyncio.Future[ModelResponse]] = {}

    @property
    def name(self) -> str:
        """The wrapped provider's name. A cache is not a different provider."""
        return self._inner.name

    @property
    def capabilities(self) -> ModelCapabilities:
        """The wrapped provider's capabilities, unchanged by caching."""
        return self._inner.capabilities

    @property
    def metrics(self) -> CacheMetrics:
        """A snapshot of the counters, taken now."""
        return self._metrics

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count tokens the way the wrapped provider does."""
        return self._inner.count_tokens(messages)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream from the wrapped provider, uncached."""
        return await self._inner.stream(request)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a completion, from the cache where the rules and the key allow it.

        Raises:
            ProviderError: Whatever the wrapped provider raises. A failed call is never
                cached, and never leaves the key wedged for the next caller.
        """
        verdict = self._policy.verdict(self._parameters)
        key = self._key_for(request)
        if not verdict.allowed:
            self._count(refusals=1)
            self._told(CacheStatus.REFUSED, key.digest, reason=verdict.reason)
            return await self._inner.complete(request)
        served = await self._served(request, key)
        if served is not None:
            return served
        return await self._live(request, key)

    async def forget(self, **by: str | None) -> int:
        """Remove this tenant's entries matching `by`, and return how many.

        Called with nothing, it removes everything stored for this tenant, which is what
        an erasure request needs. `prompt_version` and `model_version` narrow it to what a
        retirement invalidated.

        Raises:
            Exception: Whatever the store raises. Erasure that silently failed is worse
                than erasure that failed loudly.
        """
        removed = await self._store.purge(tenant=self._tenant, **by)
        if self._semantic is not None:
            await self._semantic.index.purge(tenant=self._tenant)
        return removed

    def _key_for(self, request: ModelRequest) -> CacheKey:
        """Build the key from every determinant of the answer."""
        return CacheKey(
            tenant=self._tenant,
            model=request.model,
            prompt=_digest_of([message.model_dump(mode="json") for message in request.messages]),
            tools=_digest_of([tool.model_dump(mode="json") for tool in request.tools]),
            output_schema=request.output_schema_hash or "",
            parameters=_digest_of(self._parameters),
            prompt_version=self._prompt_version,
            model_version=self._model_version,
        )

    async def _served(self, request: ModelRequest, key: CacheKey) -> ModelResponse | None:
        """Look the call up exactly, then approximately, and answer if either can."""
        try:
            entry = await self._store.get(key.digest)
        except Exception as unreachable:
            self._count(store_failures=1)
            self._told(CacheStatus.STORE_UNAVAILABLE, key.digest, reason=str(unreachable))
            return None
        if entry is not None and not entry.expired(self._now()):
            self._count(hits=1, saved=entry.response.usage)
            self._told(CacheStatus.HIT, key.digest, saved=entry.response.usage)
            return entry.response
        if entry is not None:
            await self._drop(key.digest)
        return await self._near(request, key)

    async def _near(self, request: ModelRequest, key: CacheKey) -> ModelResponse | None:
        """Consult the semantic tier, where one is configured and reaches its threshold."""
        if self._semantic is None:
            return None
        vector = await self._vector_for(request, self._semantic)
        found = await self._semantic.index.nearest(
            vector, tenant=self._tenant, embedding_model=self._semantic.model
        )
        if found is None or found[1] < self._semantic.threshold:
            return None
        entry = await self._store.get(found[0])
        if entry is None or entry.expired(self._now()):
            return None
        self._count(semantic_hits=1, saved=entry.response.usage)
        self._told(
            CacheStatus.SEMANTIC_HIT, key.digest, saved=entry.response.usage, similarity=found[1]
        )
        return entry.response

    async def _live(self, request: ModelRequest, key: CacheKey) -> ModelResponse:
        """Call the provider, or wait on the call somebody else already made."""
        waiting = self._inflight.get(key.digest)
        if waiting is not None:
            self._count(coalesced=1)
            return await asyncio.shield(waiting)
        answer: asyncio.Future[ModelResponse] = asyncio.get_running_loop().create_future()
        self._inflight[key.digest] = answer
        try:
            response = await self._inner.complete(request)
        except BaseException as failed:
            answer.set_exception(failed)
            answer.exception()
            raise
        finally:
            del self._inflight[key.digest]
        answer.set_result(response)
        self._count(misses=1)
        self._told(CacheStatus.MISS, key.digest)
        await self._stored(request, key, response)
        return response

    async def _stored(self, request: ModelRequest, key: CacheKey, response: ModelResponse) -> None:
        """Write the answer, and index it where a semantic tier is on."""
        now = self._now()
        entry = CacheEntry(
            key=key,
            response=response,
            stored_at=now,
            expires_at=now + self._policy.ttl_seconds,
            embedding_model=self._semantic.model if self._semantic else None,
            threshold=self._semantic.threshold if self._semantic else None,
        )
        try:
            await self._store.put(entry)
        except Exception as unreachable:
            self._count(store_failures=1)
            self._told(CacheStatus.STORE_UNAVAILABLE, key.digest, reason=str(unreachable))
            return
        self._count(stores=1)
        semantic = self._semantic
        if semantic is not None:
            await semantic.index.add(
                key.digest,
                await self._vector_for(request, semantic),
                tenant=self._tenant,
                embedding_model=semantic.model,
            )

    async def _vector_for(self, request: ModelRequest, semantic: SemanticConfig) -> Vector:
        """Embed what the caller asked, which is what a near match is near to."""
        vectors = await semantic.embedder.embed([_asked_text(request)], model=semantic.model)
        return vectors[0]

    async def _drop(self, digest: str) -> None:
        """Remove an expired entry, quietly: it is already not being served."""
        with contextlib.suppress(Exception):
            await self._store.purge(digest=digest)

    def _told(
        self,
        status: CacheStatus,
        digest: str,
        *,
        saved: Usage | None = None,
        reason: str = "",
        similarity: float | None = None,
    ) -> None:
        """Hand the outcome to whoever is recording it on the run."""
        if self._observer is None:
            return
        self._observer(
            CacheOutcome(
                status=status,
                digest=digest,
                saved=saved or Usage(input_tokens=0, output_tokens=0),
                reason=reason,
                similarity=similarity,
            )
        )

    def _count(self, *, saved: Usage | None = None, **counters: int) -> None:
        """Move the counters, which are a snapshot rather than a mutable record."""
        moved = {name: getattr(self._metrics, name) + value for name, value in counters.items()}
        if saved is not None:
            moved["saved"] = self._metrics.saved + saved
        self._metrics = replace(self._metrics, **moved)

    def _now(self) -> float:
        """The cache's clock, which a test owns and production does not."""
        return self._clock.now() if self._clock is not None else time.monotonic()


@contextlib.contextmanager
def not_cacheable(reason: str) -> Iterator[None]:
    """Mark every model call made inside this block as one that must not be cached.

    For the paths the rules cannot see from the request alone: a personalised memory read,
    a side-effecting tool's result, an approval-gated answer. The mark belongs to the task
    that set it and is removed when the block ends.

    Example:
        >>> with not_cacheable("read the user's own history"):
        ...     pass
    """
    token = _MARK.set(reason)
    try:
        yield
    finally:
        _MARK.reset(token)


def _matches(entry: CacheEntry, by: Mapping[str, str | None]) -> bool:
    """Whether an entry satisfies every criterion that was given a value."""
    against = {
        "tenant": entry.key.tenant,
        "prompt_version": entry.key.prompt_version,
        "model_version": entry.key.model_version,
        "digest": entry.key.digest,
    }
    return all(against.get(name) == wanted for name, wanted in by.items() if wanted is not None)


def _digest_of(value: object) -> str:
    """A stable digest of anything JSON can hold, key order included."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _asked_text(request: ModelRequest) -> str:
    """The text of the last message, which is what a near match is measured against."""
    if not request.messages:
        return ""
    parts = request.messages[-1].content
    return "".join(getattr(part, "text", "") for part in parts)


def _cosine(left: Vector, right: Vector) -> float:
    """Cosine similarity, zero for a zero vector rather than a division error."""
    if len(left) != len(right):
        return 0.0
    dot = sum(one * other for one, other in zip(left, right, strict=True))
    size = math.sqrt(sum(one * one for one in left)) * math.sqrt(sum(one * one for one in right))
    return dot / size if size else 0.0
