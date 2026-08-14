"""Turning text into vectors: batched, cached on content, and honest about failure.

Two costs dominate an ingest. The first is the provider bill, which is paid again on every
re-index unless something remembers that this exact text under this exact model already has
a vector; the cache here is keyed on precisely that and nothing else, so an unchanged corpus
re-indexes for free and an edited passage is the only one sent.

The second is paid later, by whoever debugs a corpus that has holes in it. An embedder that
answers a zero vector for the batch the provider rate-limited leaves passages that are
simply never retrieved, with nothing in the index to say so. This one stops instead, and
says which batch failed and where to resume, so an ingest is restartable rather than
restarted.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections import OrderedDict
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core import (
    AdkModel,
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    RetryConfig,
    SchemaViolationError,
    Usage,
    current_tenant,
)
from tesserix_adk.runtime import SystemClock
from tesserix_adk.runtime.retry import RetryPlan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from random import Random

    from tesserix_adk.core import Clock

__all__ = [
    "BatchVectors",
    "BatchedEmbedder",
    "EmbeddedBatch",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingModel",
    "EmbeddingStats",
    "MemoryEmbeddingCache",
    "Vector",
    "VectorSource",
    "embedding_key",
    "normalised",
]

type Vector = tuple[float, ...]
"""One embedding, of the width its model declares."""


def normalised(text: str) -> str:
    """The form of `text` that is embedded and keyed.

    Composed to NFC and stripped at the ends: two spellings of one accented word are the
    same text to a reader and should not be two cache entries with two bills, and neither
    should a chunk that picked up a trailing newline from its document.
    """
    return unicodedata.normalize("NFC", text).strip()


def embedding_key(model: EmbeddingModel, text: str, *, tenant: str) -> str:
    """The cache key for `text` under `model`, scoped to `tenant`.

    Content-addressed, so nothing has to be invalidated: a new model version or an edited
    passage simply addresses somewhere else. The text is hashed rather than carried,
    because a key is copied into logs and store browsers and the corpus is not.

    Args:
        model: The model the vector under this key was produced by.
        text: The passage, normalised before hashing so two spellings of it agree.
        tenant: The tenant the entry belongs to, or "" for a corpus shared by all of them.
    """
    parts = (model.name, model.version, str(model.dimension), tenant, normalised(text))
    return sha256("\0".join(parts).encode()).hexdigest()


class EmbeddingModel(AdkModel):
    """Which model produced a vector, in enough detail to know when it did not.

    Args:
        name: The provider's id for the model.
        version: The version of it, where the provider states one. Part of the cache key:
            a model that was retrained under its old name answers a different space, and
            reusing the old vectors mixes two spaces in one index.
        dimension: The width of the vectors it answers.
        max_batch: The most texts it will accept in one call.
    """

    name: str = Field(min_length=1)
    version: str = ""
    dimension: int = Field(gt=0)
    max_batch: int = Field(default=64, gt=0)


class BatchVectors(AdkModel):
    """One provider call's answer.

    Args:
        vectors: One per text, in the order the texts were given.
        usage: What the call consumed, for attribution to the ingest that made it.
    """

    vectors: tuple[Vector, ...]
    usage: Usage = Usage(input_tokens=0, output_tokens=0)


class EmbeddingStats(AdkModel):
    """What an embedding call did, in the terms its bill is argued in.

    Args:
        requested: Texts asked for, counting repeats.
        cached: How many were answered from the cache.
        embedded: How many reached the provider.
        batches: Provider calls made.
        retries: Attempts beyond the first.
        cache_failures: Cache reads and writes that failed and were worked around. Not an
            error, but the difference between a cheap ingest and an expensive one, so it
            is counted where somebody will see it.
    """

    requested: int = Field(default=0, ge=0)
    cached: int = Field(default=0, ge=0)
    embedded: int = Field(default=0, ge=0)
    batches: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    cache_failures: int = Field(default=0, ge=0)

    @property
    def hit_rate(self) -> float:
        """The share of texts the cache answered, or 0.0 where nothing was asked for."""
        return self.cached / self.requested if self.requested else 0.0


class EmbeddedBatch(AdkModel):
    """Vectors for the texts that were asked for, and what producing them cost.

    Args:
        vectors: One per text, in the order the texts were given, repeats included.
        usage: The total across the provider calls actually made.
        stats: How much of the work was avoided.
    """

    vectors: tuple[Vector, ...]
    usage: Usage = Usage(input_tokens=0, output_tokens=0)
    stats: EmbeddingStats = EmbeddingStats()


@runtime_checkable
class VectorSource(Protocol):
    """The vendor-facing half: one call, one batch, no batching or caching of its own."""

    @property
    def model(self) -> EmbeddingModel:
        """The model it embeds with."""
        ...

    async def vectors_for(self, texts: Sequence[str]) -> BatchVectors:
        """Embed `texts`, which are within `model.max_batch` and already normalised."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """The corpus-facing half: whatever texts there are, however many calls that takes."""

    @property
    def model(self) -> EmbeddingModel:
        """The model whose space these vectors live in."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddedBatch:
        """Embed `texts` for indexing.

        Raises:
            SchemaViolationError: Where a text is empty or whitespace alone.
            EmbeddingDimensionError: Where a vector is not the model's width.
            EmbeddingUnavailableError: Where a batch could not be embedded within the
                retry budget. Carries where to resume from; nothing is substituted.
        """
        ...

    async def embed_query(self, text: str) -> Vector:
        """Embed one query, in the same space the documents were indexed in."""
        ...


@runtime_checkable
class EmbeddingCache(Protocol):
    """Somewhere vectors survive between ingests, keyed by `embedding_key`."""

    async def get(self, keys: Sequence[str]) -> Mapping[str, Vector]:
        """Return what is held for `keys`, omitting what is not."""
        ...

    async def put(self, vectors: Mapping[str, Vector]) -> None:
        """Hold `vectors` under their keys."""
        ...


class MemoryEmbeddingCache:
    """A bounded in-process cache, for a single ingest or a test.

    Least-recently-used eviction, because an ingest that is larger than the bound should
    slow down rather than exhaust the process. Nothing survives the process; a re-index
    that must be cheap wants a shared backend behind the same protocol.

    Args:
        keep: The most entries held at once.
    """

    def __init__(self, keep: int = 100_000) -> None:
        if keep < 1:
            raise ValueError("a cache that holds nothing is not a cache")
        self._keep = keep
        self._held: OrderedDict[str, Vector] = OrderedDict()

    async def get(self, keys: Sequence[str]) -> Mapping[str, Vector]:
        """Return what is held for `keys`, counting each hit as a use."""
        found = {}
        for key in keys:
            if key in self._held:
                self._held.move_to_end(key)
                found[key] = self._held[key]
        return found

    async def put(self, vectors: Mapping[str, Vector]) -> None:
        """Hold `vectors`, evicting the least recently used to stay within the bound."""
        for key, vector in vectors.items():
            self._held[key] = vector
            self._held.move_to_end(key)
        while len(self._held) > self._keep:
            self._held.popitem(last=False)


class BatchedEmbedder:
    """An `Embedder` over a `VectorSource`: batching, caching, retries and refusals.

    Args:
        source: The thing that actually calls the provider.
        cache: Where vectors are looked for first, and written after. A backend that is
            down is worked around rather than propagated — the vectors can be bought
            again — but the fallback is counted in `EmbeddingStats.cache_failures`.
        shared: Whether cache entries are shared across tenants. False, the default, keys
            every entry to the tenant in force and refuses to embed outside a tenant
            scope: one tenant's document text is not a thing to answer another's ingest
            with. True is for a public corpus and is a deliberate statement.
        retry: What to do about a provider that refused. Only failures the kit itself
            calls retryable are retried; a rate limit that asked for a delay gets it.
        concurrency: The most provider calls in flight at once.
        clock: Where waiting happens, injected so a test does not.
        random: The source of retry jitter.
    """

    def __init__(
        self,
        source: VectorSource,
        *,
        cache: EmbeddingCache | None = None,
        shared: bool = False,
        retry: RetryConfig | None = None,
        concurrency: int = 4,
        clock: Clock | None = None,
        random: Random | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("an embedder that makes no calls at once makes none at all")
        self._source = source
        self._cache = cache
        self._shared = shared
        self._plan = RetryPlan(retry or RetryConfig(), random=random)
        self._limit = asyncio.Semaphore(concurrency)
        self._clock = clock or SystemClock()

    @property
    def model(self) -> EmbeddingModel:
        """The model whose space these vectors live in."""
        return self._source.model

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddedBatch:
        """Embed `texts`, sending only what the cache does not already hold.

        Repeats within one call are sent once and answered twice: a corpus with a
        boilerplate paragraph in every document should pay for it once.

        Raises:
            MissingTenantContextError: Where the cache is tenant-isolated and no tenant
                is in force.
            SchemaViolationError: Where a text is empty or whitespace alone.
            EmbeddingDimensionError: Where a vector is not the model's width.
            EmbeddingUnavailableError: Where a batch could not be embedded within the
                retry budget.
        """
        tenant = self._tenant()
        wanted = [self._checked(text, f"texts[{index}]") for index, text in enumerate(texts)]
        unique = list(dict.fromkeys(wanted))
        keys = {text: embedding_key(self.model, text, tenant=tenant) for text in unique}

        known, read_failures = await self._held(keys)
        missing = [text for text in unique if text not in known]
        fresh, usage, retries, write_failures = await self._embed(missing, wanted, keys)

        vectors = known | fresh
        return EmbeddedBatch(
            vectors=tuple(vectors[text] for text in wanted),
            usage=usage,
            stats=EmbeddingStats(
                requested=len(wanted),
                cached=sum(1 for text in wanted if text in known),
                embedded=sum(1 for text in wanted if text in fresh),
                batches=self._batches(len(missing)),
                retries=retries,
                cache_failures=read_failures + write_failures,
            ),
        )

    async def embed_query(self, text: str) -> Vector:
        """Embed one query in the same space, through the same cache.

        Raises:
            SchemaViolationError: Where the query is empty or whitespace alone.
        """
        embedded = await self.embed_documents((self._checked(text, "text"),))
        return embedded.vectors[0]

    def _tenant(self) -> str:
        if self._shared:
            return ""
        return current_tenant(where="the embedding cache").tenant

    def _checked(self, text: str, where: str) -> str:
        cleaned = normalised(text)
        if not cleaned:
            raise SchemaViolationError(
                f"{where} is empty, and an empty text has no meaning to embed",
                model="EmbeddedBatch",
                paths=(where,),
            )
        return cleaned

    def _batches(self, count: int) -> int:
        size = self.model.max_batch
        return -(-count // size)

    async def _held(self, keys: Mapping[str, str]) -> tuple[dict[str, Vector], int]:
        """What the cache holds for `keys`, and 1 where it could not be asked."""
        if self._cache is None or not keys:
            return {}, 0
        try:
            found = await self._cache.get([keys[text] for text in keys])
        except Exception:
            return {}, 1
        held = {text: found[key] for text, key in keys.items() if key in found}
        for vector in held.values():
            self._widthed(vector, "a cached vector")
        return held, 0

    async def _embed(
        self,
        missing: Sequence[str],
        wanted: Sequence[str],
        keys: Mapping[str, str],
    ) -> tuple[dict[str, Vector], Usage, int, int]:
        size = self.model.max_batch
        batches = [tuple(missing[at : at + size]) for at in range(0, len(missing), size)]
        answered = await asyncio.gather(
            *(self._attempt(batch) for batch in batches), return_exceptions=True
        )

        fresh: dict[str, Vector] = {}
        usage = Usage(input_tokens=0, output_tokens=0)
        retries = 0
        for batch, answer in zip(batches, answered, strict=True):
            if isinstance(answer, BaseException):
                continue
            vectors, attempts = answer
            fresh.update(zip(batch, vectors.vectors, strict=True))
            usage += vectors.usage
            retries += attempts - 1

        write_failures = await self._keep({keys[text]: vector for text, vector in fresh.items()})
        self._reraise(batches, answered, wanted)
        return fresh, usage, retries, write_failures

    def _reraise(
        self,
        batches: Sequence[tuple[str, ...]],
        answered: Sequence[object],
        wanted: Sequence[str],
    ) -> None:
        """Report the first batch that failed, after the successful ones are safely cached."""
        for answer in answered:
            if isinstance(answer, asyncio.CancelledError):
                raise answer
        for index, answer in enumerate(answered):
            if not isinstance(answer, BaseException):
                continue
            if isinstance(answer, EmbeddingDimensionError):
                raise answer
            cursor = min(wanted.index(text) for text in batches[index])
            raise EmbeddingUnavailableError(
                f"batch {index} of {len(batches)} could not be embedded; resume from text {cursor}",
                batch=index,
                cursor=cursor,
            ) from answer

    async def _attempt(self, batch: tuple[str, ...]) -> tuple[BatchVectors, int]:
        attempt = 1
        while True:
            try:
                async with self._limit:
                    answered = await self._source.vectors_for(batch)
            except asyncio.CancelledError:
                raise
            except Exception as failure:
                delay = self._delay_after(failure, attempt)
                if delay is None:
                    raise
                await self._clock.sleep(delay)
                attempt += 1
                continue
            self._checked_widths(batch, answered)
            return answered, attempt

    def _delay_after(self, failure: BaseException, attempt: int) -> float | None:
        if not self._plan.retryable(failure):
            return None
        return self._plan.delay_for(attempt, retry_after=getattr(failure, "retry_after", None))

    def _checked_widths(self, batch: tuple[str, ...], answered: BatchVectors) -> None:
        if len(answered.vectors) != len(batch):
            raise EmbeddingDimensionError(
                f"asked for {len(batch)} vectors and was given {len(answered.vectors)}",
                expected=len(batch),
                received=len(answered.vectors),
            )
        for vector in answered.vectors:
            self._widthed(vector, "a vector from the provider")

    def _widthed(self, vector: Vector, what: str) -> None:
        if len(vector) != self.model.dimension:
            raise EmbeddingDimensionError(
                f"{what} is {len(vector)} wide, and the collection is {self.model.dimension}",
                expected=self.model.dimension,
                received=len(vector),
            )

    async def _keep(self, vectors: Mapping[str, Vector]) -> int:
        if self._cache is None or not vectors:
            return 0
        try:
            await self._cache.put(vectors)
        except Exception:
            return 1
        return 0
