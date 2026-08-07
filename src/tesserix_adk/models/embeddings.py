"""Turning many one-text embedding calls into few provider calls, safely.

Indexing a document asks for one vector at a time, which is the right thing for a caller
to write and the wrong thing to put on the wire: hundreds of sequential round trips are
slow, cost more than the same work batched, and spend the rate-limit headroom the model
calls need. So concurrent requests are coalesced behind the same interface a caller
already uses.

Everything else here exists because coalescing is where vectors get mixed up. A caller
must get the vector for its own text or an error — never a neighbour's, never a zero
vector, never one the provider truncated. That is why the batch is keyed by model, width
and tenant, why each waiter is matched to its own text by digest rather than by trust in
the ordering, and why a batch that fails is bisected until the offending item is the only
caller that hears about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from tesserix_adk.core.errors import ContextWindowExceededError, ModelResponseError
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "BatchConfig",
    "BatchingEmbedder",
    "EmbeddingLimits",
    "EmbeddingMetrics",
    "EmbeddingProvider",
    "Vector",
]

type Vector = tuple[float, ...]

_BYTES_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class EmbeddingLimits:
    """What one provider says it will accept in a single embedding call.

    Declared rather than assumed: a vendor that raises its batch ceiling should improve
    throughput by saying so, not by waiting for a constant here to be edited.

    Attributes:
        max_items: Texts allowed in one call.
        max_bytes: Total encoded size allowed in one call.
        max_item_tokens: The longest single text the model will read. A text past it is
            refused here rather than truncated by the vendor, since a truncated embedding
            is a wrong answer with nothing to show that it is wrong.
        dimensions: The width every returned vector must have.
    """

    max_items: int = 64
    max_bytes: int = 1_000_000
    max_item_tokens: int = 8191
    dimensions: int = 1536


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A provider that turns texts into vectors, one call at a time."""

    @property
    def name(self) -> str:
        """The provider name, for an error message and a metric label."""
        ...

    def limits(self, model: str) -> EmbeddingLimits:
        """What this model accepts in one call."""
        ...

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        """Return one vector per text, in the order the texts were given."""
        ...


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """How long to wait for a batch to fill, and how big to let it get.

    The ceilings here may narrow what the provider declares and never widen it: a wait
    that is too short costs round trips, but a batch past what the vendor accepts costs
    the whole batch.

    Attributes:
        max_wait_seconds: How long a partly-filled batch is held. Short, because this is
            latency added to work somebody is waiting for.
        max_items: A ceiling of the caller's own, applied under the provider's.
        max_bytes: The same for total encoded size.
    """

    max_wait_seconds: float = 0.02
    max_items: int = 4096
    max_bytes: int = 1_000_000


@dataclass(frozen=True, slots=True)
class EmbeddingMetrics:
    """A snapshot of what the embedder did, for an operator tuning the window.

    Attributes:
        requests: Texts callers asked for.
        batches: Provider calls made.
        deduplicated: Requests answered from a copy already in the same batch.
        bypassed: Interactive requests that skipped the window.
        isolated: Batches re-sent in halves to find which item the provider refused.
        flushed_full: Batches sent because they filled.
        flushed_due: Batches sent because their window expired.
    """

    requests: int = 0
    batches: int = 0
    deduplicated: int = 0
    bypassed: int = 0
    isolated: int = 0
    flushed_full: int = 0
    flushed_due: int = 0


@dataclass(frozen=True, slots=True)
class _Key:
    """What must match for two requests to be allowed in one batch."""

    model: str
    tenant: str
    dimensions: int


@dataclass(slots=True)
class _Waiting:
    """One caller, its text, and the digest its answer is checked against."""

    text: str
    digest: str
    answer: asyncio.Future[Vector]


@dataclass(slots=True)
class _Batch:
    """A batch still filling, and the flush its window will trigger."""

    waiting: list[_Waiting] = field(default_factory=list)
    bytes_held: int = 0
    timer: asyncio.Task[None] | None = None


class BatchingEmbedder:
    """Coalesces concurrent single-text requests into provider batches.

    A caller asks for one vector and gets one vector; whether that crossed the wire on its
    own or with two hundred others is this class's business. Requests are grouped by
    model, tenant and vector width, held for at most `max_wait_seconds`, and sent as soon
    as the batch is full.

    Args:
        provider: Who does the embedding.
        config: The caller's own ceilings, applied under the provider's declared ones.
        clock: Where the window's time comes from, injected so a test does not sleep.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: BatchConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._provider = provider
        self._config = config or BatchConfig()
        self._clock = clock or SystemClock()
        self._filling: dict[_Key, _Batch] = {}
        self._metrics = EmbeddingMetrics()
        self._closed = False

    @property
    def metrics(self) -> EmbeddingMetrics:
        """A snapshot of the counters, taken now."""
        return self._metrics

    async def embed(
        self, text: str, *, model: str, tenant: str = "default", interactive: bool = False
    ) -> Vector:
        """Return the vector for `text`, batched with whatever else is in flight.

        Args:
            text: What to embed.
            model: Which embedding model to use.
            tenant: Whose text it is. Two tenants are never in one batch.
            interactive: Whether somebody is waiting on this answer. An interactive
                request skips the coalescing window, since a query embedding delayed
                behind a bulk flush is latency a user sees.

        Raises:
            ContextWindowExceededError: If the text is longer than the model will read.
                Refused here rather than truncated by the vendor.
            ModelResponseError: If the provider answered with the wrong number of vectors,
                or vectors of the wrong width.
            RuntimeError: If the embedder has been closed.
        """
        if self._closed:
            raise RuntimeError("this embedder is closed; nothing further will be sent")
        limits = self._provider.limits(model)
        self._refuse_if_too_long(text, limits)
        self._metrics = replace(self._metrics, requests=self._metrics.requests + 1)
        if interactive:
            self._metrics = replace(self._metrics, bypassed=self._metrics.bypassed + 1)
            return (await self._sent((text,), model=model))[0]
        return await self._queued(text, model=model, tenant=tenant, limits=limits)

    async def aclose(self) -> None:
        """Send whatever is still waiting, then refuse anything new."""
        self._closed = True
        for key in list(self._filling):
            await self._flush(key, due=False)

    async def __aenter__(self) -> Self:
        """Return self, so the window's tasks are owned by a block."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Flush and close."""
        await self.aclose()

    async def _queued(
        self, text: str, *, model: str, tenant: str, limits: EmbeddingLimits
    ) -> Vector:
        """Put one caller in the batch for its key, flushing when the batch is full."""
        key = _Key(model=model, tenant=tenant, dimensions=limits.dimensions)
        held = self._filling.get(key)
        if held is not None and self._would_overflow(held, text, limits):
            # Sent as it stands rather than one item over: a batch past what the vendor
            # accepts costs the whole batch, not the item that tipped it.
            await self._flush(key, due=False)
        batch = self._filling.setdefault(key, _Batch())
        answer: asyncio.Future[Vector] = asyncio.get_running_loop().create_future()
        batch.waiting.append(_Waiting(text=text, digest=_digest(text), answer=answer))
        batch.bytes_held += len(text.encode())
        if batch.timer is None:
            batch.timer = asyncio.ensure_future(self._after_the_window(key))
        if self._full(batch, limits):
            await self._flush(key, due=False)
        return await answer

    def _would_overflow(self, batch: _Batch, text: str, limits: EmbeddingLimits) -> bool:
        """Would adding this text put the batch past a ceiling it must stay under?"""
        return batch.bytes_held + len(text.encode()) > min(limits.max_bytes, self._config.max_bytes)

    def _full(self, batch: _Batch, limits: EmbeddingLimits) -> bool:
        """Has this batch reached the narrower of the declared and configured ceilings?"""
        distinct = len({one.text for one in batch.waiting})
        return distinct >= min(limits.max_items, self._config.max_items) or batch.bytes_held >= min(
            limits.max_bytes, self._config.max_bytes
        )

    async def _after_the_window(self, key: _Key) -> None:
        """Flush this key when its window expires, however full it happens to be."""
        with contextlib.suppress(asyncio.CancelledError):
            await self._clock.sleep(self._config.max_wait_seconds)
            await self._flush(key, due=True)

    async def _flush(self, key: _Key, *, due: bool) -> None:
        """Send the batch held for `key` and answer everyone waiting in it."""
        batch = self._filling.pop(key, None)
        if batch is None:
            # A window that expired while close was awaiting an earlier key already sent it.
            return
        if batch.timer is not None and not due:
            batch.timer.cancel()
        counted = "flushed_due" if due else "flushed_full"
        self._metrics = replace(self._metrics, **{counted: getattr(self._metrics, counted) + 1})
        await self._answer(batch.waiting, model=key.model)

    async def _answer(self, waiting: list[_Waiting], *, model: str) -> None:
        """Embed what these callers asked for, deduplicated, and hand each its own vector."""
        wanted: dict[str, list[_Waiting]] = {}
        for one in waiting:
            wanted.setdefault(one.text, []).append(one)
        self._metrics = replace(
            self._metrics, deduplicated=self._metrics.deduplicated + len(waiting) - len(wanted)
        )
        await self._resolved(tuple(wanted), wanted, model=model)

    async def _resolved(
        self, texts: tuple[str, ...], wanted: dict[str, list[_Waiting]], *, model: str
    ) -> None:
        """Send `texts`, bisecting on failure so one bad item loses only its own caller."""
        try:
            vectors = await self._sent(texts, model=model)
        except Exception as failed:
            if len(texts) == 1:
                _deliver(wanted[texts[0]], failed)
                return
            self._metrics = replace(self._metrics, isolated=self._metrics.isolated + 1)
            middle = len(texts) // 2
            await self._resolved(texts[:middle], wanted, model=model)
            await self._resolved(texts[middle:], wanted, model=model)
            return
        for text, vector in zip(texts, vectors, strict=True):
            for one in wanted[text]:
                # The digest is what makes this a match rather than a hope: a reordering
                # bug here hands a caller a plausible vector for somebody else's text.
                if one.digest != _digest(text):  # pragma: no cover - defended, not reachable
                    _deliver([one], ModelResponseError("a vector was matched to the wrong text"))
                    continue
                if not one.answer.done():
                    one.answer.set_result(vector)

    async def _sent(self, texts: tuple[str, ...], *, model: str) -> tuple[Vector, ...]:
        """One provider call, with what came back checked before anyone is given it."""
        self._metrics = replace(self._metrics, batches=self._metrics.batches + 1)
        vectors = tuple(tuple(vector) for vector in await self._provider.embed(texts, model=model))
        width = self._provider.limits(model).dimensions
        if len(vectors) != len(texts):
            raise ModelResponseError(
                f"{self._provider.name} answered {len(vectors)} vectors for {len(texts)} texts",
                details={"model": model},
            )
        wrong = next((one for one in vectors if len(one) != width), None)
        if wrong is not None:
            raise ModelResponseError(
                f"{self._provider.name} answered a vector of width {len(wrong)}, not {width}",
                details={"model": model},
            )
        return vectors

    def _refuse_if_too_long(self, text: str, limits: EmbeddingLimits) -> None:
        """Refuse a text past the model's window here, before it is silently truncated."""
        counted = len(text.encode()) // _BYTES_PER_TOKEN
        if counted > limits.max_item_tokens:
            raise ContextWindowExceededError(
                f"a text of about {counted} tokens is past this model's {limits.max_item_tokens}",
                counted=counted,
                limit=limits.max_item_tokens,
            )


def _digest(text: str) -> str:
    """A short digest of a text, for matching a returned vector to the caller that wants it."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _deliver(waiting: list[_Waiting], failed: BaseException) -> None:
    """Give every caller of one text the failure that its own item caused."""
    for one in waiting:
        if not one.answer.done():
            one.answer.set_exception(failed)
