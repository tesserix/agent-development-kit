"""The latency numbers that decide whether CPU inference is usable.

"Kit overhead under twenty milliseconds" is easy to state and says almost nothing: against
a CPU model call of five to thirty seconds it is a rounding error. What decides whether a
product feels usable is how long the first token takes, how fast tokens arrive after it,
and how much of the prompt the provider did not have to prefill — prefill is where CPU
latency goes, so the cache hit ratio belongs beside the latency rather than in a different
dashboard.

Every number here is emitted as a metric, not only as a span attribute, for the reason
`observability.metrics` gives: traces are sampled, and a latency percentile computed from
whatever the sampler kept is precise-looking and wrong.

A provider that reports no cached-token count leaves the ratio *unknown*. It is not zero
and it is not one — both of those are answers, and an unknown that reads as an answer is
how a cache that stopped working goes unnoticed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.observability.metrics import Meter

__all__ = [
    "CACHE_HIT_RATIO",
    "LATENCY_SECONDS",
    "TIME_TO_FIRST_TOKEN",
    "TOKENS_PER_SECOND",
    "CacheHits",
    "LatencyReport",
    "RunTimer",
]

TIME_TO_FIRST_TOKEN = "adk.time_to_first_token"  # noqa: S105 — a metric name, not a secret
LATENCY_SECONDS = "adk.latency_seconds"
TOKENS_PER_SECOND = "adk.tokens_per_second"
CACHE_HIT_RATIO = "adk.cache_hit_ratio"


class CacheHits(AdkModel):
    """How much of a prompt the provider did not have to prefill.

    Args:
        input_tokens: What was sent.
        cached_tokens: What was read from the provider's prefix cache, or `None` where the
            provider did not say.

    Raises:
        ValueError: If more tokens were cached than were sent, which is a reading error
            somewhere upstream and not a very good cache.
    """

    input_tokens: int = Field(default=0, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> CacheHits:
        """Refuse a ratio that would come out above one."""
        if self.cached_tokens is not None and self.cached_tokens > self.input_tokens:
            raise ValueError(
                f"cached_tokens {self.cached_tokens} is more than the {self.input_tokens} sent"
            )
        return self

    @property
    def known(self) -> bool:
        """Whether a ratio can be computed at all."""
        return self.cached_tokens is not None and self.input_tokens > 0

    @property
    def ratio(self) -> float | None:
        """Cached over sent, or `None` where nothing can be said."""
        if not self.known or self.cached_tokens is None:
            return None
        return self.cached_tokens / self.input_tokens


class LatencyReport(AdkModel):
    """What one run cost, in the terms a user experiences it.

    Args:
        time_to_first_token: Seconds from the start of the run to the first token reaching
            the caller, or `None` for a run that did not stream or produced nothing.
        seconds: The whole run, start to last token.
        output_tokens: How many tokens came back.
        tokens_per_second: The sustained rate after the first token, which is the number a
            reader perceives, or `None` where there was nothing to sustain.
        hits: What the provider said about its prefix cache.
        streaming: Whether the run streamed. A streamed and a blocking run share no
            comparable first-token number, so they are never averaged together.
        cold: Whether this was a cold start with no cache to hit. Averaging a cold start in
            with warm runs hides both.
    """

    time_to_first_token: float | None = None
    seconds: float = 0.0
    output_tokens: int = 0
    tokens_per_second: float | None = None
    hits: CacheHits = CacheHits()
    streaming: bool = True
    cold: bool = False

    def attributes(self) -> dict[str, str]:
        """Span attributes: durations and counts, never a token of content."""
        attributes = {
            "adk.latency_seconds": f"{self.seconds:.3f}",
            "adk.output_tokens": str(self.output_tokens),
            "adk.streaming": "true" if self.streaming else "false",
            "adk.cold": "true" if self.cold else "false",
        }
        if self.time_to_first_token is not None:
            attributes[TIME_TO_FIRST_TOKEN] = f"{self.time_to_first_token:.3f}"
        if self.tokens_per_second is not None:
            attributes[TOKENS_PER_SECOND] = f"{self.tokens_per_second:.2f}"
        ratio = self.hits.ratio
        if ratio is not None:
            attributes[CACHE_HIT_RATIO] = f"{ratio:.3f}"
        return attributes

    def emit(self, meter: Meter, **dimensions: str) -> None:
        """Count every number this run has, under `dimensions` plus how it was run.

        A number the run does not have is not counted. An unknown cache hit ratio recorded
        as zero is the failure this whole module exists to avoid.

        Args:
            meter: Where the counters go. Its own failures are swallowed, as a collector
                outage must never end a run.
            dimensions: What to split the series by — model and tenant bucket, typically.
        """
        labelled = {
            **dimensions,
            "streaming": "true" if self.streaming else "false",
            "cold": "true" if self.cold else "false",
        }
        counted: list[tuple[str, float | None]] = [
            (LATENCY_SECONDS, self.seconds),
            (TIME_TO_FIRST_TOKEN, self.time_to_first_token),
            (TOKENS_PER_SECOND, self.tokens_per_second),
            (CACHE_HIT_RATIO, self.hits.ratio),
        ]
        for name, value in counted:
            if value is None:
                continue
            try:
                meter.count(name, value, **labelled)
            except Exception:
                return

    def render(self) -> str:
        """One line an operator reads before deciding the machine is too small."""
        ratio = self.hits.ratio
        first = "n/a" if self.time_to_first_token is None else f"{self.time_to_first_token:.3f}s"
        rate = "n/a" if self.tokens_per_second is None else f"{self.tokens_per_second:.1f} tok/s"
        cache = "unknown" if ratio is None else f"{ratio:.0%}"
        start = "cold" if self.cold else "warm"
        mode = "stream" if self.streaming else "blocking"
        return (
            f"{start} {mode}: first token {first}, {rate}, {self.seconds:.3f}s total, cache {cache}"
        )


class RunTimer:
    """Times one run and reports it, on a clock a test can move.

    Args:
        clock: Where time comes from. `None` uses the system clock.
        streaming: Whether this run streams. A blocking run has no first token to time.
        cold: Whether this is a cold start.
    """

    def __init__(
        self, *, clock: Clock | None = None, streaming: bool = True, cold: bool = False
    ) -> None:
        self._clock = clock or SystemClock()
        self._streaming = streaming
        self._cold = cold
        self._started = self._clock.now()
        self._first: float | None = None

    def first_token(self) -> None:
        """Mark the first token reaching the caller. Later calls are ignored."""
        if self._first is None:
            self._first = self._clock.now() - self._started

    def finished(self, *, output_tokens: int, hits: CacheHits | None = None) -> LatencyReport:
        """Close the run and report it.

        Args:
            output_tokens: How many tokens came back.
            hits: What the provider said about its prefix cache, where it said anything.

        Returns:
            The report, ready to emit.
        """
        seconds = self._clock.now() - self._started
        first = self._first if self._streaming and output_tokens else None
        decoding = seconds - (first if first is not None else 0.0)
        rate = output_tokens / decoding if output_tokens and decoding > 0 else None
        return LatencyReport(
            time_to_first_token=first,
            seconds=seconds,
            output_tokens=output_tokens,
            tokens_per_second=rate,
            hits=hits or CacheHits(),
            streaming=self._streaming,
            cold=self._cold,
        )
