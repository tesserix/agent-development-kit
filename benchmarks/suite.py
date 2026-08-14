"""The paths a consuming product pays for on every call.

Every scenario runs against scripted providers and local fakes, so the numbers measure the
kit and not somebody's network. Nothing here needs a credential, and a scenario that
reached for one would be measuring the wrong thing anyway.

Token overhead is measured where the kit assembles a prompt: a refactor that adds a
sentence of boilerplate to every request shows up here as a token delta rather than as a
line on a consumer's invoice six months later.
"""

from __future__ import annotations

import asyncio
import hashlib
import statistics
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tesserix_adk.core import Agent, TextPart, ToolCall, Usage
from tesserix_adk.models import BatchingEmbedder, EmbeddingLimits
from tesserix_adk.observability import CacheHits, LatencyReport, RunTimer
from tesserix_adk.runtime import AgentRunner, ModelResponse, ToolDeclaration
from tesserix_adk.testing import CAPABLE, FakeToolRegistry, ScriptedProvider, estimate_tokens
from tesserix_adk.testing.benchmarks import Metric, Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core import Message
    from tesserix_adk.models import Vector
    from tesserix_adk.runtime import ModelRequest

NATIVE = CAPABLE.declaring(structured_output=True)
LOOKUPS = ("timetable", "weather", "hotels", "events")
ASKED = "Four nights near Kyoto, arriving from Osaka."


class TripPlan(BaseModel):
    """The shape a structured answer is validated against.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


PLANNER = Agent(
    name="planner",
    instructions="Plan trips. Cite the timetable before recommending a leg.",
    model="claude-sonnet-5",
    tools=LOOKUPS,
    free_text=True,
)
STRUCTURED = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    output_type=TripPlan,
)


class Tokens:
    """The tokens the kit assembled into requests, across every scenario that reports them."""

    def __init__(self) -> None:
        self.total = 0

    def count(self, provider: ScriptedProvider) -> None:
        """Add what this provider was sent, counted the way a provider without one would."""
        self.total += sum(estimate_tokens(one.messages) for one in provider.requests)

    def __call__(self) -> int:
        """The running total, which the harness brackets around a measured round."""
        return self.total


def answered(content: str = "Kyoto, four nights.") -> ModelResponse:
    """A plain answer with the usage a vendor would report."""
    return ModelResponse(content=content, usage=Usage(input_tokens=812, output_tokens=24))


def structured() -> ModelResponse:
    """An answer shaped for the declared output type."""
    return ModelResponse(
        content='{"destination": "Kyoto", "nights": 4}',
        usage=Usage(input_tokens=812, output_tokens=24),
    )


def asked_for(*names: str) -> ModelResponse:
    """A turn that calls every named tool at once."""
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(id=f"c{index}", name=name, arguments={"leg": "Osaka to Kyoto"})
            for index, name in enumerate(names)
        ),
        usage=Usage(input_tokens=812, output_tokens=48),
    )


def declarations() -> FakeToolRegistry:
    """Four tools that answer immediately, so the measurement is the kit's own dispatch."""
    return FakeToolRegistry(
        {name: (lambda leg: f"{leg}: 09:12, 11:40") for name in LOOKUPS},
        declarations={
            name: ToolDeclaration(name=name, parameters={"type": "object"}) for name in LOOKUPS
        },
    )


def scenarios() -> tuple[Scenario, ...]:
    """Every path the suite defends, in the order the report reads best."""
    tokens = Tokens()
    cold, warm, sustained = Latencies(), Latencies(), Latencies()
    return (
        Scenario(name="single-turn", run=_single_turn(tokens), tokens=tokens, iterations=20),
        Scenario(name="tool-turn", run=_tool_turn(tokens), tokens=tokens, iterations=20),
        Scenario(name="streaming", run=_streaming, iterations=20),
        Scenario(name="structured-output", run=_structured_output, iterations=20),
        Scenario(name="embedding-batch", run=_embedding_batch, iterations=5),
        Scenario(name="run-fanout", run=_run_fanout, iterations=5, rounds=3),
        Scenario(
            name="first-token-cold",
            run=_first_token(cold, cold_start=True),
            observed=cold,
            iterations=20,
        ),
        Scenario(
            name="first-token-warm",
            run=_first_token(warm, cold_start=False),
            observed=warm,
            iterations=20,
        ),
        Scenario(
            name="sustained-stream",
            run=_sustained_stream(sustained),
            observed=sustained,
            iterations=20,
        ),
    )


def _single_turn(tokens: Tokens) -> Any:  # noqa: ANN401 — a scenario body, typed by Scenario
    """One question, one answer, one complete run record."""

    async def body() -> None:
        provider = ScriptedProvider(answered(), capabilities=NATIVE)
        await AgentRunner(provider=provider, tools=declarations()).run(
            PLANNER, ASKED, tenant="acme"
        )
        tokens.count(provider)

    return body


def _tool_turn(tokens: Tokens) -> Any:  # noqa: ANN401 — a scenario body, typed by Scenario
    """A turn that calls four tools at once, then answers."""

    async def body() -> None:
        provider = ScriptedProvider(asked_for(*LOOKUPS), answered(), capabilities=NATIVE)
        await AgentRunner(provider=provider, tools=declarations()).run(
            PLANNER, ASKED, tenant="acme"
        )
        tokens.count(provider)

    return body


async def _streaming() -> None:
    """Every event of a streamed run, consumed to the end."""
    runner = AgentRunner(
        provider=ScriptedProvider(answered(), capabilities=NATIVE), tools=declarations()
    )
    stream = runner.stream(PLANNER, ASKED, tenant="acme")
    async with stream:
        async for _ in stream:
            pass
    await stream


async def _structured_output() -> None:
    """A declared output type, parsed and validated on the way out."""
    runner = AgentRunner(provider=ScriptedProvider(structured(), capabilities=NATIVE))
    await runner.run(STRUCTURED, ASKED, tenant="acme")


class _LocalEmbeddings:
    """An embedding provider answering from a hash, so batching is measured and not the wire."""

    @property
    def name(self) -> str:
        """What this provider is called."""
        return "local"

    def limits(self, model: str) -> EmbeddingLimits:
        """What it accepts in one call."""
        del model
        return EmbeddingLimits(max_items=32, max_bytes=100_000, max_item_tokens=200, dimensions=8)

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        """Embed a batch deterministically."""
        del model
        return [_vector(one) for one in texts]


def _vector(text: str) -> Vector:
    """A stand-in embedding, cheap enough not to dominate the batching measurement."""
    digest = hashlib.sha256(text.encode()).digest()
    return tuple(digest[index] / 255 for index in range(8))


async def _embedding_batch() -> None:
    """Two hundred concurrent single-text calls, coalesced into provider batches."""
    async with BatchingEmbedder(_LocalEmbeddings()) as embedding:
        await asyncio.gather(
            *(
                embedding.embed(f"paragraph {index}", model="text-embedding-3-small")
                for index in range(200)
            )
        )


async def _run_fanout() -> None:
    """Twenty runs in flight together, which is what a busy request path looks like."""
    runners = [
        AgentRunner(
            provider=ScriptedProvider(answered(), capabilities=NATIVE), tools=declarations()
        )
        for _ in range(20)
    ]
    await asyncio.gather(*(one.run(PLANNER, ASKED, tenant="acme") for one in runners))


class CpuProfile:
    """Declared prefill and decode rates for the CPU these objectives are stated against.

    A shared CI runner cannot measure CPU inference reproducibly — its cores are contended
    and its neighbours are invisible — so the model's own time is modelled from these rates
    and only the kit's time and the token counts are measured. The counts are the part a
    change can move: break prefix stability and the uncached count rises, and the modelled
    first token rises with it. See `docs/latency-objectives.md`.

    Args:
        prefill: Tokens per second the target CPU prefills, for tokens not already cached.
        decode: Tokens per second it emits after the first one.
    """

    def __init__(self, *, prefill: float, decode: float) -> None:
        self.prefill = prefill
        self.decode = decode

    def modelled(self, measured: LatencyReport, hits: CacheHits) -> LatencyReport:
        """Add the model time this run would have cost to the kit time it did cost."""
        uncached = hits.input_tokens - (hits.cached_tokens or 0)
        first = (measured.time_to_first_token or 0.0) + uncached / self.prefill
        seconds = measured.seconds + uncached / self.prefill + measured.output_tokens / self.decode
        decoding = seconds - first
        return measured.model_copy(
            update={
                "time_to_first_token": first,
                "seconds": seconds,
                "tokens_per_second": measured.output_tokens / decoding if decoding > 0 else None,
                "hits": hits,
            }
        )


TARGET_CPU = CpuProfile(prefill=420.0, decode=18.0)
"""Eight contemporary x86 cores, no GPU, an int8-quantized small model."""


class Latencies:
    """What every run of a scenario reported, folded into the metrics the measurement records."""

    def __init__(self) -> None:
        self.reports: list[LatencyReport] = []

    def record(self, report: LatencyReport) -> None:
        """Keep one run's report."""
        self.reports.append(report)

    def __call__(self) -> dict[Metric, float]:
        """The median of what was kept, then start again for the next measurement.

        A median, not a mean: one round descheduled by a noisy neighbour would drag a mean
        far enough to fail the gate for a reason no change caused.
        """
        kept, self.reports = self.reports, []
        measured: dict[Metric, float] = {}
        for metric, values in (
            (Metric.TIME_TO_FIRST_TOKEN, [one.time_to_first_token for one in kept]),
            (Metric.TOKENS_PER_SECOND, [one.tokens_per_second for one in kept]),
            (Metric.CACHE_HIT_RATIO, [one.hits.ratio for one in kept]),
        ):
            known = [one for one in values if one is not None]
            if known:
                measured[metric] = statistics.median(known)
        return measured


class Prefill:
    """How much of each request a prefix cache would already hold, from the last one seen."""

    def __init__(self) -> None:
        self._last = ""

    def hits(self, request: ModelRequest) -> CacheHits:
        """Count what was sent and what a prefix cache would not have to prefill again."""
        rendered = _rendered(request.messages)
        shared = 0
        for mine, theirs in zip(self._last, rendered, strict=False):
            if mine != theirs:
                break
            shared += 1
        self._last = rendered
        return CacheHits(input_tokens=len(rendered) // 4, cached_tokens=shared // 4)


def _rendered(messages: Sequence[Message]) -> str:
    """The text of a request, as a prefix cache would see it."""
    return "".join(
        part.text for one in messages for part in one.content if isinstance(part, TextPart)
    )


async def _timed(
    provider: ScriptedProvider, prefill: Prefill, latencies: Latencies, *, cold: bool, asked: str
) -> None:
    """One streamed run, timed where it is measurable and modelled where it is not."""
    runner = AgentRunner(provider=provider, tools=declarations())
    timer = RunTimer(streaming=True, cold=cold)
    stream = runner.stream(PLANNER, asked, tenant="acme")
    async with stream:
        async for _ in stream:
            timer.first_token()
    result = await stream
    hits = prefill.hits(provider.requests[-1])
    latencies.record(
        TARGET_CPU.modelled(timer.finished(output_tokens=result.usage.output_tokens), hits)
    )


FOLLOW_UPS = (
    "Which of those legs has a reserved-seat requirement I should book ahead?",
    "If the second night moves to Nara instead, what changes about the transfers?",
    "Give me the same plan again but arriving in the evening rather than at midday.",
    "What is the walking distance from each hotel to the nearest limited express stop?",
)


def _follow_up(index: int) -> str:
    """A fresh turn on a stable prefix, which is what a warm conversation actually is."""
    return f"{ASKED} {FOLLOW_UPS[index % len(FOLLOW_UPS)]}"


def _first_token(latencies: Latencies, *, cold_start: bool) -> Any:  # noqa: ANN401 — typed by Scenario
    """First-token latency, cold and warm kept apart because averaging them hides both."""
    prefill = Prefill()
    asked = iter(range(1_000_000))

    async def body() -> None:
        provider = ScriptedProvider(answered(), capabilities=NATIVE)
        await _timed(
            provider,
            Prefill() if cold_start else prefill,
            latencies,
            cold=cold_start,
            asked=ASKED if cold_start else _follow_up(next(asked)),
        )

    return body


def _sustained_stream(latencies: Latencies) -> Any:  # noqa: ANN401 — typed by Scenario
    """The rate a reader perceives once tokens are arriving, over a longer answer."""
    prefill = Prefill()
    asked = iter(range(1_000_000))
    long_answer = ModelResponse(
        content="Kyoto, four nights. " * 40, usage=Usage(input_tokens=812, output_tokens=200)
    )

    async def body() -> None:
        provider = ScriptedProvider(long_answer, capabilities=NATIVE)
        await _timed(provider, prefill, latencies, cold=False, asked=_follow_up(next(asked)))

    return body
