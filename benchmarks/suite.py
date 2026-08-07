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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tesserix_adk.core import Agent, ToolCall, Usage
from tesserix_adk.models import BatchingEmbedder, EmbeddingLimits
from tesserix_adk.runtime import AgentRunner, ModelResponse, ToolDeclaration
from tesserix_adk.testing import CAPABLE, FakeToolRegistry, ScriptedProvider, estimate_tokens
from tesserix_adk.testing.benchmarks import Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.models import Vector

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
    return (
        Scenario(name="single-turn", run=_single_turn(tokens), tokens=tokens, iterations=20),
        Scenario(name="tool-turn", run=_tool_turn(tokens), tokens=tokens, iterations=20),
        Scenario(name="streaming", run=_streaming, iterations=20),
        Scenario(name="structured-output", run=_structured_output, iterations=20),
        Scenario(name="embedding-batch", run=_embedding_batch, iterations=5),
        Scenario(name="run-fanout", run=_run_fanout, iterations=5, rounds=3),
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
