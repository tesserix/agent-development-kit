"""Assemble the kit against a fake provider, with no network and no credentials.

This is what the post-publish smoke job runs against the wheel it just installed from
the index, once per extra: an installable artefact that cannot be assembled is not a
release. Run it with `python examples/getting_started.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk import __version__
from tesserix_adk.core import (
    BudgetPolicy,
    MemoryStore,
    Message,
    ModelCapabilities,
    ModelProvider,
    Tracer,
    resolve_config,
    verify_conformance,
)
from tesserix_adk.testing import FakeBudgetPolicy, FakeMemoryStore, FakeTracer, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence


class EchoProvider:
    """A provider that answers from memory, so the example needs no endpoint."""

    @property
    def name(self) -> str:
        """Identify the provider in traces and audit records."""
        return "echo"

    @property
    def capabilities(self) -> ModelCapabilities:
        """What it declares it can do. Nothing here, which is what silence should mean."""
        return ModelCapabilities()

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimated, since this example has no tokeniser to call."""
        return estimate_tokens(messages)

    async def complete(self, request: Any) -> str:  # noqa: ANN401 — mirrors the protocol
        """Answer a request without leaving the process."""
        return f"echo: {request}"

    async def stream(self, request: Any) -> str:  # noqa: ANN401 — mirrors the protocol
        """Stream the same answer; the example does not exercise chunking."""
        return await self.complete(request)


async def main() -> None:
    """Resolve configuration, check the seams, and run one budgeted exchange."""
    resolution = resolve_config({"provider.endpoint": "http://localhost:0/unused"})
    config = resolution.config
    provider, memory, budget, tracer = (
        EchoProvider(),
        FakeMemoryStore(),
        FakeBudgetPolicy(limit=config.budget.max_tokens_per_run),
        FakeTracer(),
    )

    # Every seam is checked at construction: a missing member should not surface halfway
    # through a run, when it has already cost a provider call.
    for implementation, protocol in (
        (provider, ModelProvider),
        (memory, MemoryStore),
        (budget, BudgetPolicy),
        (tracer, Tracer),
    ):
        verify_conformance(implementation, protocol)

    with tracer.span("exchange", provider=provider.name):
        await budget.reserve(estimate=16)
        answer = await provider.complete("what is this kit for?")
        await budget.record(actual=len(answer))
        await memory.put("last-answer", answer)

    print(f"tesserix-adk {__version__}")  # noqa: T201 — the example's output is the point
    print(resolution.explain())  # noqa: T201
    print(f"provider: {provider.name}")  # noqa: T201
    print(f"answer: {await memory.get('last-answer')}")  # noqa: T201
    print(f"spans: {tracer.names()}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
