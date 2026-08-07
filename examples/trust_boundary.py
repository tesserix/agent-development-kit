"""Fallback that fails closed at the boundary, and a choice that explains itself.

Four scenarios: what the boundary drops from the chain; a run whose self-hosted model is
down and whose only alternative is a vendor; the same failure where an equivalent model
exists; and the rationale a run record keeps.

Run it with `python examples/trust_boundary.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    CHEAP,
    Agent,
    ModelCapabilities,
    ModelSpec,
    ProviderUnavailableError,
    RetryConfig,
    RunEventKind,
    TrustBoundary,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

SEALED = TrustBoundary(tier="sealed", hosting="self-hosted", residency="in-central")
VENDOR = TrustBoundary(tier="standard", hosting="vendor-api", residency="us")
CAPABLE = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=200_000)

CLERK = Agent(name="clerk", instructions="Cite the page.", free_text=True, task_class=CHEAP)


def spec(ref: str, trust: TrustBoundary) -> ModelSpec:
    """One model, with where it may be used stated alongside what it can do."""
    provider, _, model = ref.partition(":")
    return ModelSpec(provider=provider, model=model, capabilities=CAPABLE, trust=trust)


def router(*candidates: ModelSpec) -> TableRouter:
    """A table offering `candidates` for the cheap class, in preference order."""
    return TableRouter(RoutingTable(rules=(RoutingRule(task_class=CHEAP, candidates=candidates),)))


async def a_run(*candidates: ModelSpec, script: list[object]) -> object:
    """One run against a fleet whose self-hosted endpoint answers from `script`."""
    fleet = {
        "vllm": ScriptedProvider(*script, name="vllm", capabilities=CAPABLE),
        "openai": ScriptedProvider(
            ModelResponse(content="Kyoto."), name="openai", capabilities=CAPABLE
        ),
    }
    run = await AgentRunner(
        provider=fleet["vllm"],
        providers=fleet,
        router=router(*candidates),
        retry=RetryConfig(max_attempts=1),
        clock=FakeClock(),
    ).run(CLERK, "what does page 12 say?", tenant="acme")
    return run, fleet


def what_the_boundary_drops() -> None:
    """A capable vendor model is not a legal fallback for a sealed matter."""
    decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(CHEAP)
    print(f"chain {decision.chain}, excluded {decision.excluded_by_boundary}")  # noqa: T201


async def the_endpoint_is_down() -> None:
    """No alternative inside the boundary, so the run ends rather than leaving it."""
    down = ProviderUnavailableError("the endpoint is down", status=503)
    run, fleet = await a_run(
        spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR), script=[down]
    )
    detail = next(
        event.detail or ""
        for event in reversed(run.events)
        if event.kind is RunEventKind.TERMINATED
    )
    print(f"{run.state}: {detail}")  # noqa: T201
    print(f"requests the vendor never received: {len(fleet['openai'].requests)}")  # noqa: T201


async def an_equivalent_model_answers() -> None:
    """Failing closed is about the boundary, not about refusing to fall back at all."""
    down = ProviderUnavailableError("the endpoint is down", status=503)
    run, _ = await a_run(
        spec("vllm:qwen", SEALED),
        spec("vllm:qwen-2", SEALED),
        script=[down, ModelResponse(content="page 12.")],
    )
    print(f"{run.state}, answered by {run.model}")  # noqa: T201


def what_the_record_keeps() -> None:
    """One line, drawn from a closed vocabulary — no prompt content reaches it."""
    decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(CHEAP)
    print(decision.explain())  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    what_the_boundary_drops()
    await the_endpoint_is_down()
    await an_equivalent_model_answers()
    what_the_record_keeps()


if __name__ == "__main__":
    asyncio.run(main())
