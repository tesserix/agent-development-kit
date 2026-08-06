"""An endpoint you run yourself, called like any other provider.

Four scenarios: a vLLM deployment answering an agent; the same endpoint reporting no
usage, and what the kit records instead; an endpoint that cannot enforce a schema
refusing rather than pretending; and one that has not finished starting.

Run it with `python examples/self_hosted_provider.py`. A stub stands in for the endpoint,
so nothing here reaches the network. Against a real deployment, drop `transport` and point
`base_url` at the service.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    CapabilityError,
    Message,
    ModelCapabilities,
    ModelRequest,
    ProviderUnavailableError,
    TextPart,
)
from tesserix_adk.models.providers import (
    OLLAMA,
    VLLM,
    CompatibilityPreset,
    OpenAICompatibleProvider,
)
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock

MODEL = "qwen2.5-7b-instruct"
IN_CLUSTER = "http://vllm.models.svc.cluster.local:8000"

# What the deployment was started with. Nothing the endpoint reports says this, which is
# why the provider will not construct without it.
SERVES = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    streaming=True,
    context_window_tokens=32768,
    max_output_tokens=2048,
)


class Weather(BaseModel):
    """The shape the agent must answer in.

    Args:
        city: Where the reading is from.
        summary: The reading, in a few words.
    """

    city: str
    summary: str


def deployment(
    body: dict[str, Any] | None = None,
    *,
    preset: CompatibilityPreset,
    status: int = 200,
    capabilities: ModelCapabilities = SERVES,
) -> OpenAICompatibleProvider:
    """A provider pointed at a stub that answers the way the named server would."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body if body is not None else answered())

    return OpenAICompatibleProvider(
        MODEL,
        base_url=IN_CLUSTER,
        capabilities=capabilities,
        preset=preset,
        transport=httpx.MockTransport(handler),
    )


def answered(content: str = "it rained", usage: dict[str, int] | None = None) -> dict[str, Any]:
    """One completion body, as the server would send it."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": usage,
    }


async def a_run_against_the_cluster() -> None:
    """The agent does not know it is talking to a box two racks away."""
    provider = deployment(answered(json.dumps({"city": "Delhi", "summary": "clear"})), preset=VLLM)
    finished = await AgentRunner(provider=provider, clock=FakeClock()).run(
        Agent(
            name="forecaster",
            instructions="Answer with the weather.",
            model=MODEL,
            output_type=Weather,
        ),
        "what is the weather in Delhi",
        tenant="acme",
    )
    print(f"{provider.name:<8}", finished.output)  # noqa: T201


async def usage_nobody_reported() -> None:
    """Zero tokens would read as a free call, and a GPU hour is not free."""
    for preset in (VLLM, OLLAMA):
        provider = deployment(answered(usage=None), preset=preset)
        usage = (await provider.complete(_asked())).usage
        print(  # noqa: T201
            f"{provider.name:<8}",
            f"in={usage.input_tokens} out={usage.output_tokens}",
            f"estimated={usage.estimated} cost={usage.cost}",
        )


async def a_shape_the_endpoint_cannot_enforce() -> None:
    """Prompting a 7B model for JSON and parsing it back is a schema enforced by nobody."""
    provider = deployment(capabilities=ModelCapabilities(tool_calling=True), preset=OLLAMA)
    try:
        await AgentRunner(provider=provider, clock=FakeClock()).run(
            Agent(name="forecaster", instructions="Answer.", model=MODEL, output_type=Weather),
            "what is the weather in Delhi",
            tenant="acme",
        )
    except CapabilityError as refused:
        print(f"{provider.name:<8}", refused)  # noqa: T201


async def an_endpoint_still_loading() -> None:
    """A 503 from a model loading its weights is worth waiting for, not rewriting."""
    provider = deployment({"error": "model is loading"}, status=503, preset=VLLM)
    try:
        await provider.complete(_asked())
    except ProviderUnavailableError as waiting:
        print(f"{provider.name:<8}", waiting, f"retryable={waiting.retryable}")  # noqa: T201


def _asked() -> ModelRequest:
    return ModelRequest(
        model=MODEL, messages=(Message(role="user", content=[TextPart(text="did it rain")]),)
    )


async def main() -> None:
    """Run every scenario in order."""
    await a_run_against_the_cluster()
    await usage_nobody_reported()
    await a_shape_the_endpoint_cannot_enforce()
    await an_endpoint_still_loading()


if __name__ == "__main__":
    asyncio.run(main())
