"""Agents on a machine with no GPU, which is most machines.

Four scenarios: what a quantized model will need before anything loads it; a model larger
than the box, refused rather than OOM-killed; the flags a server is started with, rendered
from the same description the fit check reads; and one agent answering identically on CPU
and on vLLM.

Run it with `python examples/cpu_inference.py`. A stub stands in for `llama-server`, so
nothing here reaches the network. Against a real server, drop `transport` and point
`base_url` at it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from tesserix_adk.core import (
    Agent,
    ModelCapabilities,
    NoOutput,
    TextPart,
)
from tesserix_adk.models.gguf import (
    GgufModel,
    ModelTooLargeError,
    Quantization,
    quantization_for,
    refuse_if_it_will_not_fit,
)
from tesserix_adk.models.providers import (
    VLLM,
    LlamaCppProvider,
    LlamaCppTuning,
)
from tesserix_adk.models.providers.compatible import OpenAICompatibleProvider
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import ModelProvider

GIB = 1024**3
MODEL = "llama-3.1-8b-instruct"
LOCAL = "http://127.0.0.1:8080"
SERVES = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    streaming=True,
    context_window_tokens=8_192,
    max_output_tokens=1_024,
)


def stub() -> httpx.MockTransport:
    """A server that answers one completion, so the example needs no weights."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "clear"},
                    }
                ],
                "usage": {"prompt_tokens": 31, "completion_tokens": 2},
            },
        )

    return httpx.MockTransport(handler)


def what_it_will_need() -> None:
    """Weights are fixed by the quantization; the cache grows with the context."""
    model = GgufModel(name=MODEL, parameters_b=8.03)
    for context in (4_096, 8_192, 32_768):
        needs = model.footprint(context_tokens=context)
        print(  # noqa: T201
            f"{context:>6}-token context:",
            f"weights={needs.weights / GIB:.2f} GiB",
            f"kv={needs.kv_cache / GIB:.2f} GiB",
            f"total={needs.readable}",
        )


def a_model_bigger_than_the_box() -> None:
    """The OOM killer explains nothing, so the refusal happens before anything loads."""
    try:
        refuse_if_it_will_not_fit(
            GgufModel(name=MODEL, parameters_b=8.03),
            context_tokens=8_192,
            available_bytes=4 * GIB,
        )
    except ModelTooLargeError as refused:
        print("refused:", refused)  # noqa: T201

    chosen = quantization_for(8.03, context_tokens=8_192, available_bytes=5 * GIB)
    generous = quantization_for(8.03, context_tokens=8_192, available_bytes=64 * GIB)
    print(  # noqa: T201
        f"a tight machine gets {chosen.value} at {chosen.bits_per_weight} bits;",
        f"a generous one gets {generous.value}, never heavier than the default",
    )


def how_the_server_was_started() -> None:
    """One description, so the launch command and the fit check cannot disagree."""
    tuning = LlamaCppTuning(threads=8, batch_size=512, context_tokens=8_192)
    print("llama-server", " ".join(tuning.server_arguments()))  # noqa: T201
    try:
        LlamaCppProvider(
            MODEL,
            base_url=LOCAL,
            capabilities=SERVES,
            tuning=LlamaCppTuning(context_tokens=32_768),
            weights=GgufModel(name=MODEL, parameters_b=8.03, quantization=Quantization.Q5_K_M),
            available_bytes=8 * GIB,
            transport=stub(),
        )
    except ModelTooLargeError as refused:
        print("the context the server was started with counts:", refused)  # noqa: T201


async def the_same_agent_somewhere_else() -> None:
    """The CPU path is worth having only if it is not a fork of the agent."""
    agent: Agent[NoOutput] = Agent(
        name="forecaster", instructions="Answer in one word.", model=MODEL, free_text=True
    )

    async def answered_by(provider: ModelProvider) -> str:
        run = await AgentRunner(provider=provider, clock=FakeClock()).run(
            agent, "what is the weather in Delhi", tenant="acme"
        )
        return "".join(part.text for part in run.messages[-1].content if isinstance(part, TextPart))

    on_cpu = await answered_by(
        LlamaCppProvider(MODEL, base_url=LOCAL, capabilities=SERVES, transport=stub())
    )
    on_gpu = await answered_by(
        OpenAICompatibleProvider(
            MODEL,
            base_url="http://vllm.models.svc.cluster.local:8000",
            capabilities=SERVES,
            preset=VLLM,
            transport=stub(),
        )
    )
    print(f"on cpu {on_cpu!r}, on vllm {on_gpu!r}, same agent")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    what_it_will_need()
    a_model_bigger_than_the_box()
    how_the_server_was_started()
    await the_same_agent_somewhere_else()


if __name__ == "__main__":
    asyncio.run(main())
