"""A provider declares what it can do, and the kit checks that before it calls.

Four scenarios: a provider written against the protocol, a tool-using agent refused by a
model that does not call tools, a prompt refused against the declared window, and the same
model addressed from configuration. Run it with `python examples/providers.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core import Agent, NoOutput, TextPart
from tesserix_adk.models import (
    Capability,
    CapabilityError,
    ContextWindowExceededError,
    ModelCapabilities,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelSpec,
)
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, FakeToolRegistry, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tesserix_adk.core import Message, StreamEvent


class EchoProvider:
    """A provider that answers from memory, so the example needs no endpoint."""

    def __init__(self, capabilities: ModelCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        """Identify this provider in records and routing."""
        return "echo"

    @property
    def capabilities(self) -> ModelCapabilities:
        """What it says it can do. The kit reads this rather than trying and finding out."""
        return self._capabilities

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer with the last thing said to it."""
        said = [p.text for m in request.messages for p in m.content if isinstance(p, TextPart)]
        return ModelResponse(content=f"echo: {said[-1]}")

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:  # noqa: ARG002
        """Refused unless declared: a single buffered chunk is not a stream.

        Raises:
            CapabilityError: If this provider does not declare `streaming`.
            NotImplementedError: Otherwise — the recorded streamed path is #39.
        """
        self._capabilities.require(Capability.STREAMING, provider=self.name, model="echo-1")
        raise NotImplementedError("the recorded streamed path is #39")

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimated, since this example ships no tokeniser."""
        return estimate_tokens(messages)


def agent(**overrides: object) -> Agent[NoOutput]:
    """The same clerk throughout."""
    fields: dict[str, object] = {
        "name": "clerk",
        "instructions": "Answer from sources.",
        "model": "echo-1",
        "free_text": True,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def a_declared_model_answers() -> None:
    """The happy path: everything the run needs is on the record."""
    provider = EchoProvider(ModelCapabilities(context_window_tokens=1_000))
    run = await AgentRunner(provider=provider, clock=FakeClock()).run(
        agent(), "when is the hearing", tenant="acme"
    )
    print(f"state: {run.state}")  # noqa: T201


async def tools_without_tool_calling_fail_at_construction() -> None:
    """The wiring is wrong, and the wiring is what the caller can still change."""
    provider = EchoProvider(ModelCapabilities(context_window_tokens=1_000))
    try:
        AgentRunner(provider=provider, tools=FakeToolRegistry({"search": lambda: "x"}))
    except CapabilityError as refused:
        print(f"refused at wiring: {refused.capability} on {refused.provider}")  # noqa: T201


async def a_prompt_past_the_window_is_refused() -> None:
    """A vendor handed an over-long prompt truncates it and does not say so."""
    provider = EchoProvider(ModelCapabilities(context_window_tokens=8))
    try:
        await AgentRunner(provider=provider, clock=FakeClock()).run(
            agent(), "a much longer question " * 20, tenant="acme"
        )
    except ContextWindowExceededError as refused:
        print(f"refused before the call: {refused.counted} tokens against {refused.limit}")  # noqa: T201


def a_model_is_addressable_from_configuration() -> None:
    """Two providers serve the same model id; the reference keeps them apart."""
    spec = ModelSpec(provider="echo", model="echo-1").with_capabilities(vision=True)
    print(f"{spec.ref} declares {sorted(c.value for c in spec.capabilities.declared)}")  # noqa: T201
    print(f"parsed: {ModelRef.parse('proxy:echo-1')}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await a_declared_model_answers()
    await tools_without_tool_calling_fail_at_construction()
    await a_prompt_past_the_window_is_refused()
    a_model_is_addressable_from_configuration()


if __name__ == "__main__":
    asyncio.run(main())
