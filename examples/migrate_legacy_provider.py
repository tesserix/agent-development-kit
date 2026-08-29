"""First reversible migration step: keep a legacy client behind ``ModelProvider``.

Run with ``uv run python examples/migrate_legacy_provider.py``. This uses no network and
changes no legacy tool implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypedDict

from tesserix_adk.core import (
    Agent,
    BudgetLimits,
    Message,
    ModelCapabilities,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


class LegacyReply(TypedDict):
    """Old client's response payload."""

    text: str
    input_tokens: int
    output_tokens: int


class LegacyClient:
    """Existing provider client, deliberately left in its old shape."""

    tools_changed = False

    async def chat(self, *, model: str, prompt: str) -> LegacyReply:
        """Return the old application's response dictionary."""
        del model, prompt
        return {"text": "Legacy client still answers.", "input_tokens": 24, "output_tokens": 5}


class LegacyProvider:
    """Translate the old client at the provider-neutral model boundary."""

    name = "legacy-client"

    def __init__(self, client: LegacyClient) -> None:
        self.client = client
        self.capabilities = ModelCapabilities(context_window_tokens=8_000)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Conservatively estimate input for the runtime's pre-dispatch budget check."""
        return max(1, sum(len(str(message.content)) for message in messages) // 4)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Translate one legacy response into the kit's typed response."""
        body = await self.client.chat(model=request.model, prompt=str(request.messages))
        return ModelResponse(
            content=str(body["text"]),
            usage=Usage(
                input_tokens=body["input_tokens"],
                output_tokens=body["output_tokens"],
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Adapt the buffered legacy call to a valid two-event stream."""
        response = await self.complete(request)

        async def events() -> AsyncIterator[StreamEvent]:
            yield TextDelta(text=response.content)
            yield StreamEnd(response=response)

        return events()


async def main() -> None:
    """Run the old client through attribution and a one-call budget ceiling."""
    client = LegacyClient()
    provider = LegacyProvider(client)
    agent = Agent(
        name="legacy-support",
        instructions="Answer the support question.",
        model="legacy-model",
        free_text=True,
        budget=BudgetLimits(max_model_calls=1, max_input_tokens=1_000, max_output_tokens=100),
    )
    run = await AgentRunner(provider=provider).run(
        agent,
        "Is the legacy client still connected?",
        tenant="acme",
        user="ada",
        run_id="migration-1",
    )
    print(f"provider: {provider.name}")  # noqa: T201
    print(f"tenant attribution: {run.tenant}")  # noqa: T201
    print(f"model-call ceiling: {agent.budget.max_model_calls if agent.budget else 0}")  # noqa: T201
    print(f"legacy tools changed: {client.tools_changed}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
