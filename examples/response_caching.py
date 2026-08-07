"""What a cache serves, and the four things it refuses to serve.

A local provider stands in for a vendor and answers with a different body every time, so
"which answer came back" is the same question as "was this call made". Nothing reaches the
network. Run it with `python examples/response_caching.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core.capabilities import ModelCapabilities
from tesserix_adk.core.primitives import Message, TextPart, Usage
from tesserix_adk.core.provider import ModelRequest, ModelResponse, ToolDeclaration
from tesserix_adk.models import (
    CachePolicy,
    CachingProvider,
    MemoryCacheStore,
    not_cacheable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MODEL = "gpt-4o"


class Counting:
    """A provider whose answer says how many times it has been called."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        """What this provider is called."""
        return "counting"

    @property
    def capabilities(self) -> ModelCapabilities:
        """What it says it supports."""
        return ModelCapabilities(tool_calling=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer, differently every time, so a repeat is visible."""
        del request
        # A real call suspends, which is the only reason a stampede exists to coalesce.
        await asyncio.sleep(0)
        self.calls += 1
        return ModelResponse(
            content=f"answer {self.calls}", usage=Usage(input_tokens=1_200, output_tokens=180)
        )

    async def stream(self, request: ModelRequest) -> Sequence[object]:
        """Not used here."""
        raise NotImplementedError

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """A rough count, which caching does not change."""
        return sum(len(str(message.content)) for message in messages)


def asked(
    text: str = "what did the contract say", *, tools: tuple[ToolDeclaration, ...] = ()
) -> ModelRequest:
    """One request, varied by whatever is being demonstrated."""
    return ModelRequest(
        model=MODEL, messages=(Message(role="user", content=[TextPart(text=text)]),), tools=tools
    )


async def a_repeat_costs_nothing() -> None:
    """The same call twice is one call, and the saving is counted."""
    inner = Counting()
    model = CachingProvider(inner, MemoryCacheStore(), tenant="acme")

    first = await model.complete(asked())
    second = await model.complete(asked())

    print("\nthe same question twice")  # noqa: T201
    print(f"  provider calls: {inner.calls}")  # noqa: T201
    print(f"  same answer:    {first.content == second.content}")  # noqa: T201
    saved = model.metrics.saved
    print(f"  tokens saved:   {saved.input_tokens + saved.output_tokens}")  # noqa: T201


async def a_second_tenant_is_never_served_the_first_ones_answer() -> None:
    """One store, two customers, one question, two answers."""
    inner = Counting()
    store = MemoryCacheStore()
    theirs = CachingProvider(inner, store, tenant="acme")
    ours = CachingProvider(inner, store, tenant="globex")

    first = await theirs.complete(asked())
    second = await ours.complete(asked())

    print("\ntwo tenants asking the same question")  # noqa: T201
    print(f"  provider calls: {inner.calls}")  # noqa: T201
    print(f"  same answer:    {first.content == second.content}")  # noqa: T201


async def changing_the_tools_changes_the_answer() -> None:
    """A tool schema is a determinant, so editing one is a miss rather than a stale hit."""
    inner = Counting()
    model = CachingProvider(inner, MemoryCacheStore(), tenant="acme")
    before = ToolDeclaration(name="clause", parameters={"type": "object"})
    after = ToolDeclaration(name="clause", parameters={"type": "object", "required": ["id"]})

    await model.complete(asked(tools=(before,)))
    await model.complete(asked(tools=(after,)))

    print("\nthe same prompt against an edited tool schema")  # noqa: T201
    print(f"  provider calls: {inner.calls}")  # noqa: T201


async def what_is_refused() -> None:
    """A sampled call and a personalised one are refused rather than cached anyway."""
    sampled = Counting()
    hot = CachingProvider(
        sampled, MemoryCacheStore(), tenant="acme", parameters={"temperature": 0.7}
    )
    await hot.complete(asked())
    await hot.complete(asked())

    personal = Counting()
    model = CachingProvider(personal, MemoryCacheStore(), tenant="acme")
    with not_cacheable("read the user's own history"):
        await model.complete(asked())
    await model.complete(asked())

    print("\ncalls the rules refuse to cache")  # noqa: T201
    print(f"  sampled — provider calls: {sampled.calls}, refusals: {hot.metrics.refusals}")  # noqa: T201
    print(f"  personalised — provider calls: {personal.calls}")  # noqa: T201


async def a_stampede_is_one_call() -> None:
    """Twenty callers on a cold key wait on one call rather than making twenty."""
    inner = Counting()
    model = CachingProvider(inner, MemoryCacheStore(), tenant="acme")

    await asyncio.gather(*(model.complete(asked()) for _ in range(20)))

    print("\ntwenty concurrent callers on a cold key")  # noqa: T201
    print(f"  provider calls: {inner.calls}, coalesced: {model.metrics.coalesced}")  # noqa: T201


async def erasure_removes_what_was_cached() -> None:
    """A tenant's right to erasure reaches the cache, not only the memory store."""
    inner = Counting()
    store = MemoryCacheStore()
    model = CachingProvider(inner, store, tenant="acme", policy=CachePolicy(ttl_seconds=3_600))
    await model.complete(asked())

    removed = await model.forget()

    print("\nan erasure request for one tenant")  # noqa: T201
    print(f"  entries removed: {removed}, left in store: {len(await store.every())}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await a_repeat_costs_nothing()
    await a_second_tenant_is_never_served_the_first_ones_answer()
    await changing_the_tools_changes_the_answer()
    await what_is_refused()
    await a_stampede_is_one_call()
    await erasure_removes_what_was_cached()


if __name__ == "__main__":
    asyncio.run(main())
