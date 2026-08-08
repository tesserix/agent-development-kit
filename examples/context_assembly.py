"""A conversation too long for the window, made to fit without losing the constraints.

Run it with `python examples/context_assembly.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import ContextBudgetError, Message, ModelResponse, TextPart
from tesserix_adk.core.capabilities import ModelCapabilities
from tesserix_adk.memory import (
    ContextAssembler,
    ContextPlan,
    MemoryKind,
    MemoryQuery,
    MemoryScope,
    SectionPlan,
    SummariseSpan,
    pinned,
)
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore, ScriptedProvider

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1")


def said(text: str, at: str) -> Message:
    """One turn, identified so the report can name it."""
    return Message(role="user", content=[TextPart(text=text)], metadata={"id": at})


def small(window: int) -> ScriptedProvider:
    """A provider with a window too small for the conversation."""
    return ScriptedProvider(capabilities=ModelCapabilities(context_window_tokens=window))


PLAN = ContextPlan(
    sections=(
        SectionPlan(name="constraints", share=0.3, pinned=True),
        SectionPlan(name="recent", share=0.7, compaction="summarise-span"),
    )
)

HISTORY = {
    "constraints": [said("aisle seat, no peanuts", "c1")],
    "recent": [
        said("looking at flights " + "x" * 200, "t1"),
        said("what about baggage " + "y" * 200, "t2"),
        said("and the seat?", "t3"),
    ],
}


async def fitted() -> None:
    """The pinned constraint survives; the older turns become one summary."""
    store = InMemoryMemoryStore(clock=FakeClock())
    assembled = await ContextAssembler(
        PLAN,
        provider=small(60),
        strategies={
            "summarise-span": SummariseSpan(
                provider=ScriptedProvider(ModelResponse(content="flights and baggage discussed")),
                model="summariser",
            )
        },
        memory=store,
        scope=SCOPE,
    ).assemble(HISTORY)

    said_now = [p.text for m in assembled.messages for p in m.content if isinstance(p, TextPart)]
    print("constraint kept:", "aisle seat, no peanuts" in said_now)  # noqa: T201
    print("summarised:", assembled.sections[1].summarised)  # noqa: T201
    print("within budget:", assembled.tokens <= assembled.budget_tokens)  # noqa: T201

    kept = await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
    print("summary kept with provenance:", kept[0].record.source)  # noqa: T201


async def refused() -> None:
    """A summariser that fails takes the assembly with it, rather than overflowing."""
    broken = SummariseSpan(provider=ScriptedProvider(TimeoutError("upstream")), model="s")
    try:
        await ContextAssembler(
            PLAN, provider=small(40), strategies={"summarise-span": broken}
        ).assemble(HISTORY)
    except ContextBudgetError as stopped:
        print("failed closed:", stopped)  # noqa: T201


async def pinned_too_big() -> None:
    """What is pinned does not fit on its own, and nothing decides it was optional."""
    try:
        await ContextAssembler(
            ContextPlan(sections=(SectionPlan(name="constraints", share=1.0, pinned=True),)),
            provider=small(4),
        ).assemble({"constraints": [pinned(said("z" * 400, "c1"))]})
    except ContextBudgetError as stopped:
        print("pinned alone:", stopped.required_tokens, ">", stopped.budget_tokens)  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await fitted()
    await refused()
    await pinned_too_big()


if __name__ == "__main__":
    asyncio.run(main())
