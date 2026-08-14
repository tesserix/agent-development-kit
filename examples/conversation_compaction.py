"""Folding a long conversation without folding away where its claims came from.

Four scenarios: a conversation compacted with its sources intact; a summariser that drops
one, refused; a second pass that does nothing; and the cacheable prefix, unchanged either
way.

Run it with `python examples/conversation_compaction.py`. Nothing here reaches the network:
the summariser is a local function.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import Agent, Message, ProvenanceLostError, TextPart
from tesserix_adk.runtime import (
    assemble_prompt,
    citations_of,
    cited,
    compact_conversation,
)

AGENT = Agent(name="desk", instructions="Answer refund questions.", model="fake", free_text=True)


def said(text: str, *, role: str = "user", sources: tuple[str, ...] = ()) -> Message:
    """One turn, carrying the citation ids it rests on."""
    turn = Message(role=role, content=[TextPart(text=text)])  # type: ignore[arg-type]
    return cited(turn, sources) if sources else turn


def conversation() -> tuple[Message, ...]:
    """Twelve turns, every other one resting on a retrieved passage."""
    return tuple(
        said(
            f"Turn {index}: refunds, at some length, with the detail that makes it long. ",
            role="user" if index % 2 == 0 else "assistant",
            sources=(f"handbook-{index}",) if index % 2 else (),
        )
        for index in range(12)
    )


async def keeps_provenance(messages: tuple[Message, ...]) -> Message:
    """A summariser that carries across every source it was handed."""
    sources = tuple(dict.fromkeys(id_ for turn in messages for id_ in citations_of(turn)))
    return cited(said(f"Earlier: {len(messages)} turns about refunds.", role="system"), sources)


async def drops_provenance(messages: tuple[Message, ...]) -> Message:
    """A summariser that writes good prose and forgets where it came from."""
    return said(f"Earlier: {len(messages)} turns about refunds.", role="system")


async def sources_survive_compaction() -> None:
    """The whole point: shorter, and still checkable."""
    history = conversation()
    done = await compact_conversation(
        history, summarise=keeps_provenance, threshold_tokens=100, keep_recent=4, run_id="run-1"
    )

    print(f"turns: {len(history)} -> {len(done.history)}")  # noqa: T201
    print(f"carried: {done.citations}")  # noqa: T201
    print(f"event: {done.event.attributes() if done.event else None}")  # noqa: T201


async def a_summary_that_lost_a_source() -> None:
    """Refused, with the history left exactly as it was."""
    history = conversation()
    try:
        await compact_conversation(
            history, summarise=drops_provenance, threshold_tokens=100, keep_recent=4
        )
    except ProvenanceLostError as refused:
        print(f"refused: {refused}")  # noqa: T201
        print(f"lost: {refused.lost}")  # noqa: T201


async def a_second_pass_does_nothing() -> None:
    """Compaction is idempotent: a summary is not summarised again."""
    once = await compact_conversation(
        conversation(), summarise=keeps_provenance, threshold_tokens=100, keep_recent=4
    )
    twice = await compact_conversation(
        once.history, summarise=keeps_provenance, threshold_tokens=100, keep_recent=4
    )

    print(f"second pass ran: {twice.ran}, unchanged: {twice.history == once.history}")  # noqa: T201


async def the_prefix_is_unaffected() -> None:
    """Nothing above the conversation layer is an input, so no cache is refilled."""
    history = conversation()
    done = await compact_conversation(
        history, summarise=keeps_provenance, threshold_tokens=100, keep_recent=4
    )

    before = assemble_prompt(AGENT, "and now?", history=history, pinned=("a case file",))
    after = assemble_prompt(AGENT, "and now?", history=done.history, pinned=("a case file",))
    print(f"same fingerprint: {before.fingerprint == after.fingerprint}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await sources_survive_compaction()
    await a_summary_that_lost_a_source()
    await a_second_pass_does_nothing()
    await the_prefix_is_unaffected()


if __name__ == "__main__":
    asyncio.run(main())
