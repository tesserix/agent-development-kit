"""Measuring what compression saves and what it costs, in the same run.

Run it with `uv run python examples/compression_gate.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.evals import (
    DEFAULT_FIXTURES,
    DEFAULT_FLOORS,
    Answer,
    CompressionCase,
    measure_compression,
)
from tesserix_adk.memory import ContentRouter, ReversibleRouter
from tesserix_adk.runtime import MemoryClaimCheckStore


class Reader:
    """A stand-in for a model: it can only answer from the text in front of it.

    A compressor that removed the answer is therefore measured as having removed it, and a
    reader that will not go and fetch the detail is measured as losing the case.
    """

    def __init__(self, *, retrieves: bool = True) -> None:
        self.retrieves = retrieves

    async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
        """Read the admitted content, expanding the handle where the answer is not in it."""
        if case.expected in content:
            return Answer(text=case.expected)
        if handle and self.retrieves:
            return Answer(text=case.expected, expanded=(handle,))
        return Answer(text="I could not find it")


def router() -> ReversibleRouter:
    """Compression that can hand the original back, which is what makes retrieval possible."""
    return ReversibleRouter(ContentRouter(threshold_tokens=8), MemoryClaimCheckStore())


async def main() -> None:
    """Measure a faithful reader, then one that will not retrieve, and show the difference."""
    faithful = await measure_compression(
        DEFAULT_FIXTURES, router(), Reader(), floors=DEFAULT_FLOORS
    )
    print(faithful.table())  # noqa: T201
    print(faithful.summary())  # noqa: T201

    lazy = await measure_compression(
        DEFAULT_FIXTURES, router(), Reader(retrieves=False), floors=DEFAULT_FLOORS
    )
    print()  # noqa: T201
    print(lazy.summary())  # noqa: T201
    for measured in lazy.failing():
        print(f"{measured.kind.value}: {measured.reason} ({', '.join(measured.cases)})")  # noqa: T201
    print(f"exit code {lazy.exit_code}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
