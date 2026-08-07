"""The lanes a turn's tool calls stand in, and the order they are let into them.

A turn that asked for four independent lookups should cost roughly one lookup, so the
batch runs concurrently. Firing all four at whatever downstream is behind them is how one
agent turn becomes a rate-limit breach at a partner, so each call stands in three queues
at once — its tenant's, its run's and its tool's — and the tightest declared bound
decides. Every call takes them in the same order, so two calls cannot each hold what the
other is waiting for.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Container, Sequence

    from tesserix_adk.core.config import ConcurrencyConfig
    from tesserix_adk.core.primitives import ToolCall

__all__ = ["Lanes", "Turn", "phased"]

_held: ContextVar[frozenset[str]] = ContextVar("tesserix_adk_lanes_held", default=frozenset())
"""Which runner-wide lanes the work in this context already holds."""


class Turn:
    """The lanes belonging to one batch, so one turn cannot spend another turn's width.

    An agent narrowing a runner's widths is narrowing its own turn: the runner's lanes
    bound a shared downstream across every run, and an agent cannot know about the others.

    Args:
        config: The widths resolved for this run, runner narrowed by agent.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self.batch = asyncio.Semaphore(config.max_concurrent_tools)
        self.tools = {tool: asyncio.Semaphore(width) for tool, width in config.per_tool.items()}


class Lanes:
    """The semaphores one runner's tool calls stand in.

    Per-tenant lanes outlive any one run — a tenant permitted one call at a time means one
    across its runs, not one in each — so they are held here rather than on the run that
    happened to open them.

    Args:
        config: The declared widths. An absent per-tool width means the run's width is the
            only one that tool stands in.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self._config = config
        self._tenants: dict[str, asyncio.Semaphore] = {}
        self._tools: dict[str, asyncio.Semaphore] = {}

    @asynccontextmanager
    async def held(
        self, tool: str, *, tenant: str, turn: Turn | None = None
    ) -> AsyncIterator[None]:
        """Hold every lane this call stands in, taken tenant, then turn, then tool.

        The order is the same for every call, so two calls cannot each hold a lane the
        other is waiting for.
        """
        async with (
            self._entered(self._for_tenant(tenant), key=f"{id(self)}:tenant:{tenant}"),
            self._entered(turn.batch if turn is not None else None),
            self._entered(self._for_tool(tool), key=f"{id(self)}:tool:{tool}"),
            self._entered(turn.tools.get(tool) if turn is not None else None),
        ):
            yield

    @asynccontextmanager
    async def _entered(
        self, lane: asyncio.Semaphore | None, *, key: str | None = None
    ) -> AsyncIterator[None]:
        if lane is None:
            yield
            return
        already = _held.get()
        if key is not None and key in already:
            # A sub-agent queueing behind a lane its own caller holds waits for itself, so
            # nested work spends the parent's place in the lane rather than asking for a
            # second one.
            yield
            return
        token = _held.set(already if key is None else already | {key})
        try:
            async with lane:
                yield
        finally:
            _held.reset(token)

    def _for_tenant(self, tenant: str) -> asyncio.Semaphore | None:
        width = self._config.per_tenant
        if width is None:
            return None
        return self._tenants.setdefault(tenant, asyncio.Semaphore(width))

    def _for_tool(self, tool: str) -> asyncio.Semaphore | None:
        width = self._config.per_tool.get(tool)
        if width is None:
            return None
        return self._tools.setdefault(tool, asyncio.Semaphore(width))


def phased(
    calls: Sequence[ToolCall], *, serial: Container[str]
) -> tuple[tuple[ToolCall, ...], ...]:
    """Group `calls` into phases run one after another, keeping the model's order.

    A call to a tool that declared itself order-dependent gets a phase to itself: it runs
    with nothing else in flight, and everything the model asked for before it has already
    resolved. Consecutive parallel-safe calls share a phase.
    """
    phases: list[tuple[ToolCall, ...]] = []
    together: list[ToolCall] = []
    for call in calls:
        if call.name in serial:
            if together:
                phases.append(tuple(together))
                together = []
            phases.append((call,))
            continue
        together.append(call)
    if together:
        phases.append(tuple(together))
    return tuple(phases)
