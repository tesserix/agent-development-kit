"""Work with declared dependencies, run as wide as those dependencies allow.

Some work is neither a chain nor a flat fan-out. Two lookups feed one comparison; three
retrievals feed one summary. Expressed as nested sequential and parallel steps, that shape
serialises branches that had no reason to wait for each other — the author ends up
hand-scheduling, and the schedule goes stale the moment a step is added.

So the dependencies are declared and the schedule is derived. Each node names what it
needs; everything whose needs are met runs together; a join starts when its inputs exist,
not when a level finishes.

A graph that could never run is refused where it is written: a cycle waits forever, and at
runtime that is indistinguishable from slow work. A failure is contained rather than
fatal — its dependents are skipped, because running a join with a missing input produces
an answer built on nothing, and branches that never depended on it still finish.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/dispatch.md`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from tesserix_adk.core.errors import ConfigurationError, DependencyCycleError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

__all__ = [
    "Dispatch",
    "DispatchNode",
    "DispatchResult",
    "NodeOutcome",
    "NodeResult",
]


class NodeOutcome(StrEnum):
    """How a node ended. A string enum so a log line and a span attribute agree."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DispatchNode[T]:
    """One piece of work and what it waits for.

    Args:
        name: What the node is called. Unique within the graph, since it is how dependents
            name it and how the result is read back.
        run: The work. It is given what its dependencies returned, keyed by their names,
            and nothing else — a node that reaches around the graph for its input is a
            dependency nobody declared and nobody scheduled.
        needs: The nodes that must complete first. Empty means the node starts immediately.

    Example:
        >>> async def fetch(inputs: dict[str, str]) -> str:
        ...     return "rows"
        >>> DispatchNode("fetch", fetch).needs
        ()
    """

    name: str
    run: Callable[[Mapping[str, T]], Awaitable[T]]
    needs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NodeResult[T]:
    """What one node did.

    Args:
        name: The node this is about.
        outcome: Whether it completed, failed, or never ran.
        value: What it returned, for a node that completed.
        failure: What it raised, for a node that failed. The exception itself, so a caller
            can re-raise it or match on its type rather than parse a message.
        blocked_by: The failures upstream of a skipped node, so the report names a cause
            rather than an absence.
    """

    name: str
    outcome: NodeOutcome
    value: T | None = None
    failure: BaseException | None = None
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DispatchResult[T]:
    """What every node in one run of the graph did.

    Args:
        nodes: Every node's result, by name. A graph with a failure in it still reports
            every node, because the useful question after a partial failure is which parts
            of the answer exist.
    """

    nodes: Mapping[str, NodeResult[T]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every node completed."""
        return all(node.outcome is NodeOutcome.COMPLETED for node in self.nodes.values())

    @property
    def values(self) -> Mapping[str, T]:
        """What the completed nodes returned. Nodes that failed or were skipped are absent."""
        return {
            name: cast("T", node.value)
            for name, node in self.nodes.items()
            if node.outcome is NodeOutcome.COMPLETED
        }

    @property
    def failures(self) -> Mapping[str, BaseException]:
        """What each failed node raised, by name."""
        return {name: node.failure for name, node in self.nodes.items() if node.failure is not None}

    def value(self, name: str) -> T:
        """What `name` returned.

        Args:
            name: The node to read.

        Returns:
            Its return value.

        Raises:
            KeyError: If no such node ran, or if it failed or was skipped — a missing
                value read as `None` is how a partial answer is mistaken for a whole one.
        """
        node = self.nodes.get(name)
        if node is None:
            raise KeyError(f"no node called {name!r} was in the graph")
        if node.outcome is not NodeOutcome.COMPLETED:
            raise KeyError(f"{name!r} {node.outcome.value} and so returned nothing")
        return cast("T", node.value)


class Dispatch[T]:
    """A dependency graph, executed as wide as the dependencies allow.

    Built once and run many times: construction is where a graph that could never work is
    rejected, so the cost of checking it is not paid per run.

    Args:
        nodes: The work, each naming what it waits for.
        width: The most nodes to have in flight at once. `None` runs everything whose
            dependencies are met, which is the point of declaring them.

    Raises:
        ConfigurationError: If the graph is empty, a name is used twice, a dependency
            names nothing, or the width could never start anything.
        DependencyCycleError: If the declared dependencies close a loop.

    Example:
        >>> import asyncio
        >>> async def one(inputs: dict[str, str]) -> str:
        ...     return "1"
        >>> async def two(inputs: dict[str, str]) -> str:
        ...     return inputs["one"] + "2"
        >>> graph = Dispatch((DispatchNode("one", one), DispatchNode("two", two, ("one",))))
        >>> asyncio.run(graph.run()).value("two")
        '12'
    """

    def __init__(self, nodes: Iterable[DispatchNode[T]], *, width: int | None = None) -> None:
        declared = tuple(nodes)
        if not declared:
            raise ConfigurationError("a graph with no nodes has nothing to dispatch")
        if width is not None and width < 1:
            raise ConfigurationError(f"a width of {width} would never start anything")
        self._nodes = _by_name(declared)
        self._order = _levels(self._nodes)
        self._width = width

    @property
    def nodes(self) -> Mapping[str, DispatchNode[T]]:
        """The graph's nodes, by name."""
        return self._nodes

    @property
    def order(self) -> tuple[frozenset[str], ...]:
        """The nodes grouped by how deep they sit, earliest first.

        The grouping is what the graph permits, not what the run does: a node starts as
        soon as its own dependencies are done, without waiting for the rest of its group.
        """
        return self._order

    async def run(self) -> DispatchResult[T]:
        """Run every node whose dependencies were met.

        Returns:
            Every node's result, including the ones that failed and the ones that were
            skipped because something they needed failed.

        Raises:
            asyncio.CancelledError: If the caller cancels the run. Cancellation is the
                caller withdrawing the question, not a node failing to answer it, so it
                propagates rather than being recorded.
        """
        results: dict[str, NodeResult[T]] = {}
        lane = asyncio.Semaphore(self._width) if self._width is not None else None
        pending = {name: asyncio.Event() for name in self._nodes}
        async with asyncio.TaskGroup() as group:
            for name in self._nodes:
                group.create_task(self._settle(name, results, pending, lane))
        return DispatchResult(nodes=results)

    async def _settle(
        self,
        name: str,
        results: dict[str, NodeResult[T]],
        pending: dict[str, asyncio.Event],
        lane: asyncio.Semaphore | None,
    ) -> None:
        """Wait for one node's dependencies, run or skip it, and release its dependents."""
        node = self._nodes[name]
        for need in node.needs:
            await pending[need].wait()
        blocked = _causes(node.needs, results)
        if blocked:
            results[name] = NodeResult(name=name, outcome=NodeOutcome.SKIPPED, blocked_by=blocked)
        else:
            results[name] = await self._attempt(node, results, lane)
        pending[name].set()

    async def _attempt(
        self,
        node: DispatchNode[T],
        results: dict[str, NodeResult[T]],
        lane: asyncio.Semaphore | None,
    ) -> NodeResult[T]:
        """Run one node, keeping whatever it raised rather than ending the whole graph."""
        inputs = {need: cast("T", results[need].value) for need in node.needs}
        try:
            if lane is None:
                value = await node.run(inputs)
            else:
                async with lane:
                    value = await node.run(inputs)
        except Exception as failed:
            return NodeResult(name=node.name, outcome=NodeOutcome.FAILED, failure=failed)
        return NodeResult(name=node.name, outcome=NodeOutcome.COMPLETED, value=value)


def _causes[T](needs: tuple[str, ...], results: Mapping[str, NodeResult[T]]) -> tuple[str, ...]:
    """The failures behind a skip, carried through the skips between, so the cause is named."""
    causes: dict[str, None] = {}
    for need in needs:
        settled = results[need]
        if settled.outcome is NodeOutcome.FAILED:
            causes[need] = None
        causes.update(dict.fromkeys(settled.blocked_by))
    return tuple(causes)


def _by_name[T](nodes: tuple[DispatchNode[T], ...]) -> dict[str, DispatchNode[T]]:
    """Index the nodes, refusing a name used twice or a dependency on nothing."""
    indexed: dict[str, DispatchNode[T]] = {}
    for node in nodes:
        if node.name in indexed:
            raise ConfigurationError(
                f"{node.name!r} is declared twice, so a dependent is ambiguous"
            )
        indexed[node.name] = node
    for node in nodes:
        for need in node.needs:
            if need not in indexed:
                raise ConfigurationError(
                    f"{node.name!r} waits on {need!r}, which no node in the graph declares"
                )
    return indexed


def _levels[T](nodes: Mapping[str, DispatchNode[T]]) -> tuple[frozenset[str], ...]:
    """Group the nodes by depth, and name the cycle if the graph never runs out of nodes."""
    remaining = {name: set(node.needs) for name, node in nodes.items()}
    order: list[frozenset[str]] = []
    while remaining:
        ready = frozenset(name for name, needs in remaining.items() if not needs)
        if not ready:
            raise DependencyCycleError(
                f"these nodes wait on each other and so none of them can start: "
                f"{', '.join(sorted(remaining))}",
                cycle=tuple(sorted(remaining)),
            )
        order.append(ready)
        remaining = {name: needs - ready for name, needs in remaining.items() if name not in ready}
    return tuple(order)
