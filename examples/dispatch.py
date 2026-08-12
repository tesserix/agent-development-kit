"""Work with declared dependencies, run as wide as those dependencies allow.

Three scenarios: a diamond that runs its middle concurrently, a graph that could never
run, and one where a branch fails and the rest of the answer survives.
Run it with `python examples/dispatch.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core import ConfigurationError, Dispatch, DispatchNode

if TYPE_CHECKING:
    from collections.abc import Mapping


async def _plan(inputs: Mapping[str, str]) -> str:
    del inputs
    return "the question"


async def _statutes(inputs: Mapping[str, str]) -> str:
    await asyncio.sleep(0.05)
    return f"statutes for {inputs['plan']}"


async def _precedent(inputs: Mapping[str, str]) -> str:
    await asyncio.sleep(0.05)
    return f"precedent for {inputs['plan']}"


async def _compare(inputs: Mapping[str, str]) -> str:
    return " and ".join(inputs[name] for name in sorted(inputs))


async def _unreachable_source(inputs: Mapping[str, str]) -> str:
    del inputs
    raise ConnectionError("the source did not answer")


async def a_diamond() -> None:
    """The two middle nodes wait on the same plan, so they wait on each other for nothing."""
    graph = Dispatch(
        (
            DispatchNode("plan", _plan),
            DispatchNode("statutes", _statutes, needs=("plan",)),
            DispatchNode("precedent", _precedent, needs=("plan",)),
            DispatchNode("compare", _compare, needs=("statutes", "precedent")),
        )
    )

    started = asyncio.get_running_loop().time()
    result = await graph.run()
    elapsed = asyncio.get_running_loop().time() - started

    print("=== a diamond ===")  # noqa: T201
    print(f"order:   {[sorted(level) for level in graph.order]}")  # noqa: T201
    print(f"answer:  {result.value('compare')}")  # noqa: T201
    print(f"elapsed: {elapsed:.2f}s for two 0.05s branches")  # noqa: T201


async def one_that_could_never_run() -> None:
    """A cycle found at runtime looks like slow work, so it is found where it is written."""
    print("\n=== one that could never run ===")  # noqa: T201
    try:
        Dispatch(
            (
                DispatchNode("a", _plan, needs=("b",)),
                DispatchNode("b", _plan, needs=("a",)),
            )
        )
    except ConfigurationError as refused:
        print(refused)  # noqa: T201


async def when_a_branch_fails() -> None:
    """What depended on the failure is skipped; what did not still finishes."""
    graph = Dispatch(
        (
            DispatchNode("plan", _plan),
            DispatchNode("statutes", _unreachable_source, needs=("plan",)),
            DispatchNode("precedent", _precedent, needs=("plan",)),
            DispatchNode("compare", _compare, needs=("statutes", "precedent")),
        )
    )

    result = await graph.run()

    print("\n=== when a branch fails ===")  # noqa: T201
    for name, node in result.nodes.items():
        cause = f" (blocked by {', '.join(node.blocked_by)})" if node.blocked_by else ""
        print(f"{name}: {node.outcome.value}{cause}")  # noqa: T201
    print(f"what survived: {sorted(result.values)}")  # noqa: T201
    print(f"still ok: {result.ok}, failed: {sorted(result.failures)}")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await a_diamond()
    await one_that_could_never_run()
    await when_a_branch_fails()


if __name__ == "__main__":
    asyncio.run(main())
