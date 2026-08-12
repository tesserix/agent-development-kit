"""What a declared graph refuses to be built as, and what happens when one node fails."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    DependencyCycleError,
    Dispatch,
    DispatchNode,
    NodeOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

type Step = Callable[[Mapping[str, str]], Awaitable[str]]


def _constant(value: str) -> Step:
    """A node that ignores what came before it and returns `value`."""

    async def run(inputs: Mapping[str, str]) -> str:
        del inputs
        return value

    return run


def _joins() -> Step:
    """A node that returns its dependencies' results, in name order."""

    async def run(inputs: Mapping[str, str]) -> str:
        return "+".join(inputs[name] for name in sorted(inputs))

    return run


def _raises(message: str) -> Step:
    """A node that fails the way real work fails: after it was already running."""

    async def run(inputs: Mapping[str, str]) -> str:
        del inputs
        raise RuntimeError(message)

    return run


class TestBuildingOne:
    def test_a_cycle_is_refused_where_it_is_declared(self) -> None:
        with pytest.raises(DependencyCycleError) as declared:
            Dispatch(
                (
                    DispatchNode("a", _constant("a"), needs=("c",)),
                    DispatchNode("b", _constant("b"), needs=("a",)),
                    DispatchNode("c", _constant("c"), needs=("b",)),
                )
            )

        assert set(declared.value.cycle) == {"a", "b", "c"}

    def test_a_node_that_waits_on_itself_is_a_cycle_of_one(self) -> None:
        with pytest.raises(DependencyCycleError):
            Dispatch((DispatchNode("a", _constant("a"), needs=("a",)),))

    def test_a_dependency_nobody_declared_is_named_rather_than_ignored(self) -> None:
        with pytest.raises(ConfigurationError, match="ghost"):
            Dispatch((DispatchNode("a", _constant("a"), needs=("ghost",)),))

    def test_two_nodes_with_one_name_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="twice"):
            Dispatch((DispatchNode("a", _constant("1")), DispatchNode("a", _constant("2"))))

    def test_an_empty_graph_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="no nodes"):
            Dispatch(())

    def test_a_width_of_zero_would_never_start_anything(self) -> None:
        with pytest.raises(ConfigurationError, match="width"):
            Dispatch((DispatchNode("a", _constant("a")),), width=0)

    def test_the_graph_says_what_it_holds(self) -> None:
        graph = Dispatch((DispatchNode("plan", _constant("plan")),))

        assert graph.nodes["plan"].needs == ()

    def test_the_order_groups_what_can_run_together(self) -> None:
        graph = Dispatch(
            (
                DispatchNode("plan", _constant("plan")),
                DispatchNode("left", _constant("l"), needs=("plan",)),
                DispatchNode("right", _constant("r"), needs=("plan",)),
                DispatchNode("join", _joins(), needs=("left", "right")),
            )
        )

        assert graph.order == (
            frozenset({"plan"}),
            frozenset({"left", "right"}),
            frozenset({"join"}),
        )


class TestRunningOne:
    async def test_a_node_is_given_what_its_dependencies_returned(self) -> None:
        graph = Dispatch(
            (
                DispatchNode("left", _constant("l")),
                DispatchNode("right", _constant("r")),
                DispatchNode("join", _joins(), needs=("left", "right")),
            )
        )

        result = await graph.run()

        assert result.ok
        assert result.value("join") == "l+r"

    async def test_independent_branches_run_at_the_same_time(self) -> None:
        both = asyncio.Barrier(2)

        async def waits_for_the_other(inputs: Mapping[str, str]) -> str:
            del inputs
            await asyncio.wait_for(both.wait(), timeout=5)
            return "together"

        graph = Dispatch(
            (
                DispatchNode("plan", _constant("plan")),
                DispatchNode("left", waits_for_the_other, needs=("plan",)),
                DispatchNode("right", waits_for_the_other, needs=("plan",)),
                DispatchNode("join", _joins(), needs=("left", "right")),
            )
        )

        result = await graph.run()

        assert result.value("join") == "together+together"

    async def test_the_join_does_not_start_before_both_branches_finish(self) -> None:
        running: set[str] = set()

        def branch(name: str) -> Step:
            async def run(inputs: Mapping[str, str]) -> str:
                del inputs
                running.add(name)
                await asyncio.sleep(0)
                running.discard(name)
                return name

            return run

        async def join(inputs: Mapping[str, str]) -> str:
            del inputs
            assert not running
            return "joined"

        graph = Dispatch(
            (
                DispatchNode("left", branch("left")),
                DispatchNode("right", branch("right")),
                DispatchNode("join", join, needs=("left", "right")),
            )
        )

        assert (await graph.run()).value("join") == "joined"

    async def test_width_bounds_how_many_run_at_once(self) -> None:
        live = 0
        widest = 0

        async def counts(inputs: Mapping[str, str]) -> str:
            del inputs
            nonlocal live, widest
            live += 1
            widest = max(widest, live)
            await asyncio.sleep(0)
            live -= 1
            return "done"

        graph = Dispatch(tuple(DispatchNode(f"n{index}", counts) for index in range(6)), width=2)

        await graph.run()

        assert widest == 2


class TestWhenANodeFails:
    async def test_what_depended_on_a_failure_is_skipped_rather_than_run_without_it(self) -> None:
        graph = Dispatch(
            (
                DispatchNode("fetch", _raises("the source was down")),
                DispatchNode("parse", _joins(), needs=("fetch",)),
                DispatchNode("report", _joins(), needs=("parse",)),
            )
        )

        result = await graph.run()

        assert not result.ok
        assert result.nodes["fetch"].outcome is NodeOutcome.FAILED
        assert result.nodes["parse"].outcome is NodeOutcome.SKIPPED
        assert result.nodes["report"].blocked_by == ("fetch",)

    async def test_a_branch_that_did_not_depend_on_the_failure_still_finishes(self) -> None:
        graph = Dispatch(
            (
                DispatchNode("fetch", _raises("the source was down")),
                DispatchNode("parse", _joins(), needs=("fetch",)),
                DispatchNode("elsewhere", _constant("done")),
            )
        )

        result = await graph.run()

        assert result.nodes["elsewhere"].outcome is NodeOutcome.COMPLETED
        assert result.values == {"elsewhere": "done"}

    async def test_the_failure_itself_is_kept_rather_than_a_description_of_it(self) -> None:
        graph = Dispatch((DispatchNode("fetch", _raises("the source was down")),))

        result = await graph.run()

        assert result.failures["fetch"].args == ("the source was down",)
        assert isinstance(result.failures["fetch"], RuntimeError)

    async def test_asking_a_failed_node_for_its_value_says_why_there_is_none(self) -> None:
        graph = Dispatch((DispatchNode("fetch", _raises("the source was down")),))

        result = await graph.run()

        with pytest.raises(KeyError, match="failed"):
            result.value("fetch")

    async def test_a_name_no_node_carries_is_not_a_value_either(self) -> None:
        result = await Dispatch((DispatchNode("a", _constant("a")),)).run()

        with pytest.raises(KeyError, match="ghost"):
            result.value("ghost")

    async def test_cancelling_the_graph_cancels_the_run_rather_than_recording_a_failure(
        self,
    ) -> None:
        started = asyncio.Event()

        async def never_finishes(inputs: Mapping[str, str]) -> str:
            del inputs
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        graph = Dispatch((DispatchNode("a", never_finishes),))
        running = asyncio.ensure_future(graph.run())
        await started.wait()
        running.cancel()

        with pytest.raises(asyncio.CancelledError):
            await running
