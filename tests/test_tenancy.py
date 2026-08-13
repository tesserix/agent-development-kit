"""Tenant identity as a property of the execution context, not of developer discipline.

The bug these are about is one missed argument on one path — a background task, a
thread-pool offload, a helper somebody wrote in a hurry — producing a query with no tenant
filter that nothing in the type system or the runtime notices. So: the context is bound at
the boundary and carried, absence raises rather than defaulting, and crossing to another
tenant has to be said out loud.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    MissingTenantContextError,
    ModelCapabilities,
    Run,
    RunState,
    TenantContext,
    TenantCrossingError,
    ToolCall,
    Usage,
    bound,
    current_tenant,
    tenant_here,
    tenant_scope,
)
from tesserix_adk.memory import MemoryScope
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_reads: list[tuple[str, str | None]] = []


class TestTheContextItself:
    def test_it_carries_what_a_scoped_operation_needs_to_be_argued_about(self) -> None:
        context = TenantContext(
            tenant="acme",
            user="ada",
            locale="en-GB",
            region="eu-west-1",
            correlation_id="c-1",
        )
        assert context.tenant == "acme"
        assert context.user == "ada"
        assert context.correlation_id == "c-1"

    def test_a_blank_tenant_is_not_a_tenant(self) -> None:
        """A scope that matches whatever the adapter joins is the bug, not a wildcard."""
        with pytest.raises(ValueError, match="must name a tenant"):
            TenantContext(tenant="  ")

    def test_it_cannot_be_edited_after_it_is_established(self) -> None:
        """A mutable context is a context one helper can rewrite for everyone below it."""
        context = TenantContext(tenant="acme")
        with pytest.raises(ValueError, match="frozen"):
            context.tenant = "globex"

    def test_a_derived_context_is_a_new_one(self) -> None:
        context = TenantContext(tenant="acme").acting_as("ada")
        assert context.user == "ada"
        assert TenantContext(tenant="acme").user is None


class TestBindingAndReading:
    def test_what_is_bound_at_the_boundary_is_what_the_work_below_reads(self) -> None:
        with tenant_scope("acme"):
            assert current_tenant().tenant == "acme"

    def test_a_scope_takes_a_context_as_readily_as_a_name(self) -> None:
        with tenant_scope(TenantContext(tenant="acme", user="ada")) as bound_context:
            assert current_tenant().user == "ada"
            assert bound_context is current_tenant()

    def test_outside_a_scope_there_is_no_tenant_rather_than_a_default_one(self) -> None:
        """A default tenant is one typo away from being every tenant."""
        with pytest.raises(MissingTenantContextError) as refused:
            current_tenant()
        assert refused.value.where == "current_tenant"
        assert tenant_here() is None

    def test_the_refusal_says_where_it_happened(self) -> None:
        """'No tenant' with no location is a fact nobody can act on."""
        with pytest.raises(MissingTenantContextError) as refused:
            current_tenant(where="memory.recall")
        assert refused.value.where == "memory.recall"
        assert "memory.recall" in str(refused.value)

    def test_leaving_a_scope_restores_what_was_there_before(self) -> None:
        """A delegated run has to give the parent's context back when it returns."""
        with tenant_scope("acme"):
            with tenant_scope(TenantContext(tenant="acme", user="ada")):
                assert current_tenant().user == "ada"
            assert current_tenant().user is None
        assert tenant_here() is None

    def test_a_scope_left_by_an_exception_still_restores(self) -> None:
        with tenant_scope("acme"), pytest.raises(RuntimeError), tenant_scope("acme"):
            raise RuntimeError("the body failed")

    def test_the_outer_tenant_is_restored_after_a_failure_below(self) -> None:
        with tenant_scope("acme"):
            with pytest.raises(RuntimeError), tenant_scope("acme"):
                raise RuntimeError("the body failed")
            assert current_tenant().tenant == "acme"


class TestCrossingToAnotherTenant:
    def test_it_is_refused_unless_it_is_said_out_loud(self) -> None:
        """An administrative operation is fine; one nobody declared is the incident."""
        with (
            tenant_scope("acme"),
            pytest.raises(TenantCrossingError) as refused,
            tenant_scope("globex"),
        ):
            pass  # pragma: no cover — the crossing is refused on entry
        assert refused.value.tenant == "acme"
        assert refused.value.into == "globex"

    def test_a_declared_crossing_is_allowed_and_carries_its_reason(self) -> None:
        with tenant_scope("acme"):
            with tenant_scope("globex", crossing="registry backfill"):
                assert current_tenant().tenant == "globex"
                assert current_tenant().crossing == "registry backfill"
            assert current_tenant().tenant == "acme"

    def test_a_crossing_declared_at_the_top_is_not_a_crossing(self) -> None:
        """Nothing was crossed from, so nothing is recorded as crossed."""
        with tenant_scope("acme", crossing="backfill"):
            assert current_tenant().crossing is None

    def test_re_entering_the_same_tenant_needs_no_declaration(self) -> None:
        with tenant_scope("acme"), tenant_scope("acme", user="ada"):
            assert current_tenant().user == "ada"


class TestItSurvivesTheWaysWorkIsSpawned:
    @pytest.mark.asyncio
    async def test_it_crosses_an_await(self) -> None:
        async def below() -> str:
            await asyncio.sleep(0)
            return current_tenant().tenant

        with tenant_scope("acme"):
            assert await below() == "acme"

    @pytest.mark.asyncio
    async def test_two_concurrent_runs_never_read_each_others(self) -> None:
        """The whole point: a fan-out for A cannot observe B, whatever the scheduler does."""

        async def under(tenant: str) -> set[str]:
            with tenant_scope(tenant):
                await asyncio.sleep(0)
                seen = await asyncio.gather(*(reading() for _ in range(8)))
                return set(seen)

        async def reading() -> str:
            await asyncio.sleep(0)
            return current_tenant().tenant

        a, b = await asyncio.gather(under("acme"), under("globex"))
        assert a == {"acme"}
        assert b == {"globex"}

    @pytest.mark.asyncio
    async def test_it_crosses_a_task_group(self) -> None:
        seen: list[str] = []

        async def reading() -> None:
            seen.append(current_tenant().tenant)

        with tenant_scope("acme"):
            async with asyncio.TaskGroup() as group:
                for _ in range(3):
                    group.create_task(reading())
        assert seen == ["acme"] * 3

    @pytest.mark.asyncio
    async def test_it_crosses_a_thread_offload(self) -> None:
        with tenant_scope("acme"):
            assert await asyncio.to_thread(lambda: current_tenant().tenant) == "acme"

    @pytest.mark.asyncio
    async def test_an_executor_that_copies_no_context_is_given_one(self) -> None:
        """`run_in_executor` drops contextvars; `bound` is what carries them across."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool, tenant_scope("acme"):
            unbound = await loop.run_in_executor(pool, tenant_here)
            carried = await loop.run_in_executor(pool, bound(lambda: current_tenant().tenant))
        assert unbound is None
        assert carried == "acme"

    def test_bound_carries_arguments_and_the_result_through(self) -> None:
        with tenant_scope("acme"):
            carried = bound(lambda one, two: f"{current_tenant().tenant}:{one}{two}")
        assert carried("a", two="b") == "acme:ab"

    @pytest.mark.asyncio
    async def test_a_generator_resumed_outside_the_scope_reads_no_tenant_rather_than_one(
        self,
    ) -> None:
        """A streaming response is resumed by whoever iterates it, from wherever they are."""

        async def streaming() -> AsyncGenerator[TenantContext | None]:
            yield tenant_here()
            yield tenant_here()

        with tenant_scope("acme"):
            lines = streaming()
            inside = await anext(lines)
        outside = await anext(lines)
        assert inside is not None
        assert inside.tenant == "acme"
        assert outside is None
        await lines.aclose()

    @pytest.mark.asyncio
    async def test_a_scope_bound_inside_a_suspended_generator_cannot_mis_scope_quietly(
        self,
    ) -> None:
        """A generator holds its binding between yields; the crossing rule makes that loud."""

        async def streaming() -> AsyncGenerator[str]:
            with tenant_scope("acme"):
                yield current_tenant().tenant
                yield current_tenant().tenant

        lines = streaming()
        assert await anext(lines) == "acme"
        with pytest.raises(TenantCrossingError), tenant_scope("globex"):
            pass  # pragma: no cover — refused on entry
        assert await anext(lines) == "acme"
        await lines.aclose()

    @pytest.mark.asyncio
    async def test_a_task_spawned_outside_any_scope_has_no_tenant_to_inherit(self) -> None:
        """Absence is loud rather than borrowed from whoever the thread ran last."""
        spawned = asyncio.get_running_loop().create_task(_read_detached())
        assert await spawned is None

    @pytest.mark.asyncio
    async def test_a_task_spawned_inside_a_scope_inherits_it(self) -> None:
        with tenant_scope("acme"):
            spawned = asyncio.get_running_loop().create_task(_read_detached())
        assert await spawned == "acme"


class TestEgressReadsTheContext:
    def test_a_memory_scope_can_be_taken_from_the_context_rather_than_an_argument(self) -> None:
        with tenant_scope(TenantContext(tenant="acme", user="ada")):
            scope = MemoryScope.here(session_id="s-1", agent="planner")
        assert scope.tenant_id == "acme"
        assert scope.user_id == "ada"
        assert scope.session_id == "s-1"

    def test_an_explicit_user_wins_over_the_acting_principal(self) -> None:
        with tenant_scope(TenantContext(tenant="acme", user="ada")):
            assert MemoryScope.here(user_id="grace").user_id == "grace"

    def test_egress_with_no_context_bound_is_refused_rather_than_unscoped(self) -> None:
        """The failure mode this exists to prevent: a query with no tenant filter."""
        with pytest.raises(MissingTenantContextError) as refused:
            MemoryScope.here()
        assert refused.value.where == "MemoryScope.here"


class TestARunBindsItForEverythingBelowIt:
    @pytest.mark.asyncio
    async def test_a_tool_body_reads_the_tenant_nobody_passed_it(self) -> None:
        """The tool signature says nothing about tenancy and the body still cannot be wrong."""
        _reads.clear()
        run = await _running("acme")

        assert run.state is RunState.COMPLETED
        assert _reads == [("acme", "ada"), ("acme", "ada")]

    @pytest.mark.asyncio
    async def test_the_binding_does_not_outlive_the_run(self) -> None:
        _reads.clear()
        await _running("acme")

        assert tenant_here() is None

    @pytest.mark.asyncio
    async def test_a_run_for_another_tenant_under_a_bound_one_is_refused(self) -> None:
        """A handler that binds A and then runs an agent for B is the incident, not a feature."""
        _reads.clear()
        with tenant_scope("globex"), pytest.raises(TenantCrossingError) as refused:
            await _running("acme")
        assert refused.value.into == "acme"


class TestCost:
    def test_reading_the_context_is_not_a_measurable_per_call_cost(self) -> None:
        """A lookup on every egress point has to be a contextvar read, not a search."""
        with tenant_scope("acme"):
            assert [current_tenant().tenant for _ in range(10_000)].count("acme") == 10_000


@tool
async def looking_up(what: str) -> str:
    """Read the tenant from the context rather than from an argument.

    Args:
        what: What is being looked up.
    """
    here = current_tenant(where="looking_up")
    _reads.append((here.tenant, here.user))
    offloaded = await asyncio.to_thread(lambda: current_tenant().tenant)
    _reads.append((offloaded, here.user))
    return f"{what} for {here.tenant}"


async def _running(tenant: str) -> Run[Any]:
    """A one-tool run, so the tool body is the thing reading the context."""
    registry = ToolRegistry((looking_up,), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="call_1", name="looking_up", arguments={"what": "fares"}),),
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            ModelResponse(content="40 EUR.", usage=Usage(input_tokens=10, output_tokens=5)),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("looking_up",), agent="planner"),
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Plan trips.",
        free_text=True,
        model="scripted-1",
        tools=("looking_up",),
    )
    return await runner.run(agent, "look it up", tenant=tenant, user="ada", run_id="run_1")


async def _read_detached() -> str | None:
    """What a task sees, which is whatever was bound where it was spawned."""
    here = tenant_here()
    return None if here is None else here.tenant
