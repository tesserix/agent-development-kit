"""Asserting what an agent called, under whose tenant, and what it did when a tool failed."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import ToolExecutionError, ToolNotPermittedError, current_tenant
from tesserix_adk.core.hooks import ApprovalRecord, digest_of
from tesserix_adk.testing import (
    ApprovalStub,
    FakeToolRegistry,
    ToolSpy,
    approving,
    assert_context_propagated,
    assert_idempotency_key_stable,
    assert_no_tool_called,
    assert_tool_called_once_with,
    assert_tool_sequence,
    denying,
    failing_tool,
    peak_concurrency,
    scoped_run,
    slow_tool,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.anyio


def _registry(**tools: Any) -> FakeToolRegistry:
    return FakeToolRegistry(tools or {"search": lambda **kw: dict(kw)})


def _record(record_id: str, arguments: Mapping[str, Any]) -> ApprovalRecord:
    return ApprovalRecord(
        id=record_id,
        run_id="run-1",
        tenant="tenant-alpha",
        agent_name="booking",
        tool_name="refund",
        arguments_digest=digest_of(arguments),
        reason="effectful tool",
    )


class TestTheSequence:
    async def test_the_order_the_agent_called_in_is_what_is_asserted(self) -> None:
        spy = ToolSpy(_registry(search=lambda **kw: kw, refund=lambda **kw: kw))
        await spy.invoke("search", {"q": "a"})
        await spy.invoke("refund", {"id": "1"})
        assert_tool_sequence(spy, "search", "refund")

    async def test_the_failure_names_the_first_place_the_two_diverged(self) -> None:
        spy = ToolSpy(_registry(search=lambda **kw: kw, refund=lambda **kw: kw))
        await spy.invoke("search", {"q": "a"})
        await spy.invoke("refund", {"id": "1"})
        with pytest.raises(AssertionError, match="call 2"):
            assert_tool_sequence(spy, "search", "search")

    async def test_a_run_that_stopped_early_says_what_never_happened(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "a"})
        with pytest.raises(AssertionError, match="nothing was called"):
            assert_tool_sequence(spy, "search", "refund")

    async def test_a_run_that_kept_going_says_what_else_happened(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "a"})
        await spy.invoke("search", {"q": "b"})
        with pytest.raises(AssertionError, match="expected nothing"):
            assert_tool_sequence(spy, "search")


class TestOneCallWithTheseArguments:
    async def test_the_arguments_are_compared_not_just_the_name(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "renewals", "limit": 5})
        assert_tool_called_once_with(spy, "search", q="renewals", limit=5)

    async def test_a_differing_argument_is_named_rather_than_dumped(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "renewals", "limit": 5})
        with pytest.raises(AssertionError, match="limit"):
            assert_tool_called_once_with(spy, "search", q="renewals", limit=10)

    async def test_a_second_call_fails_the_assertion_even_with_the_same_arguments(self) -> None:
        """Twice is a duplicated side effect, which is the defect this exists to catch."""
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "renewals"})
        await spy.invoke("search", {"q": "renewals"})
        with pytest.raises(AssertionError, match="2 times"):
            assert_tool_called_once_with(spy, "search", q="renewals")

    async def test_never_calling_it_fails_with_what_was_called_instead(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "renewals"})
        with pytest.raises(AssertionError, match="search"):
            assert_tool_called_once_with(spy, "refund", id="1")


class TestNothingWasCalled:
    async def test_a_run_that_called_nothing_passes(self) -> None:
        assert_no_tool_called(ToolSpy(_registry()))

    async def test_naming_a_tool_ignores_the_others(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "a"})
        assert_no_tool_called(spy, "refund")

    async def test_the_call_that_should_not_have_happened_is_quoted(self) -> None:
        spy = ToolSpy(_registry())
        await spy.invoke("search", {"q": "a"})
        with pytest.raises(AssertionError, match="q"):
            assert_no_tool_called(spy, "search")


class TestIdempotencyKeys:
    async def test_a_retried_call_reusing_its_key_passes(self) -> None:
        spy = ToolSpy(_registry(charge=failing_tool(ToolExecutionError("upstream flaked"))))
        with scoped_run(tenant="tenant-alpha", idempotency_key="key-1"):
            for _ in range(2):
                with pytest.raises(ToolExecutionError):
                    await spy.invoke("charge", {"amount": 10})
        assert_idempotency_key_stable(spy, "charge")

    async def test_a_retry_that_minted_a_new_key_is_a_duplicate_side_effect(self) -> None:
        spy = ToolSpy(_registry(charge=lambda **kw: kw))
        with scoped_run(tenant="tenant-alpha", idempotency_key="key-1"):
            await spy.invoke("charge", {"amount": 10})
        with scoped_run(tenant="tenant-alpha", idempotency_key="key-2"):
            await spy.invoke("charge", {"amount": 10})
        with pytest.raises(AssertionError, match="key-1"):
            assert_idempotency_key_stable(spy, "charge")

    async def test_different_arguments_are_allowed_different_keys(self) -> None:
        spy = ToolSpy(_registry(charge=lambda **kw: kw))
        with scoped_run(tenant="tenant-alpha", idempotency_key="key-1"):
            await spy.invoke("charge", {"amount": 10})
        with scoped_run(tenant="tenant-alpha", idempotency_key="key-2"):
            await spy.invoke("charge", {"amount": 20})
        assert_idempotency_key_stable(spy, "charge")

    async def test_a_repeated_effectful_call_with_no_key_at_all_is_refused(self) -> None:
        """No key is not a stable key; it is a duplicate waiting for a retry."""
        spy = ToolSpy(_registry(charge=lambda **kw: kw))
        with scoped_run(tenant="tenant-alpha"):
            await spy.invoke("charge", {"amount": 10})
            await spy.invoke("charge", {"amount": 10})
        with pytest.raises(AssertionError, match="no idempotency key"):
            assert_idempotency_key_stable(spy, "charge")

    async def test_a_tool_nobody_called_cannot_be_asserted_about(self) -> None:
        with pytest.raises(AssertionError, match="never called"):
            assert_idempotency_key_stable(ToolSpy(_registry()), "charge")


class TestTheScopedRun:
    def test_the_tenant_is_really_bound_not_merely_recorded(self) -> None:
        with scoped_run(tenant="tenant-alpha", user="ada"):
            assert current_tenant().tenant == "tenant-alpha"
            assert current_tenant().user == "ada"

    def test_the_context_is_restored_when_the_run_ends(self) -> None:
        with scoped_run(tenant="tenant-alpha") as run:
            assert run.tenant == "tenant-alpha"
        with scoped_run(tenant="tenant-beta") as second:
            assert second.tenant == "tenant-beta"

    def test_the_run_carries_a_tool_context_a_tool_would_have_been_handed(self) -> None:
        with scoped_run(tenant="tenant-alpha", user="ada", run_id="run-7") as run:
            assert run.context.run_id == "run-7"
            assert run.context.tenant == "tenant-alpha"
            assert run.context.user == "ada"

    async def test_the_spy_records_the_tenant_every_call_ran_under(self) -> None:
        spy = ToolSpy(_registry())
        with scoped_run(tenant="tenant-alpha", user="ada"):
            await spy.invoke("search", {"q": "a"})
        assert_context_propagated(spy, tenant="tenant-alpha", user="ada")

    async def test_a_call_that_escaped_the_tenant_is_named(self) -> None:
        spy = ToolSpy(_registry())
        with scoped_run(tenant="tenant-alpha"):
            await spy.invoke("search", {"q": "a"})
        await spy.invoke("search", {"q": "b"})
        with pytest.raises(AssertionError, match="search"):
            assert_context_propagated(spy, tenant="tenant-alpha")

    async def test_a_run_that_called_nothing_proves_no_propagation(self) -> None:
        with pytest.raises(AssertionError, match="no tool"):
            assert_context_propagated(ToolSpy(_registry()), tenant="tenant-alpha")


class TestScopesNarrowerThanTheTool:
    async def test_a_scope_the_run_does_not_hold_fails_authorisation(self) -> None:
        """Not proceeding: an allowlist enforced after dispatch is a side effect already made."""
        spy = ToolSpy(_registry(search=lambda **kw: kw, refund=lambda **kw: kw))
        with scoped_run(tenant="tenant-alpha", scopes=("search",)) as run:
            with pytest.raises(ToolNotPermittedError):
                run.allowlist.check("refund")
            assert run.allowlist.permits("search")
        assert_no_tool_called(spy)

    def test_a_run_that_states_no_scopes_permits_what_it_declares(self) -> None:
        with scoped_run(tenant="tenant-alpha", declares=("search",)) as run:
            assert run.allowlist.permits("search")


class TestApprovals:
    async def test_an_approving_gate_grants_the_payload_it_was_shown(self) -> None:
        gate = approving(decided_by="human:reviewer")
        decision = await gate.request(_record("a1", {"amount": 10}))
        assert decision.granted
        assert decision.decided_by == "human:reviewer"

    async def test_a_denying_gate_refuses_with_a_reason(self) -> None:
        gate = denying(reason="too large for the desk")
        decision = await gate.request(_record("a1", {"amount": 10}))
        assert not decision.granted
        assert "too large" in decision.reason

    async def test_the_gate_records_what_it_was_asked_so_a_test_can_assert_on_it(self) -> None:
        gate = approving()
        await gate.request(_record("a1", {"amount": 10}))
        assert gate.asked[0].tool_name == "refund"

    async def test_a_gate_nobody_answered_denies_rather_than_hangs(self) -> None:
        """Silence is not permission, and a test that hangs reports nothing at all."""
        gate = ApprovalStub(granted=None)
        decision = await gate.request(_record("a1", {"amount": 10}))
        assert not decision.granted
        assert "nobody" in decision.reason


class TestFailureInjection:
    async def test_a_tool_that_raises_a_domain_error_surfaces_it_unchanged(self) -> None:
        spy = ToolSpy(_registry(refund=failing_tool(ToolExecutionError("the ledger is closed"))))
        with pytest.raises(ToolExecutionError, match="the ledger is closed"):
            await spy.invoke("refund", {"id": "1"})

    async def test_exactly_one_attempt_is_recorded_and_no_result_is_fabricated(self) -> None:
        spy = ToolSpy(_registry(refund=failing_tool(ToolExecutionError("the ledger is closed"))))
        with pytest.raises(ToolExecutionError):
            await spy.invoke("refund", {"id": "1"})
        assert len(spy.calls) == 1
        assert spy.calls[0].result is None
        assert isinstance(spy.calls[0].error, ToolExecutionError)

    async def test_a_slow_tool_exceeds_the_timeout_the_test_gives_it(self) -> None:
        spy = ToolSpy(_registry(slow=slow_tool(seconds=5.0)))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await spy.invoke("slow", {})

    async def test_the_timed_out_call_is_still_recorded_as_an_attempt(self) -> None:
        spy = ToolSpy(_registry(slow=slow_tool(seconds=5.0)))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await spy.invoke("slow", {})
        assert len(spy.calls) == 1

    async def test_a_probe_reports_how_many_calls_really_overlapped(self) -> None:
        probe = peak_concurrency()
        spy = ToolSpy(_registry(fetch=probe))
        await asyncio.gather(*(spy.invoke("fetch", {"n": n}) for n in range(4)))
        assert probe.peak == 4

    async def test_a_bounded_dispatch_is_provably_bounded(self) -> None:
        probe = peak_concurrency()
        spy = ToolSpy(_registry(fetch=probe))
        lane = asyncio.Semaphore(2)

        async def held(n: int) -> None:
            async with lane:
                await spy.invoke("fetch", {"n": n})

        await asyncio.gather(*(held(n) for n in range(6)))
        assert probe.peak == 2


class TestMixedRegistries:
    async def test_sync_and_async_tools_live_in_one_registry(self) -> None:
        async def looked_up(**kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0)
            return dict(kwargs)

        spy = ToolSpy(_registry(sync=lambda **kw: dict(kw), asynchronous=looked_up))
        assert await spy.invoke("sync", {"a": 1}) == {"a": 1}
        assert await spy.invoke("asynchronous", {"b": 2}) == {"b": 2}
        assert_tool_sequence(spy, "sync", "asynchronous")

    async def test_the_declarations_of_the_registry_it_wraps_are_passed_through(self) -> None:
        spy = ToolSpy(_registry(search=lambda **kw: kw))
        assert [declared.name for declared in spy.declarations()] == ["search"]
