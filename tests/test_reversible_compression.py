"""Compressed content that can be pulled back, and the boundaries a handle cannot cross."""

from __future__ import annotations

import json

import pytest

from tesserix_adk.core import AuditDecision, ClaimUnavailableError
from tesserix_adk.memory import ContentRouter, ReversibleRouter
from tesserix_adk.runtime import MemoryAuditSink, MemoryClaimCheckStore
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import ToolContext, expand_content_tool

pytestmark = pytest.mark.anyio

ROWS = json.dumps(
    [{"id": index, "region": "apac", "host": f"node-{index:03d}"} for index in range(200)]
)


def reversible(
    *, ttl_seconds: float = 3_600.0, clock: FakeClock | None = None
) -> tuple[ReversibleRouter, MemoryClaimCheckStore, MemoryAuditSink]:
    """A reversible router over a real store and a real audit trail."""
    store = MemoryClaimCheckStore(clock or FakeClock())
    audit = MemoryAuditSink()
    router = ReversibleRouter(
        ContentRouter(threshold_tokens=8),
        store,
        ttl_seconds=ttl_seconds,
        audit=audit,
        clock=clock or FakeClock(),
    )
    return router, store, audit


class TestCompressionThatCanBeUndone:
    """The bet that the model will not need the elided part is occasionally wrong."""

    async def test_compressed_content_carries_a_handle(self) -> None:
        router, _, _ = reversible()

        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        assert admitted.handle != ""
        assert admitted.handle in admitted.content

    async def test_the_content_says_how_to_pull_the_original_back(self) -> None:
        router, _, _ = reversible()

        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        assert "expand_content" in admitted.content

    async def test_the_exact_original_bytes_come_back(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        expanded = await router.expand(admitted.handle, tenant="acme", run_id="run-1")

        assert expanded.content == ROWS
        assert expanded.truncated is False
        assert expanded.chars == len(ROWS)

    async def test_the_retrieval_is_recorded_with_the_handle_and_the_size(self) -> None:
        router, _, audit = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        await router.expand(admitted.handle, tenant="acme", run_id="run-1", user="u-1")

        written = await audit.records(tenant="acme")
        assert written[0].decision is AuditDecision.EXECUTED
        assert written[0].user == "u-1"
        assert admitted.handle in written[0].reason
        assert str(len(ROWS)) in written[0].reason

    async def test_content_nothing_compressed_gets_no_handle(self) -> None:
        router, store, _ = reversible()

        admitted = await router.admit(
            "<<<<>>>> ||| ###" * 40, budget_tokens=1_000, tenant="acme", run_id="run-1"
        )

        assert admitted.handle == ""
        assert store.held == 0

    async def test_the_handle_is_counted_in_what_the_prompt_pays(self) -> None:
        router, _, _ = reversible()

        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        assert admitted.compressed_tokens > 0
        assert admitted.compressed_tokens < admitted.original_tokens

    async def test_identical_content_in_one_run_is_stored_once(self) -> None:
        router, store, _ = reversible()

        first = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")
        second = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        assert first.handle == second.handle
        assert store.held == 1


class TestWhereAHandleDoesNotReach:
    """A handle must never be a way to read across an isolation boundary."""

    async def test_another_tenant_gets_nothing(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        with pytest.raises(ClaimUnavailableError) as refused:
            await router.expand(admitted.handle, tenant="other", run_id="run-1")

        assert refused.value.handle == admitted.handle

    async def test_another_run_of_the_same_tenant_gets_nothing(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        with pytest.raises(ClaimUnavailableError):
            await router.expand(admitted.handle, tenant="acme", run_id="run-2")

    async def test_a_handle_past_its_retention_window_gets_nothing(self) -> None:
        clock = FakeClock()
        router, _, _ = reversible(ttl_seconds=60.0, clock=clock)
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        clock.advance(61.0)

        with pytest.raises(ClaimUnavailableError):
            await router.expand(admitted.handle, tenant="acme", run_id="run-1")

    async def test_a_handle_the_model_invented_reads_the_same_nothing(self) -> None:
        router, _, _ = reversible()
        await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        with pytest.raises(ClaimUnavailableError):
            await router.expand("claim:0123456789abcdef", tenant="acme", run_id="run-1")

    async def test_something_that_is_not_a_handle_at_all_is_refused(self) -> None:
        router, _, _ = reversible()

        with pytest.raises(ClaimUnavailableError):
            await router.expand("the whole document please", tenant="acme", run_id="run-1")

    async def test_a_refused_expansion_is_recorded_too(self) -> None:
        router, _, audit = reversible()

        with pytest.raises(ClaimUnavailableError):
            await router.expand("claim:0123456789abcdef", tenant="acme", run_id="run-1")

        written = await audit.records(tenant="acme")
        assert written[0].decision is AuditDecision.REFUSED

    async def test_the_original_never_reaches_the_audit_record(self) -> None:
        router, _, audit = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        await router.expand(admitted.handle, tenant="acme", run_id="run-1")

        assert "node-199" not in (await audit.records(tenant="acme"))[0].model_dump_json()


class TestWhatComesBackIsAdmittedToo:
    """Retrieval that blows the budget has moved the problem rather than solved it."""

    async def test_an_expansion_over_the_budget_is_windowed_and_says_so(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        expanded = await router.expand(
            admitted.handle, tenant="acme", run_id="run-1", budget_tokens=100
        )

        assert expanded.truncated is True
        assert expanded.chars == len(ROWS)
        assert ROWS.startswith(expanded.content)

    async def test_an_expansion_inside_the_budget_is_whole(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")

        expanded = await router.expand(
            admitted.handle, tenant="acme", run_id="run-1", budget_tokens=100_000
        )

        assert expanded.content == ROWS
        assert expanded.truncated is False


class TestTheToolTheModelCalls:
    """The handle is only useful if the model can redeem it from where it stands."""

    async def test_the_tool_returns_the_original(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")
        expand = expand_content_tool(router)

        try:
            returned = await expand.invoke(
                {"handle": admitted.handle}, ToolContext(run_id="run-1", tenant="acme")
            )
        finally:
            expand.release()

        assert returned == ROWS

    async def test_the_tool_refuses_a_handle_that_resolves_to_nothing(self) -> None:
        router, _, _ = reversible()
        expand = expand_content_tool(router)

        try:
            with pytest.raises(Exception, match="expand_content"):
                await expand.invoke(
                    {"handle": "claim:0123456789abcdef"},
                    ToolContext(run_id="run-1", tenant="acme"),
                )
        finally:
            expand.release()

    async def test_the_tool_reads_within_the_budget_it_was_built_with(self) -> None:
        router, _, _ = reversible()
        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")
        expand = expand_content_tool(router, budget_tokens=100)

        try:
            returned = await expand.invoke(
                {"handle": admitted.handle}, ToolContext(run_id="run-1", tenant="acme")
            )
        finally:
            expand.release()

        assert len(returned) < len(ROWS)


class TestARouterWithoutAnAuditSink:
    """Auditing is what a deployment owes its reviewers, not what the router needs."""

    async def test_it_still_stores_and_expands(self) -> None:
        router = ReversibleRouter(ContentRouter(threshold_tokens=8), MemoryClaimCheckStore())

        admitted = await router.admit(ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1")
        expanded = await router.expand(admitted.handle, tenant="acme", run_id="run-1")

        assert expanded.content == ROWS

    async def test_it_still_fails_closed(self) -> None:
        router = ReversibleRouter(ContentRouter(threshold_tokens=8), MemoryClaimCheckStore())

        with pytest.raises(ClaimUnavailableError):
            await router.expand("claim:0123456789abcdef", tenant="acme", run_id="run-1")
