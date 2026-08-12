"""A large tool result is paid for on every turn after the one that fetched it.

The failure this file exists to prevent is a tool returning a hundred-kilobyte document
and that document sitting in the prompt for the rest of the run, re-prefilled on every
iteration. The result is replaced by a head and a handle; the detail is fetched only if
the model actually needs it. Every assertion here is about what the model receives, what
the store holds, who may dereference a handle, and what happens when the payload is gone.
"""

from __future__ import annotations

from typing import Any

import pytest

from tesserix_adk.core import (
    HANDLE_PREFIX,
    Agent,
    ClaimCheckPolicy,
    ClaimTicket,
    ClaimUnavailableError,
    ConfigurationError,
    ModelCapabilities,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    ToolRefusal,
    Usage,
    claim_handle,
)
from tesserix_adk.runtime import (
    AgentRunner,
    ClaimCheck,
    MemoryClaimCheckStore,
    ModelResponse,
    ToolResult,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext, ToolRegistry, claim_check_tool, tool

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)

BIG = "x" * 6_000


class TestThePolicy:
    def test_a_head_no_smaller_than_the_threshold_saves_nothing(self) -> None:
        with pytest.raises(ConfigurationError, match="head"):
            ClaimCheckPolicy(threshold_chars=512, head_chars=512)

    def test_a_threshold_of_nothing_would_store_every_result(self) -> None:
        with pytest.raises(ConfigurationError, match="threshold"):
            ClaimCheckPolicy(threshold_chars=0)

    def test_a_retention_window_of_nothing_stores_what_cannot_be_read(self) -> None:
        with pytest.raises(ConfigurationError, match="retention"):
            ClaimCheckPolicy(ttl_seconds=0)


class TestWhatTheModelReceives:
    async def test_a_result_above_the_threshold_becomes_a_head_and_a_handle(self) -> None:
        check, store = _check()

        ticket = await check.stored(_result(BIG), tenant="acme", run_id="run_1")

        assert ticket is not None
        assert ticket.chars == 6_000
        assert len(ticket.head) <= 512
        assert BIG not in ticket.rendered()
        assert ticket.handle in ticket.rendered()
        assert await store.fetch(ticket.handle, tenant="acme", run_id="run_1") == BIG

    async def test_the_rendering_says_how_much_was_held_back_and_how_to_get_it(self) -> None:
        check, _ = _check()

        ticket = await check.stored(_result(BIG), tenant="acme", run_id="run_1")

        assert ticket is not None
        rendered = ticket.rendered()
        assert "6000" in rendered
        assert "fetch_result" in rendered

    async def test_content_just_under_the_threshold_passes_through_untouched(self) -> None:
        check, store = _check()

        assert await check.stored(_result("y" * 4_095), tenant="acme", run_id="run_1") is None
        assert store.held == 0

    async def test_content_exactly_at_the_threshold_passes_through_untouched(self) -> None:
        check, _ = _check()

        assert await check.stored(_result("y" * 4_096), tenant="acme", run_id="run_1") is None

    async def test_the_head_is_cut_at_a_line_boundary_rather_than_mid_word(self) -> None:
        check, _ = _check()
        content = "first line\nsecond line\n" + "z" * 6_000

        ticket = await check.stored(_result(content), tenant="acme", run_id="run_1")

        assert ticket is not None
        assert ticket.head.endswith("second line")

    async def test_a_head_with_no_boundary_to_cut_at_is_cut_at_the_ceiling(self) -> None:
        check, _ = _check()

        ticket = await check.stored(_result(BIG), tenant="acme", run_id="run_1")

        assert ticket is not None
        assert ticket.head == "x" * 512

    async def test_a_tool_may_carry_its_own_threshold(self) -> None:
        check, _ = _check(
            per_tool={"read_page": ClaimCheckPolicy(threshold_chars=100, head_chars=50)}
        )

        ticket = await check.stored(_result("y" * 200, tool="read_page"), tenant="a", run_id="r")

        assert ticket is not None

    async def test_the_same_content_twice_is_stored_once(self) -> None:
        check, store = _check()

        first = await check.stored(_result(BIG), tenant="acme", run_id="run_1")
        second = await check.stored(_result(BIG), tenant="acme", run_id="run_1")

        assert first is not None
        assert second is not None
        assert first.handle == second.handle
        assert store.held == 1


class TestWhoMayDereferenceAHandle:
    async def test_a_handle_is_scoped_to_the_run_that_made_it(self) -> None:
        check, store = _check()
        ticket = await check.stored(_result(BIG), tenant="acme", run_id="run_1")
        assert ticket is not None

        with pytest.raises(ClaimUnavailableError, match="run_2"):
            await store.fetch(ticket.handle, tenant="acme", run_id="run_2")

    async def test_a_handle_is_scoped_to_the_tenant_that_made_it(self) -> None:
        check, store = _check()
        ticket = await check.stored(_result(BIG), tenant="acme", run_id="run_1")
        assert ticket is not None

        with pytest.raises(ClaimUnavailableError):
            await store.fetch(ticket.handle, tenant="other", run_id="run_1")

    async def test_the_same_content_in_two_tenants_derives_two_handles(self) -> None:
        first = claim_handle(tenant="acme", run_id="run_1", content=BIG)
        second = claim_handle(tenant="other", run_id="run_1", content=BIG)

        assert first != second

    async def test_a_handle_nobody_stored_is_unavailable_rather_than_empty(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())

        with pytest.raises(ClaimUnavailableError):
            await store.fetch("claim:nothing", tenant="acme", run_id="run_1")

    async def test_a_payload_past_its_retention_window_is_gone_not_stale(self) -> None:
        clock = FakeClock()
        store = MemoryClaimCheckStore(clock=clock)
        await store.put("claim:one", BIG, tenant="acme", run_id="run_1", ttl_seconds=60)

        clock.advance(61)

        with pytest.raises(ClaimUnavailableError):
            await store.fetch("claim:one", tenant="acme", run_id="run_1")

    async def test_erasure_reaches_the_stored_payloads_of_a_tenant(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put("claim:one", BIG, tenant="acme", run_id="run_1", ttl_seconds=60)
        await store.put("claim:two", BIG, tenant="other", run_id="run_1", ttl_seconds=60)

        assert await store.forget(tenant="acme") == 1
        with pytest.raises(ClaimUnavailableError):
            await store.fetch("claim:one", tenant="acme", run_id="run_1")
        assert await store.fetch("claim:two", tenant="other", run_id="run_1") == BIG

    async def test_erasure_can_be_narrowed_to_one_run(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put("claim:one", BIG, tenant="acme", run_id="run_1", ttl_seconds=60)
        await store.put("claim:two", BIG, tenant="acme", run_id="run_2", ttl_seconds=60)

        assert await store.forget(tenant="acme", run_id="run_1") == 1
        assert await store.fetch("claim:two", tenant="acme", run_id="run_2") == BIG


class TestTheRetrievalTool:
    async def test_the_tool_returns_the_stored_payload(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put(
            "claim:one", "the whole document", tenant="acme", run_id="run_1", ttl_seconds=60
        )
        fetch = claim_check_tool(store)

        answer = await fetch.invoke(
            {"handle": "claim:one"}, ToolContext(run_id="run_1", tenant="acme")
        )

        assert answer == "the whole document"
        fetch.release()

    async def test_a_payload_that_is_gone_is_a_typed_refusal_not_a_substitute(self) -> None:
        fetch = claim_check_tool(MemoryClaimCheckStore(clock=FakeClock()))

        with pytest.raises(ToolRefusal) as refused:
            await fetch.invoke(
                {"handle": "claim:missing"}, ToolContext(run_id="run_1", tenant="acme")
            )

        assert refused.value.code == "claim_unavailable"
        assert "claim:missing" in str(refused.value)
        fetch.release()

    async def test_the_tool_reads_a_window_so_one_call_cannot_undo_the_saving(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put("claim:one", "abcdefghij", tenant="acme", run_id="run_1", ttl_seconds=60)
        fetch = claim_check_tool(store, max_chars=4)

        first = await fetch.invoke(
            {"handle": "claim:one"}, ToolContext(run_id="run_1", tenant="acme")
        )
        second = await fetch.invoke(
            {"handle": "claim:one", "offset": 4}, ToolContext(run_id="run_1", tenant="acme")
        )

        assert first == "abcd"
        assert second == "efgh"
        fetch.release()

    async def test_an_offset_past_the_end_reads_empty_rather_than_failing(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put("claim:one", "abc", tenant="acme", run_id="run_1", ttl_seconds=60)
        fetch = claim_check_tool(store)

        answer = await fetch.invoke(
            {"handle": "claim:one", "offset": 99}, ToolContext(run_id="run_1", tenant="acme")
        )

        assert answer == ""
        fetch.release()

    async def test_a_negative_offset_is_refused_rather_than_read_from_the_end(self) -> None:
        store = MemoryClaimCheckStore(clock=FakeClock())
        await store.put("claim:one", "abc", tenant="acme", run_id="run_1", ttl_seconds=60)
        fetch = claim_check_tool(store)

        with pytest.raises(ToolRefusal) as refused:
            await fetch.invoke(
                {"handle": "claim:one", "offset": -1}, ToolContext(run_id="run_1", tenant="acme")
            )

        assert refused.value.code == "invalid_offset"
        fetch.release()

    async def test_something_that_is_not_a_handle_is_refused_without_touching_the_store(
        self,
    ) -> None:
        fetch = claim_check_tool(MemoryClaimCheckStore(clock=FakeClock()))

        with pytest.raises(ToolRefusal) as refused:
            await fetch.invoke(
                {"handle": "/etc/passwd"}, ToolContext(run_id="run_1", tenant="acme")
            )

        assert refused.value.code == "claim_unavailable"
        fetch.release()

    async def test_the_tool_will_not_run_without_a_run_to_scope_the_handle_to(self) -> None:
        fetch = claim_check_tool(MemoryClaimCheckStore(clock=FakeClock()))

        assert fetch.context_required
        fetch.release()


class TestWhatTheRunLoopDoes:
    async def test_an_oversized_result_reaches_the_model_as_a_handle(self) -> None:
        _, provider = await _run(_calling("fetch_page"), _answer())

        prompt = _last_prompt(provider)
        assert BIG not in prompt
        assert HANDLE_PREFIX in prompt

    async def test_the_substitution_is_recorded_as_its_own_event(self) -> None:
        run, _ = await _run(_calling("fetch_page"), _answer())

        stored = [event for event in run.events if event.kind is RunEventKind.TOOL_RESULT_STORED]
        assert len(stored) == 1
        assert "6000 chars" in (stored[0].detail or "")
        assert stored[0].name == "fetch_page"

    async def test_a_stored_result_is_not_also_reported_as_truncated(self) -> None:
        run, _ = await _run(_calling("fetch_page"), _answer())

        assert not [
            event for event in run.events if event.kind is RunEventKind.TOOL_RESULT_TRUNCATED
        ]

    async def test_a_small_result_is_left_alone_by_the_loop(self) -> None:
        run, provider = await _run(_calling("fetch_note"), _answer())

        assert not [event for event in run.events if event.kind is RunEventKind.TOOL_RESULT_STORED]
        assert "a short note" in _last_prompt(provider)

    async def test_a_runner_without_a_claim_check_behaves_as_it_always_did(self) -> None:
        _, provider = await _run(_calling("fetch_page"), _answer(), claim_check=None)

        assert HANDLE_PREFIX not in _last_prompt(provider)

    async def test_the_model_can_fetch_what_was_held_back(self) -> None:
        handle = claim_handle(tenant="acme", run_id="run_1", content=BIG)

        run, provider = await _run(
            _calling("fetch_page"),
            _calling("fetch_result", handle=handle),
            _answer(),
        )

        assert run.state is RunState.COMPLETED
        assert "x" * 1_000 in _last_prompt(provider)

    async def test_a_handle_the_model_invented_is_refused_rather_than_answered(self) -> None:
        run, _ = await _run(
            _calling("fetch_result", handle="claim:invented"),
            _answer(),
        )

        refused = [event for event in run.events if event.kind is RunEventKind.TOOL_REFUSED]
        assert len(refused) == 1
        assert "claim_unavailable" in (refused[0].detail or "")


def _check(
    per_tool: dict[str, ClaimCheckPolicy] | None = None,
) -> tuple[ClaimCheck, MemoryClaimCheckStore]:
    """A claim check over an in-process store, thresholded at 4096 characters."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    return ClaimCheck(store=store, per_tool=per_tool or {}), store


def _result(text: str, *, tool: str = "answer") -> ToolResult:
    """What the result boundary hands on: a rendered result and the tool that returned it."""
    return ToolResult(tool=tool, payload=text, text=text, tenant="acme")


def _calling(name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id=f"call_{name}", name=name, arguments=arguments),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _answer() -> ModelResponse:
    return ModelResponse(content="done", usage=Usage(input_tokens=1, output_tokens=1))


async def _run(*responses: ModelResponse, **overrides: Any) -> tuple[Run[Any], ScriptedProvider]:
    """A run whose `read_page` returns a document nobody wants in the prompt twice."""
    store = MemoryClaimCheckStore(clock=FakeClock())
    fetch = claim_check_tool(store)
    check = overrides.pop("claim_check", ClaimCheck(store=store))
    provider = ScriptedProvider(*responses, capabilities=CAPABLE)
    allowed = ("fetch_page", "fetch_note", "fetch_result")
    registry = ToolRegistry((read_page, read_note, fetch), clock=FakeClock())
    runner = AgentRunner(
        provider=provider,
        clock=FakeClock(),
        tools=registry.view(allow=allowed, agent="planner"),
        claim_check=check,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Read documents.",
        free_text=True,
        model="scripted-1",
        tools=allowed,
    )
    try:
        return await runner.run(agent, "read it", tenant="acme", run_id="run_1"), provider
    finally:
        fetch.release()


def _last_prompt(provider: ScriptedProvider) -> str:
    """Everything the model was sent on its final call, which is what a turn costs."""
    return "\n".join(str(message.content) for message in provider.requests[-1].messages)


@tool(name="fetch_page")
async def read_page() -> str:
    """Fetch a document far larger than anyone wants in a prompt."""
    return BIG


@tool(name="fetch_note")
async def read_note() -> str:
    """Fetch something small enough to keep."""
    return "a short note"


def test_a_ticket_renders_without_a_run() -> None:
    """The rendering is pure, so a consumer can log it outside a run."""
    ticket = ClaimTicket(handle="claim:one", tool="read_page", chars=6_000, head="the head")

    assert "claim:one" in ticket.rendered()
    assert "the head" in ticket.rendered()
