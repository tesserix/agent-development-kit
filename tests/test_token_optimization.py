"""RTK and Headroom are selected by origin, with isolation at the external boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tesserix_adk.memory import (
    HeadroomMcpOptimizer,
    OptimizationBackend,
    OptimizationChannel,
    OptimizationError,
    TokenOptimizer,
)


class HeadroomSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.fail = False
        self.is_error = False
        self.retrieved_hash = "h-1"

    async def call_tool(self, name: str, arguments: dict[str, str]) -> object:
        self.calls.append((name, arguments))
        if self.fail:
            raise OSError("headroom is unavailable")
        if name == "headroom_compress":
            payload = {
                "compressed": "3 active hosts in apac [hash=h-1]",
                "hash": "h-1",
                "original_tokens": 900,
                "compressed_tokens": 12,
                "tokens_saved": 888,
                "savings_percent": 98.7,
                "transforms": ["json-table"],
                "note": "original retained",
            }
        else:
            payload = {
                "hash": self.retrieved_hash,
                "source": "local",
                "original_content": "the exact original",
                "original_item_count": 3,
                "compressed_item_count": 1,
                "retrieval_count": 1,
            }
        block = SimpleNamespace(type="text", text=json.dumps(payload))
        return SimpleNamespace(content=[block], isError=self.is_error)


def optimizer(session: HeadroomSession | None = None) -> tuple[TokenOptimizer, HeadroomSession]:
    connected = session or HeadroomSession()
    headroom = HeadroomMcpOptimizer(connected, tenant="acme", run_id="run-1")
    return TokenOptimizer(headroom=headroom), connected


@pytest.mark.parametrize(
    ("channel", "backend"),
    [
        (OptimizationChannel.JSON, OptimizationBackend.HEADROOM),
        (OptimizationChannel.API, OptimizationBackend.HEADROOM),
        (OptimizationChannel.MCP, OptimizationBackend.HEADROOM),
        (OptimizationChannel.RAG, OptimizationBackend.HEADROOM),
        (OptimizationChannel.CONVERSATION, OptimizationBackend.HEADROOM),
        (OptimizationChannel.GATEWAY, OptimizationBackend.HEADROOM),
        (OptimizationChannel.MULTI_AGENT, OptimizationBackend.HEADROOM),
        (OptimizationChannel.UNKNOWN, OptimizationBackend.NONE),
    ],
)
def test_content_origin_selects_the_expected_backend(
    channel: OptimizationChannel, backend: OptimizationBackend
) -> None:
    routed, _ = optimizer()

    assert routed.select(channel, headroom_allowed=True).backend is backend


@pytest.mark.parametrize("command", ["git", "kubectl", "pytest", "docker", "rg"])
def test_supported_cli_commands_are_planned_through_rtk(command: str) -> None:
    routed, _ = optimizer()

    plan = routed.plan_command((command, "--version"))

    assert plan.backend is OptimizationBackend.RTK
    assert plan.argv == ("rtk", command, "--version")


def test_an_unsupported_command_is_not_sent_to_a_guessed_filter() -> None:
    routed, _ = optimizer()

    plan = routed.plan_command(("custom-admin", "status"))

    assert plan.backend is OptimizationBackend.NONE
    assert plan.argv == ("custom-admin", "status")


def test_an_absolute_executable_is_not_replaced_by_a_different_binary() -> None:
    routed, _ = optimizer()

    plan = routed.plan_command(("/private/admin/git", "status"))

    assert plan.backend is OptimizationBackend.NONE
    assert plan.argv == ("/private/admin/git", "status")


def test_an_already_optimised_command_is_not_wrapped_twice() -> None:
    routed, _ = optimizer()

    plan = routed.plan_command(("rtk", "git", "status"))

    assert plan.backend is OptimizationBackend.RTK
    assert plan.argv == ("rtk", "git", "status")


def test_an_empty_command_is_refused_before_a_process_can_start() -> None:
    routed, _ = optimizer()

    with pytest.raises(ValueError, match="executable"):
        routed.plan_command(())


async def test_headroom_requires_explicit_permission_to_cross_its_boundary() -> None:
    routed, session = optimizer()

    result = await routed.optimize(
        "sensitive tenant data" * 200,
        channel=OptimizationChannel.MCP,
        tenant="acme",
        run_id="run-1",
    )

    assert result.backend is OptimizationBackend.NONE
    assert result.content == "sensitive tenant data" * 200
    assert "not permitted" in result.reason
    assert session.calls == []


async def test_headroom_compresses_eligible_content_and_reports_savings() -> None:
    routed, session = optimizer()

    result = await routed.optimize(
        "large JSON response" * 200,
        channel=OptimizationChannel.JSON,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
    )

    assert result.backend is OptimizationBackend.HEADROOM
    assert result.content == "3 active hosts in apac [hash=h-1]"
    assert result.handle == "h-1"
    assert result.original_tokens == 900
    assert result.optimized_tokens == 12
    assert result.saved_tokens == 888
    assert result.transforms == ("json-table",)
    assert session.calls[0][0] == "headroom_compress"


async def test_small_content_stays_local_because_the_mcp_call_costs_more() -> None:
    routed, session = optimizer()

    result = await routed.optimize(
        "three rows",
        channel=OptimizationChannel.JSON,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
    )

    assert result.backend is OptimizationBackend.NONE
    assert "threshold" in result.reason
    assert session.calls == []


async def test_headroom_preserves_the_untrusted_label() -> None:
    routed, _ = optimizer()

    result = await routed.optimize(
        "untrusted MCP output" * 200,
        channel=OptimizationChannel.MCP,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
        untrusted=True,
    )

    assert result.untrusted is True


async def test_headroom_outage_degrades_to_the_original_content() -> None:
    session = HeadroomSession()
    session.fail = True
    routed, _ = optimizer(session)
    original = "large API response" * 200

    result = await routed.optimize(
        original,
        channel=OptimizationChannel.API,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
    )

    assert result.backend is OptimizationBackend.NONE
    assert result.content == original
    assert "unavailable" in result.reason


async def test_headroom_mcp_error_degrades_to_the_original_content() -> None:
    session = HeadroomSession()
    session.is_error = True
    routed, _ = optimizer(session)
    original = "large API response" * 200

    result = await routed.optimize(
        original,
        channel=OptimizationChannel.API,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
    )

    assert result.backend is OptimizationBackend.NONE
    assert result.content == original


async def test_retrieval_returns_the_exact_original_inside_the_bound_scope() -> None:
    routed, _ = optimizer()
    compressed = await routed.optimize(
        "large RAG context" * 200,
        channel=OptimizationChannel.RAG,
        tenant="acme",
        run_id="run-1",
        headroom_allowed=True,
    )

    original = await routed.retrieve(compressed.handle, tenant="acme", run_id="run-1")

    assert original == "the exact original"


async def test_retrieval_refuses_a_response_for_a_different_hash() -> None:
    session = HeadroomSession()
    session.retrieved_hash = "different"
    routed, _ = optimizer(session)

    with pytest.raises(OptimizationError, match="different retrieval hash"):
        await routed.retrieve("h-1", tenant="acme", run_id="run-1")


async def test_retrieval_from_another_tenant_fails_without_calling_headroom() -> None:
    routed, session = optimizer()

    with pytest.raises(OptimizationError):
        await routed.retrieve("h-1", tenant="other", run_id="run-1")

    assert session.calls == []


async def test_retrieval_from_another_run_fails_without_calling_headroom() -> None:
    routed, session = optimizer()

    with pytest.raises(OptimizationError):
        await routed.retrieve("h-1", tenant="acme", run_id="run-2")

    assert session.calls == []
