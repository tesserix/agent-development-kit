"""Kit tools published over MCP, held to what the in-process path enforces, with no network."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from tesserix_adk.adapters.mcp import McpClient
from tesserix_adk.adapters.mcp_server import (
    APPROVAL_REQUIRED,
    ExportedSession,
    McpExportError,
    McpExportReason,
    McpServer,
    published,
)
from tesserix_adk.adapters.mcp_transport import PROTOCOL_VERSION
from tesserix_adk.core.config import McpServerConfig
from tesserix_adk.core.errors import McpAuthError, ToolRefusal
from tesserix_adk.core.hooks import ApprovalPolicy
from tesserix_adk.mcp import META_PREFIX
from tesserix_adk.tools import ToolContext, ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import Mapping

_TENANT = {f"{META_PREFIX}/tenant": "acme", f"{META_PREFIX}/run": "run-1"}


class Fare(BaseModel):
    """What a tool answering with a model returns."""

    leg: str
    eur: int


def _tools() -> tuple[Any, ...]:
    """Six tools, of which two are meant to be published."""

    @tool(name="export_fare_for")
    def export_fare_for(leg: str) -> dict[str, Any]:
        """Price one leg."""
        return {"leg": leg, "eur": 40}

    @tool(name="export_whose", description="Say whose tenant the call ran under.")
    def export_whose(context: ToolContext) -> dict[str, Any]:
        """Report the bound tenant."""
        return {"tenant": context.tenant, "run": context.run_id}

    @tool(
        name="export_refund", requires_approval=ApprovalPolicy(required=True, reason="money leaves")
    )
    def export_refund(order: str, amount: int) -> dict[str, Any]:
        """Refund an order."""
        return {"order": order, "amount": amount}

    @tool(name="export_decline")
    def export_decline(order: str) -> dict[str, Any]:
        """Decline every call."""
        raise ToolRefusal("export_decline", "not_cancellable", f"{order} has shipped")

    @tool(name="export_leak")
    def export_leak() -> dict[str, Any]:
        """Answer with something the local path would redact."""
        return {"note": "the key is sk-live-9f2c8a71b4e6"}

    @tool(name="export_unpublished")
    def export_unpublished() -> str:
        """Never exported."""
        return "internal"

    return (
        export_fare_for,
        export_whose,
        export_refund,
        export_decline,
        export_leak,
        export_unpublished,
    )


def _server(
    *, exports: tuple[str, ...] = ("export_fare_for", "export_refund"), **overrides: Any
) -> McpServer:
    """A registry of the six tools, published under an explicit export allowlist."""
    registered = _tools()
    registry = ToolRegistry(registered, timeouts=overrides.pop("timeouts", None))
    view = registry.view(allow=tuple(each.name for each in registered), agent="planner")
    return McpServer(view, exports=exports, name="handbook", version="1.2.0", **overrides)


def _calls(server: McpServer, **overrides: Any) -> ExportedSession:
    """A connected session, with a supported protocol version unless one is given."""
    return server.connect(**overrides)


class TestWhatIsPublished:
    """Only the allowlist is served, and its schemas come from the definitions."""

    @pytest.mark.asyncio
    async def test_lists_only_the_exported_tools(self) -> None:
        listed = await _calls(_server()).list_tools()
        assert [descriptor.name for descriptor in listed] == ["export_fare_for", "export_refund"]

    @pytest.mark.asyncio
    async def test_publishes_the_generated_schema(self) -> None:
        registered = _tools()[0]
        listed = await _calls(_server()).list_tools()
        assert listed[0].input_schema == registered.parameters_schema
        assert listed[0].description == registered.description

    @pytest.mark.asyncio
    async def test_an_unpublished_tool_is_refused_as_if_it_did_not_exist(self) -> None:
        session = _calls(_server())
        with pytest.raises(McpExportError) as export_unpublished:
            await session.call_tool("export_unpublished", {}, meta=_TENANT, timeout_seconds=1)
        with pytest.raises(McpExportError) as nonsense:
            await session.call_tool("export_no_such_tool", {}, meta=_TENANT, timeout_seconds=1)
        assert export_unpublished.value.reason is McpExportReason.NOT_FOUND
        assert str(export_unpublished.value).replace("export_unpublished", "X") == str(
            nonsense.value
        ).replace("export_no_such_tool", "X")

    def test_exporting_something_the_view_cannot_call_is_refused(self) -> None:
        with pytest.raises(McpExportError) as refused:
            _server(exports=("export_fare_for", "export_nothing_registered"))
        assert refused.value.reason is McpExportReason.NOT_FOUND

    def test_a_descriptor_is_generated_from_the_definition(self) -> None:
        descriptor = published(_tools()[0])
        assert descriptor.name == "export_fare_for"
        assert descriptor.input_schema["type"] == "object"


class TestWhoTheCallIsFor:
    """A call names its tenant or it does not run."""

    @pytest.mark.asyncio
    async def test_an_unscoped_call_never_enters_the_tool(self) -> None:
        session = _calls(_server(exports=("export_whose",)))
        with pytest.raises(McpAuthError):
            await session.call_tool("export_whose", {}, meta={}, timeout_seconds=1)

    @pytest.mark.asyncio
    async def test_the_tenant_from_the_request_reaches_the_tool(self) -> None:
        session = _calls(_server(exports=("export_whose",)))
        answered = await session.call_tool("export_whose", {}, meta=_TENANT, timeout_seconds=1)
        assert answered.structured_content == {"tenant": "acme", "run": "run-1"}

    @pytest.mark.asyncio
    async def test_concurrent_callers_do_not_see_each_others_tenant(self) -> None:
        session = _calls(_server(exports=("export_whose",)))

        async def called(tenant: str) -> Mapping[str, Any] | None:
            meta = {f"{META_PREFIX}/tenant": tenant, f"{META_PREFIX}/run": f"run-{tenant}"}
            answered = await session.call_tool("export_whose", {}, meta=meta, timeout_seconds=1)
            return answered.structured_content

        both = await asyncio.gather(called("acme"), called("globex"))
        assert [each["tenant"] for each in both if each] == ["acme", "globex"]


class TestArgumentsFromOutside:
    """Arguments are held to the published schema before the body is entered."""

    @pytest.mark.asyncio
    async def test_arguments_outside_the_schema_are_refused(self) -> None:
        session = _calls(_server())
        with pytest.raises(McpExportError) as refused:
            await session.call_tool(
                "export_fare_for", {"leg": "Osaka", "extra": 1}, meta=_TENANT, timeout_seconds=1
            )
        assert refused.value.reason is McpExportReason.INVALID_ARGUMENTS

    @pytest.mark.asyncio
    async def test_a_refusal_does_not_repeat_the_argument_it_refused(self) -> None:
        session = _calls(_server())
        with pytest.raises(McpExportError) as refused:
            await session.call_tool(
                "export_fare_for",
                {"leg": ["sk-live-9f2c8a71b4e6"]},
                meta=_TENANT,
                timeout_seconds=1,
            )
        assert "sk-live-9f2c8a71b4e6" not in str(refused.value)


class TestApprovalIsNotWaivedForARemoteCaller:
    """The gate is enforced here, because the caller cannot be asked to enforce it."""

    @pytest.mark.asyncio
    async def test_an_approval_required_tool_answers_pending_rather_than_running(self) -> None:
        session = _calls(_server())
        answered = await session.call_tool(
            "export_refund", {"order": "A-1", "amount": 50}, meta=_TENANT, timeout_seconds=1
        )
        structured = answered.structured_content or {}
        assert answered.is_error
        assert structured["refusal"] == {"code": APPROVAL_REQUIRED}

    @pytest.mark.asyncio
    async def test_the_pending_response_carries_a_digest_rather_than_the_arguments(self) -> None:
        session = _calls(_server())
        answered = await session.call_tool(
            "export_refund", {"order": "A-1", "amount": 50}, meta=_TENANT, timeout_seconds=1
        )
        approval = (answered.structured_content or {})["approval"]
        assert isinstance(approval, dict)
        assert len(str(approval["arguments_digest"])) == 64
        assert "A-1" not in json.dumps(approval)


class TestWhatComesBack:
    """A result leaves the process the way the local path would have left it."""

    @pytest.mark.asyncio
    async def test_a_result_is_redacted_before_serialisation(self) -> None:
        session = _calls(_server(exports=("export_leak",)))
        answered = await session.call_tool("export_leak", {}, meta=_TENANT, timeout_seconds=1)
        assert "sk-live-9f2c8a71b4e6" not in json.dumps(answered.structured_content)

    @pytest.mark.asyncio
    async def test_a_refusal_is_answered_as_a_refusal_with_its_code(self) -> None:
        session = _calls(_server(exports=("export_decline",)))
        answered = await session.call_tool(
            "export_decline", {"order": "A-1"}, meta=_TENANT, timeout_seconds=1
        )
        assert answered.is_error
        assert (answered.structured_content or {})["refusal"] == {"code": "not_cancellable"}

    @pytest.mark.asyncio
    async def test_a_timeout_is_answered_as_a_failure_rather_than_hanging(self) -> None:
        @tool(name="export_slow")
        async def export_slow() -> str:
            """Take longer than the ceiling allows."""
            await asyncio.sleep(0.5)
            return "late"

        registry = ToolRegistry((export_slow,), timeouts={"export_slow": 0.01})
        view = registry.view(allow=("export_slow",), agent="planner")
        session = McpServer(view, exports=("export_slow",)).connect()
        answered = await session.call_tool("export_slow", {}, meta=_TENANT, timeout_seconds=1)
        assert answered.is_error
        assert (answered.structured_content or {})["failure"] == {"code": "tool_timed_out"}


class TestAToolThatAnswersWithAModel:
    """A pydantic result is published as structured content rather than as its repr."""

    @pytest.mark.asyncio
    async def test_a_model_result_is_structured(self) -> None:
        @tool(name="export_priced")
        def export_priced(leg: str) -> Fare:
            """Price one leg, as a model."""
            return Fare(leg=leg, eur=40)

        registry = ToolRegistry((export_priced,))
        session = McpServer(
            registry.view(allow=("export_priced",), agent="planner"), exports=("export_priced",)
        ).connect()
        answered = await session.call_tool(
            "export_priced", {"leg": "Osaka"}, meta=_TENANT, timeout_seconds=1
        )
        assert answered.structured_content == {"leg": "Osaka", "eur": 40}


class TestTheProtocolItself:
    """A version outside the supported range is refused rather than best-effort."""

    @pytest.mark.asyncio
    async def test_a_supported_version_negotiates(self) -> None:
        info = await _calls(_server()).initialize()
        assert info.protocol_version == PROTOCOL_VERSION
        assert info.capabilities == ("tools",)
        assert info.name == "handbook"

    @pytest.mark.asyncio
    async def test_an_unsupported_version_is_refused(self) -> None:
        session = _calls(_server(), protocol_version="2019-01-01")
        with pytest.raises(McpExportError) as refused:
            await session.initialize()
        assert refused.value.reason is McpExportReason.UNSUPPORTED_PROTOCOL


class TestChangingWhatIsPublished:
    """A widened allowlist reaches new sessions and not open ones."""

    @pytest.mark.asyncio
    async def test_an_open_session_does_not_widen(self) -> None:
        server = _server()
        open_session = _calls(server)
        server.reload(("export_fare_for", "export_refund", "export_leak"))
        assert [each.name for each in await open_session.list_tools()] == [
            "export_fare_for",
            "export_refund",
        ]

    def test_the_published_set_is_readable(self) -> None:
        server = _server()
        server.reload(("export_fare_for",))
        assert server.exports == ("export_fare_for",)

    @pytest.mark.asyncio
    async def test_a_new_session_sees_the_reloaded_allowlist(self) -> None:
        server = _server()
        server.reload(("export_fare_for", "export_refund", "export_leak"))
        listed = await _calls(server).list_tools()
        assert [each.name for each in listed] == ["export_fare_for", "export_refund", "export_leak"]


class TestOneCallerDoesNotStarveAnother:
    """Concurrency is bounded per tenant, not only across the process."""

    @pytest.mark.asyncio
    async def test_one_tenants_calls_are_bounded(self) -> None:
        running, peak = 0, 0

        @tool(name="export_wait")
        async def export_wait() -> str:
            """Occupy a lane for long enough to be counted."""
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0)
            running -= 1
            return "done"

        registry = ToolRegistry((export_wait,))
        view = registry.view(allow=("export_wait",), agent="planner")
        session = McpServer(view, exports=("export_wait",), per_tenant_calls=1).connect()
        await asyncio.gather(
            *(
                session.call_tool("export_wait", {}, meta=_TENANT, timeout_seconds=1)
                for _ in range(4)
            )
        )
        assert peak == 1

    @pytest.mark.asyncio
    async def test_a_second_tenant_is_not_held_behind_the_first(self) -> None:
        entered: list[str] = []
        held = asyncio.Event()

        @tool(name="export_hold")
        async def export_hold(context: ToolContext) -> str:
            """Block the first tenant until the second has been let in."""
            entered.append(context.tenant)
            if context.tenant == "acme":
                await held.wait()
            else:
                held.set()
            return context.tenant

        registry = ToolRegistry((export_hold,))
        view = registry.view(allow=("export_hold",), agent="planner")
        session = McpServer(view, exports=("export_hold",), per_tenant_calls=1).connect()

        async def called(tenant: str) -> None:
            await session.call_tool(
                "export_hold", {}, meta={f"{META_PREFIX}/tenant": tenant}, timeout_seconds=1
            )

        await asyncio.wait_for(asyncio.gather(called("acme"), called("globex")), timeout=2)
        assert sorted(entered) == ["acme", "globex"]


class TestTheSameToolBothWays:
    """What a remote caller gets and what the in-process caller gets are the same."""

    @pytest.mark.asyncio
    async def test_a_success_matches_the_in_process_result(self) -> None:
        registered = _tools()
        registry = ToolRegistry(registered)
        view = registry.view(allow=tuple(each.name for each in registered), agent="planner")
        locally = await view.invoke("export_fare_for", {"leg": "Osaka"})
        session = McpServer(view, exports=("export_fare_for",)).connect()
        remotely = await session.call_tool(
            "export_fare_for", {"leg": "Osaka"}, meta=_TENANT, timeout_seconds=1
        )
        assert remotely.structured_content == locally

    @pytest.mark.asyncio
    async def test_a_refusal_matches_the_in_process_refusal(self) -> None:
        registered = _tools()
        registry = ToolRegistry(registered)
        view = registry.view(allow=tuple(each.name for each in registered), agent="planner")
        with pytest.raises(ToolRefusal) as locally:
            await view.invoke("export_decline", {"order": "A-1"})
        server = McpServer(view, exports=("export_decline",))
        client = McpClient(
            server.connect(meta=_TENANT),
            config=McpServerConfig(
                name="handbook", endpoint="https://handbook.internal/mcp", allow=("*",)
            ),
        )
        discovered = await client.discover()
        with pytest.raises(ToolRefusal) as remotely:
            await discovered.tools[0].invoke({"order": "A-1"})
        assert remotely.value.code == locally.value.code

    @pytest.mark.asyncio
    async def test_the_server_is_usable_wherever_a_session_is(self) -> None:
        registered = _tools()
        registry = ToolRegistry(registered)
        view = registry.view(allow=tuple(each.name for each in registered), agent="planner")
        server = McpServer(view, exports=("export_fare_for",))
        client = McpClient(
            server.connect(meta=_TENANT),
            config=McpServerConfig(
                name="handbook", endpoint="https://handbook.internal/mcp", allow=("*",)
            ),
        )
        async with client:
            discovered = await client.discover()
            answered = await discovered.tools[0].invoke({"leg": "Osaka"})
        assert "Osaka" in answered
