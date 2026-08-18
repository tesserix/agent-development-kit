"""The optional MCP SDK stays behind the AgentGateway transport boundary."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from datetime import timedelta

from tesserix_adk.adapters import McpStreamableHttpTransport
from tesserix_adk.mcp import McpGatewayError, McpGatewayReason


class _Session:
    pages: ClassVar[dict[str | None, dict[str, object]]]
    result: ClassVar[dict[str, object]]
    calls: ClassVar[list[tuple[str, dict[str, object] | None, timedelta | None, object]]]

    def __init__(self, read: object, write: object, **kwargs: object) -> None:
        del read, write, kwargs
        self.initialized = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self, cursor: str | None = None) -> object:
        assert self.initialized
        return _Dumped(self.pages[cursor])

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        read_timeout_seconds: timedelta | None = None,
        *,
        meta: dict[str, object] | None = None,
    ) -> object:
        assert self.initialized
        self.calls.append((name, arguments, read_timeout_seconds, meta))
        return _Dumped(self.result)


def _tool_page(*, next_cursor: str | None = None) -> dict[str, object]:
    return {
        "tools": [
            {
                "name": "get_booking",
                "description": "Remote description",
                "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
            }
        ],
        "nextCursor": next_cursor,
    }


class _Dumped:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def model_dump(self, *, mode: str, by_alias: bool) -> dict[str, object]:
        assert mode == "json"
        assert by_alias
        return self.value


class _HttpClient(Protocol):
    headers: Mapping[str, str]


@asynccontextmanager
async def _streamable_http_client(
    url: str, *, http_client: object
) -> AsyncIterator[tuple[object, object, object]]:
    assert url == "https://agentgateway.example.test/mcp/bookings"
    assert cast("_HttpClient", http_client).headers["authorization"] == "[secure]"
    yield object(), object(), lambda: "session-1"


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    _Session.pages = {None: _tool_page()}
    _Session.result = {"content": [], "structuredContent": None, "isError": False}
    _Session.calls = []
    mcp = ModuleType("mcp")
    mcp.__dict__["ClientSession"] = _Session
    client = ModuleType("mcp.client")
    streamable = ModuleType("mcp.client.streamable_http")
    streamable.__dict__["streamable_http_client"] = _streamable_http_client
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable)


async def test_streamable_http_transport_initializes_and_removes_vendor_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    transport = McpStreamableHttpTransport()

    tools = await transport.list_tools(
        endpoint="https://agentgateway.example.test/mcp/bookings",
        headers={"Authorization": "[secure]"},
        meta={"tesserix/adk/tenant": "acme"},
        timeout_seconds=5,
        max_result_bytes=4096,
        max_tools=40,
    )

    assert tools[0].name == "get_booking"
    assert tools[0].input_schema == {"type": "object"}
    assert type(tools[0]).__module__ == "tesserix_adk.mcp.gateway"


async def test_streamable_http_transport_follows_tool_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    _Session.pages = {
        None: _tool_page(next_cursor="page-2"),
        "page-2": {
            "tools": [
                {
                    "name": "cancel_booking",
                    "inputSchema": {"type": "object"},
                }
            ],
            "nextCursor": None,
        },
    }

    tools = await McpStreamableHttpTransport().list_tools(
        endpoint="https://agentgateway.example.test/mcp/bookings",
        headers={"Authorization": "[secure]"},
        meta={},
        timeout_seconds=5,
        max_result_bytes=4096,
        max_tools=40,
    )

    assert [tool.name for tool in tools] == ["get_booking", "cancel_booking"]


async def test_streamable_http_transport_refuses_a_repeated_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    _Session.pages = {
        None: _tool_page(next_cursor="same"),
        "same": {"tools": [], "nextCursor": "same"},
    }

    with pytest.raises(McpGatewayError) as caught:
        await McpStreamableHttpTransport().list_tools(
            endpoint="https://agentgateway.example.test/mcp/bookings",
            headers={"Authorization": "[secure]"},
            meta={},
            timeout_seconds=5,
            max_result_bytes=4096,
            max_tools=40,
        )

    assert caught.value.reason is McpGatewayReason.DISCOVERY


async def test_streamable_http_transport_refuses_an_oversized_discovery_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)

    with pytest.raises(McpGatewayError) as caught:
        await McpStreamableHttpTransport().list_tools(
            endpoint="https://agentgateway.example.test/mcp/bookings",
            headers={"Authorization": "[secure]"},
            meta={},
            timeout_seconds=5,
            max_result_bytes=8,
            max_tools=40,
        )

    assert caught.value.reason is McpGatewayReason.PAYLOAD


async def test_streamable_http_transport_stops_discovery_at_the_tool_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    _Session.pages = {
        None: {
            "tools": [
                {"name": "first", "inputSchema": {"type": "object"}},
                {"name": "second", "inputSchema": {"type": "object"}},
            ],
            "nextCursor": None,
        }
    }

    with pytest.raises(McpGatewayError) as caught:
        await McpStreamableHttpTransport().list_tools(
            endpoint="https://agentgateway.example.test/mcp/bookings",
            headers={"Authorization": "[secure]"},
            meta={},
            timeout_seconds=5,
            max_result_bytes=4096,
            max_tools=1,
        )

    assert caught.value.reason is McpGatewayReason.LIMIT


async def test_streamable_http_transport_converts_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    _Session.result = {
        "content": [{"type": "text", "text": "confirmed"}],
        "structuredContent": {"status": "confirmed"},
        "isError": False,
    }

    result = await McpStreamableHttpTransport().call_tool(
        endpoint="https://agentgateway.example.test/mcp/bookings",
        tool="get_booking",
        arguments={"booking_id": "B-7"},
        headers={"Authorization": "[secure]"},
        meta={"tesserix/adk/run": "run_7"},
        timeout_seconds=5,
        max_result_bytes=4096,
    )

    assert result.structured_content == {"status": "confirmed"}
    assert result.content == ({"type": "text", "text": "confirmed"},)
    assert _Session.calls[0][0:2] == ("get_booking", {"booking_id": "B-7"})
    assert _Session.calls[0][3] == {"tesserix/adk/run": "run_7"}


async def test_streamable_http_transport_refuses_an_oversized_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    _Session.result = {
        "content": [{"type": "text", "text": "x" * 100}],
        "structuredContent": None,
        "isError": False,
    }

    with pytest.raises(McpGatewayError) as caught:
        await McpStreamableHttpTransport().call_tool(
            endpoint="https://agentgateway.example.test/mcp/bookings",
            tool="get_booking",
            arguments={"booking_id": "B-7"},
            headers={"Authorization": "[secure]"},
            meta={},
            timeout_seconds=5,
            max_result_bytes=8,
        )

    assert caught.value.reason is McpGatewayReason.PAYLOAD
