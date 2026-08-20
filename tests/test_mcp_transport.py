"""One MCP server reached two ways, with the agent unable to tell which."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tesserix_adk.adapters.mcp import McpClient
from tesserix_adk.adapters.mcp_transport import (
    HttpTransport,
    McpTransportError,
    McpTransportReason,
    RecordingTransport,
    StdioTransport,
    TransportSession,
    transport_for,
)
from tesserix_adk.core.config import McpServerConfig
from tesserix_adk.core.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from pydantic import JsonValue

_SEARCH: dict[str, JsonValue] = {
    "name": "search",
    "description": "Search the handbook.",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

_TOOLS: list[JsonValue] = [_SEARCH]

_SERVER = """
import json, sys

TOOLS = {tools}

def reply(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    method, ident = request.get("method"), request.get("id")
    if ident is None:
        continue
    if method == "initialize":
        result = {{
            "protocolVersion": "2025-06-18",
            "capabilities": {{"tools": {{}}}},
            "serverInfo": {{"name": "handbook", "version": "1.2.0"}},
        }}
    elif method == "tools/list":
        result = {{"tools": TOOLS}}
    elif method == "tools/call":
        query = request["params"]["arguments"]["query"]
        result = {{"content": [{{"type": "text", "text": "found " + query}}]}}
    elif method == "shout":
        reply({{"jsonrpc": "2.0", "id": ident, "result": {{"noise": "x" * 200000}}}})
        continue
    elif method == "leak":
        result = {{"seen": sorted(k for k in os.environ if k.startswith("MCP_TEST_"))}}
    elif method == "sleep":
        continue
    elif method == "die":
        sys.exit(3)
    else:
        reply({{"jsonrpc": "2.0", "id": ident, "error": {{"code": -32601, "message": "no"}}}})
        continue
    reply({{"jsonrpc": "2.0", "id": ident, "result": result}})
"""


def _script(tmp_path: Path) -> Path:
    script = tmp_path / "server.py"
    script.write_text("import os\n" + _SERVER.format(tools=json.dumps(_TOOLS)))
    return script


def _stdio(tmp_path: Path, **overrides: Any) -> McpServerConfig:
    return McpServerConfig(
        name="handbook",
        transport="stdio",
        command=(sys.executable, "-u", str(_script(tmp_path))),
        **overrides,
    )


def _endpoint(**overrides: Any) -> McpServerConfig:
    return McpServerConfig(name="handbook", endpoint="http://handbook.internal/mcp", **overrides)


def _http(handler: Any, **overrides: Any) -> HttpTransport:
    config = _endpoint(**overrides)
    return HttpTransport(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _json_rpc(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    method, ident = body["method"], body.get("id")
    if ident is None:
        return httpx.Response(202)
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "handbook", "version": "1.2.0"},
        }
    elif method == "tools/list":
        result = {"tools": _TOOLS}
    else:
        query = body["params"]["arguments"]["query"]
        result = {"content": [{"type": "text", "text": f"found {query}"}]}
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": ident, "result": result})


class TestOneServerReachedTwoWays:
    @pytest.mark.asyncio
    async def test_discovery_and_a_call_agree_across_transports(self, tmp_path: Path) -> None:
        stdio_config = _stdio(tmp_path)
        http_config = _endpoint()
        http = HttpTransport(
            http_config, client=httpx.AsyncClient(transport=httpx.MockTransport(_json_rpc))
        )
        answers = []
        schemas = []
        for config, transport in (
            (stdio_config, StdioTransport(stdio_config)),
            (http_config, http),
        ):
            async with McpClient(
                TransportSession(transport, config=config), config=config
            ) as client:
                discovery = await client.discover()
                schemas.append([(tool.name, tool.parameters_schema) for tool in discovery.tools])
                answers.append(await discovery.tools[0].invoke({"query": "leave"}))

        assert schemas[0] == schemas[1]
        assert "found leave" in answers[0]
        assert answers[0] == answers[1]

    @pytest.mark.asyncio
    async def test_the_server_reports_itself_the_same_way(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        session = TransportSession(StdioTransport(config), config=config)

        info = await session.initialize()
        await session.close()

        assert (info.name, info.version, info.capabilities) == ("handbook", "1.2.0", ("tools",))


class TestFailuresAreTyped:
    @pytest.mark.asyncio
    async def test_a_child_that_exits_mid_call_is_a_disconnect(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        transport = StdioTransport(config)
        await transport.open()

        with pytest.raises(McpTransportError) as failure:
            await transport.request("die", {}, timeout_seconds=5.0)

        assert failure.value.reason is McpTransportReason.DISCONNECTED
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_stream_that_goes_quiet_is_a_timeout(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path, read_timeout_seconds=0.2)
        transport = StdioTransport(config)
        await transport.open()

        with pytest.raises(McpTransportError) as failure:
            await transport.request("sleep", {}, timeout_seconds=0.2)

        assert failure.value.reason is McpTransportReason.TIMEOUT
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_child_that_writes_without_end_is_bounded(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path, max_message_bytes=2048)
        transport = StdioTransport(config)
        await transport.open()

        with pytest.raises(McpTransportError) as failure:
            await transport.request("shout", {}, timeout_seconds=5.0)

        assert failure.value.reason is McpTransportReason.LIMIT
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_redirect_is_never_followed(self) -> None:
        transport = _http(
            lambda _request: httpx.Response(302, headers={"location": "http://elsewhere/mcp"})
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.PROTOCOL
        await transport.close()

    @pytest.mark.asyncio
    async def test_an_html_error_page_is_not_read_as_protocol(self) -> None:
        transport = _http(
            lambda _request: httpx.Response(
                200, text="<html>gateway</html>", headers={"content-type": "text/html"}
            )
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.PROTOCOL
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_server_error_is_a_protocol_failure_carrying_no_prose(self) -> None:
        transport = _http(
            lambda request: httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json.loads(request.content)["id"],
                    "error": {"code": -32000, "message": "secret internal detail"},
                },
            )
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert "secret internal detail" not in str(failure.value)
        await transport.close()


class TestAnUnreachableEndpoint:
    @pytest.mark.asyncio
    async def test_attempts_are_bounded_and_then_reported(self) -> None:
        attempts = 0

        def refuse(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("refused", request=request)

        transport = _http(refuse)

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=2.0)

        assert failure.value.reason is McpTransportReason.UNAVAILABLE
        assert 1 < attempts <= 4
        await transport.close()


class TestTheChildProcess:
    @pytest.mark.asyncio
    async def test_only_allowlisted_variables_reach_it(self, tmp_path: Path) -> None:
        os.environ["MCP_TEST_ALLOWED"] = "yes"
        os.environ["MCP_TEST_WITHHELD"] = "no"
        config = _stdio(tmp_path, env_allow=("MCP_TEST_ALLOWED",))
        transport = StdioTransport(config)
        await transport.open()

        seen = await transport.request("leak", {}, timeout_seconds=5.0)

        assert seen["seen"] == ["MCP_TEST_ALLOWED"]
        await transport.close()

    @pytest.mark.asyncio
    async def test_closing_leaves_nothing_behind(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        transport = StdioTransport(config)
        await transport.open()
        await transport.request("initialize", {}, timeout_seconds=5.0)

        await transport.close()

        assert transport.healthy is False
        assert transport.returncode is not None

    @pytest.mark.asyncio
    async def test_a_cancelled_call_still_tears_the_child_down(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        transport = StdioTransport(config)
        await transport.open()
        call = asyncio.ensure_future(transport.request("sleep", {}, timeout_seconds=30.0))
        await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        await transport.close()

        assert transport.returncode is not None

    @pytest.mark.asyncio
    async def test_a_command_that_does_not_exist_is_unavailable(self, tmp_path: Path) -> None:
        config = McpServerConfig(
            name="handbook", transport="stdio", command=(str(tmp_path / "nothing"),)
        )
        transport = StdioTransport(config)

        with pytest.raises(McpTransportError) as failure:
            await transport.open()

        assert failure.value.reason is McpTransportReason.UNAVAILABLE


class TestLimitsHoldOnEveryTransport:
    @pytest.mark.asyncio
    async def test_an_oversized_http_message_is_refused(self) -> None:
        transport = _http(
            lambda request: httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": json.loads(request.content)["id"],
                    "result": {"noise": "x" * 5000},
                },
            ),
            max_message_bytes=2048,
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.LIMIT
        await transport.close()

    @pytest.mark.asyncio
    async def test_no_more_requests_are_in_flight_than_declared(self) -> None:
        live = 0
        peak = 0

        async def slow(request: httpx.Request) -> httpx.Response:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.05)
            live -= 1
            return _json_rpc(request)

        transport = _http(slow, max_in_flight=2)

        await asyncio.gather(
            *(transport.request("tools/list", {}, timeout_seconds=5.0) for _ in range(6))
        )

        assert peak <= 2
        await transport.close()


class TestDeclaringTheTransport:
    def test_a_stdio_server_needs_an_argv(self) -> None:
        with pytest.raises(ValueError, match="argv"):
            McpServerConfig(name="handbook", transport="stdio")

    def test_an_http_server_is_not_given_an_argv(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            McpServerConfig(name="handbook", transport="http", command=("python",))

    def test_the_factory_builds_what_was_declared(self, tmp_path: Path) -> None:
        stdio = transport_for(_stdio(tmp_path))
        http = transport_for(McpServerConfig(name="handbook", endpoint="http://x/mcp"))

        assert isinstance(stdio, StdioTransport)
        assert isinstance(http, HttpTransport)

    def test_an_http_server_without_an_endpoint_cannot_be_built(self) -> None:
        with pytest.raises(ConfigurationError, match="endpoint"):
            transport_for(McpServerConfig(name="handbook"))


class TestRecordingTransport:
    @pytest.mark.asyncio
    async def test_it_answers_from_a_script_and_records_what_was_asked(self) -> None:
        transport = RecordingTransport(
            {
                "initialize": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "handbook", "version": "1.2.0"},
                },
                "tools/list": {"tools": _TOOLS},
                "tools/call": {"content": [{"type": "text", "text": "found leave"}]},
            }
        )
        config = McpServerConfig(name="handbook", endpoint="http://x/mcp")

        async with McpClient(TransportSession(transport, config=config), config=config) as client:
            discovery = await client.discover()
            answer = await discovery.tools[0].invoke({"query": "leave"})

        assert "found leave" in answer
        assert [method for method, _ in transport.requests] == [
            "initialize",
            "tools/list",
            "tools/call",
        ]

    @pytest.mark.asyncio
    async def test_a_method_it_was_not_given_is_a_protocol_failure(self) -> None:
        transport = RecordingTransport({})

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.PROTOCOL


class TestSessionOverAnyTransport:
    @pytest.mark.asyncio
    async def test_pagination_is_followed_to_the_end(self) -> None:
        listed: list[dict[str, JsonValue]] = [
            {"tools": _TOOLS, "nextCursor": "second"},
            {"tools": [{**_SEARCH, "name": "lookup"}]},
        ]
        pages: Iterator[dict[str, JsonValue]] = iter(listed)

        class Paged(RecordingTransport):
            async def request(
                self,
                method: str,
                params: Mapping[str, JsonValue],
                *,
                timeout_seconds: float,
            ) -> dict[str, JsonValue]:
                if method == "tools/list":
                    self.requests.append((method, dict(params)))
                    return next(pages)
                return await super().request(method, params, timeout_seconds=timeout_seconds)

        transport = Paged(
            {
                "initialize": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "handbook", "version": "1.2.0"},
                }
            }
        )
        config = McpServerConfig(name="handbook", endpoint="http://x/mcp")
        session = TransportSession(transport, config=config)

        tools = await session.list_tools()

        assert [tool.name for tool in tools] == ["search", "lookup"]

    @pytest.mark.asyncio
    async def test_a_tool_error_comes_back_as_a_result_the_client_can_read(self) -> None:
        transport = RecordingTransport(
            {"tools/call": {"content": [{"type": "text", "text": "no"}], "isError": True}}
        )
        config = McpServerConfig(name="handbook", endpoint="http://x/mcp")
        session = TransportSession(transport, config=config)

        result = await session.call_tool(
            "search", {"query": "x"}, meta={"run-id": "run-1"}, timeout_seconds=1.0
        )

        assert result.is_error is True
        assert transport.requests[0][1]["_meta"] == {"run-id": "run-1"}


class TestAnEndpointThatStreams:
    @pytest.mark.asyncio
    async def test_a_data_event_is_read_as_the_reply(self) -> None:
        def stream(request: httpx.Request) -> httpx.Response:
            body = {"jsonrpc": "2.0", "id": json.loads(request.content)["id"], "result": {"ok": 1}}
            return httpx.Response(
                200,
                text=f"event: message\ndata: {json.dumps(body)}\n\n",
                headers={"content-type": "text/event-stream"},
            )

        transport = _http(stream)

        assert await transport.request("tools/list", {}, timeout_seconds=1.0) == {"ok": 1}
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_stream_that_carries_nothing_is_a_disconnect(self) -> None:
        transport = _http(
            lambda _request: httpx.Response(
                200, text=": keep-alive\n\n", headers={"content-type": "text/event-stream"}
            )
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.DISCONNECTED
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_stream_past_the_ceiling_is_refused(self) -> None:
        transport = _http(
            lambda _request: httpx.Response(
                200,
                text="data: " + "x" * 5000 + "\n\n",
                headers={"content-type": "text/event-stream"},
            ),
            max_message_bytes=1024,
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.LIMIT
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_body_that_is_not_json_is_not_read_as_protocol(self) -> None:
        transport = _http(
            lambda _request: httpx.Response(
                200, text="not json", headers={"content-type": "application/json"}
            )
        )

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.PROTOCOL
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_a_result_is_refused(self) -> None:
        transport = _http(
            lambda request: httpx.Response(
                200, json={"jsonrpc": "2.0", "id": json.loads(request.content)["id"], "result": 7}
            )
        )

        with pytest.raises(McpTransportError, match="not a result"):
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        await transport.close()

    @pytest.mark.asyncio
    async def test_a_server_error_status_is_reported_as_unavailable(self) -> None:
        transport = _http(lambda _request: httpx.Response(503))

        with pytest.raises(McpTransportError) as failure:
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert failure.value.reason is McpTransportReason.UNAVAILABLE
        assert failure.value.retryable is True
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_notification_expects_nothing_back(self) -> None:
        posted: list[str] = []

        def accept(request: httpx.Request) -> httpx.Response:
            posted.append(json.loads(request.content)["method"])
            return httpx.Response(202)

        transport = _http(accept)

        await transport.notify("notifications/initialized", {})

        assert posted == ["notifications/initialized"]
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_closed_transport_refuses_to_send(self) -> None:
        transport = _http(_json_rpc)
        await transport.close()
        await transport.close()

        with pytest.raises(McpTransportError):
            await transport.request("tools/list", {}, timeout_seconds=1.0)

        assert transport.healthy is False


class TestAChildThatMisbehaves:
    @pytest.mark.asyncio
    async def test_an_error_reply_carries_no_server_prose(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        transport = StdioTransport(config)

        with pytest.raises(McpTransportError) as failure:
            await transport.request("unknown", {}, timeout_seconds=5.0)

        assert failure.value.reason is McpTransportReason.PROTOCOL
        await transport.close()

    @pytest.mark.asyncio
    async def test_what_it_wrote_to_stderr_is_kept(self, tmp_path: Path) -> None:
        script = tmp_path / "noisy.py"
        script.write_text("import sys\nsys.stderr.write('it went wrong\\n')\n")
        config = McpServerConfig(
            name="handbook", transport="stdio", command=(sys.executable, "-u", str(script))
        )
        transport = StdioTransport(config)
        await transport.open()

        with pytest.raises(McpTransportError):
            await transport.request("initialize", {}, timeout_seconds=5.0)

        await transport.close()
        assert transport.stderr_tail == ("it went wrong",)

    @pytest.mark.asyncio
    async def test_a_reopened_transport_stays_closed(self, tmp_path: Path) -> None:
        config = _stdio(tmp_path)
        transport = StdioTransport(config)
        await transport.close()

        with pytest.raises(McpTransportError):
            await transport.open()


class TestDiscoveryOverATransport:
    @pytest.mark.asyncio
    async def test_a_page_that_is_not_a_list_is_refused(self) -> None:
        session = TransportSession(
            RecordingTransport({"tools/list": {"tools": "everything"}}), config=_endpoint()
        )

        with pytest.raises(McpTransportError, match="invalid tool page"):
            await session.list_tools()

    @pytest.mark.asyncio
    async def test_a_descriptor_that_is_not_an_object_is_refused(self) -> None:
        session = TransportSession(
            RecordingTransport({"tools/list": {"tools": ["search"]}}), config=_endpoint()
        )

        with pytest.raises(McpTransportError, match="invalid tool descriptor"):
            await session.list_tools()

    @pytest.mark.asyncio
    async def test_more_tools_than_the_ceiling_allows_is_refused(self) -> None:
        many: list[JsonValue] = [{**_SEARCH, "name": f"search{index}"} for index in range(5)]
        session = TransportSession(
            RecordingTransport({"tools/list": {"tools": many}}),
            config=_endpoint(max_tools=2),
        )

        with pytest.raises(McpTransportError) as failure:
            await session.list_tools()

        assert failure.value.reason is McpTransportReason.LIMIT

    @pytest.mark.asyncio
    async def test_a_repeated_cursor_is_refused(self) -> None:
        session = TransportSession(
            RecordingTransport({"tools/list": {"tools": [], "nextCursor": "again"}}),
            config=_endpoint(),
        )

        with pytest.raises(McpTransportError, match="repeated a discovery cursor"):
            await session.list_tools()

    @pytest.mark.asyncio
    async def test_a_server_that_says_nothing_about_itself_is_named_as_declared(self) -> None:
        transport = RecordingTransport({"initialize": {}})
        session = TransportSession(transport, config=_endpoint())

        info = await session.initialize()

        assert (info.name, info.capabilities) == ("handbook", ())
        assert transport.notifications == [("notifications/initialized", {})]

    @pytest.mark.asyncio
    async def test_noise_on_stdout_is_stepped_over(self, tmp_path: Path) -> None:
        script = tmp_path / "chatty.py"
        script.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    ident = json.loads(line)['id']\n"
            "    sys.stdout.write('starting up\\n')\n"
            "    sys.stdout.write('[1, 2]\\n')\n"
            "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': 99, 'result': {}}) + '\\n')\n"
            "    sys.stdout.write(\n"
            "        json.dumps({'jsonrpc': '2.0', 'id': ident, 'result': {'ok': 1}}) + '\\n'\n"
            "    )\n"
            "    sys.stdout.flush()\n"
        )
        config = McpServerConfig(
            name="handbook", transport="stdio", command=(sys.executable, "-u", str(script))
        )
        transport = StdioTransport(config)

        assert await transport.request("initialize", {}, timeout_seconds=5.0) == {"ok": 1}
        await transport.close()

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_a_result_is_refused(self, tmp_path: Path) -> None:
        script = tmp_path / "wrong.py"
        script.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    ident = json.loads(line)['id']\n"
            "    sys.stdout.write(json.dumps({'id': ident, 'result': 7}) + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        config = McpServerConfig(
            name="handbook", transport="stdio", command=(sys.executable, "-u", str(script))
        )
        transport = StdioTransport(config)

        with pytest.raises(McpTransportError, match="not a result"):
            await transport.request("initialize", {}, timeout_seconds=5.0)

        await transport.close()
