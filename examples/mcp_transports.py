"""One MCP server reached two ways, with the agent unable to tell which.

Run it with `uv run python examples/mcp_transports.py`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

from tesserix_adk.adapters import (
    HttpTransport,
    McpClient,
    McpTransport,
    RecordingTransport,
    StdioTransport,
    TransportSession,
)
from tesserix_adk.core.config import McpServerConfig

TOOLS = [
    {
        "name": "search",
        "description": "Search the handbook.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]

SERVER = """
import json, sys

TOOLS = json.loads(sys.argv[1])

for line in sys.stdin:
    request = json.loads(line)
    method, ident = request["method"], request.get("id")
    if ident is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "handbook", "version": "1.2.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    else:
        query = request["params"]["arguments"]["query"]
        result = {"content": [{"type": "text", "text": "found " + query}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}) + "\\n")
    sys.stdout.flush()
"""


def endpoint(request: httpx.Request) -> httpx.Response:
    """Answer as an in-cluster MCP endpoint would, with no socket involved."""
    body = json.loads(request.content)
    identifier = body.get("id")
    if identifier is None:
        return httpx.Response(202)
    if body["method"] == "initialize":
        result: dict[str, object] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "handbook", "version": "1.2.0"},
        }
    elif body["method"] == "tools/list":
        result = {"tools": TOOLS}
    else:
        query = body["params"]["arguments"]["query"]
        result = {"content": [{"type": "text", "text": f"found {query}"}]}
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": identifier, "result": result})


async def answered(config: McpServerConfig, transport: McpTransport) -> str:
    """Discover this server's tools and call one, whichever way it is reached."""
    async with McpClient(TransportSession(transport, config=config), config=config) as client:
        discovery = await client.discover()
        return await discovery.tools[0].invoke({"query": "leave policy"})


async def main() -> None:
    """Reach the same server as a child process, as an endpoint, and from a recording."""
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "server.py"
        script.write_text(SERVER)
        local = McpServerConfig(
            name="handbook",
            transport="stdio",
            command=(sys.executable, "-u", str(script), json.dumps(TOOLS)),
        )
        cluster = McpServerConfig(name="handbook", endpoint="http://handbook.internal/mcp")

        over_stdio = await answered(local, StdioTransport(local))
        over_http = await answered(
            cluster,
            HttpTransport(
                cluster, client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
            ),
        )
        recorded = await answered(
            cluster,
            RecordingTransport(
                {
                    "initialize": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "handbook", "version": "1.2.0"},
                    },
                    "tools/list": {"tools": TOOLS},
                    "tools/call": {"content": [{"type": "text", "text": "found leave policy"}]},
                }
            ),
        )

    print("over stdio:", over_stdio)  # noqa: T201
    print("identical over http:", over_http == over_stdio)  # noqa: T201
    print("identical from a recording:", recorded == over_stdio)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
