"""Resolve one agent's tool surface across two servers that both advertise `search`.

Run it with `uv run --extra mcp python examples/mcp_surface.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.adapters import (
    McpClient,
    McpServerInfo,
    McpSurfaceDriftError,
    McpToolConflictError,
    ToolSurface,
)
from tesserix_adk.core.config import McpConfig, McpServerConfig
from tesserix_adk.mcp import GatewayToolResult, McpToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue


def tool(name: str, *, description: str) -> McpToolDescriptor:
    """One advertised tool, with a schema simple enough to read."""
    return McpToolDescriptor(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


class ExampleSession:
    """An in-process stand-in for a connected server, so this runs with no network."""

    def __init__(self, name: str, tools: tuple[McpToolDescriptor, ...]) -> None:
        self.name = name
        self.tools = list(tools)

    async def initialize(self) -> McpServerInfo:
        """Report what the server says about itself."""
        return McpServerInfo(name=self.name, version="1.0.0", capabilities=("tools",))

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """Advertise whatever the server has right now, which may have changed."""
        return tuple(self.tools)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Answer, which this example never gets as far as needing."""
        del name, arguments, meta, timeout_seconds
        return GatewayToolResult(content=({"type": "text", "text": "ok"},))

    async def close(self) -> None:
        """Release the session."""


WIKI = ExampleSession(
    "wiki", (tool("search", description="Search the wiki."), tool("edit", description="Edit it."))
)
DOCS = ExampleSession("docs", (tool("search", description="Search the manuals."),))


async def resolved(config: McpConfig, session: ExampleSession, server: str) -> ToolSurface:
    """One server's contribution to the surface, under the declaration for it."""
    async with McpClient(session, config=config.server(server)) as client:
        return (await client.discover()).surface


async def main() -> None:
    """Collide two servers, prefix them apart, then hold the result to a pin."""
    unprefixed = McpConfig(
        servers=(
            McpServerConfig(name="wiki", allow=("search",)),
            McpServerConfig(name="docs", allow=("search",)),
        )
    )
    try:
        ToolSurface.merged(
            await resolved(unprefixed, WIKI, "wiki"),
            await resolved(unprefixed, DOCS, "docs"),
        )
    except McpToolConflictError as clash:
        print("collision:", clash)  # noqa: T201

    prefixed = McpConfig(
        servers=(
            McpServerConfig(name="wiki", allow=("*",), deny=("edit",), prefix="wiki"),
            McpServerConfig(name="docs", allow=("*",), prefix="docs"),
        )
    )
    surface = ToolSurface.merged(
        await resolved(prefixed, WIKI, "wiki"),
        await resolved(prefixed, DOCS, "docs"),
    )
    print("resolved surface:")  # noqa: T201
    print(surface.report(), end="")  # noqa: T201
    print("edit was never adopted:", "edit" not in surface.names())  # noqa: T201

    pin = surface.of_server("docs").pin()
    DOCS.tools.append(tool("draft", description="Draft a manual."))
    try:
        (await resolved(prefixed, DOCS, "docs")).check(pin)
    except McpSurfaceDriftError as drift:
        print("drift:", drift)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
