"""Two of six kit tools published over MCP, with the same guarantees the local path gives.

Run it with `uv run --extra mcp python examples/mcp_server.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.adapters import McpExportError, McpServer
from tesserix_adk.core.errors import McpAuthError
from tesserix_adk.core.hooks import ApprovalPolicy
from tesserix_adk.mcp import META_PREFIX
from tesserix_adk.tools import ToolContext, ToolRegistry, tool

CALLER = {f"{META_PREFIX}/tenant": "acme", f"{META_PREFIX}/run": "run-1"}


@tool(name="fare_for")
def fare_for(leg: str) -> dict[str, Any]:
    """Price one leg."""
    return {"leg": leg, "eur": 40}


@tool(name="whose")
def whose(context: ToolContext) -> dict[str, Any]:
    """Report the tenant the call ran under."""
    return {"tenant": context.tenant}


@tool(name="refund", requires_approval=ApprovalPolicy(required=True, reason="money leaves"))
def refund(order: str, amount: int) -> dict[str, Any]:
    """Refund an order, once a human has said so."""
    return {"order": order, "amount": amount}


@tool(name="internal_ledger")
def internal_ledger() -> str:
    """Registered for the agent's own use and never published."""
    return "not for remote callers"


async def main() -> None:
    """Publish three of four tools and watch the fourth stay invisible."""
    registry = ToolRegistry((fare_for, whose, refund, internal_ledger))
    view = registry.view(allow=registry.names, agent="planner")
    server = McpServer(view, exports=("fare_for", "whose", "refund"), name="handbook")
    session = server.connect()

    info = await session.initialize()
    print("negotiated:", info.name, info.protocol_version)  # noqa: T201
    print("published:", [each.name for each in await session.list_tools()])  # noqa: T201

    priced = await session.call_tool("fare_for", {"leg": "Osaka"}, meta=CALLER, timeout_seconds=5)
    print("answered:", priced.structured_content)  # noqa: T201

    scoped = await session.call_tool("whose", {}, meta=CALLER, timeout_seconds=5)
    print("ran under:", scoped.structured_content)  # noqa: T201

    held = await session.call_tool(
        "refund", {"order": "A-1", "amount": 50}, meta=CALLER, timeout_seconds=5
    )
    print("approval gate:", (held.structured_content or {})["refusal"])  # noqa: T201

    try:
        await session.call_tool("internal_ledger", {}, meta=CALLER, timeout_seconds=5)
    except McpExportError as refused:
        print("not published:", refused)  # noqa: T201

    try:
        await session.call_tool("fare_for", {"leg": "Osaka"}, meta={}, timeout_seconds=5)
    except McpAuthError as unscoped:
        print("no tenant, no call:", unscoped)  # noqa: T201

    try:
        await session.call_tool("fare_for", {"leg": 1}, meta=CALLER, timeout_seconds=5)
    except McpExportError as invalid:
        print("outside the schema:", invalid.reason)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
