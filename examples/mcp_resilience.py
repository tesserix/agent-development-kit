"""One unreachable optional server, one healthy required one, and a breaker that opens.

Run it with `uv run --extra mcp python examples/mcp_resilience.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.adapters import (
    McpClient,
    McpServerUnavailableError,
    ResilientSession,
    assembled,
)
from tesserix_adk.adapters.mcp_transport import McpTransportError
from tesserix_adk.core.config import McpServerConfig, RetryConfig
from tesserix_adk.mcp import McpToolDescriptor
from tesserix_adk.testing import FakeClock, FaultyMcpServer, McpFault

FORECAST = McpToolDescriptor(
    name="forecast",
    description="Tomorrow's weather.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)

SEARCH = McpToolDescriptor(
    name="search",
    description="Search the handbook.",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)


def declared(name: str, *, required: bool) -> McpServerConfig:
    """One server, with the degradation policy this example is about."""
    return McpServerConfig(
        name=name,
        endpoint=f"https://{name}.internal/mcp",
        allow=("*",),
        prefix=name,
        required=required,
        connect_timeout_seconds=1,
        retry=RetryConfig(max_attempts=2, base_delay_seconds=0.1),
        breaker_failures=2,
        breaker_reset_seconds=30,
    )


def client(server: FaultyMcpServer, config: McpServerConfig, clock: FakeClock) -> McpClient:
    """The server behind its policy, adopted as ordinary kit tools."""
    return McpClient(ResilientSession(server, config=config, clock=clock), config=config)


async def main() -> None:
    """Assemble an agent whose optional server is down, then watch its breaker open."""
    clock = FakeClock()
    handbook = FaultyMcpServer((SEARCH,))
    weather = FaultyMcpServer((FORECAST,), fault=McpFault.UNREACHABLE)

    fleet = await assembled(
        (
            client(handbook, declared("handbook", required=True), clock),
            client(weather, declared("weather", required=False), clock),
        )
    )
    print("tools:", [tool.name for tool in fleet.tools])  # noqa: T201
    print("degraded:", [gap.server for gap in fleet.degraded])  # noqa: T201
    print("stated to the model as data:", "<untrusted-data" in fleet.notice())  # noqa: T201

    failing = ResilientSession(
        FaultyMcpServer((SEARCH,), fault=McpFault.UNREACHABLE),
        config=declared("handbook", required=True),
        clock=clock,
    )
    for _ in range(2):
        try:
            await failing.call_tool("search", {}, meta={}, timeout_seconds=1)
        except McpTransportError as fault:
            print("fault:", fault)  # noqa: T201
    try:
        await failing.call_tool("search", {}, meta={}, timeout_seconds=1)
    except McpServerUnavailableError as refused:
        print("refused without calling:", refused)  # noqa: T201
    print("health:", failing.health.model_dump(include={"server", "state", "outcome"}))  # noqa: T201

    required = declared("weather", required=True)
    try:
        await assembled((client(weather, required, clock),))
    except McpServerUnavailableError as absent:
        print("a required server is not optional:", absent)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
