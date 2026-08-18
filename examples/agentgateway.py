"""Adopt a pinned AgentGateway MCP tool surface without making a network call.

Run it with `uv run --extra mcp python examples/agentgateway.py`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import SecretStr

from tesserix_adk.adapters import agent_gateway_tools
from tesserix_adk.core import AgentIdentity, Principal
from tesserix_adk.mcp import (
    AgentGatewayConfig,
    AgentGatewayRoute,
    AgentGatewayRouter,
    AgentGatewayToolConfig,
    GatewayToolResult,
    McpToolDescriptor,
)
from tesserix_adk.tools import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pydantic import JsonValue

READ = "bookings:read"


@dataclass(frozen=True)
class ExampleCredential:
    """A credential shaped like one returned by a deployment's credential broker."""

    token: SecretStr
    scopes: frozenset[str]

    def headers(self) -> dict[str, str]:
        """Render authority only into the transport headers."""
        return {"Authorization": f"Bearer {self.token.get_secret_value()}"}


class ExampleCredentials:
    """Mint a demonstration credential; production uses a real CredentialSource."""

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> ExampleCredential:
        """Return only requested scopes the caller actually holds."""
        del audience, run_id, agent_version
        return ExampleCredential(
            SecretStr("example-token"), frozenset(needs) & frozenset(identity.effective)
        )


class ExampleGatewayTransport:
    """Stand in for McpStreamableHttpTransport so the example stays network-free."""

    async def list_tools(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        meta: Mapping[str, str],
        timeout_seconds: float,
        max_result_bytes: int,
        max_tools: int,
    ) -> tuple[McpToolDescriptor, ...]:
        """Return what AgentGateway would discover from the routed server."""
        del endpoint, headers, meta, timeout_seconds, max_result_bytes, max_tools
        return (
            McpToolDescriptor(
                name="get_booking",
                description="untrusted remote prose",
                input_schema={
                    "type": "object",
                    "properties": {"booking_id": {"type": "string"}},
                    "required": ["booking_id"],
                    "additionalProperties": False,
                },
            ),
        )

    async def call_tool(
        self,
        *,
        endpoint: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
        headers: Mapping[str, str],
        meta: Mapping[str, str],
        timeout_seconds: float,
        max_result_bytes: int,
    ) -> GatewayToolResult:
        """Return one bounded, vendor-neutral result."""
        del endpoint, tool, headers, meta, timeout_seconds, max_result_bytes
        return GatewayToolResult(
            structured_content={"booking_id": arguments["booking_id"], "status": "confirmed"}
        )


async def main() -> None:
    """Discover, adopt, and invoke one namespaced gateway tool."""
    identity = AgentIdentity.resolve(
        agent="desk",
        declared=(READ,),
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset({READ})),
    )
    router = AgentGatewayRouter(
        AgentGatewayConfig(
            base_url="https://agentgateway.example.test",
            routes=(
                AgentGatewayRoute(
                    server="bookings",
                    audience="https://bookings.mcp",
                    scopes=(READ,),
                    tools=(
                        AgentGatewayToolConfig(
                            name="get_booking",
                            description="Look up one booking by its identifier.",
                            scopes=(READ,),
                            requires_approval=False,
                        ),
                    ),
                ),
            ),
        ),
        credentials=ExampleCredentials(),
        transport=ExampleGatewayTransport(),
    )
    routed = await router.tools_for(identity=identity, run_id="run_1")
    adopted = agent_gateway_tools(routed)
    registry = ToolRegistry(adopted)
    try:
        result = await registry.invoke("bookings__get_booking", {"booking_id": "B-7"})
        print(result)  # noqa: T201
    finally:
        for adopted_tool in adopted:
            adopted_tool.release()


if __name__ == "__main__":
    asyncio.run(main())
