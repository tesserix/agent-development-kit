"""An MCP server that fails on purpose, so degradation can be tested without a network.

Every interesting MCP failure is a failure of a third party: a server that never answers,
one that answers every other time, one that answers with half of what it promised. None of
them can be provoked reliably against a real server, and all of them are what a consumer's
degradation policy has to be tested against. So the failures are declared here instead.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING

from tesserix_adk.adapters.mcp import McpServerInfo
from tesserix_adk.mcp import GatewayToolResult, McpGatewayError, McpGatewayReason

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

    from tesserix_adk.mcp import McpToolDescriptor

__all__ = ["FaultyMcpServer", "McpFault"]


class McpFault(StrEnum):
    """How a `FaultyMcpServer` misbehaves."""

    NONE = "none"
    """It works, which is the control every fault is compared against."""

    UNREACHABLE = "unreachable"
    """Nothing connects, so the handshake never completes."""

    SLOW = "slow"
    """Everything answers eventually, which for a deadline is not answering."""

    FLAPPING = "flapping"
    """It fails, then recovers, which is what makes a retry policy worth having."""

    MALFORMED = "malformed"
    """It answers with structured content its own result schema does not allow."""

    TRUNCATED = "truncated"
    """The response stops mid-frame, which is a protocol failure rather than a result."""


class FaultyMcpServer:
    """An in-process MCP server with a declared fault, usable anywhere a session is.

    Args:
        tools: What it advertises. A `MALFORMED` server needs one with an output schema,
            since answering outside a promise requires having made one.
        fault: How it misbehaves.
        delay: How long a `SLOW` server takes, in real seconds. Keep it well above the
            deadline under test and the test stays fast.
        recover_after: How many operations a `FLAPPING` server fails before it works.

    Example:
        >>> server = FaultyMcpServer(fault=McpFault.UNREACHABLE)
        >>> server.fault
        <McpFault.UNREACHABLE: 'unreachable'>
    """

    def __init__(
        self,
        tools: Sequence[McpToolDescriptor] = (),
        *,
        fault: McpFault = McpFault.NONE,
        delay: float = 0.0,
        recover_after: int = 1,
    ) -> None:
        self.tools = list(tools)
        self.fault = fault
        self.delay = delay
        self.recover_after = recover_after
        self.attempts = 0
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.closed = 0

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session, or fail in whatever way this server was told to.

        Raises:
            ConnectionError: The server is unreachable, or is flapping and has not yet
                recovered.
        """
        await self._misbehave()
        return McpServerInfo(
            name="faulty", version="0", protocol_version="2025-06-18", capabilities=("tools",)
        )

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """Advertise the declared tools, after misbehaving in the declared way.

        Raises:
            ConnectionError: As `initialize`.
        """
        await self._misbehave()
        return tuple(self.tools)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Answer a call, badly where that is the declared fault.

        Raises:
            ConnectionError: The server is unreachable, or flapping and not yet recovered.
            McpGatewayError: The response was truncated.
        """
        del arguments, timeout_seconds
        self.calls.append((name, dict(meta)))
        await self._misbehave()
        if self.fault is McpFault.TRUNCATED:
            raise McpGatewayError(
                "the response ended mid-frame", reason=McpGatewayReason.PAYLOAD, tool=name
            )
        if self.fault is McpFault.MALFORMED:
            return GatewayToolResult(structured_content={"unexpected": True})
        return GatewayToolResult(content=({"type": "text", "text": "ok"},))

    async def close(self) -> None:
        """Record that the session was given back."""
        self.closed += 1

    async def _misbehave(self) -> None:
        """The declared fault, applied to whichever operation asked."""
        self.attempts += 1
        if self.fault is McpFault.SLOW:
            await asyncio.sleep(self.delay)
        if self.fault is McpFault.UNREACHABLE:
            raise ConnectionError("connection refused")
        if self.fault is McpFault.FLAPPING and self.attempts <= self.recover_after:
            raise ConnectionError("connection reset")
