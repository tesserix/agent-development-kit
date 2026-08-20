"""How an MCP server is reached, kept behind one protocol so nothing above it knows.

A server runs as a subprocess on a developer's machine and as an HTTP endpoint in the
cluster, and the agent that uses it should not be able to tell. Everything above this
module speaks `McpTransport`; moving a server between the two is a configuration change.

The two implementations are held to the same ceilings — one message size, one read
deadline, one in-flight count — because a limit that only one transport enforces is a
limit nobody can rely on.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

from tesserix_adk import __version__
from tesserix_adk.adapters.mcp import McpServerInfo
from tesserix_adk.core.errors import AdkError, ConfigurationError
from tesserix_adk.mcp import GatewayToolResult, McpToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from tesserix_adk.core.config import McpServerConfig

__all__ = [
    "HttpTransport",
    "McpTransport",
    "McpTransportError",
    "McpTransportReason",
    "RecordingTransport",
    "StdioTransport",
    "TransportSession",
    "transport_for",
]

PROTOCOL_VERSION = "2025-06-18"
"""The MCP protocol revision this client negotiates."""

_ATTEMPTS = 3
"""How many times an endpoint is tried before an outage is reported as one."""

_BACKOFF_SECONDS = 0.05
"""The first pause between attempts. Jittered, so replicas do not retry in step."""

_STDERR_LINES = 50
"""How much of a child's stderr is kept for the failure that needs it."""

_DISCARD_BYTES = 64 * 1024
"""How much of a failed stream is read at a time while it is drained to its end."""

_TERMINATE_SECONDS = 5.0
"""How long a child is given to exit politely before it is killed."""


class McpTransportReason(StrEnum):
    """Why a transport failed, in the terms a caller can act on."""

    DISCONNECTED = "disconnected"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"
    LIMIT = "limit"
    UNAVAILABLE = "unavailable"


class McpTransportError(AdkError):
    """A server could not be reached, or answered with something that was not protocol.

    The reason separates a connection that dropped from a read that never came, because
    the two call for different handling and a single "transport failed" hides which.

    Args:
        server: The server as configured here, never as the server names itself.
        reason: What went wrong, from the fixed set above.
    """

    def __init__(self, message: str, *, server: str, reason: McpTransportReason) -> None:
        self.server = server
        self.reason = reason
        super().__init__(message, details={"server": server, "reason": reason.value})

    @property
    def retryable(self) -> bool:
        """Whether trying again could plausibly help."""
        return self.reason in {McpTransportReason.UNAVAILABLE, McpTransportReason.TIMEOUT}


class McpTransport(Protocol):
    """The operations a session needs from whatever carries its messages.

    Stability: this protocol is public and changes only under the kit's deprecation
    policy, so a transport written outside the kit survives minor versions.
    """

    async def open(self) -> None:
        """Make the transport ready to carry messages. Called more than once safely."""

    async def request(
        self, method: str, params: Mapping[str, JsonValue], *, timeout_seconds: float
    ) -> dict[str, JsonValue]:
        """Send one request and return its result.

        Raises:
            McpTransportError: The server could not be reached, went quiet, exceeded a
                ceiling, or answered with something that was not a result.
        """

    async def notify(self, method: str, params: Mapping[str, JsonValue]) -> None:
        """Send one message that has no reply."""

    async def close(self) -> None:
        """Release the connection or the child process. Called once, whatever happened."""

    @property
    def healthy(self) -> bool:
        """Whether the transport is open and has not failed."""


class StdioTransport:
    """One MCP server as a child process, framed as newline-delimited JSON-RPC.

    The child gets an explicit argv and only the environment variables the declaration
    allows: a subprocess that inherits this process's environment inherits its secrets.
    It is torn down on close, on cancellation and on failure, so no orphan survives a run.

    Args:
        config: The declared server, which owns the argv, the allowlist and the ceilings.
    """

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, JsonValue]]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._errors: asyncio.Task[None] | None = None
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        self._gate = asyncio.Semaphore(config.max_in_flight)
        self._next = 0
        self._closed = False

    @property
    def healthy(self) -> bool:
        """Whether the child is running and its stream has not failed."""
        return not self._closed and self._process is not None and self._process.returncode is None

    @property
    def returncode(self) -> int | None:
        """The child's exit status, or `None` while it is still running."""
        return None if self._process is None else self._process.returncode

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        """The last lines the child wrote to stderr, for the failure that needs them."""
        return tuple(self._stderr)

    async def open(self) -> None:
        """Spawn the child, or do nothing if it is already running.

        Raises:
            McpTransportError: The command could not be started.
        """
        if self._process is not None:
            return
        if self._closed:
            raise self._failure("the transport is closed", McpTransportReason.DISCONNECTED)
        command, *arguments = self._config.command
        try:
            self._process = await asyncio.create_subprocess_exec(
                command,
                *arguments,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment(),
                limit=self._config.max_message_bytes,
            )
        except OSError as unstartable:
            raise self._failure(
                "the MCP server could not be started", McpTransportReason.UNAVAILABLE
            ) from unstartable
        self._reader = asyncio.create_task(self._read())
        self._errors = asyncio.create_task(self._drain())

    async def request(
        self, method: str, params: Mapping[str, JsonValue], *, timeout_seconds: float
    ) -> dict[str, JsonValue]:
        """Send one request down the child's stdin and await its reply.

        Raises:
            McpTransportError: The child exited, went quiet past the read deadline, wrote
                more than one message may carry, or answered with an error.
        """
        await self.open()
        async with self._gate:
            identifier = self._identifier()
            waiting: asyncio.Future[dict[str, JsonValue]] = (
                asyncio.get_running_loop().create_future()
            )
            self._pending[identifier] = waiting
            try:
                await self._write(
                    {"jsonrpc": "2.0", "id": identifier, "method": method, "params": dict(params)}
                )
                deadline = min(timeout_seconds, self._config.read_timeout_seconds)
                async with asyncio.timeout(deadline):
                    return await waiting
            except TimeoutError as quiet:
                raise self._failure(
                    "the MCP server did not answer within the read deadline",
                    McpTransportReason.TIMEOUT,
                ) from quiet
            finally:
                self._pending.pop(identifier, None)

    async def notify(self, method: str, params: Mapping[str, JsonValue]) -> None:
        """Send one message the child will not answer."""
        await self.open()
        await self._write({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    async def close(self) -> None:
        """Stop reading, fail anything outstanding, and make sure the child is gone."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.returncode is None:
                if process.stdin is not None:
                    process.stdin.close()
                process.terminate()
                try:
                    async with asyncio.timeout(_TERMINATE_SECONDS):
                        await process.wait()
                except TimeoutError:
                    process.kill()
            await process.wait()
        await self._settle()
        self._fail_pending("the MCP server connection was closed", McpTransportReason.DISCONNECTED)

    async def _settle(self) -> None:
        """Let the readers see the end of the pipes, so no pipe is left open behind them."""
        for task in (self._reader, self._errors):
            if task is None:
                continue
            try:
                async with asyncio.timeout(_TERMINATE_SECONDS):
                    await task
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()

    def _environment(self) -> dict[str, str]:
        """Only what the declaration allows, so nothing ambient reaches the child."""
        return {name: os.environ[name] for name in self._config.env_allow if name in os.environ}

    def _identifier(self) -> int:
        self._next += 1
        return self._next

    async def _write(self, message: dict[str, JsonValue]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise self._failure("the MCP server is not running", McpTransportReason.DISCONNECTED)
        process.stdin.write(json.dumps(message).encode() + b"\n")
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as gone:
            raise self._failure(
                "the MCP server closed its input", McpTransportReason.DISCONNECTED
            ) from gone

    async def _read(self) -> None:
        """Deliver replies to whoever is waiting for them until the child stops talking."""
        process = self._process
        if process is None or process.stdout is None:
            return
        while True:
            try:
                line = await process.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                self._fail_pending(
                    "the MCP server wrote a message past the size ceiling",
                    McpTransportReason.LIMIT,
                )
                await self._discard(process.stdout)
                return
            if not line:
                self._fail_pending(
                    "the MCP server closed the connection", McpTransportReason.DISCONNECTED
                )
                return
            if len(line) > self._config.max_message_bytes:
                self._fail_pending(
                    "the MCP server wrote a message past the size ceiling",
                    McpTransportReason.LIMIT,
                )
                return
            self._deliver(line)

    async def _discard(self, stream: asyncio.StreamReader) -> None:
        """Read a failed stream to its end so the pipe closes rather than lingering open."""
        while await stream.read(_DISCARD_BYTES):
            pass

    def _deliver(self, line: bytes) -> None:
        """Match one line to its waiting caller, ignoring anything that is not protocol."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict):
            return
        identifier = message.get("id")
        waiting = self._pending.get(identifier) if isinstance(identifier, int) else None
        if waiting is None or waiting.done():
            return
        if "error" in message:
            waiting.set_exception(
                self._failure("the MCP server returned an error", McpTransportReason.PROTOCOL)
            )
            return
        result = message.get("result")
        if not isinstance(result, dict):
            waiting.set_exception(
                self._failure(
                    "the MCP server returned a reply that was not a result",
                    McpTransportReason.PROTOCOL,
                )
            )
            return
        waiting.set_result(cast("dict[str, JsonValue]", result))

    async def _drain(self) -> None:
        """Keep the child's stderr from filling its pipe, and keep the last of it."""
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            try:
                line = await process.stderr.readline()
            except (ValueError, asyncio.LimitOverrunError):
                return
            if not line:
                return
            self._stderr.append(line.decode(errors="replace").rstrip())

    def _fail_pending(self, message: str, reason: McpTransportReason) -> None:
        for waiting in self._pending.values():
            if not waiting.done():
                waiting.set_exception(self._failure(message, reason))

    def _failure(self, message: str, reason: McpTransportReason) -> McpTransportError:
        return McpTransportError(message, server=self._config.name, reason=reason)


class HttpTransport:
    """One MCP server as an HTTP endpoint, reading a JSON or an SSE reply.

    A redirect, an HTML error page or any other content type is a transport failure and
    is never read as protocol: an intercepting proxy answering in place of the server is
    exactly the case where parsing whatever arrived is the wrong thing to do.

    Args:
        config: The declared server, which owns the endpoint and the ceilings.
        client: An HTTP client to use, mostly so a test can supply one. When it is not
            given the transport owns its own and closes it.
        headers: Headers sent with every message. Authority belongs to its own story;
            nothing here mints a credential.
    """

    def __init__(
        self,
        config: McpServerConfig,
        *,
        client: httpx.AsyncClient | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not config.endpoint:
            raise ConfigurationError(f"the MCP server {config.name} declares no endpoint")
        self._config = config
        self._headers = dict(headers or {})
        self._owned = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False, timeout=config.read_timeout_seconds
        )
        self._gate = asyncio.Semaphore(config.max_in_flight)
        self._next = 0
        self._closed = False

    @property
    def healthy(self) -> bool:
        """Whether the transport is still open."""
        return not self._closed

    async def open(self) -> None:
        """Nothing is held open ahead of the first message; the client pools its own."""
        if self._closed:
            raise self._failure("the transport is closed", McpTransportReason.DISCONNECTED)

    async def request(
        self, method: str, params: Mapping[str, JsonValue], *, timeout_seconds: float
    ) -> dict[str, JsonValue]:
        """Post one request and read its reply, whether it arrives as JSON or as a stream.

        Raises:
            McpTransportError: The endpoint was unreachable across its bounded attempts,
                answered with something that was not protocol, or exceeded a ceiling.
        """
        await self.open()
        self._next += 1
        message: dict[str, JsonValue] = {
            "jsonrpc": "2.0",
            "id": self._next,
            "method": method,
            "params": dict(params),
        }
        async with self._gate:
            body = await self._posted(message, timeout_seconds=timeout_seconds)
        if "error" in body:
            raise self._failure("the MCP server returned an error", McpTransportReason.PROTOCOL)
        result = body.get("result")
        if not isinstance(result, dict):
            raise self._failure(
                "the MCP server returned a reply that was not a result",
                McpTransportReason.PROTOCOL,
            )
        return cast("dict[str, JsonValue]", result)

    async def notify(self, method: str, params: Mapping[str, JsonValue]) -> None:
        """Post one message that has no reply and read nothing back."""
        await self.open()
        async with self._gate:
            message: dict[str, JsonValue] = {
                "jsonrpc": "2.0",
                "method": method,
                "params": dict(params),
            }
            await self._posted(
                message,
                timeout_seconds=self._config.read_timeout_seconds,
                answered=False,
            )

    async def close(self) -> None:
        """Close the client, where this transport is the one that opened it."""
        if self._closed:
            return
        self._closed = True
        if self._owned:
            await self._client.aclose()

    async def _posted(
        self,
        message: dict[str, JsonValue],
        *,
        timeout_seconds: float,
        answered: bool = True,
    ) -> dict[str, Any]:
        """Try the endpoint a bounded number of times, then report the outage as one."""
        last: Exception | None = None
        for attempt in range(_ATTEMPTS):
            try:
                async with asyncio.timeout(timeout_seconds):
                    return await self._once(message, answered=answered)
            except (httpx.TransportError, httpx.HTTPError) as unreachable:
                last = unreachable
            except TimeoutError as quiet:
                raise self._failure(
                    "the MCP endpoint did not answer within the read deadline",
                    McpTransportReason.TIMEOUT,
                ) from quiet
            await asyncio.sleep(_BACKOFF_SECONDS * (2**attempt) * _jitter())
        raise self._failure(
            "the MCP endpoint could not be reached", McpTransportReason.UNAVAILABLE
        ) from last

    async def _once(self, message: dict[str, JsonValue], *, answered: bool) -> dict[str, Any]:
        """Post once and read at most one message back."""
        request = self._client.build_request(
            "POST",
            self._config.endpoint,
            json=message,
            headers={
                **self._headers,
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
            },
        )
        response = await self._client.send(request, stream=True)
        try:
            self._acceptable(response)
            if not answered:
                return {}
            payload = await self._body(response)
        finally:
            await response.aclose()
        return payload

    def _acceptable(self, response: httpx.Response) -> None:
        """Refuse anything that is not this server answering in protocol."""
        if response.is_redirect:
            raise self._failure(
                "the MCP endpoint answered with a redirect", McpTransportReason.PROTOCOL
            )
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise self._failure(
                "the MCP endpoint is not serving requests", McpTransportReason.UNAVAILABLE
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise self._failure(
                "the MCP endpoint rejected the request", McpTransportReason.PROTOCOL
            )

    async def _body(self, response: httpx.Response) -> dict[str, Any]:
        """Read one reply, bounded, from either shape the endpoint may answer in."""
        media = response.headers.get("content-type", "").split(";")[0].strip()
        if media == "text/event-stream":
            return _parsed(await self._streamed(response), server=self._config.name)
        if media != "application/json":
            raise self._failure(
                "the MCP endpoint answered with something other than protocol",
                McpTransportReason.PROTOCOL,
            )
        return _parsed(await self._bounded(response), server=self._config.name)

    async def _bounded(self, response: httpx.Response) -> str:
        """Read a body up to the message ceiling, refusing rather than buffering past it."""
        read = bytearray()
        async for chunk in response.aiter_bytes():
            read += chunk
            if len(read) > self._config.max_message_bytes:
                raise self._failure(
                    "the MCP endpoint answered past the size ceiling", McpTransportReason.LIMIT
                )
        return read.decode(errors="replace")

    async def _streamed(self, response: httpx.Response) -> str:
        """Take the first data event off an SSE reply and stop reading."""
        read = 0
        data: list[str] = []
        async for line in response.aiter_lines():
            read += len(line)
            if read > self._config.max_message_bytes:
                raise self._failure(
                    "the MCP endpoint streamed past the size ceiling", McpTransportReason.LIMIT
                )
            if line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
                continue
            if not line.strip() and data:
                break
        if not data:
            raise self._failure(
                "the MCP endpoint closed the stream without answering",
                McpTransportReason.DISCONNECTED,
            )
        return "".join(data)

    def _failure(self, message: str, reason: McpTransportReason) -> McpTransportError:
        return McpTransportError(message, server=self._config.name, reason=reason)


class RecordingTransport:
    """A transport that answers from a script and remembers what it was asked.

    It exists so tool behaviour can be tested with no subprocess and no socket, which is
    the only way those tests stay fast and deterministic.

    Args:
        responses: One result per method name. A method that is not there is a protocol
            failure, so a test cannot pass by accident on a call nobody scripted.
        server: The name failures are reported under.
    """

    def __init__(self, responses: Mapping[str, JsonValue], *, server: str = "recorded") -> None:
        self.requests: list[tuple[str, dict[str, JsonValue]]] = []
        self.notifications: list[tuple[str, dict[str, JsonValue]]] = []
        self._responses = dict(responses)
        self._server = server
        self._closed = False

    @property
    def healthy(self) -> bool:
        """Whether the transport is still open."""
        return not self._closed

    async def open(self) -> None:
        """Nothing to open."""

    async def request(
        self, method: str, params: Mapping[str, JsonValue], *, timeout_seconds: float
    ) -> dict[str, JsonValue]:
        """Record the call and answer from the script.

        Raises:
            McpTransportError: No answer was scripted for that method.
        """
        del timeout_seconds
        self.requests.append((method, dict(params)))
        scripted = self._responses.get(method)
        if not isinstance(scripted, dict):
            raise McpTransportError(
                f"no reply was recorded for {method}",
                server=self._server,
                reason=McpTransportReason.PROTOCOL,
            )
        return dict(scripted)

    async def notify(self, method: str, params: Mapping[str, JsonValue]) -> None:
        """Record a message that has no reply."""
        self.notifications.append((method, dict(params)))

    async def close(self) -> None:
        """Mark the transport closed."""
        self._closed = True


class TransportSession:
    """An MCP session spoken over any transport, which is where the two shapes converge.

    Args:
        transport: What carries the messages.
        config: The declared server, for its name and its discovery ceiling.
    """

    def __init__(self, transport: McpTransport, *, config: McpServerConfig) -> None:
        self._transport = transport
        self._config = config

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session and say what the server said about itself.

        Raises:
            McpTransportError: The server could not be reached or did not answer in
                protocol.
        """
        await self._transport.open()
        result = await self._transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tesserix-adk", "version": __version__},
            },
            timeout_seconds=self._config.timeout_seconds,
        )
        await self._transport.notify("notifications/initialized", {})
        described = result.get("serverInfo")
        described = described if isinstance(described, dict) else {}
        capabilities = result.get("capabilities")
        return McpServerInfo(
            name=str(described.get("name", self._config.name)),
            version=str(described.get("version", "")),
            protocol_version=str(result.get("protocolVersion", "")),
            capabilities=tuple(sorted(capabilities)) if isinstance(capabilities, dict) else (),
        )

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """Read every page of the server's tool list, bounded by the discovery ceiling.

        Raises:
            McpTransportError: A page was not protocol, or the server paged without end.
        """
        found: list[McpToolDescriptor] = []
        seen: set[str] = set()
        cursor: str | None = None
        while True:
            params: dict[str, JsonValue] = {"cursor": cursor} if cursor else {}
            page = await self._transport.request(
                "tools/list", params, timeout_seconds=self._config.timeout_seconds
            )
            advertised = page.get("tools")
            if not isinstance(advertised, list):
                raise self._failure("the MCP server returned an invalid tool page")
            found.extend(_descriptor(entry, server=self._config.name) for entry in advertised)
            if len(found) > self._config.max_tools:
                raise McpTransportError(
                    "the MCP server advertised more tools than the ceiling allows",
                    server=self._config.name,
                    reason=McpTransportReason.LIMIT,
                )
            following = page.get("nextCursor")
            if following is None:
                return tuple(found)
            if not isinstance(following, str) or following in seen:
                raise self._failure("the MCP server repeated a discovery cursor")
            seen.add(following)
            cursor = following

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Call one tool and return its bounded result.

        Raises:
            McpTransportError: The call did not reach the server or did not come back in
                protocol. A tool that failed is a result, not an exception.
        """
        result = await self._transport.request(
            "tools/call",
            {"name": name, "arguments": dict(arguments), "_meta": dict(meta)},
            timeout_seconds=timeout_seconds,
        )
        content = result.get("content")
        structured = result.get("structuredContent")
        return GatewayToolResult(
            content=tuple(part for part in content if isinstance(part, dict))
            if isinstance(content, list)
            else (),
            structured_content=structured if isinstance(structured, dict) else None,
            is_error=bool(result.get("isError", False)),
        )

    async def close(self) -> None:
        """Release the transport."""
        await self._transport.close()

    def _failure(self, message: str) -> McpTransportError:
        return McpTransportError(
            message, server=self._config.name, reason=McpTransportReason.PROTOCOL
        )


def transport_for(config: McpServerConfig) -> McpTransport:
    """Build the transport the declaration asks for.

    A caller with its own HTTP client or headers builds `HttpTransport` directly; this is
    the path for a server that is entirely configuration.

    Args:
        config: The declared server.

    Returns:
        A stdio or an HTTP transport, which the session above cannot tell apart.

    Raises:
        ConfigurationError: An HTTP server is declared without an endpoint to reach.
    """
    if config.transport == "stdio":
        return StdioTransport(config)
    return HttpTransport(config)


def _jitter() -> float:
    """A multiplier between 1 and 2, so replicas do not retry in step."""
    return 1.0 + secrets.randbelow(1000) / 1000


def _parsed(text: str, *, server: str) -> dict[str, Any]:
    """Read one JSON-RPC message, refusing anything that is not an object."""
    try:
        message = json.loads(text)
    except json.JSONDecodeError as malformed:
        raise McpTransportError(
            "the MCP endpoint answered with something that was not JSON",
            server=server,
            reason=McpTransportReason.PROTOCOL,
        ) from malformed
    if not isinstance(message, dict):
        raise McpTransportError(
            "the MCP endpoint answered with something that was not a message",
            server=server,
            reason=McpTransportReason.PROTOCOL,
        )
    return cast("dict[str, Any]", message)


def _descriptor(entry: object, *, server: str) -> McpToolDescriptor:
    """Read one advertised tool, refusing a descriptor the protocol does not allow."""
    if not isinstance(entry, dict):
        raise McpTransportError(
            "the MCP server returned an invalid tool descriptor",
            server=server,
            reason=McpTransportReason.PROTOCOL,
        )
    schema = entry.get("inputSchema")
    output = entry.get("outputSchema")
    return McpToolDescriptor(
        name=str(entry.get("name", "")),
        description=str(entry.get("description", "")),
        input_schema=schema if isinstance(schema, dict) else {},
        output_schema=output if isinstance(output, dict) else None,
    )
