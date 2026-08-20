"""What happens to an agent when one of its MCP servers is slow, absent or wrong.

An MCP server is a third party in the hot path of a run, so its bad day is the agent's bad
day unless the policy is declared: startup blocks on an unreachable server, a hanging call
spends the whole run, and a truncated answer arrives downstream as a confusing error rather
than as the protocol failure it is.

The policy here is per server and ordinary configuration. Every reach for a server is
bounded — the handshake, the tool list, and the call, which is additionally bounded by
whatever is left of the run. A transport fault is worth another attempt after a jittered
wait; a refusal, a validation failure and an effectful call nobody gave an idempotency key
are not. Consecutive faults shut a breaker, so a server that is down stops costing model
turns, and it is reopened by one probe rather than by assumption.

The result is that a server declared `optional` degrades a capability instead of a run: its
tools are absent, the absence is stated to the model inside the untrusted-data envelope,
and nothing is guessed in its place. A `required` server that cannot be reached fails
assembly, typed and named.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from random import Random
from typing import TYPE_CHECKING

from tesserix_adk.adapters.mcp import McpProtocolError
from tesserix_adk.adapters.mcp_transport import McpTransportError, McpTransportReason
from tesserix_adk.core import (
    AdkError,
    AdkModel,
    BudgetExceededError,
    CapabilityError,
    ContentSource,
    Origin,
    SpanKind,
    ToolTimedOutError,
    sealed,
)
from tesserix_adk.mcp import McpGatewayError
from tesserix_adk.runtime import RetryPlan, SystemClock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
    from contextlib import AbstractContextManager

    from pydantic import JsonValue

    from tesserix_adk.adapters.mcp import McpClient, McpServerInfo, McpSession
    from tesserix_adk.core.config import McpServerConfig
    from tesserix_adk.core.instrumentation import Instrumentation, Span
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.mcp import GatewayToolResult, McpToolDescriptor
    from tesserix_adk.runtime.cancellation import Deadline
    from tesserix_adk.tools import Tool

__all__ = [
    "BreakerState",
    "Degraded",
    "McpFleet",
    "McpServerUnavailableError",
    "ResilientSession",
    "ServerBreaker",
    "ServerHealth",
    "assembled",
]

_IDEMPOTENCY_KEY = "idempotency-key"
_REACHING = frozenset({"connect", "discover"})
_NOT_PROTOCOL = frozenset(
    {McpTransportReason.DISCONNECTED, McpTransportReason.TIMEOUT, McpTransportReason.UNAVAILABLE}
)


class BreakerState(StrEnum):
    """Whether this process is currently willing to reach for a server."""

    CLOSED = "closed"
    """Calls go through, which is a server nobody has reason to doubt."""

    OPEN = "open"
    """Calls are refused without being made, because the last several failed."""

    HALF_OPEN = "half_open"
    """One probe is let through, so recovery is measured rather than assumed."""


class McpServerUnavailableError(CapabilityError):
    """A capability an MCP server was to provide is not available, and why it is not.

    Raised where a server is unreachable after its attempts are spent and where its breaker
    is open. A capability error rather than a tool failure: the answer is that the tool
    cannot be called at all, and nothing is silently put in its place.

    Args:
        server: The server as configured here.
        state: What its breaker was doing when the call was refused.
    """

    def __init__(self, message: str, *, server: str, state: BreakerState) -> None:
        self.server = server
        self.state = state
        super().__init__(
            message,
            capability=f"mcp:{server}",
            details={"server": server, "breaker": state.value},
        )


class ServerHealth(AdkModel):
    """What one server has just done, in the terms an alert is written in.

    Args:
        server: The server as configured here.
        state: What its breaker is doing.
        outcome: How the last operation ended — `ok`, `fault`, `timeout`, `protocol` or
            `refused`, which is the class an alert groups on.
        operation: What that operation was.
        consecutive_failures: Faults since the last answer.
        retries: Extra attempts the last operation cost.
        latency_seconds: How long it took, by the injected clock.
    """

    server: str
    state: BreakerState = BreakerState.CLOSED
    outcome: str = ""
    operation: str = ""
    consecutive_failures: int = 0
    retries: int = 0
    latency_seconds: float = 0.0


class Degraded(AdkModel):
    """One capability an agent was assembled without, and why it is not there.

    Args:
        server: The server as configured here.
        reason: Why it is absent, in the kit's own words rather than the server's.
    """

    server: str
    reason: str


class ServerBreaker:
    """One server's breaker: shut by consecutive faults, reopened by a probe.

    Args:
        failures: How many faults in a row shut it.
        reset_seconds: How long it stays shut before a probe is let through.
        clock: What measures the window, injected so a test never sleeps.

    Example:
        >>> from tesserix_adk.testing.fakes import FakeClock
        >>> breaker = ServerBreaker(failures=1, reset_seconds=30, clock=FakeClock())
        >>> breaker.failed()
        >>> breaker.state
        <BreakerState.OPEN: 'open'>
    """

    __slots__ = ("_clock", "_failures", "_opened_at", "_reset_seconds", "_since")

    def __init__(self, *, failures: int, reset_seconds: float, clock: Clock) -> None:
        self._failures = failures
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._since = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        """What the breaker is doing now, which the reset window may have just changed."""
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock.now() - self._opened_at >= self._reset_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def consecutive_failures(self) -> int:
        """Faults since the last answer, which is the number that shuts it."""
        return self._since

    def allows(self) -> bool:
        """Whether a call may be made at all. A half-open breaker allows the probe."""
        return self.state is not BreakerState.OPEN

    def succeeded(self) -> None:
        """Record an answer, which closes the breaker and forgets the run of faults."""
        self._since = 0
        self._opened_at = None

    def failed(self) -> None:
        """Record a fault, shutting the breaker where enough have come in a row."""
        self._since += 1
        if self._since >= self._failures:
            self._opened_at = self._clock.now()


class ResilientSession:
    """One MCP session with its degradation policy applied, still an `McpSession`.

    Wraps a transport session so the client above it is unchanged: deadlines, bounded
    retries and the breaker apply to every operation, and what comes back is either the
    server's own answer or a typed error saying which of those stopped it.

    Args:
        session: The session being protected.
        config: The declared server, whose deadlines, retry policy and breaker thresholds
            are the policy.
        clock: What measures waits and windows. Injected, so a test never sleeps.
        random: The jitter source for backoff. Two replicas retrying a failing server on
            their own schedules is what keeps a recovery from being stampeded.
        deadline: The run's deadline, where the caller has one. A call is bounded by
            whatever is left of it, and a run with nothing left is reported as budget
            rather than as this server's fault.
        instrumentation: Where spans go. None records nowhere, which is tracing off.
    """

    __slots__ = (
        "_breaker",
        "_clock",
        "_config",
        "_deadline",
        "_health",
        "_instrumentation",
        "_retry",
        "_session",
    )

    def __init__(
        self,
        session: McpSession,
        *,
        config: McpServerConfig,
        clock: Clock | None = None,
        random: Random | None = None,
        deadline: Deadline | None = None,
        instrumentation: Instrumentation | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._clock: Clock = clock or SystemClock()
        self._deadline = deadline
        self._instrumentation = instrumentation
        self._retry = RetryPlan(config.retry, random=random or Random())  # noqa: S311 — jitter
        self._breaker = ServerBreaker(
            failures=config.breaker_failures,
            reset_seconds=config.breaker_reset_seconds,
            clock=self._clock,
        )
        self._health = ServerHealth(server=config.name)

    @property
    def server(self) -> str:
        """What this server is called here."""
        return self._config.name

    @property
    def state(self) -> BreakerState:
        """Whether this process is currently willing to reach for the server."""
        return self._breaker.state

    @property
    def health(self) -> ServerHealth:
        """The last operation's outcome and the breaker's state, for logs and alerts."""
        return self._health.model_copy(
            update={
                "state": self._breaker.state,
                "consecutive_failures": self._breaker.consecutive_failures,
            }
        )

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session inside the connect deadline.

        Raises:
            McpServerUnavailableError: The server could not be reached inside its deadline,
                or its breaker is open.
        """
        return await self._attempt(
            "connect",
            self._session.initialize,
            ceiling=self._config.connect_timeout_seconds,
            repeatable=True,
        )

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """Read the server's tool list inside the discovery deadline.

        Raises:
            McpServerUnavailableError: The tool list could not be read inside its deadline,
                or the breaker is open.
        """
        return await self._attempt(
            "discover",
            self._session.list_tools,
            ceiling=self._config.discovery_timeout_seconds,
            repeatable=True,
        )

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Call one tool, bounded by its own ceiling and by what is left of the run.

        Repeated only where `meta` carries an idempotency key: without one, nobody can say
        whether a second attempt repeats a side effect.

        Raises:
            BudgetExceededError: The run had no time left to spend on this call.
            ToolTimedOutError: The server did not answer inside the bound.
            McpTransportError: It could not be reached, with the attempts spent.
            McpProtocolError: It answered with something that is not an answer.
            McpServerUnavailableError: Its breaker is open, so the call was not made.
        """
        timeout = self._bounded(timeout_seconds, tool=name)
        return await self._attempt(
            name,
            lambda: self._session.call_tool(name, arguments, meta=meta, timeout_seconds=timeout),
            ceiling=timeout,
            repeatable=bool(meta.get(_IDEMPOTENCY_KEY)),
        )

    async def close(self) -> None:
        """Release the wrapped session."""
        await self._session.close()

    def _bounded(self, timeout_seconds: float, *, tool: str) -> float:
        """The call's ceiling, narrowed to what the run has left to spend on it."""
        if self._deadline is None:
            return timeout_seconds
        remaining = self._deadline.remaining(self._clock.now())
        if remaining <= 0:
            raise BudgetExceededError(
                f"the run had no time left to call {tool} on {self._config.name}",
                breached="run_seconds",
                remaining=Decimal(0),
                details={"server": self._config.name, "tool": tool},
            )
        return min(timeout_seconds, remaining)

    async def _attempt[T](
        self,
        operation: str,
        work: Callable[[], Awaitable[T]],
        *,
        ceiling: float,
        repeatable: bool,
    ) -> T:
        """One operation, with the breaker, the deadline and the retry policy applied."""
        self._guard(operation)
        started = self._clock.now()
        attempt = 1
        while True:
            try:
                with self._span(operation, attempt):
                    result = await self._inside(operation, work, ceiling=ceiling)
            except AdkError as failure:
                if _is_fault(failure):
                    self._breaker.failed()
                delay = self._delay(failure, attempt, repeatable=repeatable)
                if delay is None:
                    self._record(operation, _outcome(failure), started, attempt)
                    raise self._final(operation, failure) from failure
                await self._clock.sleep(delay)
                attempt += 1
                continue
            self._breaker.succeeded()
            self._record(operation, "ok", started, attempt)
            return result

    def _guard(self, operation: str) -> None:
        """Refuse the operation outright where the breaker is open."""
        if self._breaker.allows():
            return
        self._record(operation, "refused", self._clock.now(), 1)
        raise McpServerUnavailableError(
            f"{self._config.name} is not being called: its breaker is open after "
            f"{self._breaker.consecutive_failures} consecutive failures",
            server=self._config.name,
            state=BreakerState.OPEN,
        )

    async def _inside[T](
        self, operation: str, work: Callable[[], Awaitable[T]], *, ceiling: float
    ) -> T:
        """The wrapped call, with its failures put into the kit's own taxonomy."""
        try:
            async with asyncio.timeout(ceiling):
                return await work()
        except TimeoutError as overran:
            raise self._overran(operation, ceiling) from overran
        except McpGatewayError as protocol:
            raise self._malformed(operation, protocol) from protocol
        except McpTransportError as transport:
            if transport.reason in _NOT_PROTOCOL:
                raise
            raise self._malformed(operation, transport) from transport
        except AdkError:
            raise
        except Exception as unreachable:
            raise McpTransportError(
                f"{self._config.name} could not be reached for {operation}",
                server=self._config.name,
                reason=McpTransportReason.UNAVAILABLE,
            ) from unreachable

    def _malformed(self, operation: str, failure: Exception) -> McpProtocolError:
        """A server that answered, but not with an answer."""
        return McpProtocolError(
            f"{self._config.name} answered {operation} with something that is not an answer",
            server=self._config.name,
            tool=operation,
            payload=str(failure),
        )

    def _overran(self, operation: str, timeout: float) -> AdkError:
        """A deadline that passed, named as the operation that passed it."""
        if operation in _REACHING:
            return McpTransportError(
                f"{self._config.name} did not answer {operation} inside {timeout:g}s",
                server=self._config.name,
                reason=McpTransportReason.TIMEOUT,
            )
        return ToolTimedOutError(operation, timeout)

    def _delay(self, failure: AdkError, attempt: int, *, repeatable: bool) -> float | None:
        """How long to wait before trying again, or None where there is not to be one."""
        if not repeatable or not self._retry.retryable(failure):
            return None
        return self._retry.delay_for(attempt)

    def _final(self, operation: str, failure: AdkError) -> AdkError:
        """The error the caller sees once the attempts are spent.

        Reaching a server at all is this session's job, so a spent connect or discovery is
        reported as the capability being unavailable. A spent call is the call's own
        failure, and keeps the type the tool taxonomy above it reads.
        """
        if operation in _REACHING and isinstance(failure, McpTransportError):
            return McpServerUnavailableError(
                str(failure),
                server=self._config.name,
                state=self._breaker.state,
            )
        return failure

    def _record(self, operation: str, outcome: str, started: float, attempt: int) -> None:
        """What the last operation did, kept where an alert can read it."""
        self._health = ServerHealth(
            server=self._config.name,
            state=self._breaker.state,
            outcome=outcome,
            operation=operation,
            consecutive_failures=self._breaker.consecutive_failures,
            retries=attempt - 1,
            latency_seconds=self._clock.now() - started,
        )

    def _span(self, operation: str, attempt: int) -> AbstractContextManager[Span | None]:
        """One span per reach for the server, or nothing where tracing is off."""
        if self._instrumentation is None:
            return contextlib.nullcontext()
        return self._instrumentation.step(
            SpanKind.TOOL,
            f"mcp.{self._config.name}.{operation}",
            attempt=attempt,
            **{
                "adk.mcp.server": self._config.name,
                "adk.mcp.operation": operation,
                "adk.mcp.breaker": self._breaker.state.value,
            },
        )


@dataclass(frozen=True, slots=True)
class McpFleet:
    """Every MCP server an agent was assembled with, and every one it was assembled without.

    Args:
        tools: The tools that resolved, across every server that answered.
        degraded: The servers that did not, each with the reason it is absent.
    """

    tools: tuple[Tool[..., str], ...] = ()
    degraded: tuple[Degraded, ...] = ()

    def notice(self) -> str:
        """What is missing, as data a model may read but not be instructed by.

        Empty where nothing is missing: an agent with all its capabilities should not be
        told about capabilities at all.
        """
        if not self.degraded:
            return ""
        lines = "\n".join(f"{gap.server}: {gap.reason}" for gap in self.degraded)
        return sealed(
            f"These capabilities are unavailable for this run:\n{lines}",
            source=ContentSource(origin=Origin.MCP_RESULT, name="mcp/unavailable"),
        )


async def assembled(clients: Sequence[McpClient], *, known: Collection[str] = ()) -> McpFleet:
    """Adopt every server's tools, tolerating the absence of the optional ones.

    Args:
        clients: The declared servers, each already wrapped in whatever session policy.
        known: Tool names the process already defines itself, so a remote tool answering to
            one of them is a conflict rather than a shadow.

    Returns:
        The tools that resolved and the servers that did not, in declaration order.

    Raises:
        McpServerUnavailableError: A required server could not be reached.
    """
    tools: list[Tool[..., str]] = []
    degraded: list[Degraded] = []
    taken = list(known)
    for client in clients:
        try:
            discovery = await client.discover(known=taken)
        except (McpServerUnavailableError, McpTransportError) as absent:
            if client.required:
                raise
            degraded.append(Degraded(server=client.server, reason=str(absent)))
            continue
        tools.extend(discovery.tools)
        taken.extend(tool.name for tool in discovery.tools)
    return McpFleet(tools=tuple(tools), degraded=tuple(degraded))


def _is_fault(failure: AdkError) -> bool:
    """Whether a failure says anything about the server's health.

    A decision the server made — a refusal, a validation failure — is an answer, and
    answers must not open a breaker. Only the failures of reaching it count.
    """
    return isinstance(failure, McpTransportError | McpProtocolError | ToolTimedOutError)


def _outcome(failure: AdkError) -> str:
    """The class an alert groups a failure on, rather than its message."""
    if isinstance(failure, ToolTimedOutError):
        return "timeout"
    if isinstance(failure, McpProtocolError):
        return "protocol"
    return "fault"
