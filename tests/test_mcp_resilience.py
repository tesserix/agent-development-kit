"""One unhealthy MCP server degrading a capability rather than a run, with no network."""

from __future__ import annotations

import asyncio
from random import Random
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters.mcp import McpClient, McpProtocolError, McpServerInfo
from tesserix_adk.adapters.mcp_resilience import (
    BreakerState,
    McpFleet,
    McpServerUnavailableError,
    ResilientSession,
    assembled,
)
from tesserix_adk.adapters.mcp_transport import McpTransportError, McpTransportReason
from tesserix_adk.core.config import McpServerConfig, RetryConfig
from tesserix_adk.core.errors import (
    BudgetExceededError,
    CapabilityError,
    ToolRefusal,
    ToolTimedOutError,
)
from tesserix_adk.core.instrumentation import Instrumentation
from tesserix_adk.mcp.gateway import (
    GatewayToolResult,
    McpGatewayError,
    McpGatewayReason,
    McpToolDescriptor,
)
from tesserix_adk.runtime.cancellation import Deadline
from tesserix_adk.testing.fakes import FakeClock, FakeTracer
from tesserix_adk.testing.mcp import FaultyMcpServer, McpFault

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

_SEARCH = McpToolDescriptor(
    name="search",
    description="Search the handbook.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

_LOOKUP = McpToolDescriptor(
    name="lookup",
    description="Look one thing up.",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    output_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title", "body"],
    },
)


def _text(text: str) -> GatewayToolResult:
    return GatewayToolResult(content=({"type": "text", "text": text},))


class _Session:
    """An in-process MCP server whose faults a test declares."""

    def __init__(
        self,
        tools: Sequence[McpToolDescriptor] = (_SEARCH,),
        *,
        initialize_fails: int = 0,
        list_fails: int = 0,
        call_fails: int = 0,
        results: Mapping[str, GatewayToolResult | Exception] | None = None,
        stall: float = 0.0,
    ) -> None:
        self.tools = list(tools)
        self.initialize_fails = initialize_fails
        self.list_fails = list_fails
        self.call_fails = call_fails
        self.results = dict(results or {})
        self.stall = stall
        self.initialised = 0
        self.listed = 0
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.closed = 0

    async def initialize(self) -> McpServerInfo:
        self.initialised += 1
        await self._stalled()
        if self.initialised <= self.initialize_fails:
            raise ConnectionError("connection refused")
        return McpServerInfo(name="handbook", capabilities=("tools",))

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        self.listed += 1
        await self._stalled()
        if self.listed <= self.list_fails:
            raise ConnectionError("connection reset")
        return tuple(self.tools)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        del arguments, timeout_seconds
        self.calls.append((name, dict(meta)))
        await self._stalled()
        if len(self.calls) <= self.call_fails:
            raise ConnectionError("connection reset")
        outcome = self.results.get(name, _text("ok"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed += 1

    async def _stalled(self) -> None:
        if self.stall:
            await asyncio.sleep(self.stall)


def _config(**overrides: Any) -> McpServerConfig:
    overrides.setdefault("allow", ("*",))
    overrides.setdefault("name", "handbook")
    return McpServerConfig(**overrides)


def _resilient(session: _Session | FaultyMcpServer, **overrides: Any) -> ResilientSession:
    clock = overrides.pop("clock", FakeClock())
    return ResilientSession(
        session,
        config=_config(**overrides),
        clock=clock,
        random=Random(7),  # noqa: S311 — a seeded jitter source, not cryptography
    )


class TestDeadlines:
    """Every reach for a server is bounded, and by the run before by the server."""

    @pytest.mark.asyncio
    async def test_a_server_that_never_answers_the_handshake_fails_at_the_deadline(self) -> None:
        session = _Session(stall=0.5)
        resilient = _resilient(session, connect_timeout_seconds=0.01)

        with pytest.raises(McpServerUnavailableError) as unavailable:
            await resilient.initialize()

        assert unavailable.value.server == "handbook"
        assert "handbook" in str(unavailable.value)

    @pytest.mark.asyncio
    async def test_a_server_that_never_answers_discovery_fails_at_its_own_deadline(self) -> None:
        session = _Session(stall=0.5)
        resilient = _resilient(session, connect_timeout_seconds=5, discovery_timeout_seconds=0.01)

        await resilient.initialize()
        with pytest.raises(McpServerUnavailableError):
            await resilient.list_tools()

    @pytest.mark.asyncio
    async def test_a_call_cannot_outlive_the_run_it_belongs_to(self) -> None:
        session = _Session(stall=0.5)
        resilient = ResilientSession(
            session,
            config=_config(timeout_seconds=30),
            clock=FakeClock(),
            deadline=Deadline.in_seconds(0.01, now=0.0),
        )

        with pytest.raises(ToolTimedOutError) as timeout:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=30)

        assert timeout.value.seconds < 30

    @pytest.mark.asyncio
    async def test_a_run_with_nothing_left_is_budget_rather_than_a_server_fault(self) -> None:
        session = _Session()
        resilient = ResilientSession(
            session,
            config=_config(),
            clock=FakeClock(start=100.0),
            deadline=Deadline(at=100.0),
        )

        with pytest.raises(BudgetExceededError) as spent:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=15)

        assert spent.value.breached == "run_seconds"
        assert session.calls == []


class TestRetries:
    """Tried again where trying again is safe, and not otherwise."""

    @pytest.mark.asyncio
    async def test_a_transport_fault_during_discovery_is_tried_again(self) -> None:
        session = _Session(list_fails=1)
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        assert await resilient.list_tools() == (_SEARCH,)
        assert session.listed == 2

    @pytest.mark.asyncio
    async def test_retries_are_bounded_by_the_declared_attempts(self) -> None:
        session = _Session(list_fails=9)
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        with pytest.raises(McpServerUnavailableError):
            await resilient.list_tools()

        assert session.listed == 3

    @pytest.mark.asyncio
    async def test_a_call_that_cannot_be_repeated_safely_is_not_retried(self) -> None:
        session = _Session(call_fails=1)
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={"run-id": "r1"}, timeout_seconds=1)

        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_a_call_carrying_an_idempotency_key_is_tried_again(self) -> None:
        session = _Session(call_fails=1)
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        result = await resilient.call_tool(
            "search", {}, meta={"idempotency-key": "k1"}, timeout_seconds=1
        )

        assert not result.is_error
        assert len(session.calls) == 2

    @pytest.mark.asyncio
    async def test_a_server_that_declined_is_answered_rather_than_asked_again(self) -> None:
        declined = GatewayToolResult(is_error=True, structured_content={"refusal": {"code": "no"}})
        session = _Session(results={"search": declined})
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        result = await resilient.call_tool(
            "search", {}, meta={"idempotency-key": "k1"}, timeout_seconds=1
        )

        assert result.is_error
        assert len(session.calls) == 1
        assert resilient.state is BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_a_decision_the_server_made_reaches_the_caller_unchanged(self) -> None:
        declined = ToolRefusal("search", "not_permitted", "the server declined")
        session = _Session(results={"search": declined})
        resilient = _resilient(session, breaker_failures=1)

        with pytest.raises(ToolRefusal) as refusal:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert refusal.value.code == "not_permitted"
        assert resilient.state is BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_the_wait_between_attempts_is_jittered(self) -> None:
        waits = []
        for seed in (1, 2):
            clock = FakeClock()
            session = _Session(list_fails=1)
            resilient = ResilientSession(
                session,
                config=_config(retry=RetryConfig(max_attempts=3, base_delay_seconds=1.0)),
                clock=clock,
                random=Random(seed),  # noqa: S311 — jitter, not cryptography
            )
            await resilient.list_tools()
            waits.append(clock.slept)

        assert all(0.0 <= wait <= 1.0 for slept in waits for wait in slept)
        assert waits[0] != waits[1]


class TestTheBreaker:
    """A server that keeps failing stops being asked, and is probed rather than assumed."""

    @pytest.mark.asyncio
    async def test_it_opens_on_call_failures_not_only_on_connect(self) -> None:
        session = _Session(call_fails=9)
        resilient = _resilient(session, breaker_failures=2)

        for _ in range(2):
            with pytest.raises(McpTransportError):
                await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert resilient.state is BreakerState.OPEN
        with pytest.raises(McpServerUnavailableError) as refused:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert len(session.calls) == 2
        assert refused.value.state is BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_a_refusal_to_call_is_a_capability_error_never_a_substitution(self) -> None:
        session = _Session(call_fails=9)
        resilient = _resilient(session, breaker_failures=1)

        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
        with pytest.raises(CapabilityError) as refused:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert refused.value.capability == "mcp:handbook"

    @pytest.mark.asyncio
    async def test_an_answer_between_faults_leaves_it_closed(self) -> None:
        session = _Session(call_fails=1)
        resilient = _resilient(session, breaker_failures=2)

        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
        await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert resilient.state is BreakerState.CLOSED
        assert resilient.health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_it_probes_once_the_reset_window_has_passed(self) -> None:
        clock = FakeClock()
        session = _Session(call_fails=1)
        resilient = _resilient(session, clock=clock, breaker_failures=1, breaker_reset_seconds=30)

        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
        clock.advance(31)

        probing = resilient.state
        await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert (probing, resilient.state) == (BreakerState.HALF_OPEN, BreakerState.CLOSED)

    @pytest.mark.asyncio
    async def test_a_failed_probe_shuts_it_again(self) -> None:
        clock = FakeClock()
        session = _Session(call_fails=9)
        resilient = _resilient(session, clock=clock, breaker_failures=1, breaker_reset_seconds=30)

        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
        clock.advance(31)
        with pytest.raises(McpTransportError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert resilient.state is BreakerState.OPEN
        with pytest.raises(McpServerUnavailableError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

    @pytest.mark.asyncio
    async def test_each_session_holds_its_own_breaker(self) -> None:
        failing, healthy = _Session(call_fails=9), _Session()
        one = _resilient(failing, breaker_failures=1)
        other = _resilient(healthy, breaker_failures=1)

        with pytest.raises(McpTransportError):
            await one.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert one.state is BreakerState.OPEN
        assert other.state is BreakerState.CLOSED


class TestRequiredAndOptionalServers:
    """One server's absence is a missing capability, not a failed agent."""

    @pytest.mark.asyncio
    async def test_an_unreachable_optional_server_leaves_the_rest_working(self) -> None:
        healthy = _Session()
        unreachable = _Session(initialize_fails=9)
        fleet = await assembled(
            (
                McpClient(_resilient(healthy), config=_config(name="handbook")),
                McpClient(
                    _resilient(unreachable, name="weather"),
                    config=_config(name="weather", required=False),
                ),
            )
        )

        assert [tool.name for tool in fleet.tools] == ["search"]
        assert [degraded.server for degraded in fleet.degraded] == ["weather"]

    @pytest.mark.asyncio
    async def test_an_unreachable_required_server_fails_assembly(self) -> None:
        unreachable = _Session(initialize_fails=9)

        with pytest.raises(McpServerUnavailableError):
            await assembled((McpClient(_resilient(unreachable), config=_config()),))

    @pytest.mark.asyncio
    async def test_what_is_missing_is_stated_to_the_model_as_data(self) -> None:
        unreachable = _Session(initialize_fails=9)
        fleet = await assembled(
            (
                McpClient(
                    _resilient(unreachable, name="weather"),
                    config=_config(name="weather", required=False),
                ),
            )
        )

        notice = fleet.notice()
        assert "<untrusted-data" in notice
        assert "weather" in notice

    @pytest.mark.asyncio
    async def test_a_fleet_with_nothing_missing_says_nothing(self) -> None:
        fleet = await assembled((McpClient(_resilient(_Session()), config=_config()),))

        assert fleet.notice() == ""
        assert fleet.degraded == ()

    @pytest.mark.asyncio
    async def test_an_empty_fleet_is_an_agent_with_no_servers(self) -> None:
        fleet = McpFleet()

        assert (fleet.tools, fleet.degraded, fleet.notice()) == ((), (), "")


class TestResultsThatAreNotWhatWasPromised:
    """Nothing partially parsed is accepted, and nothing plausible is invented."""

    @pytest.mark.asyncio
    async def test_a_result_that_violates_the_tools_own_schema_is_typed(self) -> None:
        truncated = GatewayToolResult(structured_content={"title": "Rota"})
        session = _Session((_LOOKUP,), results={"lookup": truncated})
        client = McpClient(session, config=_config())

        (adapted,) = await client.tools()
        with pytest.raises(McpProtocolError) as malformed:
            await adapted.invoke({"id": "1"})

        assert malformed.value.tool == "lookup"
        assert "title" in malformed.value.payload

    @pytest.mark.asyncio
    async def test_a_result_the_schema_required_and_the_server_omitted_is_typed(self) -> None:
        session = _Session((_LOOKUP,), results={"lookup": _text("Rota")})
        client = McpClient(session, config=_config())

        (adapted,) = await client.tools()
        with pytest.raises(McpProtocolError):
            await adapted.invoke({"id": "1"})

    @pytest.mark.asyncio
    async def test_a_result_that_keeps_its_promise_is_returned(self) -> None:
        whole = GatewayToolResult(structured_content={"title": "Rota", "body": "Monday"})
        session = _Session((_LOOKUP,), results={"lookup": whole})
        client = McpClient(session, config=_config())

        (adapted,) = await client.tools()

        assert "Rota" in await adapted.invoke({"id": "1"})

    @pytest.mark.asyncio
    async def test_a_protocol_failure_carries_the_payload_for_debugging(self) -> None:
        broken = McpGatewayError("truncated frame", reason=McpGatewayReason.PAYLOAD, tool="search")
        session = _Session(results={"search": broken})
        resilient = _resilient(session)

        with pytest.raises(McpProtocolError) as protocol:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert protocol.value.server == "handbook"
        assert resilient.state is BreakerState.CLOSED


class TestWhatOnCallSees:
    """Health a person can alert on, and a span per reach for the server."""

    @pytest.mark.asyncio
    async def test_health_reports_the_outcome_the_breaker_and_the_retries(self) -> None:
        session = _Session(call_fails=1)
        resilient = _resilient(session, retry=RetryConfig(max_attempts=3))

        await resilient.call_tool("search", {}, meta={"idempotency-key": "k"}, timeout_seconds=1)
        health = resilient.health

        assert (health.server, health.state, health.outcome) == (
            "handbook",
            BreakerState.CLOSED,
            "ok",
        )
        assert health.retries == 1

    @pytest.mark.asyncio
    async def test_each_reach_for_the_server_is_a_span(self) -> None:
        tracer = FakeTracer()
        clock = FakeClock()
        instrumentation = Instrumentation(tracer, clock=clock)
        session = _Session()
        resilient = ResilientSession(
            session, config=_config(), clock=clock, instrumentation=instrumentation
        )

        with instrumentation.run("run-1"):
            await resilient.initialize()
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        recorded = [event for event in tracer.recorded if "mcp" in str(event.attributes)]
        assert recorded
        attributes = recorded[0].attributes
        assert attributes["adk.mcp.server"] == "handbook"
        assert attributes["adk.mcp.breaker"] == "closed"


class TestTheFaultInjectingServer:
    """A server a consumer can point at their own degradation tests, with no network."""

    @pytest.mark.asyncio
    async def test_an_unreachable_server_never_completes_the_handshake(self) -> None:
        resilient = _resilient(FaultyMcpServer((_SEARCH,), fault=McpFault.UNREACHABLE))

        with pytest.raises(McpServerUnavailableError):
            await resilient.initialize()

    @pytest.mark.asyncio
    async def test_a_flapping_server_recovers_where_the_policy_allows_a_second_attempt(
        self,
    ) -> None:
        server = FaultyMcpServer((_SEARCH,), fault=McpFault.FLAPPING, recover_after=1)
        resilient = _resilient(server, retry=RetryConfig(max_attempts=3))

        assert await resilient.list_tools() == (_SEARCH,)

    @pytest.mark.asyncio
    async def test_a_slow_server_fails_at_the_deadline(self) -> None:
        server = FaultyMcpServer((_SEARCH,), fault=McpFault.SLOW, delay=0.5)
        resilient = _resilient(server, connect_timeout_seconds=0.01)

        with pytest.raises(McpServerUnavailableError):
            await resilient.initialize()

    @pytest.mark.asyncio
    async def test_a_malformed_result_never_reaches_a_model(self) -> None:
        server = FaultyMcpServer((_LOOKUP,), fault=McpFault.MALFORMED)
        client = McpClient(server, config=_config())

        (adapted,) = await client.tools()
        with pytest.raises(McpProtocolError):
            await adapted.invoke({"id": "1"})

    @pytest.mark.asyncio
    async def test_a_transport_that_read_something_that_was_not_protocol_is_typed(self) -> None:
        broken = McpTransportError(
            "not a JSON-RPC frame", server="handbook", reason=McpTransportReason.PROTOCOL
        )
        session = _Session(results={"search": broken})
        resilient = _resilient(session)

        with pytest.raises(McpProtocolError) as protocol:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert protocol.value.server == resilient.server

    @pytest.mark.asyncio
    async def test_a_transport_that_could_not_reach_the_server_stays_a_transport_fault(
        self,
    ) -> None:
        dropped = McpTransportError(
            "connection dropped", server="handbook", reason=McpTransportReason.DISCONNECTED
        )
        session = _Session(results={"search": dropped})
        resilient = _resilient(session)

        with pytest.raises(McpTransportError) as fault:
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)

        assert fault.value.reason is McpTransportReason.DISCONNECTED

    @pytest.mark.asyncio
    async def test_a_server_with_no_fault_is_the_control_every_fault_is_read_against(
        self,
    ) -> None:
        server = FaultyMcpServer((_SEARCH,))
        resilient = _resilient(server)

        result = await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
        await resilient.close()

        assert not result.is_error
        assert (server.calls, server.closed) == ([("search", {})], 1)

    @pytest.mark.asyncio
    async def test_a_truncated_response_is_a_protocol_failure(self) -> None:
        server = FaultyMcpServer((_SEARCH,), fault=McpFault.TRUNCATED)
        resilient = _resilient(server)

        with pytest.raises(McpProtocolError):
            await resilient.call_tool("search", {}, meta={}, timeout_seconds=1)
