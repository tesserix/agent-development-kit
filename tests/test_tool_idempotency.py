"""A tool that books a flight books it once, however many times the runtime tries.

The failure this file exists to prevent is the second booking: the downstream committed,
the response never arrived, and the retry looked to the runtime exactly like a first
attempt. The kit derives the key, holds the record, and refuses to guess where it cannot.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tesserix_adk.core import (
    Agent,
    Claim,
    Idempotency,
    IdempotencyPolicy,
    ModelCapabilities,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
    idempotency_key,
)
from tesserix_adk.runtime import AgentRunner, MemoryIdempotencyStore, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


class Unreachable(MemoryIdempotencyStore):
    """A store that is down when the dispatcher reaches it."""

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:  # noqa: ARG002
        raise ConnectionError("idempotency store unreachable")


class _Busy(MemoryIdempotencyStore):
    """A store where somebody else always holds the key and never says how it went."""

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:  # noqa: ARG002
        return Claim(in_flight=True)


class _Amnesiac(MemoryIdempotencyStore):
    """A store that takes the reservation and then loses the answer."""

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:
        raise ConnectionError(f"lost {outcome!r} for {key} in {tenant} after {ttl_seconds}s")


class TestDeclaringWhatRepeatingACallDoes:
    def test_a_tool_says_which_of_the_three_it_is(self) -> None:
        @tool(idempotency="effectful")
        async def book(flight: str) -> str:
            """Book a seat.

            Args:
                flight: Which flight.
            """
            return f"booked {flight}"

        assert book.idempotency is not None
        assert book.idempotency.kind is Idempotency.EFFECTFUL
        book.release()

    def test_a_tool_that_says_nothing_keeps_the_behaviour_it_had(self) -> None:
        """Declaring nothing is not declaring safety, and it is not a breaking change."""

        @tool
        async def look_up(flight: str) -> str:
            """Check a schedule.

            Args:
                flight: Which flight.
            """
            return f"on time: {flight}"

        assert look_up.idempotency is None
        look_up.release()

    def test_the_key_arguments_are_named_on_the_policy(self) -> None:
        @tool(idempotency=IdempotencyPolicy(Idempotency.EFFECTFUL, key_arguments=("flight",)))
        async def book(flight: str, seat: str) -> str:
            """Book a seat.

            Args:
                flight: Which flight.
                seat: Which seat.
            """
            return f"booked {flight} {seat}"

        assert book.idempotency is not None
        assert book.idempotency.key_arguments == ("flight",)
        book.release()

    def test_key_arguments_a_tool_does_not_take_are_refused_at_import(self) -> None:
        """A key over an argument that does not exist is a key that is never derivable."""
        with pytest.raises(Exception, match="seat"):

            @tool(idempotency=IdempotencyPolicy(Idempotency.EFFECTFUL, key_arguments=("seat",)))
            async def book(flight: str) -> str:
                """Book a seat.

                Args:
                    flight: Which flight.
                """
                return f"booked {flight}"


class TestDerivingTheKey:
    def test_the_same_call_derives_the_same_key(self) -> None:
        first = idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments={"flight": "BA117", "seat": "3A"}
        )
        second = idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments={"seat": "3A", "flight": "BA117"}
        )

        assert first == second

    def test_a_different_run_is_a_different_key(self) -> None:
        """One run's decision to book is not another run's."""
        arguments = {"flight": "BA117"}

        assert idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments=arguments
        ) != idempotency_key(tenant="acme", run_id="run_2", tool="book", arguments=arguments)

    def test_a_different_tenant_is_a_different_key(self) -> None:
        arguments = {"flight": "BA117"}

        assert idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments=arguments
        ) != idempotency_key(tenant="rival", run_id="run_1", tool="book", arguments=arguments)

    def test_a_float_written_two_ways_is_one_key(self) -> None:
        assert idempotency_key(
            tenant="acme", run_id="run_1", tool="pay", arguments={"amount": 1.50}
        ) == idempotency_key(tenant="acme", run_id="run_1", tool="pay", arguments={"amount": 1.5})

    def test_the_same_text_composed_two_ways_is_one_key(self) -> None:
        """Unicode has more than one spelling for the same string, and a payload picks one."""
        assert idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments={"name": "Angélique"}
        ) == idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments={"name": "Angélique"}
        )

    def test_only_the_named_arguments_count(self) -> None:
        """A retry that renumbers a request id is the same booking."""
        assert idempotency_key(
            tenant="acme",
            run_id="run_1",
            tool="book",
            arguments={"flight": "BA117", "request_id": "a"},
            key_arguments=("flight",),
        ) == idempotency_key(
            tenant="acme",
            run_id="run_1",
            tool="book",
            arguments={"flight": "BA117", "request_id": "b"},
            key_arguments=("flight",),
        )

    def test_a_named_argument_that_is_absent_has_no_key(self) -> None:
        assert (
            idempotency_key(
                tenant="acme",
                run_id="run_1",
                tool="book",
                arguments={"seat": "3A"},
                key_arguments=("flight",),
            )
            is None
        )

    def test_a_nested_payload_is_ordered_all_the_way_down(self) -> None:
        """A key that depends on how a provider serialised a nested object is not a key."""
        assert idempotency_key(
            tenant="acme",
            run_id="run_1",
            tool="book",
            arguments={"legs": [{"to": "JFK", "from": "LHR"}], "hold": True, "seats": 2},
        ) == idempotency_key(
            tenant="acme",
            run_id="run_1",
            tool="book",
            arguments={"seats": 2, "hold": True, "legs": [{"from": "LHR", "to": "JFK"}]},
        )

    def test_a_boolean_is_not_the_number_it_would_digest_as(self) -> None:
        assert idempotency_key(
            tenant="acme", run_id="run_1", tool="book", arguments={"hold": True}
        ) != idempotency_key(tenant="acme", run_id="run_1", tool="book", arguments={"hold": 1.0})

    def test_the_key_does_not_carry_the_arguments(self) -> None:
        """A key is written to a store and read by operators; a card number must not be."""
        key = idempotency_key(
            tenant="acme", run_id="run_1", tool="pay", arguments={"card": "4111111111111111"}
        )

        assert key is not None
        assert "4111111111111111" not in key


class TestNotRunningTheSameEffectTwice:
    async def test_a_timeout_is_not_retried_into_a_second_booking(self) -> None:
        """The tool declares itself retryable; being effectful is what overrules that."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        run = await _run(store, attempts, fail_first=True)

        assert run.state is RunState.FAILED
        assert attempts == ["BA117"]
        assert RunEventKind.TOOL_INDETERMINATE in {event.kind for event in run.events}

    async def test_a_replay_after_a_restart_books_nothing_further(self) -> None:
        """The worker died holding the answer. The store is what remembers it."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        await _run(store, attempts)
        replayed = await _run(store, attempts)

        assert attempts == ["BA117"]
        assert replayed.state is RunState.COMPLETED
        assert RunEventKind.TOOL_DEDUPLICATED in {event.kind for event in replayed.events}

    async def test_the_recorded_outcome_is_what_the_agent_is_told(self) -> None:
        """A deduplicated call is answered, not skipped: the agent sees the first result."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        await _run(store, attempts)
        replayed = await _run(store, attempts)

        result = next(message for message in replayed.messages if message.role == "tool")
        assert "booked BA117" in result.content[0].text  # type: ignore[union-attr]

    async def test_two_concurrent_identical_calls_book_once(self) -> None:
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        run = await _run(store, attempts, calls=2, slow=True)

        assert run.state is RunState.COMPLETED
        assert attempts == ["BA117"]

    async def test_a_read_only_tool_is_not_deduplicated(self) -> None:
        """Two identical lookups are two lookups; caching them is a different feature."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        await _run(store, attempts, kind="read_only", calls=2)

        assert attempts == ["BA117", "BA117"]


class TestFailingClosedRatherThanBookingTwice:
    async def test_a_store_that_cannot_be_reached_stops_the_call(self) -> None:
        attempts: list[str] = []

        run = await _run(Unreachable(), attempts)

        assert run.state is RunState.FAILED
        assert attempts == []
        assert "IndeterminateOutcomeError" in (run.events[-1].detail or "")

    async def test_an_effectful_call_whose_key_cannot_be_derived_never_goes_out(self) -> None:
        """No key means no record, and no record means a retry is a coin toss."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        run = await _run(store, attempts, key_arguments=("passenger",))

        assert run.state is RunState.FAILED
        assert attempts == []
        assert "passenger" in (run.events[-1].detail or "")

    async def test_an_outcome_that_cannot_be_recorded_stops_the_run(self) -> None:
        """The seat is booked and nothing remembers it; the next attempt must not decide."""
        attempts: list[str] = []

        run = await _run(_Amnesiac(), attempts)

        assert run.state is RunState.FAILED
        assert attempts == ["BA117"]

    async def test_a_caller_that_never_finishes_does_not_release_the_key(self) -> None:
        """Waiting has an end; what it ends in is 'unknown', never 'go ahead and run it'."""
        attempts: list[str] = []

        run = await _run(_Busy(), attempts)

        assert run.state is RunState.FAILED
        assert attempts == []

    async def test_a_call_the_validator_refuses_leaves_the_key_free(self) -> None:
        """Nothing ran, so holding the key would strand every later attempt at the same call."""
        store = MemoryIdempotencyStore()
        attempts: list[str] = []

        refused = {"flight": "BA117", "seat": "3A"}

        await _run(store, attempts, arguments=refused)

        assert attempts == []
        assert (await store.begin(_key(**refused), tenant="acme", ttl_seconds=_TTL)).in_flight is (
            False
        )

    async def test_a_record_that_expired_is_unknown_rather_than_a_success(self) -> None:
        clock = FakeClock()
        store = MemoryIdempotencyStore(clock)
        attempts: list[str] = []

        await _run(store, attempts, clock=clock)
        clock.advance(_TTL * 4)

        reserved = await store.begin(_key(), tenant="acme", ttl_seconds=_TTL)
        assert reserved.outcome is None


class TestWhatTheStoreHolds:
    async def test_a_record_belongs_to_one_tenant(self) -> None:
        store = MemoryIdempotencyStore()
        await store.record("k", tenant="acme", outcome="booked", ttl_seconds=_TTL)

        reserved = await store.begin("k", tenant="rival", ttl_seconds=_TTL)

        assert reserved.outcome is None

    async def test_erasure_removes_one_tenant_and_leaves_the_other(self) -> None:
        store = MemoryIdempotencyStore()
        await store.record("k", tenant="acme", outcome="booked", ttl_seconds=_TTL)
        await store.record("k", tenant="rival", outcome="booked", ttl_seconds=_TTL)

        assert await store.forget(tenant="acme") == 1
        assert (await store.begin("k", tenant="acme", ttl_seconds=_TTL)).outcome is None
        assert (await store.begin("k", tenant="rival", ttl_seconds=_TTL)).outcome == "booked"

    async def test_an_abandoned_reservation_lets_the_next_attempt_run(self) -> None:
        """A call that failed left nothing behind; holding its key would strand the tool."""
        store = MemoryIdempotencyStore()
        await store.begin("k", tenant="acme", ttl_seconds=_TTL)

        await store.abandon("k", tenant="acme")

        assert (await store.begin("k", tenant="acme", ttl_seconds=_TTL)).in_flight is False


_TTL = 900.0


def _key(**arguments: Any) -> str:
    """The key the helper's run derives, for a test that outlives the run."""
    derived = idempotency_key(
        tenant="acme",
        run_id="run_1",
        tool="book",
        arguments=arguments or {"flight": "BA117"},
    )
    assert derived is not None
    return derived


async def _run(
    store: Any,
    attempts: list[str],
    *,
    fail_first: bool = False,
    calls: int = 1,
    slow: bool = False,
    kind: str = "effectful",
    key_arguments: tuple[str, ...] = (),
    clock: FakeClock | None = None,
    arguments: dict[str, Any] | None = None,
) -> Run[Any]:
    """A run over a booking tool, returning the run and recording every body entry."""
    failed: list[bool] = []
    policy = IdempotencyPolicy(Idempotency(kind), key_arguments=key_arguments)

    @tool(idempotency=policy, timeout=30.0)
    async def book(flight: str, passenger: str | None = None) -> str:
        """Book a seat.

        Args:
            flight: Which flight.
            passenger: Who is flying, where the caller says.
        """
        if slow:
            await asyncio.sleep(0)
        attempts.append(flight)
        if fail_first and not failed:
            failed.append(True)
            raise ConnectionError("the response never arrived")
        return f"booked {flight}" if passenger is None else f"booked {flight} for {passenger}"

    ticking = clock or FakeClock()
    registry = ToolRegistry((book,), clock=ticking)
    runner = AgentRunner(
        provider=ScriptedProvider(
            _calling(calls, arguments or {"flight": "BA117"}), _answering(), capabilities=CAPABLE
        ),
        clock=ticking,
        tools=registry.view(allow=("book",), agent="planner"),
        idempotency=store,
        idempotency_ttl_seconds=_TTL,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Book it.",
        free_text=True,
        model="scripted-1",
        tools=("book",),
        idempotent_tools=("book",),
    )
    try:
        return await runner.run(agent, "book BA117", tenant="acme", run_id="run_1")
    finally:
        book.release()


def _calling(calls: int, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=tuple(
            ToolCall(id=f"call_{index}", name="book", arguments=arguments) for index in range(calls)
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _answering() -> ModelResponse:
    return ModelResponse(content="Booked.", usage=Usage(input_tokens=1, output_tokens=1))
