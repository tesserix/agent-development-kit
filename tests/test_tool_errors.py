"""A tool that failed and a tool that declined are different answers.

The failure this file exists to prevent is a run retrying a refusal until the iteration cap
fires — spending the budget to be told the same thing, and in the worst case re-attempting
an action the downstream system already declined. Every assertion here is about what the
run loop is allowed to conclude from a raised exception.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tesserix_adk.core import (
    Agent,
    ModelCapabilities,
    RetryConfig,
    Run,
    RunEventKind,
    RunState,
    TextPart,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolCallSpan, ToolRegistry, tool
from tesserix_adk.tools.errors import (
    ToolErrorMap,
    ToolFailure,
    ToolRefusal,
    permanent,
    refusal,
    transient,
)

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


class UnavailableError(Exception):
    """A library exception a tool author did not write."""


class DeclinedError(Exception):
    """A library exception meaning the downstream said no."""

    status_code = 409


class TestWhatTheTaxonomySays:
    def test_a_transient_failure_declares_itself_worth_another_attempt(self) -> None:
        failure = ToolFailure("book", "upstream_unavailable", transient=True)

        assert failure.retryable
        assert failure.code == "upstream_unavailable"
        assert failure.tool == "book"

    def test_a_permanent_failure_is_never_worth_another_attempt(self) -> None:
        assert not ToolFailure("book", "malformed_upstream_payload").retryable

    def test_a_refusal_is_an_answer_rather_than_a_fault(self) -> None:
        declined = ToolRefusal("cancel", "booking_not_cancellable", "This fare is non-refundable.")

        assert not declined.retryable
        assert declined.code == "booking_not_cancellable"
        assert declined.message == "This fare is non-refundable."

    def test_a_wait_the_upstream_asked_for_is_carried_rather_than_guessed(self) -> None:
        waiting = ToolFailure("book", "rate_limited", transient=True, retry_after=30.0)

        assert waiting.retry_after == 30.0

    def test_a_code_is_required_because_an_unnamed_failure_cannot_be_acted_on(self) -> None:
        with pytest.raises(ValueError, match="code"):
            ToolFailure("book", "")


class TestTranslatingWhatAToolAuthorDidNotWrite:
    def test_a_mapped_exception_becomes_the_failure_the_author_declared(self) -> None:
        mapped = ToolErrorMap(
            {UnavailableError: transient("upstream_unavailable", retry_after=2.0)}
        )

        failure = mapped.classify(UnavailableError("connection reset"), tool="book")

        assert isinstance(failure, ToolFailure)
        assert failure.retryable
        assert failure.retry_after == 2.0

    def test_an_unmapped_exception_is_permanent_rather_than_optimistically_retried(self) -> None:
        failure = ToolErrorMap({}).classify(UnavailableError("connection reset"), tool="book")

        assert not failure.retryable
        assert failure.code == "unmapped_failure"

    def test_the_most_specific_rule_wins_over_a_base_class(self) -> None:
        class SlowError(UnavailableError):
            pass

        mapped = ToolErrorMap(
            {UnavailableError: permanent("upstream_broken"), SlowError: transient("slow")}
        )

        assert mapped.classify(SlowError(), tool="book").code == "slow"

    def test_a_status_carrying_exception_maps_on_its_status(self) -> None:
        mapped = ToolErrorMap({}, statuses={409: refusal("booking_not_cancellable", "Cannot.")})

        assert isinstance(mapped.classify(DeclinedError(), tool="cancel"), ToolRefusal)

    def test_a_cancelled_call_is_cancellation_rather_than_a_retryable_fault(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            ToolErrorMap({}).classify(asyncio.CancelledError(), tool="book")

    def test_a_credential_in_the_original_message_never_survives_translation(self) -> None:
        original = UnavailableError("401 for token sk-live-abcdefghijklmnopqrst")

        failure = ToolErrorMap({}).classify(original, tool="book")

        assert "sk-live-abcdefghijklmnopqrst" not in str(failure)

    def test_an_already_typed_error_is_passed_through_untouched(self) -> None:
        declared = ToolFailure("book", "upstream_unavailable", transient=True)

        assert ToolErrorMap({}).classify(declared, tool="book") is declared


class TestWhatTheRunLoopRetries:
    async def test_a_transient_failure_is_retried_and_the_second_attempt_stands(self) -> None:
        run = await _run(_calling("flaky"), _answer(), fails=1)

        assert run.state is RunState.COMPLETED
        assert _events(run, RunEventKind.ATTEMPT_FAILED)

    async def test_a_permanent_failure_is_not_retried(self) -> None:
        run = await _run(_calling("broken"), _answer())

        assert not _events(run, RunEventKind.ATTEMPT_FAILED)
        assert _events(run, RunEventKind.TOOL_ERROR)

    async def test_a_refusal_reaches_the_model_once_and_is_never_retried(self) -> None:
        run = await _run(_calling("cancel", booking="AB-1"), _answer())

        assert run.state is RunState.COMPLETED
        assert not _events(run, RunEventKind.ATTEMPT_FAILED)
        assert [event.detail for event in _events(run, RunEventKind.TOOL_REFUSED)] == [
            "booking_not_cancellable: this fare is non-refundable"
        ]

    async def test_a_refusal_is_delivered_as_data_rather_than_as_an_instruction(self) -> None:
        run = await _run(_calling("cancel", booking="AB-1"), _answer())
        delivered = [message for message in run.messages if message.role == "tool"]
        said = "".join(part.text for part in delivered[0].content if isinstance(part, TextPart))

        assert said.startswith('<untrusted-data source="tool_refusal">')
        assert "booking_not_cancellable" in said

    async def test_a_wait_longer_than_the_run_has_left_fails_closed_without_sleeping(self) -> None:
        clock = FakeClock()

        run = await _run(_calling("slow"), _answer(), clock=clock)

        assert not _events(run, RunEventKind.ATTEMPT_FAILED)
        assert clock.now() == FakeClock().now()

    async def test_one_flaky_tool_cannot_spend_the_whole_iteration_budget(self) -> None:
        run = await _run(
            _calling("flaky"),
            _calling("flaky"),
            _calling("flaky"),
            _answer(),
            fails=99,
            max_tool_attempts=4,
        )

        assert len(_events(run, RunEventKind.ATTEMPT_FAILED)) <= 4


class TestWhatIsRecordedAboutAFailedCall:
    async def test_the_error_event_names_the_code_and_how_often_it_was_tried(self) -> None:
        run = await _run(_calling("flaky"), _answer(), fails=99)
        errored = _events(run, RunEventKind.TOOL_ERROR)

        assert "upstream_unavailable" in (errored[0].detail or "")
        assert "attempts" in (errored[0].detail or "")

    async def test_a_credential_in_a_failure_never_reaches_the_run_record(self) -> None:
        run = await _run(_calling("leaky"), _answer())
        errored = _events(run, RunEventKind.TOOL_ERROR)

        assert "sk-live-abcdefghijklmnopqrst" not in (errored[0].detail or "")

    async def test_a_span_tells_a_refusal_apart_from_a_failure(self) -> None:
        spans: list[ToolCallSpan] = []
        registry = _registry(spans=spans)

        with pytest.raises(ToolRefusal):
            await registry.invoke("cancel", {"booking": "AB-1"})
        with pytest.raises(ToolFailure):
            await registry.invoke("broken", {})

        assert [span.outcome for span in spans] == ["declined", "error"]
        assert [span.code for span in spans] == ["booking_not_cancellable", "upstream_broken"]


def _events(run: Run[Any], kind: RunEventKind) -> list[Any]:
    return [event for event in run.events if event.kind is kind]


def _registry(*, spans: list[ToolCallSpan] | None = None, fails: int = 0) -> ToolRegistry:
    registry = ToolRegistry(_tools(fails=fails), clock=FakeClock())
    if spans is not None:
        registry.observe(spans.append)
    return registry


def _tools(*, fails: int) -> tuple[Any, ...]:
    attempts = {"flaky": 0}
    mapped = ToolErrorMap(
        {UnavailableError: transient("upstream_unavailable")},
        statuses={409: refusal("booking_not_cancellable", "this fare is non-refundable")},
    )

    @tool
    async def flaky() -> str:
        """Fail a declared number of times, then work."""
        attempts["flaky"] += 1
        if attempts["flaky"] <= fails:
            raise mapped.classify(UnavailableError("connection reset"), tool="flaky")
        return "booked"

    @tool
    async def broken() -> str:
        """Fail in a way no retry will fix."""
        raise ToolFailure("broken", "upstream_broken")

    @tool
    async def cancel(booking: str) -> str:
        """Decline, because the downstream declined.

        Args:
            booking: What was asked about.
        """
        if not booking:
            return "nothing to cancel"
        raise mapped.classify(DeclinedError(), tool="cancel")

    @tool
    async def slow() -> str:
        """Ask for a wait longer than any run has left."""
        raise ToolFailure("slow", "rate_limited", transient=True, retry_after=3600.0)

    @tool
    async def leaky() -> str:
        """Fail with a credential in the message."""
        raise mapped.classify(
            UnavailableError("401 for token sk-live-abcdefghijklmnopqrst"), tool="leaky"
        )

    return (flaky, broken, cancel, slow, leaky)


async def _run(
    *responses: ModelResponse,
    fails: int = 0,
    clock: FakeClock | None = None,
    **overrides: object,
) -> Run[Any]:
    """A run over tools that fail, decline and recover."""
    names = ("flaky", "broken", "cancel", "slow", "leaky")
    registry = _registry(fails=fails)
    runner = AgentRunner(
        provider=ScriptedProvider(*responses, capabilities=CAPABLE),
        clock=clock or FakeClock(),
        tools=registry.view(allow=names, agent="planner"),
        **{key: value for key, value in overrides.items() if key == "max_tool_attempts"},  # type: ignore[arg-type]
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Book trips.",
        free_text=True,
        model="scripted-1",
        tools=names,
        idempotent_tools=names,
        retry=RetryConfig(max_attempts=3, base_delay_seconds=0.001),
    )
    return await runner.run(agent, "book it", tenant="acme", run_id="run_1")


def _calling(name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id=f"call_{name}", name=name, arguments=arguments),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _answer() -> ModelResponse:
    return ModelResponse(
        content="Done.",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
