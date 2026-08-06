"""A run, the states it may move between, and the context it carries.

A run is checkpointed mid-flight and rehydrated by a different process, so it holds data
and never a collaborator. Its state machine is explicit: every terminal state says why
the run ended, and a transition nobody declared legal is refused rather than logged.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from tesserix_adk.core import (
    Message,
    Run,
    RunContext,
    RunEvent,
    RunEventKind,
    RunState,
    TenantContext,
    TextPart,
    Usage,
    legal_transitions,
)

_FIELDS: dict[str, object] = {
    "id": "run_1",
    "tenant": "acme",
    "agent_name": "planner",
    "agent_version": "1.0.0",
    "model": "claude-sonnet-5",
}


class TripPlan(BaseModel):
    """A trip the model proposes.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


def run(**overrides: object) -> Run:
    return Run(**{**_FIELDS, **overrides})  # type: ignore[arg-type]


class TestRunState:
    def test_every_terminal_state_says_why_the_run_ended(self) -> None:
        """ "Failed" for a budget ceiling and for a provider outage is two bugs in one bucket."""
        terminal = {state for state in RunState if state.is_terminal}
        assert terminal == {
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.BUDGET_EXHAUSTED,
            RunState.MAX_ITERATIONS_EXCEEDED,
            RunState.LOOP_LIMIT_EXCEEDED,
        }

    def test_a_terminal_state_goes_nowhere(self) -> None:
        assert legal_transitions(RunState.COMPLETED) == frozenset()

    def test_a_pending_run_can_only_start_or_be_cancelled(self) -> None:
        """A run that never started cannot have exhausted a budget."""
        assert legal_transitions(RunState.PENDING) == frozenset(
            {RunState.RUNNING, RunState.CANCELLED}
        )

    def test_a_running_run_can_reach_every_terminal_state(self) -> None:
        assert legal_transitions(RunState.RUNNING) == frozenset(
            state for state in RunState if state.is_terminal
        )


class TestTransitions:
    def test_a_legal_transition_returns_a_new_run(self) -> None:
        started = run().transition_to(RunState.RUNNING)
        assert started.state is RunState.RUNNING

    def test_the_original_run_is_unchanged(self) -> None:
        """A frozen run means a checkpoint taken before a step still describes that step."""
        pending = run()
        pending.transition_to(RunState.RUNNING)
        assert pending.state is RunState.PENDING

    def test_an_undeclared_transition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pending"):
            run().transition_to(RunState.COMPLETED)

    def test_the_refusal_names_what_was_allowed(self) -> None:
        """The legal set is the fix; without it the caller guesses."""
        with pytest.raises(ValueError, match="running"):
            run().transition_to(RunState.COMPLETED)

    def test_a_terminal_run_cannot_be_restarted(self) -> None:
        """Reopening a finished run makes its audit trail a lie."""
        finished = run().transition_to(RunState.RUNNING).transition_to(RunState.COMPLETED)
        with pytest.raises(ValueError, match="terminal"):
            finished.transition_to(RunState.RUNNING)

    def test_a_transition_to_the_same_state_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already"):
            run().transition_to(RunState.PENDING)


class TestTheRunItself:
    def test_a_run_starts_pending_with_nothing_spent(self) -> None:
        assert run().state is RunState.PENDING
        assert run().usage == Usage(input_tokens=0, output_tokens=0)

    def test_a_run_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            run().state = RunState.RUNNING

    def test_a_run_round_trips_for_checkpointing(self) -> None:
        """Serialised by one process mid-flight, rehydrated by another."""
        mid = run(
            messages=[Message(role="user", content=[TextPart(text="plan a trip")])]
        ).transition_to(RunState.RUNNING)
        assert Run.model_validate_json(mid.model_dump_json()) == mid

    def test_a_run_cannot_hold_a_live_collaborator(self) -> None:
        """A socket does not survive a checkpoint, so no field may accept one."""
        with pytest.raises(ValidationError):
            run(provider=object())

    def test_recording_usage_totals_it(self) -> None:
        spent = run().record(Usage(input_tokens=10, output_tokens=2))
        assert spent.usage == Usage(input_tokens=10, output_tokens=2)

    def test_recording_usage_twice_accumulates(self) -> None:
        once = run().record(Usage(input_tokens=10, output_tokens=2))
        assert once.record(Usage(input_tokens=5, output_tokens=1)).input_tokens_spent == 15

    def test_a_run_without_a_tenant_is_refused(self) -> None:
        """Cost attribution and isolation both key off it; there is no default tenant."""
        with pytest.raises(ValidationError, match="tenant"):
            run(tenant="")


class TestEvents:
    def test_a_run_starts_with_no_events(self) -> None:
        assert run().events == []

    def test_recording_an_event_appends_it(self) -> None:
        recorded = run().record_event(RunEvent(kind=RunEventKind.MODEL_CALL, name="sonnet"))
        assert [event.kind for event in recorded.events] == [RunEventKind.MODEL_CALL]

    def test_events_keep_the_order_they_happened_in(self) -> None:
        """The event list is the record of what happened; out of order it is fiction."""
        recorded = (
            run()
            .record_event(RunEvent(kind=RunEventKind.PROMPT_ASSEMBLED))
            .record_event(RunEvent(kind=RunEventKind.MODEL_CALL))
            .record_event(RunEvent(kind=RunEventKind.TOOL_CALL, name="search"))
        )
        assert [event.kind for event in recorded.events] == [
            RunEventKind.PROMPT_ASSEMBLED,
            RunEventKind.MODEL_CALL,
            RunEventKind.TOOL_CALL,
        ]

    def test_recording_an_event_leaves_the_original_run_alone(self) -> None:
        original = run()
        original.record_event(RunEvent(kind=RunEventKind.MODEL_CALL))
        assert original.events == []

    def test_an_event_is_frozen(self) -> None:
        event = RunEvent(kind=RunEventKind.MODEL_CALL)
        with pytest.raises(ValidationError):
            event.kind = RunEventKind.TOOL_CALL

    def test_an_event_carries_what_it_cost_where_there_is_a_cost(self) -> None:
        """Cost attribution reads the events, not a per-project wiring of its own."""
        event = RunEvent(
            kind=RunEventKind.MODEL_RESPONSE, usage=Usage(input_tokens=9, output_tokens=1)
        )
        assert event.usage is not None
        assert event.usage.input_tokens == 9

    def test_a_run_with_events_round_trips(self) -> None:
        recorded = run().record_event(
            RunEvent(kind=RunEventKind.TOOL_RESULT, name="search", detail="3 rows")
        )
        assert Run.model_validate_json(recorded.model_dump_json()) == recorded


class TestOutput:
    def test_a_run_has_no_output_until_one_is_validated(self) -> None:
        assert run().output is None

    def test_the_output_is_the_declared_type_and_still_survives_a_checkpoint(self) -> None:
        """The type parameter is what rehydrates JSON as that type rather than as a dict."""
        started = Run[TripPlan](**_FIELDS)  # type: ignore[arg-type]
        finished = started.with_output(TripPlan(destination="Kyoto", nights=3))
        assert finished.output == TripPlan(destination="Kyoto", nights=3)
        assert Run[TripPlan].model_validate_json(finished.model_dump_json()) == finished


class TestContext:
    def test_a_tenant_context_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            TenantContext(tenant="acme").tenant = "other"

    def test_a_run_context_carries_its_tenant(self) -> None:
        """The runtime threads it; a tool never receives the tenant by hand."""
        context = RunContext(run_id="run_1", tenant=TenantContext(tenant="acme"))
        assert context.tenant.tenant == "acme"

    def test_a_run_context_round_trips(self) -> None:
        context = RunContext(run_id="run_1", tenant=TenantContext(tenant="acme", user="ada"))
        assert RunContext.model_validate_json(context.model_dump_json()) == context

    def test_a_run_provides_its_own_context(self) -> None:
        assert run().context.run_id == "run_1"
        assert run().context.tenant.tenant == "acme"
