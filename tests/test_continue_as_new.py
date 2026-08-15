"""An execution that has run long enough to be worth starting again, without losing the run."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Checkpoint,
    CheckpointBoundary,
    ConfigurationError,
    Usage,
)
from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.workflows import (
    DEFAULT_CONTINUATION,
    Continuation,
    ContinuationPolicy,
    Journal,
    ModelCallResult,
    WorkflowState,
    continued,
)
from tesserix_adk.workflows.durable import ToolCallResult


def journal_of(steps: int) -> Journal:
    """A journal with `steps` model activities recorded on it."""
    journal = Journal()
    answered = ModelCallResult(
        response=ModelResponse(content="done", usage=Usage(input_tokens=10, output_tokens=10)),
        history="h-1",
    )
    for step in range(steps):
        journal = journal.with_model(f"model:{step}", answered)
    return journal


class TestWhenToStartAgain:
    def test_a_short_execution_carries_on(self) -> None:
        assert DEFAULT_CONTINUATION.due(journal_of(3)) is False

    def test_an_execution_past_the_step_ceiling_is_due(self) -> None:
        policy = ContinuationPolicy(max_steps=4)

        assert policy.due(journal_of(4)) is True

    def test_an_engine_history_past_the_ceiling_is_due_on_its_own(self) -> None:
        """The journal counts activities; the engine's history counts everything else too."""
        assert DEFAULT_CONTINUATION.due(Journal(), history_bytes=2_000_000) is True

    def test_a_deployment_that_cannot_measure_its_history_uses_the_steps_alone(self) -> None:
        policy = ContinuationPolicy(max_steps=4, max_history_bytes=0)

        assert policy.due(Journal(), history_bytes=9_000_000) is False

    def test_tool_steps_count_towards_the_ceiling_too(self) -> None:
        policy = ContinuationPolicy(max_steps=2)
        done = ToolCallResult(call_id="c0", content="ok", history="h-1")
        journal = journal_of(1).with_tool("tool:0:c0", done)

        assert policy.due(journal) is True


class TestWhatCrossesOver:
    def test_the_transcript_travels_as_a_handle(self) -> None:
        carried = continued(
            WorkflowState(run_id="r1", history="h-9", iteration=4),
            tenant="acme",
            agent_name="booking",
        )

        assert carried.checkpoint.history_handle == "h-9"
        assert carried.checkpoint.messages == ()

    def test_the_ledger_and_the_iteration_count_survive(self) -> None:
        state = WorkflowState(
            run_id="r1",
            history="h-9",
            iteration=4,
            usage=Usage(input_tokens=1_200, output_tokens=300),
        )

        carried = continued(state, tenant="acme", agent_name="booking")

        assert carried.checkpoint.iterations == 4
        assert carried.checkpoint.usage.input_tokens == 1_200

    def test_the_approval_the_run_waits_on_crosses_with_it(self) -> None:
        state = WorkflowState(run_id="r1", history="h-9", pending_approval="req-9", grant="grant-2")

        carried = continued(state, tenant="acme", agent_name="booking")

        assert carried.checkpoint.pending_approval == "req-9"
        assert carried.checkpoint.grant_id == "grant-2"
        assert carried.checkpoint.boundary is CheckpointBoundary.BEFORE_APPROVAL

    def test_a_run_waiting_on_nothing_crosses_at_the_model_boundary(self) -> None:
        carried = continued(
            WorkflowState(run_id="r1", history="h-9"), tenant="acme", agent_name="booking"
        )

        assert carried.checkpoint.boundary is CheckpointBoundary.AFTER_MODEL_CALL

    def test_attribution_crosses_because_a_resumed_run_still_bills_someone(self) -> None:
        carried = continued(
            WorkflowState(run_id="r1", history="h-9"),
            tenant="acme",
            agent_name="booking",
            model="claude-opus-5",
            user="ada",
            scopes=("booking:write",),
        )

        assert carried.checkpoint.tenant == "acme"
        assert carried.checkpoint.user == "ada"
        assert carried.checkpoint.model == "claude-opus-5"
        assert carried.checkpoint.scopes == ("booking:write",)

    def test_the_journal_does_not_cross(self) -> None:
        """Its results are already folded into the state; replaying them is the cost being cut."""
        carried = continued(
            WorkflowState(run_id="r1", history="h-9", iteration=4),
            tenant="acme",
            agent_name="booking",
        )

        assert "journal" not in carried.model_dump()

    def test_the_frontier_is_the_shape_the_resumer_reads(self) -> None:
        carried = continued(
            WorkflowState(run_id="r1", history="h-9"), tenant="acme", agent_name="booking"
        )

        assert isinstance(carried.checkpoint, Checkpoint)
        assert carried.checkpoint.resumable_by(carried.checkpoint.format_version)


class TestRefusingToLoseTheBinding:
    def test_a_continuation_that_dropped_the_approval_is_refused(self) -> None:
        state = WorkflowState(run_id="r1", history="h-9", pending_approval="req-9")
        frontier = Checkpoint(run_id="r1", tenant="acme", agent_name="booking")

        with pytest.raises(ConfigurationError, match="waiting on approval"):
            Continuation(state=state, checkpoint=frontier)

    def test_a_continuation_that_dropped_the_grant_is_refused(self) -> None:
        state = WorkflowState(run_id="r1", history="h-9", grant="grant-2")
        frontier = Checkpoint(run_id="r1", tenant="acme", agent_name="booking")

        with pytest.raises(ConfigurationError, match="acting under grant"):
            Continuation(state=state, checkpoint=frontier)

    def test_the_binding_built_here_always_agrees_with_itself(self) -> None:
        state = WorkflowState(run_id="r1", history="h-9", pending_approval="req-9", grant="grant-2")

        carried = continued(state, tenant="acme", agent_name="booking")

        assert carried.state.pending_approval == carried.checkpoint.pending_approval
