"""What a checkpoint promises, and what a resume refuses to guess at."""

from __future__ import annotations

import pytest

from tesserix_adk.core.agent import Agent
from tesserix_adk.core.checkpoint import (
    CHECKPOINT_FORMAT,
    Checkpoint,
    CheckpointBoundary,
    CheckpointPolicy,
    CheckpointStore,
    PendingCall,
    ResumePlan,
    Resumption,
    ToolDisposition,
)
from tesserix_adk.core.definition import AgentDefinition, Owner
from tesserix_adk.core.errors import (
    CheckpointFormatError,
    CheckpointTooLargeError,
    ConfigurationError,
    IndeterminateToolCallError,
    ResumeConflictError,
    StateNotFoundError,
)
from tesserix_adk.core.primitives import Message, TextPart, ToolCall, Usage
from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.core.run import RunEventKind, RunState
from tesserix_adk.runtime import (
    AgentRunner,
    Checkpointer,
    MemoryCheckpointStore,
    MemoryIdempotencyStore,
    claim_resume,
    plan_resume,
    refuse_if_undecidable,
)
from tesserix_adk.runtime.checkpoint import RESUME_TTL_SECONDS
from tesserix_adk.testing import (
    CheckpointStoreConformance,
    FakeClock,
    FakeToolRegistry,
    ScriptedProvider,
)

TENANT = "acme"


def a_call(name: str = "book", call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"flight": "BA117"})


def a_checkpoint(
    *,
    run_id: str = "r1",
    boundary: CheckpointBoundary = CheckpointBoundary.AFTER_MODEL_CALL,
    pending: tuple[PendingCall, ...] = (),
    messages: tuple[Message, ...] = (),
    format_version: int = CHECKPOINT_FORMAT,
) -> Checkpoint:
    return Checkpoint(
        run_id=run_id,
        tenant=TENANT,
        agent_name="planner",
        boundary=boundary,
        pending=pending,
        messages=messages,
        format_version=format_version,
    )


class TestWhatACheckpointKnowsAboutItself:
    def test_it_measures_itself_the_way_it_will_be_written(self) -> None:
        assert a_checkpoint().size_bytes == len(a_checkpoint().model_dump_json().encode())

    def test_a_longer_conversation_is_a_larger_checkpoint(self) -> None:
        talkative = a_checkpoint(
            messages=(Message(role="user", content=[TextPart(text="book it" * 100)]),)
        )
        assert talkative.size_bytes > a_checkpoint().size_bytes

    def test_a_reader_of_the_same_format_can_read_it(self) -> None:
        assert a_checkpoint().resumable_by(CHECKPOINT_FORMAT) is True

    def test_an_older_checkpoint_is_still_readable(self) -> None:
        assert a_checkpoint(format_version=1).resumable_by(9) is True

    def test_a_reader_will_not_read_a_newer_format(self) -> None:
        """Reading fields it has to guess at is how a resume replays a call that ran."""
        assert a_checkpoint(format_version=99).resumable_by(CHECKPOINT_FORMAT) is False

    def test_a_checkpoint_cannot_lose_its_run(self) -> None:
        with pytest.raises(ValueError, match="at least 1 character"):
            Checkpoint(run_id="", tenant=TENANT, agent_name="planner")


class TestWhatAPolicySays:
    def test_every_boundary_is_written_by_default(self) -> None:
        policy = CheckpointPolicy()
        assert all(policy.writes_at(boundary) for boundary in CheckpointBoundary)

    def test_a_narrowed_policy_skips_the_boundaries_it_left_out(self) -> None:
        policy = CheckpointPolicy(boundaries=frozenset({CheckpointBoundary.AFTER_TOOL_RESULT}))
        assert policy.writes_at(CheckpointBoundary.AFTER_MODEL_CALL) is False

    def test_a_policy_that_writes_nowhere_is_allowed(self) -> None:
        """Checkpointing off is a configuration, not an error."""
        assert (
            CheckpointPolicy(boundaries=frozenset()).writes_at(CheckpointBoundary.BEFORE_APPROVAL)
            is False
        )

    def test_a_cap_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            CheckpointPolicy(max_bytes=0)

    def test_a_checkpoint_that_never_goes_stale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            CheckpointPolicy(ttl_seconds=0)


class TestWhatAResumptionMayClaim:
    def test_a_completed_call_carries_what_it_recorded(self) -> None:
        done = Resumption(call=a_call(), disposition=ToolDisposition.COMPLETED, outcome="booked")
        assert done.outcome == "booked"

    def test_a_completed_call_with_nothing_to_replay_is_refused(self) -> None:
        """Without the outcome there is nothing to resume with but a second call."""
        with pytest.raises(ValueError, match="must carry what it recorded"):
            Resumption(call=a_call(), disposition=ToolDisposition.COMPLETED)

    def test_a_call_that_never_ran_needs_no_outcome(self) -> None:
        assert Resumption(call=a_call(), disposition=ToolDisposition.NEVER_RAN).outcome is None


class TestWhatAPlanSays:
    def test_a_plan_with_nothing_outstanding_is_safe(self) -> None:
        assert ResumePlan(checkpoint=a_checkpoint()).safe is True

    def test_a_plan_sorts_calls_by_what_is_known_about_them(self) -> None:
        plan = ResumePlan(
            checkpoint=a_checkpoint(),
            resumptions=(
                Resumption(call=a_call("a"), disposition=ToolDisposition.NEVER_RAN),
                Resumption(call=a_call("b"), disposition=ToolDisposition.COMPLETED, outcome="done"),
                Resumption(call=a_call("c"), disposition=ToolDisposition.INDETERMINATE),
            ),
        )
        assert [one.call.name for one in plan.to_dispatch] == ["a"]
        assert [one.call.name for one in plan.completed] == ["b"]
        assert [one.call.name for one in plan.indeterminate] == ["c"]

    def test_a_plan_with_an_undecidable_call_is_not_safe(self) -> None:
        plan = ResumePlan(
            checkpoint=a_checkpoint(),
            resumptions=(Resumption(call=a_call(), disposition=ToolDisposition.INDETERMINATE),),
        )
        assert plan.safe is False


class TestWritingAFrontierDown:
    async def test_a_checkpoint_at_a_named_boundary_is_written(self) -> None:
        store = MemoryCheckpointStore()
        assert await Checkpointer(store).record(a_checkpoint()) is True
        assert await store.latest("r1", tenant=TENANT) is not None

    async def test_a_checkpoint_at_a_boundary_the_policy_skips_is_not(self) -> None:
        store = MemoryCheckpointStore()
        policy = CheckpointPolicy(boundaries=frozenset({CheckpointBoundary.BEFORE_APPROVAL}))
        assert await Checkpointer(store, policy).record(a_checkpoint()) is False
        assert await store.latest("r1", tenant=TENANT) is None

    async def test_a_written_checkpoint_records_when_it_happened(self) -> None:
        checkpointer = Checkpointer(MemoryCheckpointStore(), clock=FakeClock(start=1_000.0))
        await checkpointer.record(a_checkpoint())
        written = await checkpointer.latest("r1", tenant=TENANT)
        assert written is not None
        assert written.created_at == 1_000.0

    async def test_a_checkpointer_with_no_clock_says_nothing_about_when(self) -> None:
        """Nothing orders by it, so a checkpointer without one is still correct."""
        checkpointer = Checkpointer(MemoryCheckpointStore())
        await checkpointer.record(a_checkpoint())
        written = await checkpointer.latest("r1", tenant=TENANT)
        assert written is not None
        assert written.created_at == 0.0

    async def test_a_checkpoint_over_the_cap_is_refused_rather_than_truncated(self) -> None:
        """Half a frontier resumes into a conversation that never happened."""
        store = MemoryCheckpointStore()
        checkpointer = Checkpointer(store, CheckpointPolicy(max_bytes=10))
        assert await checkpointer.record(a_checkpoint()) is False
        assert await store.latest("r1", tenant=TENANT) is None

    async def test_a_refusal_names_the_cap_it_was_measured_against(self) -> None:
        checkpointer = Checkpointer(MemoryCheckpointStore(), CheckpointPolicy(max_bytes=10))
        await checkpointer.record(a_checkpoint())
        refused = checkpointer.last_error
        assert isinstance(refused, CheckpointTooLargeError)
        assert refused.max_bytes == 10

    async def test_a_store_that_could_not_take_the_write_does_not_fail_the_run(self) -> None:
        """Losing the ability to resume is worth knowing about, not worth stopping over."""
        checkpointer = Checkpointer(_ABrokenStore())
        assert await checkpointer.record(a_checkpoint()) is False
        assert isinstance(checkpointer.last_error, RuntimeError)

    async def test_a_later_success_clears_the_earlier_failure(self) -> None:
        checkpointer = Checkpointer(MemoryCheckpointStore(), CheckpointPolicy(max_bytes=10))
        await checkpointer.record(a_checkpoint())
        checkpointer._policy = CheckpointPolicy()
        assert await checkpointer.record(a_checkpoint()) is True
        assert checkpointer.last_error is None

    async def test_a_checkpointer_reports_the_policy_it_applies(self) -> None:
        policy = CheckpointPolicy(max_bytes=99)
        assert Checkpointer(MemoryCheckpointStore(), policy).policy is policy

    async def test_a_checkpoint_from_a_newer_kit_is_refused_rather_than_guessed_at(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(a_checkpoint(format_version=CHECKPOINT_FORMAT + 1))
        with pytest.raises(CheckpointFormatError) as refused:
            await Checkpointer(store).latest("r1", tenant=TENANT)
        assert refused.value.readable_version == CHECKPOINT_FORMAT

    async def test_a_run_that_was_never_checkpointed_reads_as_nothing(self) -> None:
        assert await Checkpointer(MemoryCheckpointStore()).latest("r9", tenant=TENANT) is None

    async def test_a_finished_run_lets_its_frontier_go(self) -> None:
        store = MemoryCheckpointStore()
        checkpointer = Checkpointer(store)
        await checkpointer.record(a_checkpoint())
        await checkpointer.forget("r1", tenant=TENANT)
        assert await store.latest("r1", tenant=TENANT) is None


class TestWorkingOutWhatToCarryOn:
    async def test_a_call_that_was_never_dispatched_is_dispatched_now(self) -> None:
        checkpoint = a_checkpoint(pending=(PendingCall(call=a_call()),))
        plan = await plan_resume(checkpoint, MemoryIdempotencyStore())
        assert [one.call.name for one in plan.to_dispatch] == ["book"]

    async def test_a_call_whose_outcome_was_recorded_is_replayed(self) -> None:
        """Re-executing it is the second booking checkpointing exists to prevent."""
        idempotency = MemoryIdempotencyStore()
        await idempotency.record("k1", tenant=TENANT, outcome="booked", ttl_seconds=600)
        checkpoint = a_checkpoint(
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),)
        )
        plan = await plan_resume(checkpoint, idempotency)
        assert [one.outcome for one in plan.completed] == ["booked"]
        assert plan.safe is True

    async def test_a_dispatched_call_nothing_holds_never_started(self) -> None:
        checkpoint = a_checkpoint(
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),)
        )
        plan = await plan_resume(checkpoint, MemoryIdempotencyStore())
        assert [one.call.name for one in plan.to_dispatch] == ["book"]

    async def test_planning_leaves_no_claim_behind_on_a_call_it_will_dispatch(self) -> None:
        """A plan that claimed the key would deadlock the resume it was planning for."""
        idempotency = MemoryIdempotencyStore()
        checkpoint = a_checkpoint(
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),)
        )
        await plan_resume(checkpoint, idempotency)
        assert (await idempotency.begin("k1", tenant=TENANT, ttl_seconds=600)).in_flight is False

    async def test_a_call_held_in_flight_by_a_process_that_is_gone_is_undecidable(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await idempotency.begin("k1", tenant=TENANT, ttl_seconds=600)
        checkpoint = a_checkpoint(
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),)
        )
        plan = await plan_resume(checkpoint, idempotency)
        assert [one.call.name for one in plan.indeterminate] == ["book"]
        assert plan.safe is False

    async def test_a_dispatched_call_with_no_key_is_undecidable(self) -> None:
        """The absence of a key is the absence of a guarantee, not permission to retry."""
        checkpoint = a_checkpoint(pending=(PendingCall(call=a_call(), dispatched=True),))
        plan = await plan_resume(checkpoint, MemoryIdempotencyStore())
        assert plan.safe is False

    async def test_with_no_idempotency_store_every_dispatched_call_is_undecidable(self) -> None:
        checkpoint = a_checkpoint(
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),)
        )
        assert (await plan_resume(checkpoint)).safe is False

    async def test_calls_keep_the_order_the_model_asked_for_them_in(self) -> None:
        checkpoint = a_checkpoint(
            pending=(
                PendingCall(call=a_call("first", "c1")),
                PendingCall(call=a_call("second", "c2")),
            )
        )
        plan = await plan_resume(checkpoint, MemoryIdempotencyStore())
        assert [one.call.name for one in plan.resumptions] == ["first", "second"]

    async def test_a_plan_carries_the_checkpoint_it_was_made_from(self) -> None:
        checkpoint = a_checkpoint()
        assert (await plan_resume(checkpoint)).checkpoint == checkpoint


class TestRefusingToGuess:
    async def test_a_safe_plan_passes(self) -> None:
        refuse_if_undecidable(await plan_resume(a_checkpoint()))

    async def test_an_undecidable_plan_names_every_call_nobody_can_decide(self) -> None:
        checkpoint = a_checkpoint(
            pending=(
                PendingCall(call=a_call("book", "c1"), dispatched=True),
                PendingCall(call=a_call("pay", "c2"), dispatched=True),
            )
        )
        with pytest.raises(IndeterminateToolCallError) as refused:
            refuse_if_undecidable(await plan_resume(checkpoint))
        assert refused.value.calls == ("book", "pay")

    async def test_an_undecidable_call_is_never_worth_retrying(self) -> None:
        """A retry is the second booking. It needs a person or the tool's own status."""
        assert IndeterminateToolCallError("stuck", run_id="r1", calls=("book",)).retryable is False


class TestOnlyOneWorkerCarriesARunOn:
    async def test_the_first_worker_takes_the_run(self) -> None:
        await claim_resume("r1", tenant=TENANT, idempotency=MemoryIdempotencyStore())

    async def test_the_second_worker_is_refused(self) -> None:
        """Two workers on one run is one budget spent twice and every call dispatched twice."""
        idempotency = MemoryIdempotencyStore()
        await claim_resume("r1", tenant=TENANT, idempotency=idempotency)
        with pytest.raises(ResumeConflictError) as refused:
            await claim_resume("r1", tenant=TENANT, idempotency=idempotency)
        assert refused.value.run_id == "r1"

    async def test_a_worker_on_a_different_run_is_not_blocked(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await claim_resume("r1", tenant=TENANT, idempotency=idempotency)
        await claim_resume("r2", tenant=TENANT, idempotency=idempotency)

    async def test_one_tenant_s_claim_does_not_hold_another_s_run(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await claim_resume("r1", tenant="one", idempotency=idempotency)
        await claim_resume("r1", tenant="two", idempotency=idempotency)

    async def test_a_claim_a_dead_worker_left_behind_expires(self) -> None:
        """Otherwise one crash strands the run forever."""
        clock = FakeClock(start=0.0)
        idempotency = MemoryIdempotencyStore(clock)
        await claim_resume("r1", tenant=TENANT, idempotency=idempotency)
        clock.advance(RESUME_TTL_SECONDS + 1)
        await claim_resume("r1", tenant=TENANT, idempotency=idempotency)

    async def test_a_run_that_already_finished_resuming_is_not_resumed_again(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await idempotency.record("resume:r1", tenant=TENANT, outcome="done", ttl_seconds=600)
        with pytest.raises(ResumeConflictError):
            await claim_resume("r1", tenant=TENANT, idempotency=idempotency)


class TestKillAndCarryOn:
    async def test_a_run_killed_mid_dispatch_books_the_flight_exactly_once(self) -> None:
        idempotency = MemoryIdempotencyStore()
        booked: list[str] = []

        async def book(key: str) -> None:
            claim = await idempotency.begin(key, tenant=TENANT, ttl_seconds=600)
            if claim.outcome is not None:
                return
            booked.append("BA117")
            await idempotency.record(key, tenant=TENANT, outcome="BA117", ttl_seconds=600)

        await book("k1")
        checkpoint = a_checkpoint(
            boundary=CheckpointBoundary.AFTER_TOOL_RESULT,
            pending=(PendingCall(call=a_call(), idempotency_key="k1", dispatched=True),),
        )
        store = MemoryCheckpointStore()
        await Checkpointer(store).record(checkpoint)

        restored = await Checkpointer(store).latest("r1", tenant=TENANT)
        assert restored is not None
        plan = await plan_resume(restored, idempotency)
        refuse_if_undecidable(plan)
        for one in plan.to_dispatch:
            await book(one.call.arguments["flight"])
        assert booked == ["BA117"]

    async def test_what_the_run_had_spent_survives_the_restart(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                usage=Usage(input_tokens=900, output_tokens=120),
                cost_micros=4_200,
                iterations=3,
            )
        )
        restored = await Checkpointer(store).latest("r1", tenant=TENANT)
        assert restored is not None
        assert (restored.iterations, restored.cost_micros) == (3, 4_200)
        assert restored.usage.input_tokens == 900

    async def test_a_guard_that_already_passed_is_not_paid_for_twice(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                guardrails_passed=("pii", "jailbreak"),
            )
        )
        restored = await store.latest("r1", tenant=TENANT)
        assert restored is not None
        assert restored.guardrails_passed == ("pii", "jailbreak")


class _ABrokenStore:
    """A store that cannot take a write, which must not take the run down with it."""

    async def put(self, checkpoint: Checkpoint) -> None:  # noqa: ARG002 — a store that refuses everything
        raise RuntimeError("the store is gone")

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:  # noqa: ARG002
        return None

    async def forget(self, run_id: str, *, tenant: str) -> None:  # noqa: ARG002
        return None


class TestTheMemoryStoreConforms(CheckpointStoreConformance):
    def make_store(self) -> CheckpointStore:
        return MemoryCheckpointStore()


class TestARunThatChecksItselfIn:
    def a_runner(self, *responses: ModelResponse, **overrides: object) -> AgentRunner:
        fields: dict[str, object] = {
            "provider": ScriptedProvider(*responses),
            "clock": FakeClock(),
        }
        return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]

    def an_agent(self, **overrides: object) -> Agent:
        fields: dict[str, object] = {
            "name": "planner",
            "instructions": "Plan trips.",
            "free_text": True,
            "model": "claude-sonnet-5",
        }
        return Agent(**{**fields, **overrides})  # type: ignore[arg-type]

    async def test_a_run_with_no_checkpointer_still_runs(self) -> None:
        """Checkpointing is something a deployment opts into, not something it needs."""
        runner = self.a_runner(ModelResponse(content="Kyoto."))
        run = await runner.run(self.an_agent(), "plan", tenant=TENANT, run_id="r1")
        assert run.state is RunState.COMPLETED

    async def test_a_run_writes_its_frontier_as_it_goes(self) -> None:
        store = MemoryCheckpointStore()
        seen: list[Checkpoint] = []
        runner = self.a_runner(
            ModelResponse(tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "kyoto"}),)),
            ModelResponse(content="Kyoto."),
            tools=FakeToolRegistry({"search": lambda q: f"3 results {q}"}),
            checkpoints=_ARecordingCheckpointer(store, seen),
        )
        await runner.run(self.an_agent(tools=("search",)), "plan", tenant=TENANT, run_id="r1")
        assert [one.boundary for one in seen] == [
            CheckpointBoundary.AFTER_MODEL_CALL,
            CheckpointBoundary.AFTER_TOOL_RESULT,
            CheckpointBoundary.AFTER_MODEL_CALL,
        ]

    async def test_a_frontier_carries_the_conversation_so_far(self) -> None:
        store = MemoryCheckpointStore()
        seen: list[Checkpoint] = []
        runner = self.a_runner(
            ModelResponse(content="Kyoto.", usage=Usage(input_tokens=10, output_tokens=5)),
            checkpoints=_ARecordingCheckpointer(store, seen),
        )
        await runner.run(self.an_agent(), "plan", tenant=TENANT, run_id="r1")
        assert seen[0].messages
        assert seen[0].iterations == 1
        assert seen[0].model == "claude-sonnet-5"

    async def test_a_run_that_reached_the_end_leaves_no_frontier_behind(self) -> None:
        """A finished run resumed is a finished run charged for twice."""
        store = MemoryCheckpointStore()
        runner = self.a_runner(ModelResponse(content="Kyoto."), checkpoints=Checkpointer(store))
        await runner.run(self.an_agent(), "plan", tenant=TENANT, run_id="r1")
        assert await store.latest("r1", tenant=TENANT) is None

    async def test_a_run_that_failed_leaves_no_frontier_behind(self) -> None:
        store = MemoryCheckpointStore()
        runner = self.a_runner(
            ModelResponse(tool_calls=(ToolCall(id="c1", name="wire_money"),)),
            tools=FakeToolRegistry({"wire_money": lambda **_: "sent"}),
            checkpoints=Checkpointer(store),
        )
        run = await runner.run(self.an_agent(tools=("search",)), "plan", tenant=TENANT, run_id="r1")
        assert run.state is RunState.FAILED
        assert await store.latest("r1", tenant=TENANT) is None


class TestCarryingARunOn:
    def an_agent(self, **overrides: object) -> Agent:
        fields: dict[str, object] = {
            "name": "planner",
            "instructions": "Plan trips.",
            "free_text": True,
            "model": "claude-sonnet-5",
        }
        return Agent(**{**fields, **overrides})  # type: ignore[arg-type]

    def a_runner(self, store: MemoryCheckpointStore, *responses: ModelResponse) -> AgentRunner:
        return AgentRunner(
            provider=ScriptedProvider(*responses),
            clock=FakeClock(),
            checkpoints=Checkpointer(store),
        )

    async def test_a_runner_with_nowhere_to_read_from_says_so(self) -> None:
        runner = AgentRunner(provider=ScriptedProvider(), clock=FakeClock())
        with pytest.raises(ConfigurationError, match="without a checkpointer"):
            await runner.resume(self.an_agent(), "r1", tenant=TENANT)

    async def test_a_run_that_was_never_checkpointed_cannot_be_carried_on(self) -> None:
        runner = self.a_runner(MemoryCheckpointStore())
        with pytest.raises(StateNotFoundError):
            await runner.resume(self.an_agent(), "r1", tenant=TENANT)

    async def test_a_run_carries_on_from_the_conversation_it_left(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan a trip")]),),
                iterations=1,
            )
        )
        run = await self.a_runner(store, ModelResponse(content="Kyoto.")).resume(
            self.an_agent(), "r1", tenant=TENANT
        )
        assert run.state is RunState.COMPLETED
        assert run.id == "r1"

    async def test_what_the_run_had_already_spent_comes_back_with_it(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
                usage=Usage(input_tokens=900, output_tokens=120),
            )
        )
        run = await self.a_runner(store, ModelResponse(content="Kyoto.")).resume(
            self.an_agent(), "r1", tenant=TENANT
        )
        assert run.usage.input_tokens >= 900

    async def test_a_resumed_run_says_it_was_resumed(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
            )
        )
        run = await self.a_runner(store, ModelResponse(content="Kyoto.")).resume(
            self.an_agent(), "r1", tenant=TENANT
        )
        assert any(event.kind is RunEventKind.RUN_RESUMED for event in run.events)

    async def test_a_call_that_never_ran_is_dispatched_by_the_resume(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
                pending=(
                    PendingCall(call=ToolCall(id="c1", name="search", arguments={"q": "kyoto"})),
                ),
            )
        )
        tools = FakeToolRegistry({"search": lambda q: f"3 results {q}"})
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            tools=tools,
            checkpoints=Checkpointer(store),
        )
        run = await runner.resume(self.an_agent(tools=("search",)), "r1", tenant=TENANT)
        assert tools.calls == [("search", {"q": "kyoto"})]
        assert run.state is RunState.COMPLETED

    async def test_a_call_that_already_ran_is_replayed_rather_than_called_again(self) -> None:
        """The whole point: the flight is booked once however many times the process dies."""
        store = MemoryCheckpointStore()
        idempotency = MemoryIdempotencyStore()
        await idempotency.record("k1", tenant=TENANT, outcome="3 results kyoto", ttl_seconds=600)
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
                pending=(
                    PendingCall(
                        call=ToolCall(id="c1", name="search", arguments={"q": "kyoto"}),
                        idempotency_key="k1",
                        dispatched=True,
                    ),
                ),
            )
        )
        tools = FakeToolRegistry({"search": lambda q: f"3 results {q}"})
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            tools=tools,
            idempotency=idempotency,
            checkpoints=Checkpointer(store),
        )
        run = await runner.resume(self.an_agent(tools=("search",)), "r1", tenant=TENANT)
        assert tools.calls == []
        assert run.state is RunState.COMPLETED

    async def test_a_call_nobody_can_decide_stops_the_resume(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                pending=(
                    PendingCall(call=ToolCall(id="c1", name="book", arguments={}), dispatched=True),
                ),
            )
        )
        with pytest.raises(IndeterminateToolCallError):
            await self.a_runner(store).resume(self.an_agent(), "r1", tenant=TENANT)

    async def test_a_second_worker_is_refused_the_run(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
            )
        )
        idempotency = MemoryIdempotencyStore()
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            idempotency=idempotency,
            checkpoints=Checkpointer(store),
        )
        await claim_resume("r1", tenant=TENANT, idempotency=idempotency)
        with pytest.raises(ResumeConflictError):
            await runner.resume(self.an_agent(), "r1", tenant=TENANT)

    async def test_a_run_carried_on_at_a_different_revision_is_refused(self) -> None:
        """Resuming into a changed agent is a different run wearing the first one's name."""
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                agent_revision="rev-1",
            )
        )
        definition = AgentDefinition(
            agent=self.an_agent(),
            owner=Owner(team="platform", service="planner", contact="platform@example.com"),
            evaluation_suite="suites/planner.yaml",
        )
        with pytest.raises(ConfigurationError, match="different run"):
            await self.a_runner(store).resume(definition, "r1", tenant=TENANT)

    async def test_a_resumed_run_that_finished_leaves_no_frontier_behind(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
            )
        )
        await self.a_runner(store, ModelResponse(content="Kyoto.")).resume(
            self.an_agent(), "r1", tenant=TENANT
        )
        assert await store.latest("r1", tenant=TENANT) is None

    async def test_a_resume_that_fails_still_lets_its_frontier_go(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            Checkpoint(
                run_id="r1",
                tenant=TENANT,
                agent_name="planner",
                model="claude-sonnet-5",
                messages=(Message(role="user", content=[TextPart(text="plan")]),),
                pending=(PendingCall(call=ToolCall(id="c1", name="wire_money", arguments={})),),
            )
        )
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            tools=FakeToolRegistry({"wire_money": lambda **_: "sent"}),
            checkpoints=Checkpointer(store),
        )
        run = await runner.resume(self.an_agent(tools=("search",)), "r1", tenant=TENANT)
        assert run.state is RunState.FAILED
        assert await store.latest("r1", tenant=TENANT) is None


class _ARecordingCheckpointer(Checkpointer):
    """A checkpointer that also keeps every frontier, so a test can see the order."""

    def __init__(self, store: MemoryCheckpointStore, seen: list[Checkpoint]) -> None:
        super().__init__(store)
        self._seen = seen

    async def record(self, checkpoint: Checkpoint) -> bool:
        self._seen.append(checkpoint)
        return await super().record(checkpoint)
