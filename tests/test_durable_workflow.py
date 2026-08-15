"""A run that survives the worker it started on."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from tesserix_adk.core import (
    CancelledError,
    ConfigurationError,
    MissingTenantContextError,
    ModelResponse,
    PayloadTooLargeError,
    ProviderUnavailableError,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import CancellationToken
from tesserix_adk.workflows import (
    PAYLOAD_LIMIT_BYTES,
    STREAMING_UNSUPPORTED,
    Activities,
    ActivityContext,
    AgentWorkflow,
    Journal,
    ModelCallInput,
    ModelCallResult,
    ToolCallInput,
    ToolCallResult,
    WorkflowState,
)

pytestmark = pytest.mark.anyio

CONTEXT = ActivityContext(run_id="r1", tenant="acme", user="ada", scopes=("read",), trace_id="t1")


def answer(content: str, *, tokens: int = 10) -> ModelResponse:
    """A response with no tool calls, which ends the run."""
    return ModelResponse(content=content, usage=Usage(input_tokens=tokens, output_tokens=tokens))


def asks(*names: str, tokens: int = 10) -> ModelResponse:
    """A response asking for tools, by name."""
    return ModelResponse(
        tool_calls=tuple(ToolCall(id=f"c{index}", name=name) for index, name in enumerate(names)),
        usage=Usage(input_tokens=tokens, output_tokens=tokens),
    )


class Worker:
    """A worker that records every activity it actually executed."""

    def __init__(self, *, responses: list[ModelResponse] | None = None) -> None:
        self.responses = responses or [answer("done")]
        self.model_steps: list[str] = []
        self.tool_steps: list[str] = []
        self.inputs: list[ModelCallInput | ToolCallInput] = []
        self.kill_after: int = 0

    async def model_call(self, request: ModelCallInput) -> ModelCallResult:
        """Answer from the scripted list, in order."""
        self._alive()
        self.model_steps.append(request.step)
        self.inputs.append(request)
        iteration = int(request.step.split(":")[1])
        response = self.responses[min(iteration, len(self.responses) - 1)]
        return ModelCallResult(response=response, history=f"{request.history}+{request.step}")

    async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
        """Run the tool, which here means recording that it ran."""
        self._alive()
        self.tool_steps.append(request.step)
        self.inputs.append(request)
        return ToolCallResult(
            call_id=request.call_id,
            content=f"{request.tool} ran",
            history=f"h+{request.step}",
        )

    def _alive(self) -> None:
        """Die once the test says the pod rolled."""
        if self.kill_after and len(self.model_steps) + len(self.tool_steps) >= self.kill_after:
            message = "worker killed"
            raise RuntimeError(message)

    @property
    def executed(self) -> int:
        """How many activities this worker actually ran."""
        return len(self.model_steps) + len(self.tool_steps)


def workflow(worker: Worker, **kwargs: object) -> AgentWorkflow:
    """One workflow over that worker."""
    return AgentWorkflow(activities=worker, model="claude-opus-5", **kwargs)  # type: ignore[arg-type]


class TestDrivingTheRun:
    """The same loop the in-process runtime drives, with the I/O taken out of it."""

    async def test_a_run_with_no_tool_calls_answers(self) -> None:
        worker = Worker(responses=[answer("Kyoto in April")])

        final = await workflow(worker).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.terminal == "answered"
        assert final.answer == "Kyoto in April"
        assert final.iteration == 1

    async def test_tool_calls_are_executed_before_the_next_model_call(self) -> None:
        worker = Worker(responses=[asks("search", "book"), answer("booked")])

        final = await workflow(worker).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert worker.model_steps == ["model:0", "model:1"]
        assert worker.tool_steps == ["tool:0:c0", "tool:0:c1"]
        assert final.answer == "booked"

    async def test_usage_is_summed_from_the_activity_results(self) -> None:
        worker = Worker(responses=[asks("search", tokens=7), answer("done", tokens=5)])

        final = await workflow(worker).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.usage.input_tokens == 12
        assert final.usage.output_tokens == 12

    async def test_the_history_handle_moves_forward_with_each_step(self) -> None:
        worker = Worker(responses=[answer("done")])

        final = await workflow(worker).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.history == "h0+model:0"

    async def test_a_run_that_never_stops_asking_is_exhausted_not_endless(self) -> None:
        worker = Worker(responses=[asks("search")])

        final = await workflow(worker, max_iterations=3).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.terminal == "exhausted"
        assert worker.model_steps == ["model:0", "model:1", "model:2"]


class TestSurvivingTheWorker:
    """The point of the whole thing: what is done stays done."""

    async def test_a_resumed_run_executes_only_what_the_journal_lacks(self) -> None:
        first = Worker(responses=[asks("search", "book"), answer("booked")])
        first.kill_after = 3
        started = workflow(first)
        with pytest.raises(RuntimeError):
            await started.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        second = Worker(responses=[asks("search", "book"), answer("booked")])
        resumed = workflow(second, journal=started.journal)
        final = await resumed.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert final.answer == "booked"
        assert second.model_steps == ["model:1"]
        assert second.tool_steps == []

    async def test_a_resumed_run_matches_the_run_that_was_never_interrupted(self) -> None:
        straight = Worker(responses=[asks("search"), answer("done")])
        expected = await workflow(straight).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        first = Worker(responses=[asks("search"), answer("done")])
        first.kill_after = 2
        started = workflow(first)
        with pytest.raises(RuntimeError):
            await started.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)
        resumed = await workflow(Worker(responses=[answer("done")]), journal=started.journal).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert resumed.answer == expected.answer
        assert resumed.usage == expected.usage
        assert resumed.iteration == expected.iteration

    async def test_a_completed_tool_call_is_never_applied_twice(self) -> None:
        journal = Journal().with_tool(
            "tool:0:c0", ToolCallResult(call_id="c0", content="charged", history="h1")
        )
        worker = Worker(responses=[asks("charge"), answer("done")])

        await workflow(worker, journal=journal).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert worker.tool_steps == []

    async def test_the_journal_counts_what_it_would_skip(self) -> None:
        worker = Worker(responses=[asks("search"), answer("done")])
        driven = workflow(worker)

        await driven.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert driven.journal.steps == 3

    async def test_step_ids_do_not_depend_on_anything_a_replay_cannot_reproduce(self) -> None:
        one = Worker(responses=[asks("search"), answer("done")])
        two = Worker(responses=[asks("search"), answer("done")])

        first = workflow(one)
        second = workflow(two)
        await first.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)
        await second.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert sorted(first.journal.models) == sorted(second.journal.models)
        assert sorted(first.journal.tools) == sorted(second.journal.tools)


class TestWhatTravels:
    """A payload is small, typed, and carries who it is for."""

    async def test_every_activity_input_carries_tenant_user_scope_and_trace(self) -> None:
        worker = Worker(responses=[asks("search"), answer("done")])

        await workflow(worker).run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert all(sent.context == CONTEXT for sent in worker.inputs)

    async def test_an_input_without_a_tenant_fails_closed(self) -> None:
        with pytest.raises(MissingTenantContextError) as refused:
            ActivityContext(run_id="r1", user="ada")

        assert refused.value.where == "activity input"

    async def test_history_travels_as_a_handle_and_never_as_a_transcript(self) -> None:
        worker = Worker()

        await workflow(worker).run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        sent = worker.inputs[0]
        assert isinstance(sent, ModelCallInput)
        assert sent.history == "h0"
        assert "messages" not in sent.model_dump()

    async def test_an_unresolvable_history_handle_is_refused_rather_than_dropped(self) -> None:
        with pytest.raises(ConfigurationError):
            ModelCallInput(context=CONTEXT, step="model:0", model="m", history="")

    async def test_an_oversized_result_fails_typed_rather_than_truncating(self) -> None:
        class Verbose(Worker):
            async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
                """Return more than the transport carries."""
                return ToolCallResult(call_id=request.call_id, content="x" * PAYLOAD_LIMIT_BYTES)

        worker = Verbose(responses=[asks("retrieve"), answer("done")])

        with pytest.raises(PayloadTooLargeError) as refused:
            await workflow(worker).run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert refused.value.payload == "result"
        assert refused.value.step == "tool:0:c0"
        assert refused.value.limit == PAYLOAD_LIMIT_BYTES

    async def test_an_oversized_input_is_refused_before_it_is_sent(self) -> None:
        worker = Worker()
        driven = AgentWorkflow(
            activities=worker,
            model="claude-opus-5",
            tools=(),
        )

        with pytest.raises(PayloadTooLargeError) as refused:
            await driven.run(
                WorkflowState(run_id="r1", history="h" * (PAYLOAD_LIMIT_BYTES + 1)),
                context=CONTEXT,
            )

        assert refused.value.payload == "input"
        assert worker.executed == 0

    async def test_streaming_is_documented_as_unavailable_rather_than_degraded(self) -> None:
        assert "streaming" in STREAMING_UNSUPPORTED


class TestWhenItGoesWrong:
    """A failed run keeps its state, and never invents a completion."""

    async def test_an_unavailable_provider_is_retried_then_reported_with_the_count(self) -> None:
        class Down(Worker):
            async def model_call(self, request: ModelCallInput) -> ModelCallResult:
                """Never answer."""
                self.model_steps.append(request.step)
                raise ProviderUnavailableError("gateway cold", provider="anthropic")

        worker = Down()

        with pytest.raises(ProviderUnavailableError) as failure:
            await workflow(worker, attempts=3).run(
                WorkflowState(run_id="r1", history="h0"), context=CONTEXT
            )

        assert failure.value.details["attempts"] == "3"
        assert failure.value.details["step"] == "model:0"
        assert worker.model_steps == ["model:0"] * 3

    async def test_a_provider_that_recovers_inside_the_policy_is_not_a_failure(self) -> None:
        class Flaky(Worker):
            async def model_call(self, request: ModelCallInput) -> ModelCallResult:
                """Fail once, then answer."""
                self.model_steps.append(request.step)
                if len(self.model_steps) == 1:
                    raise ProviderUnavailableError("cold", provider="anthropic")
                return ModelCallResult(response=answer("warm"), history="h1")

        final = await workflow(Flaky()).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.answer == "warm"

    async def test_the_attempt_number_reaches_the_activity(self) -> None:
        class Flaky(Worker):
            def __init__(self) -> None:
                super().__init__()
                self.attempts: list[int] = []

            async def model_call(self, request: ModelCallInput) -> ModelCallResult:
                """Record which attempt this is, failing the first."""
                self.model_steps.append(request.step)
                self.attempts.append(request.attempt)
                if request.attempt == 1:
                    raise ProviderUnavailableError("cold", provider="anthropic")
                return ModelCallResult(response=answer("warm"), history="h1")

        worker = Flaky()
        await workflow(worker).run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert worker.attempts == [1, 2]

    async def test_a_failed_run_leaves_the_completed_steps_in_the_journal(self) -> None:
        worker = Worker(responses=[asks("search"), answer("done")])
        worker.kill_after = 2
        driven = workflow(worker)

        with pytest.raises(RuntimeError):
            await driven.run(WorkflowState(run_id="r1", history="h0"), context=CONTEXT)

        assert "model:0" in driven.journal.models


class TestStopping:
    """Cancelling has to reach the call, not wait for it."""

    async def test_no_further_activity_starts_once_the_run_is_cancelled(self) -> None:
        worker = Worker(responses=[asks("search"), answer("done")])
        token = CancellationToken()
        token.cancel("caller went away")

        with pytest.raises(CancelledError):
            await workflow(worker, token=token).run(
                WorkflowState(run_id="r1", history="h0"), context=CONTEXT
            )

        assert worker.executed == 0

    async def test_cancellation_stops_a_call_that_is_already_in_flight(self) -> None:
        token = CancellationToken()

        class Streaming(Worker):
            def __init__(self) -> None:
                super().__init__()
                self.consumed = 0

            async def model_call(self, request: ModelCallInput) -> ModelCallResult:
                """Consume tokens until somebody stops it."""
                del request
                for _ in range(1000):
                    await asyncio.sleep(0)
                    self.consumed += 1
                return ModelCallResult(response=answer("never"), history="h1")

        worker = Streaming()
        cancelling = asyncio.get_running_loop().call_later(0, token.cancel, "user closed the tab")

        with pytest.raises(CancelledError) as stopped:
            await workflow(worker, token=token).run(
                WorkflowState(run_id="r1", history="h0"), context=CONTEXT
            )

        cancelling.cancel()
        assert str(stopped.value) == "user closed the tab"
        assert worker.consumed < 1000

    async def test_an_uncancelled_run_is_unaffected_by_the_race(self) -> None:
        worker = Worker(responses=[answer("done")])

        final = await workflow(worker, token=CancellationToken()).run(
            WorkflowState(run_id="r1", history="h0"), context=CONTEXT
        )

        assert final.answer == "done"


class TestTheContract:
    """What a deployment binds to Temporal, without the kit importing it."""

    def test_the_activities_protocol_is_satisfied_by_a_plain_object(self) -> None:
        assert isinstance(Worker(), Activities)

    def test_the_kit_imports_with_no_workflow_engine_installed(self) -> None:
        assert importlib.util.find_spec("tesserix_adk.workflows") is not None
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("temporalio")
