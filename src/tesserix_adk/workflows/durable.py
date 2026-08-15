"""An agent run as a durable workflow, where every call is an activity.

A long run lives in one process today: a pod roll, an OOM kill or a scale-down loses it, and
the choices are to start again — paying for every model call a second time — or to hand-roll
the durable wiring, which two teams have now done differently.

`AgentWorkflow` drives the same loop the in-process runtime drives, with one difference that
matters: it performs no I/O itself. Model calls and tool calls are the only execution points,
each a typed serialisable input and a validated result, and each recorded in a `Journal` under
a step id derived from the run's position rather than from a clock or a random. A worker that
dies and restarts replays the loop, finds the completed steps in the journal, and pays for
none of them again.

Nothing here imports a workflow engine. The kit supplies the deterministic driver and the
activity payloads; binding those to Temporal activities belongs to the deployment and to
`tesserix-adk[temporal]`, which is why this module imports cleanly with nothing installed.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
from collections.abc import (  # noqa: TC003 — pydantic needs the runtime types
    Awaitable,
    Callable,
    Mapping,
)
from typing import Any, Protocol, runtime_checkable

from pydantic import model_validator

from tesserix_adk.core.errors import (
    CancelledError,
    ConfigurationError,
    MissingTenantContextError,
    PayloadTooLargeError,
    ProviderUnavailableError,
)
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import ToolCall, Usage
from tesserix_adk.core.provider import (  # noqa: TC001 — pydantic needs the runtime types
    ModelResponse,
    ToolDeclaration,
)

__all__ = [
    "PAYLOAD_LIMIT_BYTES",
    "STREAMING_UNSUPPORTED",
    "Activities",
    "ActivityContext",
    "AgentWorkflow",
    "Cancellation",
    "Journal",
    "ModelCallInput",
    "ModelCallResult",
    "ToolCallInput",
    "ToolCallResult",
    "WorkflowState",
]

PAYLOAD_LIMIT_BYTES = 2_000_000
"""What a Temporal payload carries. Anything larger travels as a handle or not at all."""

STREAMING_UNSUPPORTED = "token streaming is not available on the workflow path"
"""An activity returns once. A run that needs tokens as they arrive stays in process."""

_DEFAULT_ATTEMPTS = 3
_DEFAULT_MAX_ITERATIONS = 8


class ActivityContext(AdkModel):
    """Who an activity is acting for, carried on every input rather than looked up.

    A worker serves every tenant, so an activity that arrives without one cannot be given
    the worker's own identity as a default: that is how one tenant's run reads another's
    data. It fails closed instead.

    Args:
        run_id: The run this activity belongs to.
        tenant: Whose run it is. Required.
        user: Who asked, where a person did.
        scopes: What the run was granted, so an activity cannot widen its own authority.
        trace_id: The trace to attach the activity's span to.
    """

    run_id: str
    tenant: str = ""
    user: str = ""
    scopes: tuple[str, ...] = ()
    trace_id: str = ""

    @model_validator(mode="after")
    def _tenant_is_never_inferred(self) -> ActivityContext:
        """Refuse an input a worker would otherwise complete from its own identity."""
        if not self.tenant:
            raise MissingTenantContextError(
                "an activity input carries its own tenant; a worker's identity is not a default",
                where="activity input",
            )
        return self


class ModelCallInput(AdkModel):
    """One model call, as the activity receives it.

    Args:
        context: Who it is for.
        step: The step id, derived from the run's position so a replay computes the same one.
        attempt: Which attempt this is, from one.
        model: Which model to call.
        history: A handle to the transcript in the run's store. The transcript itself never
            travels: it outgrows the payload limit long before the run ends.
        tools: What the model is told it may call.
    """

    context: ActivityContext
    step: str
    attempt: int = 1
    model: str
    history: str
    tools: tuple[ToolDeclaration, ...] = ()

    @model_validator(mode="after")
    def _history_travels_as_a_handle(self) -> ModelCallInput:
        """Refuse an input with nothing to resolve, rather than calling with no history."""
        if not self.history:
            raise ConfigurationError("a model activity receives a history handle, not a transcript")
        return self


class ModelCallResult(AdkModel):
    """What a model activity returned.

    Args:
        response: The provider's answer, validated.
        history: The handle to the transcript with this turn appended, which the next step
            reads. The activity appends and stores; the workflow only carries the handle.
    """

    response: ModelResponse
    history: str


class ToolCallInput(AdkModel):
    """One tool call, as the activity receives it.

    Args:
        context: Who it is for.
        step: The step id, so a retry of a completed call is served from the journal.
        attempt: Which attempt this is, from one.
        tool: Which tool.
        call_id: The model's own id for the call, which is what the result is matched to.
        arguments: The arguments, as data.
    """

    context: ActivityContext
    step: str
    attempt: int = 1
    tool: str
    call_id: str
    arguments: Mapping[str, Any] = {}


class ToolCallResult(AdkModel):
    """What a tool activity returned.

    Args:
        call_id: Which call this answers.
        content: The result, where it is small enough to travel.
        handle: Where the result was stored, where it was not.
        failed: Whether the tool failed. A failure is a result the model is told about,
            not an exception that loses the run.
        history: The handle to the transcript with the result appended.
    """

    call_id: str
    content: str = ""
    handle: str = ""
    failed: bool = False
    history: str = ""


@runtime_checkable
class Activities(Protocol):
    """The two execution points. Everything else on the workflow path is arithmetic."""

    async def model_call(self, request: ModelCallInput) -> ModelCallResult:
        """Call the model, append the turn to the run's history, return the new handle."""
        ...

    async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
        """Run the tool, append the result to the run's history, return the new handle."""
        ...


@runtime_checkable
class Cancellation(Protocol):
    """What the workflow needs of a cancellation token, satisfied by the runtime's own."""

    def raise_if_cancelled(self) -> None:
        """Raise where the run has been cancelled, so no further activity starts."""
        ...

    async def wait(self) -> str:
        """Suspend until cancelled, so an in-flight activity can be raced against it."""
        ...


class Journal(AdkModel):
    """What is already done, by step id.

    The workflow's memory across a restart. A step found here is not executed again, which
    is what makes a resumed run cost the difference rather than the whole.

    Args:
        models: Completed model calls.
        tools: Completed tool calls.
    """

    models: Mapping[str, ModelCallResult] = {}
    tools: Mapping[str, ToolCallResult] = {}

    def with_model(self, step: str, result: ModelCallResult) -> Journal:
        """Return this journal with one model step recorded."""
        return self.model_copy(update={"models": {**self.models, step: result}})

    def with_tool(self, step: str, result: ToolCallResult) -> Journal:
        """Return this journal with one tool step recorded."""
        return self.model_copy(update={"tools": {**self.tools, step: result}})

    @property
    def steps(self) -> int:
        """How many activities have completed, which is what a resumed run skips."""
        return len(self.models) + len(self.tools)


class WorkflowState(AdkModel):
    """Everything the workflow decides, and nothing it fetches.

    Args:
        run_id: The run.
        history: The current transcript handle.
        iteration: How many model calls have completed.
        usage: The ledger, summed from activity results rather than from a provider client.
        pending_approval: The approval the run is waiting on, where it is waiting.
        grant: The autonomy grant the run is acting under.
        terminal: Empty while running, then `answered`, `exhausted` or `cancelled`.
        answer: The final content, once there is one.
    """

    run_id: str
    history: str
    iteration: int = 0
    usage: Usage = Usage(input_tokens=0, output_tokens=0)
    pending_approval: str = ""
    grant: str = ""
    terminal: str = ""
    answer: str = ""


class AgentWorkflow:
    """The run loop, on the workflow path.

    Args:
        activities: The two execution points, bound by the deployment to whatever runs them.
        model: Which model the run calls.
        tools: What the model is told it may call.
        journal: What is already done. A resumed run is constructed with the journal the
            engine replayed, and executes only what is missing from it.
        max_iterations: How many model calls the run may make before it is exhausted.
        attempts: How many times an unavailable provider is asked again before the run
            fails with the provider's own error, carrying the attempt count.
        token: Cancellation, checked before every activity and raced against each one, so a
            cancelled run stops consuming rather than finishing the call it is inside.

    A worked run, including the restart, is in `examples/durable_run.py`.
    """

    def __init__(
        self,
        *,
        activities: Activities,
        model: str,
        tools: tuple[ToolDeclaration, ...] = (),
        journal: Journal | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        attempts: int = _DEFAULT_ATTEMPTS,
        token: Cancellation | None = None,
    ) -> None:
        self._activities = activities
        self._model = model
        self._tools = tools
        self._max_iterations = max_iterations
        self._attempts = attempts
        self._token = token
        self.journal = journal or Journal()

    async def run(self, state: WorkflowState, *, context: ActivityContext) -> WorkflowState:
        """Drive the run to a terminal state, executing only what the journal lacks.

        Args:
            state: Where the run is. A fresh run passes iteration zero; a resumed one
                passes the state the engine replayed.
            context: Who the run is for, copied onto every activity input.

        Returns:
            The terminal state: the answer, the iteration count and the usage ledger, which
            match a non-durable run over the same provider traffic.

        Raises:
            CancelledError: The run was cancelled. The state stays inspectable.
            ProviderUnavailableError: The provider was still unavailable after `attempts`.
            PayloadTooLargeError: An activity input or result outgrew the transport.
        """
        while state.iteration < self._max_iterations:
            self._stop_if_cancelled()
            step = f"model:{state.iteration}"
            result = await self._model_step(step, state, context)
            state = state.model_copy(
                update={
                    "iteration": state.iteration + 1,
                    "history": result.history,
                    "usage": state.usage + result.response.usage,
                }
            )
            if not result.response.tool_calls:
                return state.model_copy(
                    update={"terminal": "answered", "answer": result.response.content}
                )
            for call in result.response.tool_calls:
                self._stop_if_cancelled()
                tool_step = f"tool:{state.iteration - 1}:{call.id}"
                tool_result = await self._tool_step(tool_step, context, call)
                state = state.model_copy(update={"history": tool_result.history or state.history})
        return state.model_copy(update={"terminal": "exhausted"})

    async def _model_step(
        self, step: str, state: WorkflowState, context: ActivityContext
    ) -> ModelCallResult:
        """One model call, served from the journal where it already completed."""
        recorded = self.journal.models.get(step)
        if recorded is not None:
            return recorded

        def call(attempt: int) -> Awaitable[ModelCallResult]:
            return self._activities.model_call(
                _sized(
                    ModelCallInput(
                        context=context,
                        step=step,
                        attempt=attempt,
                        model=self._model,
                        history=state.history,
                        tools=self._tools,
                    ),
                    kind="input",
                    step=step,
                )
            )

        result = await self._attempted(step, call)
        _sized(result, kind="result", step=step)
        self.journal = self.journal.with_model(step, result)
        return result

    async def _tool_step(
        self, step: str, context: ActivityContext, requested: ToolCall
    ) -> ToolCallResult:
        """One tool call, served from the journal where it already completed."""
        recorded = self.journal.tools.get(step)
        if recorded is not None:
            return recorded

        def call(attempt: int) -> Awaitable[ToolCallResult]:
            return self._activities.tool_call(
                _sized(
                    ToolCallInput(
                        context=context,
                        step=step,
                        attempt=attempt,
                        tool=requested.name,
                        call_id=requested.id,
                        arguments=dict(requested.arguments),
                    ),
                    kind="input",
                    step=step,
                )
            )

        result = await self._attempted(step, call)
        _sized(result, kind="result", step=step)
        self.journal = self.journal.with_tool(step, result)
        return result

    async def _attempted[ResultT](
        self, step: str, activity: Callable[[int], Awaitable[ResultT]]
    ) -> ResultT:
        """Run one activity, retrying an unavailable provider and racing cancellation."""
        unavailable: ProviderUnavailableError | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                return await self._raceable(activity(attempt))
            except ProviderUnavailableError as failure:
                unavailable = failure
                failure.details["attempts"] = str(attempt)
                failure.details["step"] = step
        assert unavailable is not None  # noqa: S101 — the loop only exits by returning or failing
        raise unavailable

    async def _raceable[ResultT](self, work: Awaitable[ResultT]) -> ResultT:
        """Await the activity, cancelling it the moment the run is cancelled.

        Cancellation has to reach the call rather than wait for it: a streaming completion
        nobody is listening to still consumes tokens, and still bills for them.
        """
        if self._token is None:
            return await work
        activity: asyncio.Task[ResultT] = asyncio.ensure_future(work)
        cancelling: asyncio.Task[str] = asyncio.ensure_future(self._token.wait())
        done, _ = await asyncio.wait({activity, cancelling}, return_when=asyncio.FIRST_COMPLETED)
        if activity in done:
            cancelling.cancel()
            return await activity
        activity.cancel()
        raise CancelledError(await cancelling)

    def _stop_if_cancelled(self) -> None:
        """Refuse to start another activity once the run has been cancelled."""
        if self._token is not None:
            self._token.raise_if_cancelled()


def _sized[PayloadT: AdkModel](payload: PayloadT, *, kind: str, step: str) -> PayloadT:
    """Return the payload, or refuse it for being larger than the transport carries."""
    size = len(payload.model_dump_json().encode("utf-8"))
    if size > PAYLOAD_LIMIT_BYTES:
        raise PayloadTooLargeError(
            f"the {kind} for {step} is {size} bytes, over the {PAYLOAD_LIMIT_BYTES} the "
            f"transport carries; store it and pass a handle",
            payload=kind,
            step=step,
            size=size,
            limit=PAYLOAD_LIMIT_BYTES,
        )
    return payload
