"""The run loop: prompt in, exactly one terminal state out.

`AgentRunner.run` always returns a `Run`, and that run's state is always terminal. A
failure is a state, not an escaped exception — the partially built run comes back with
everything recorded so far, because a failure that discards the record leaves nobody able
to say what happened.

Configuration failures are the exception to that, deliberately: an agent that declares a
guardrail the runner was never given is refused before the run starts, since starting
anyway would run it without a check it declared.

`ModelRequest` and `ModelResponse` are owned here until the provider protocol lands its
own types (#49), and `ToolDeclaration` until the tools epic lands a registry (#130).

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tesserix_adk.core import (
    BudgetExceededError,
    CancelledError,
    ConfigurationError,
    DeadlineConfig,
    Message,
    ModelProvider,
    Run,
    RunEvent,
    RunEventKind,
    RunState,
    TextPart,
    ToolCall,
    ToolExecutionError,
    ToolFailurePolicy,
    Usage,
    deduplicate,
    verify_conformance,
)
from tesserix_adk.runtime.cancellation import CancellationToken, Deadline
from tesserix_adk.runtime.prompt import ToolDeclaration, assemble_prompt, wrap_untrusted

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable, Mapping

    from tesserix_adk.core import Agent, BudgetPolicy, Clock, Guardrail, ToolRegistry

__all__ = ["AgentRunner", "ModelRequest", "ModelResponse", "SystemClock"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_MAX_TOOL_RESULT_CHARS = 8_000
_CHARS_PER_TOKEN = 4
_TRUNCATION_MARKER = "\n[truncated]"

# A cancelled coroutine needs a loop turn or two to unwind; only then is the grace window
# the honest measure of whether it is going to stop at all.
_UNWIND_TURNS = 3

_T = TypeVar("_T")


class ModelRequest(BaseModel):
    """One call to a provider, as data.

    Provisional and owned by the runtime until the provider protocol declares its own
    request type (#49).
    """

    model_config = _FROZEN

    model: str = Field(min_length=1)
    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...] = ()
    output_schema: dict[str, Any] | None = None


class ModelResponse(BaseModel):
    """What a provider returned for one call.

    A response with neither content nor tool calls is not retried: asking again for the
    same nothing is how a loop wedges.
    """

    model_config = _FROZEN

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))


class SystemClock:
    """Wall-clock time, for callers that did not inject one."""

    def now(self) -> float:
        """Return Unix seconds."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Suspend for `seconds`."""
        await asyncio.sleep(seconds)


@dataclass(frozen=True, slots=True)
class _Bounds:
    """What limits one run: the caller's switch, the run's instant, the step ceilings."""

    token: CancellationToken
    deadline: Deadline | None
    deadlines: DeadlineConfig


class AgentRunner:
    """Drives one agent to one terminal state.

    Args:
        provider: Where completions come from.
        tools: The registry backing the agent's allowlist. Required if the agent names
            any tool.
        guardrails: Guardrails by name. Every name the agent declares must appear here.
        budget: The spend policy. Required if the agent declares a budget.
        deadlines: Wall-clock ceilings for runs this runner drives. An agent that
            declares its own overrides these; nothing is bounded by default.
        clock: Injected time. Defaults to wall-clock.
        max_iterations: How many model calls one run may make before it is capped.
        max_tool_result_chars: Where an oversized tool result is cut. Truncation is
            recorded as its own event; silently dropping half a result is a wrong answer
            nobody can account for. Compaction rather than cutting is #192.

    Raises:
        ProtocolConformanceError: If a collaborator is missing a member its protocol
            requires, which is a wiring mistake and fails here rather than mid-run.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: ToolRegistry | None = None,
        guardrails: Mapping[str, Guardrail] | None = None,
        budget: BudgetPolicy | None = None,
        deadlines: DeadlineConfig | None = None,
        clock: Clock | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        max_tool_result_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
    ) -> None:
        verify_conformance(provider, ModelProvider)
        self._provider = provider
        self._tools = tools
        self._guardrails = dict(guardrails or {})
        misfiled = [name for name, guardrail in self._guardrails.items() if guardrail.name != name]
        if misfiled:
            raise ConfigurationError(
                f"guardrails filed under a name they do not answer to: "
                f"{', '.join(sorted(misfiled))}. An agent declaring that name would get a "
                f"different check than the one it asked for"
            )
        self._budget = budget
        self._deadlines = deadlines or DeadlineConfig()
        self._clock: Clock = clock or SystemClock()
        self._max_iterations = max_iterations
        self._max_tool_result_chars = max_tool_result_chars
        self._orphans: set[asyncio.Task[Any]] = set()

    def run_sync(
        self,
        agent: Agent,
        user_input: str,
        *,
        tenant: str,
        user: str | None = None,
        run_id: str | None = None,
        history: Iterable[Message] = (),
        memory: Iterable[str] = (),
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
    ) -> Run:
        """Run `agent` from a synchronous caller. Arguments are `run`'s.

        A deliberate wrapper, not an afterthought: not every consumer is async. It drives
        a loop of its own rather than `asyncio.run`, which clears the thread's event loop
        on exit and so would break whatever else had set one.

        Raises:
            RuntimeError: If called from inside a running event loop. Await `run` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.run(
                        agent,
                        user_input,
                        tenant=tenant,
                        user=user,
                        run_id=run_id,
                        history=history,
                        memory=memory,
                        cancellation=cancellation,
                        deadline=deadline,
                    )
                )
            finally:
                loop.close()
        raise RuntimeError("run_sync cannot be called from a running event loop; await run")

    async def run(
        self,
        agent: Agent,
        user_input: str,
        *,
        tenant: str,
        user: str | None = None,
        run_id: str | None = None,
        history: Iterable[Message] = (),
        memory: Iterable[str] = (),
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
    ) -> Run:
        """Drive `agent` until it reaches a terminal state, and return the run.

        Args:
            agent: What to run.
            user_input: What is being asked.
            tenant: The isolation boundary. Every record of the run keys off it.
            user: The acting principal, where there is one.
            run_id: Identity, generated if absent.
            history: Prior conversation, in order.
            memory: Recalled text, handed to the model as untrusted data.
            cancellation: The caller's switch. Flipping it aborts the step in flight and
                stops the run, which resolves `cancelled` rather than raising.
            deadline: A ceiling from the caller, typically a parent run's. It narrows the
                agent's own and never extends it, so a sub-agent cannot outlive its parent.

        Returns:
            The run, in a terminal state, carrying every event recorded on the way.

        Raises:
            ConfigurationError: If the agent declares a collaborator the runner was not
                given, or a task class, which needs a router (#53).
            asyncio.CancelledError: If the surrounding task is cancelled. It is never
                swallowed — a cancelled task that returns normally leaves its canceller
                waiting forever.
        """
        self._refuse_incomplete_wiring(agent)
        bounds = self._bounds_for(agent, cancellation, deadline)
        declared = self._declarations_for(agent)
        prompt = assemble_prompt(agent, user_input, history=history, memory=memory, tools=declared)

        model = agent.model or ""
        run = Run(
            id=run_id or uuid.uuid4().hex,
            tenant=tenant,
            user=user,
            agent_name=agent.name,
            agent_version=agent.version,
            model=model,
            prompt_version=prompt.version,
        ).transition_to(RunState.RUNNING, at=self._clock.now())
        run = run.record_event(self._event(RunEventKind.PROMPT_ASSEMBLED, name=prompt.version))

        messages = list(prompt.messages)
        try:
            for _ in range(self._max_iterations):
                self._stop_if_over(run, bounds)
                request = ModelRequest(
                    model=model,
                    messages=tuple(messages),
                    tools=prompt.tools,
                    output_schema=(
                        agent.output_type.model_json_schema() if agent.output_type else None
                    ),
                )
                run = run.record_event(self._event(RunEventKind.MODEL_CALL, name=model))

                run, response = await self._call_model(run, request, bounds)
                run = await self._settle(run, response, messages)
                if run.state.is_terminal:
                    return run

                run = await self._check_guardrails(run, agent, response, bounds)
                run, done = await self._advance(run, agent, response, messages, bounds)
                if done:
                    return run
        except _Terminal as stop:
            return self._terminate(stop.run, stop.state, stop.detail)

        return self._terminate(
            run,
            RunState.MAX_ITERATIONS_EXCEEDED,
            f"stopped after {self._max_iterations} model calls",
        )

    def _refuse_incomplete_wiring(self, agent: Agent) -> None:
        if agent.task_class:
            raise ConfigurationError(
                f"agent {agent.name!r} selects its model by task_class "
                f"{agent.task_class!r}; this runner has no router (#53), and guessing a "
                f"model would attribute the run to one that never ran it"
            )
        if agent.tools and self._tools is None:
            raise ConfigurationError(
                f"agent {agent.name!r} declares tools ({', '.join(agent.tools)}) but the "
                f"runner was given no registry"
            )
        missing = [name for name in agent.guardrails if name not in self._guardrails]
        if missing:
            raise ConfigurationError(
                f"agent {agent.name!r} declares guardrails the runner was not given: "
                f"{', '.join(missing)}. Running without a declared check is worse than "
                f"not starting"
            )
        if agent.budget is not None and self._budget is None:
            raise ConfigurationError(
                f"agent {agent.name!r} declares a budget but the runner was given no "
                f"budget policy, so the ceiling would not be enforced"
            )

    def _bounds_for(
        self, agent: Agent, token: CancellationToken | None, caller: Deadline | None
    ) -> _Bounds:
        deadlines = agent.deadlines or self._deadlines
        ceiling = (
            Deadline.in_seconds(deadlines.run_seconds, now=self._clock.now())
            if deadlines.run_seconds is not None
            else None
        )
        return _Bounds(
            token=token or CancellationToken(),
            deadline=ceiling.narrowed_to(caller) if ceiling is not None else caller,
            deadlines=deadlines,
        )

    def _stop_if_over(self, run: Run, bounds: _Bounds) -> None:
        """Refuse to start a step nobody is waiting for the answer to.

        Raises:
            _Terminal: If the run was cancelled or its deadline has elapsed.
        """
        if bounds.token.cancelled:
            reason = bounds.token.reason or "cancelled"
            raise _Terminal(
                run.record_event(self._event(RunEventKind.CANCELLATION_REQUESTED, detail=reason)),
                RunState.CANCELLED,
                reason,
            )
        if bounds.deadline is not None and bounds.deadline.expired(self._clock.now()):
            reason = "the run deadline elapsed"
            raise _Terminal(
                run.record_event(self._event(RunEventKind.DEADLINE_EXCEEDED, detail=reason)),
                RunState.CANCELLED,
                reason,
            )

    def _limit(self, bounds: _Bounds, step: float | None) -> float | None:
        """The tighter of a step's own ceiling and what is left of the run's."""
        remaining = (
            bounds.deadline.remaining(self._clock.now()) if bounds.deadline is not None else None
        )
        ceilings = [seconds for seconds in (step, remaining) if seconds is not None]
        return min(ceilings) if ceilings else None

    async def _bounded(
        self,
        work: Coroutine[Any, Any, _T],
        *,
        limit: float | None,
        bounds: _Bounds,
        what: str,
    ) -> _T:
        """Await `work`, racing it against cancellation and against its own ceiling.

        Raises:
            _Aborted: If cancellation or the ceiling won the race. `work` is cancelled
                first and given a grace window; if it does not stop, it is reported
                orphaned rather than waited on.
        """
        if bounds.token.cancelled:
            work.close()
            raise _Aborted(bounds.token.reason or "cancelled", deadline=False, orphaned=False)
        if limit is not None and limit <= 0:
            work.close()
            raise _Aborted(f"{what} had no time left", deadline=True, orphaned=False)

        task: asyncio.Task[_T] = asyncio.ensure_future(work)
        watchers: list[asyncio.Task[Any]] = [asyncio.ensure_future(bounds.token.wait())]
        if limit is not None:
            watchers.append(asyncio.ensure_future(self._clock.sleep(limit)))
        done, _ = await asyncio.wait([task, *watchers], return_when=asyncio.FIRST_COMPLETED)
        for watcher in watchers:
            watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)
        if task in done:
            return task.result()

        expired = not bounds.token.cancelled
        reason = (
            f"{what} exceeded its {limit}s ceiling"
            if expired
            else bounds.token.reason or "cancelled"
        )
        raise _Aborted(reason, deadline=expired, orphaned=not await self._unwind(task, bounds))

    async def _unwind(self, task: asyncio.Task[Any], bounds: _Bounds) -> bool:
        """Cancel `task`, and say whether it actually stopped inside the grace window."""
        task.cancel()
        for _ in range(_UNWIND_TURNS):
            if task.done():
                return True
            await asyncio.sleep(0)

        grace = asyncio.ensure_future(self._clock.sleep(bounds.deadlines.grace_seconds))
        done, _ = await asyncio.wait([task, grace], return_when=asyncio.FIRST_COMPLETED)
        grace.cancel()
        await asyncio.gather(grace, return_exceptions=True)
        if task in done:
            return True

        # Keeping the reference is what makes "orphaned" honest: a dropped task can be
        # destroyed mid-flight, and then nobody can say whether its effect landed.
        self._orphans.add(task)
        task.add_done_callback(self._orphans.discard)
        return False

    def _cancelled(self, run: Run, abort: _Aborted, *, name: str | None = None) -> _Terminal:
        """Turn an aborted step into the one terminal state a cancelled run ends in."""
        requested = RunEventKind.CANCELLATION_REQUESTED
        kind = RunEventKind.DEADLINE_EXCEEDED if abort.deadline else requested
        run = run.record_event(self._event(kind, name=name, detail=abort.reason))
        if abort.orphaned:
            run = run.record_event(
                self._event(
                    RunEventKind.WORK_ORPHANED,
                    name=name,
                    detail="did not stop inside the grace window; still running when abandoned",
                )
            )
        return _Terminal(run, RunState.CANCELLED, abort.reason)

    def _declarations_for(self, agent: Agent) -> tuple[ToolDeclaration, ...]:
        """Only the allowlist is declared: a tool never named cannot be called for."""
        if self._tools is None or not agent.tools:
            return ()
        declared: Iterable[ToolDeclaration] = self._tools.declarations()
        return tuple(tool for tool in declared if tool.name in agent.tools)

    def _event(
        self,
        kind: RunEventKind,
        *,
        name: str | None = None,
        detail: str | None = None,
        usage: Usage | None = None,
    ) -> RunEvent:
        return RunEvent(kind=kind, at=self._clock.now(), name=name, detail=detail, usage=usage)

    def _terminate(self, run: Run, state: RunState, detail: str | None = None) -> Run:
        recorded = run.record_event(self._event(RunEventKind.TERMINATED, detail=detail))
        return recorded.transition_to(state, at=self._clock.now())

    async def _call_model(
        self, run: Run, request: ModelRequest, bounds: _Bounds
    ) -> tuple[Run, ModelResponse]:
        estimate = sum(_length(message) for message in request.messages) // _CHARS_PER_TOKEN
        try:
            if self._budget is not None:
                await self._budget.reserve(estimate)
            response: ModelResponse = await self._bounded(
                self._provider.complete(request),
                limit=self._limit(bounds, bounds.deadlines.model_call_seconds),
                bounds=bounds,
                what="model call",
            )
        except _Aborted as abort:
            raise self._cancelled(run, abort, name=request.model) from None
        except BudgetExceededError as exceeded:
            raise _Terminal(run, RunState.BUDGET_EXHAUSTED, str(exceeded)) from exceeded
        except CancelledError as cancelled:
            raise _Terminal(run, RunState.CANCELLED, str(cancelled)) from cancelled
        except Exception as failure:
            raise _Terminal(run, RunState.FAILED, f"{type(failure).__name__}: {failure}") from (
                failure
            )
        return run, response

    async def _settle(self, run: Run, response: ModelResponse, messages: list[Message]) -> Run:
        """Record what the response cost, and stop if it said nothing at all."""
        run = run.record(response.usage).record_event(
            self._event(
                RunEventKind.MODEL_RESPONSE,
                name=self._provider.name,
                usage=response.usage,
            )
        )
        if self._budget is not None:
            await self._budget.record(response.usage.input_tokens + response.usage.output_tokens)
        if not response.content and not response.tool_calls:
            return self._terminate(
                run, RunState.FAILED, "provider returned no content and no tool calls"
            )
        if response.content:
            messages.append(Message(role="assistant", content=[TextPart(text=response.content)]))
        return _carrying(run, messages)

    async def _check_guardrails(
        self, run: Run, agent: Agent, response: ModelResponse, bounds: _Bounds
    ) -> Run:
        """Guardrails fail closed: a check that did not run is not a check that passed."""
        for name in agent.guardrails:
            guardrail = self._guardrails[name]
            try:
                verdict = await self._bounded(
                    guardrail.check(response.content),
                    limit=self._limit(bounds, None),
                    bounds=bounds,
                    what=f"guardrail {name}",
                )
            except _Aborted as abort:
                raise self._cancelled(run, abort, name=name) from None
            except Exception as failure:
                raise _Terminal(
                    run.record_event(
                        self._event(
                            RunEventKind.GUARDRAIL_REFUSAL,
                            name=name,
                            detail=f"could not evaluate: {failure}",
                        )
                    ),
                    RunState.FAILED,
                    f"guardrail {name} could not be evaluated",
                ) from failure
            if not verdict:
                raise _Terminal(
                    run.record_event(self._event(RunEventKind.GUARDRAIL_REFUSAL, name=name)),
                    RunState.FAILED,
                    f"guardrail {name} refused the response",
                )
        return run

    async def _advance(
        self,
        run: Run,
        agent: Agent,
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> tuple[Run, bool]:
        """Dispatch any tool calls, or finish. Returns the run and whether it is over."""
        if response.tool_calls:
            return await self._dispatch(run, agent, response, messages, bounds), False
        return self._finish(run, agent, response), True

    async def _dispatch(
        self,
        run: Run,
        agent: Agent,
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> Run:
        for call in deduplicate(list(response.tool_calls)):
            if call.name not in agent.tools:
                raise _Terminal(
                    run.record_event(self._event(RunEventKind.TOOL_REFUSED, name=call.name)),
                    RunState.FAILED,
                    f"model called {call.name!r}, which is not on the agent's allowlist",
                )
            run = run.model_copy(update={"tool_calls": [*run.tool_calls, call]})
            run = run.record_event(self._event(RunEventKind.TOOL_CALL, name=call.name))
            run, text, source = await self._invoke(run, agent, call, bounds)
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=[TextPart(text=wrap_untrusted(text, source=source))],
                )
            )
            run = _carrying(run, messages)
        return run

    async def _invoke(
        self, run: Run, agent: Agent, call: ToolCall, bounds: _Bounds
    ) -> tuple[Run, str, str]:
        assert self._tools is not None  # noqa: S101 — guarded by _refuse_incomplete_wiring
        try:
            result = await self._bounded(
                self._tools.invoke(call.name, call.arguments),
                limit=self._limit(bounds, bounds.deadlines.tool_call_seconds),
                bounds=bounds,
                what=f"tool {call.name}",
            )
        except _Aborted as abort:
            stopped = run.record_event(self._indeterminacy(agent, call))
            raise self._cancelled(stopped, abort, name=call.name) from None
        except Exception as failure:
            wrapped = ToolExecutionError(
                f"tool {call.name!r} failed: {failure}", run_id=run.id, tenant=run.tenant
            )
            run = run.record_event(
                self._event(RunEventKind.TOOL_ERROR, name=call.name, detail=str(failure))
            )
            if agent.on_tool_error is ToolFailurePolicy.FAIL_RUN:
                raise _Terminal(run, RunState.FAILED, str(wrapped)) from failure
            return run, f"error: {failure}", "tool_error"

        text = _render(result)
        if len(text) > self._max_tool_result_chars:
            run = run.record_event(
                self._event(
                    RunEventKind.TOOL_RESULT_TRUNCATED,
                    name=call.name,
                    detail=f"{len(text)} chars cut to {self._max_tool_result_chars}",
                )
            )
            text = text[: self._max_tool_result_chars] + _TRUNCATION_MARKER
        return (
            run.record_event(self._event(RunEventKind.TOOL_RESULT, name=call.name)),
            text,
            "tool_result",
        )

    def _indeterminacy(self, agent: Agent, call: ToolCall) -> RunEvent:
        """A tool stopped mid-flight is unknown, not undone — unless it said it is safe to retry."""
        if call.name in agent.idempotent_tools:
            return self._event(
                RunEventKind.TOOL_ERROR,
                name=call.name,
                detail="stopped before it returned; declared idempotent, so safe to retry",
            )
        return self._event(
            RunEventKind.TOOL_INDETERMINATE,
            name=call.name,
            detail="stopped after dispatch; whether its effect landed cannot be known",
        )

    def _finish(self, run: Run, agent: Agent, response: ModelResponse) -> Run:
        if agent.output_type is None:
            return self._terminate(run, RunState.COMPLETED)
        try:
            validated = agent.output_type.model_validate(json.loads(response.content))
        except (ValidationError, ValueError) as violation:
            recorded = run.record_event(
                self._event(RunEventKind.SCHEMA_VIOLATION, detail=str(violation))
            )
            return self._terminate(
                recorded, RunState.FAILED, f"output did not satisfy {agent.output_type.__name__}"
            )
        finished = run.with_output(validated.model_dump(mode="json")).record_event(
            self._event(RunEventKind.OUTPUT_VALIDATED, name=agent.output_type.__name__)
        )
        return self._terminate(finished, RunState.COMPLETED)


class _Aborted(Exception):  # noqa: N818 — control flow, not an error the caller sees
    """One step stopped short: the caller cancelled, or the step ran out of time.

    Args:
        reason: What to record on the run.
        deadline: Whether time ran out, as opposed to a caller cancelling.
        orphaned: Whether the work was still running when the run stopped waiting.
    """

    def __init__(self, reason: str, *, deadline: bool, orphaned: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.deadline = deadline
        self.orphaned = orphaned


class _Terminal(Exception):  # noqa: N818 — control flow, not an error the caller sees
    """Unwinds the loop to one terminal state, carrying the run recorded so far."""

    def __init__(self, run: Run, state: RunState, detail: str | None = None) -> None:
        super().__init__(detail or state)
        self.run = run
        self.state = state
        self.detail = detail


def _carrying(run: Run, messages: list[Message]) -> Run:
    """The conversation belongs on the run: a checkpoint without it cannot be resumed."""
    return run.model_copy(update={"messages": list(messages)})


def _length(message: Message) -> int:
    return sum(len(part.text) for part in message.content if isinstance(part, TextPart))


def _render(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
