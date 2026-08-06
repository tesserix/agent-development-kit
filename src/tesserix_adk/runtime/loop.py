"""The run loop: prompt in, exactly one terminal state out.

`AgentRunner.run` always returns a `Run`, and that run's state is always terminal. A
failure is a state, not an escaped exception — the partially built run comes back with
everything recorded so far, because a failure that discards the record leaves nobody able
to say what happened.

Configuration failures are the exception to that, deliberately: an agent that declares a
guardrail the runner was never given is refused before the run starts, since starting
anyway would run it without a check it declared.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from tesserix_adk.core import (
    AdkError,
    ApprovalDecision,
    ApprovalDeniedError,
    ApprovalExpiredError,
    ApprovalGate,
    ApprovalRecord,
    BinaryPart,
    BudgetExceededError,
    CancelledError,
    Capability,
    ConfigurationError,
    ContextWindowExceededError,
    DeadlineConfig,
    DeclaresEmulation,
    FanOutLimitError,
    HookAction,
    HookChain,
    HookDecision,
    HookEvaluationError,
    HookPoint,
    HookRefusedError,
    HookSubject,
    LoopConfig,
    MaxIterationsError,
    Message,
    ModelProvider,
    ModelResponseError,
    RecursionLimitError,
    RepeatedCallError,
    RetryConfig,
    Run,
    RunContext,
    RunEvent,
    RunEventKind,
    RunState,
    SchemaViolationError,
    TextPart,
    ToolCall,
    ToolExecutionError,
    ToolFailurePolicy,
    Usage,
    deduplicate,
    resolve_hooks,
    verify_conformance,
)
from tesserix_adk.core.provider import ModelRequest, ModelResponse
from tesserix_adk.runtime.cancellation import CancellationToken, Deadline
from tesserix_adk.runtime.prompt import ToolDeclaration, assemble_prompt, wrap_untrusted
from tesserix_adk.runtime.retry import RetryPlan
from tesserix_adk.runtime.structured import OutputContract, unwrap_fenced

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable, Mapping, Sequence
    from random import Random

    from tesserix_adk.core import (
        Agent,
        BudgetPolicy,
        Clock,
        Guardrail,
        Hook,
        IdFactory,
        ToolRegistry,
    )

__all__ = ["AgentRunner", "ModelRequest", "ModelResponse", "SystemClock"]


_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_MAX_TOOL_RESULT_CHARS = 8_000
_CHARS_PER_TOKEN = 4
_TRUNCATION_MARKER = "\n[truncated]"

# A cancelled coroutine needs a loop turn or two to unwind; only then is the grace window
# the honest measure of whether it is going to stop at all.
_UNWIND_TURNS = 3

_T = TypeVar("_T")


def _random_id() -> str:
    """Ids for callers that did not inject a factory."""
    return uuid.uuid4().hex


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
    retry: RetryConfig
    loop: LoopConfig


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
        retry: Which failures are worth another attempt, and how long to wait first. An
            agent that declares its own overrides this; nothing is retried by default.
        loop: Caps on the shape of a run — depth, fan-out, repetition. An agent that
            declares its own narrows these and never widens them.
        hooks: Where policy attaches to the loop. Sealed on the way in: the chain a run
            starts with is the chain it is judged by.
        approvals: Where a held tool call waits for a human decision. Required if any
            agent this runner drives declares `approval_required_tools`.
        approval_ttl_seconds: How long a request stays answerable. A decision outside the
            window is refused, because an approval is permission at a moment rather than
            a standing licence. Unbounded by default.
        jitter: The source the backoff is drawn from. Injected so a test can seed it and
            assert the exact schedule instead of waiting it out.
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
        retry: RetryConfig | None = None,
        loop: LoopConfig | None = None,
        hooks: HookChain | Iterable[Hook] | None = None,
        approvals: ApprovalGate | None = None,
        approval_ttl_seconds: float | None = None,
        jitter: Random | None = None,
        clock: Clock | None = None,
        ids: IdFactory | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        max_tool_result_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
    ) -> None:
        verify_conformance(provider, ModelProvider)
        self._provider = provider
        if tools is not None:
            provider.capabilities.require(
                Capability.TOOL_CALLING, provider=provider.name, model="<any>"
            )
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
        self._retry = retry or RetryConfig()
        self._loop = loop or LoopConfig()
        self._hooks = (
            hooks.sealed() if isinstance(hooks, HookChain) else HookChain(hooks or ()).sealed()
        )
        self._approvals = approvals
        self._approval_ttl = approval_ttl_seconds
        self._jitter = jitter
        self._clock: Clock = clock or SystemClock()
        self._ids: IdFactory = ids or _random_id
        self._max_iterations = max_iterations
        self._max_tool_result_chars = max_tool_result_chars
        self._orphans: set[asyncio.Task[Any]] = set()

    def run_sync[OutputT: BaseModel](
        self,
        agent: Agent[OutputT],
        user_input: str,
        *,
        tenant: str,
        user: str | None = None,
        run_id: str | None = None,
        history: Iterable[Message] = (),
        memory: Iterable[str] = (),
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
        parent: RunContext | None = None,
    ) -> Run[OutputT]:
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
                        parent=parent,
                    )
                )
            finally:
                loop.close()
        raise RuntimeError("run_sync cannot be called from a running event loop; await run")

    async def run[OutputT: BaseModel](
        self,
        agent: Agent[OutputT],
        user_input: str,
        *,
        tenant: str,
        user: str | None = None,
        run_id: str | None = None,
        history: Iterable[Message] = (),
        memory: Iterable[str] = (),
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
        parent: RunContext | None = None,
    ) -> Run[OutputT]:
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
            parent: The context of the run that called this one, where one did. It carries
                the depth, so a chain of agents calling agents cannot outrun its ceiling.

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
        depth = parent.depth + 1 if parent is not None else 0
        if depth > bounds.loop.max_depth:
            return await self._too_deep(
                agent, depth, bounds, tenant=tenant, user=user, run_id=run_id
            )
        model = agent.model or ""
        run: Run[Any] = _carrier_for(agent.output_type)(
            id=run_id or self._ids(),
            tenant=tenant,
            user=user,
            agent_name=agent.name,
            agent_version=agent.version,
            model=model,
            depth=depth,
        ).transition_to(RunState.RUNNING, at=self._clock.now())

        try:
            run, asked = await self._asked(run, agent, user_input, bounds)
            contract = self._contract_for(agent)
            prompt = assemble_prompt(
                agent,
                asked,
                history=history,
                memory=memory,
                tools=self._declarations_for(agent),
                output=contract,
            )
            run = run.model_copy(update={"prompt_version": prompt.version}).record_event(
                self._event(RunEventKind.PROMPT_ASSEMBLED, name=prompt.version)
            )
            messages = list(prompt.messages)
            for _ in range(self._max_iterations):
                self._stop_if_over(run, bounds)
                run, messages = await self._before_the_call(run, agent, messages, bounds)
                self._refuse_unreadable_prompt(messages, model)
                request = ModelRequest(
                    model=model,
                    messages=tuple(messages),
                    tools=prompt.tools,
                    output_schema=contract.schema
                    if contract is not None and contract.native
                    else None,
                    output_schema_hash=contract.hash if contract is not None else None,
                )
                run = run.record_event(self._event(RunEventKind.MODEL_CALL, name=model))

                run, response = await self._call_model(run, request, bounds)
                run, response = await self._after_the_response(run, agent, response, bounds)
                run = await self._settle(run, agent, response, messages, bounds)
                if run.state.is_terminal:
                    return run

                run = await self._check_guardrails(run, agent, response, bounds)
                run, done = await self._advance(run, agent, response, messages, bounds)
                if done:
                    return run
        except _Terminal as stop:
            return await self._terminate(stop.run, agent, stop.state, stop.detail, bounds)

        return await self._terminate(
            run,
            agent,
            RunState.MAX_ITERATIONS_EXCEEDED,
            _named(MaxIterationsError(f"stopped after {self._max_iterations} model calls")),
            bounds,
        )

    async def _asked(
        self, run: Run[Any], agent: Agent[Any], user_input: str, bounds: _Bounds
    ) -> tuple[Run[Any], str]:
        """What is actually asked, after policy has had its say about it."""
        run, asked, decision, _ = await self._ask_hooks(
            run, agent, HookPoint.BEFORE_PROMPT_ASSEMBLY, bounds, content=user_input
        )
        self._stop_on_refusal(run, decision)
        return run, asked

    async def _before_the_call(
        self, run: Run[Any], agent: Agent[Any], messages: list[Message], bounds: _Bounds
    ) -> tuple[Run[Any], list[Message]]:
        """Policy sees what is about to go upstream, and may redact it before it does."""
        run, rewritten, decision, _ = await self._ask_hooks(
            run, agent, HookPoint.BEFORE_MODEL_CALL, bounds, content=_text_of(messages[-1])
        )
        self._stop_on_refusal(run, decision)
        if rewritten != _text_of(messages[-1]):
            messages = [*messages[:-1], _retexted(messages[-1], rewritten)]
        return run, messages

    async def _after_the_response(
        self, run: Run[Any], agent: Agent[Any], response: ModelResponse, bounds: _Bounds
    ) -> tuple[Run[Any], ModelResponse]:
        run, rewritten, decision, _ = await self._ask_hooks(
            run, agent, HookPoint.AFTER_MODEL_RESPONSE, bounds, content=response.content
        )
        self._stop_on_refusal(run, decision)
        if rewritten != response.content:
            response = response.model_copy(update={"content": rewritten})
        return run, response

    async def _ask_hooks(
        self,
        run: Run[Any],
        agent: Agent[Any],
        point: HookPoint,
        bounds: _Bounds,
        *,
        content: str = "",
        tool_name: str | None = None,
        tool_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Run[Any], str, HookDecision, str]:
        """Ask every hook at `point`, in declaration order, and resolve what they said.

        Every hook is asked even after one refuses: stopping at the first refusal hides
        the second, and a chain that reports different things on different runs is a
        policy nobody can audit.
        """
        verdicts: list[tuple[str, HookDecision]] = []
        for hook in self._hooks.at(point):
            subject = HookSubject(
                point=point,
                run_id=run.id,
                tenant=run.tenant,
                user=run.user,
                agent_name=agent.name,
                content=content,
                tool_name=tool_name,
                tool_arguments=dict(tool_arguments or {}),
            )
            decision = await self._ask(run, hook, subject, bounds)
            verdicts.append((hook.name, decision))
            if decision.action is HookAction.REWRITE and decision.replacement is not None:
                run = run.record_event(
                    self._event(
                        RunEventKind.HOOK_REWRITE,
                        name=hook.name,
                        detail=f"{point}: {_short(content)} → {_short(decision.replacement)}",
                    )
                )
                content = decision.replacement
        held = resolve_hooks(decision for _, decision in verdicts)
        by = next((name for name, decision in verdicts if decision == held), "")
        if held.action is not HookAction.CONTINUE and held.action is not HookAction.REWRITE:
            run = run.record_event(
                self._event(RunEventKind.HOOK_REFUSAL, name=by, detail=held.reason)
            )
        return run, content, held, by

    async def _ask(
        self, run: Run[Any], hook: Hook, subject: HookSubject, bounds: _Bounds
    ) -> HookDecision:
        """One hook, bounded and fail-closed: a check that did not finish did not pass."""
        try:
            decision: HookDecision = await self._bounded(
                hook.on(subject),
                limit=self._limit(bounds, bounds.deadlines.hook_seconds),
                bounds=bounds,
                what=f"hook {hook.name}",
            )
        except _Aborted as abort:
            raise self._cancelled(run, abort, name=hook.name) from None
        except Exception as failure:
            raise _Terminal(
                run.record_event(
                    self._event(
                        RunEventKind.HOOK_REFUSAL,
                        name=hook.name,
                        detail=f"could not be evaluated: {failure}",
                    )
                ),
                RunState.FAILED,
                _named(HookEvaluationError(f"hook {hook.name!r} could not be evaluated")),
            ) from failure
        else:
            return decision

    def _stop_on_refusal(self, run: Run[Any], decision: HookDecision) -> None:
        """Refuse the step, or read a request for approval as the refusal it amounts to.

        Anywhere but a tool dispatch there is no call to hold.
        """
        if decision.action in (HookAction.REFUSE, HookAction.REQUIRE_APPROVAL):
            raise _Terminal(run, RunState.FAILED, _named(HookRefusedError(decision.reason)))

    def _refuse_incomplete_wiring(self, agent: Agent[Any]) -> None:
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
        if agent.tools:
            self._require(Capability.TOOL_CALLING, model=agent.model or "")
        missing = [name for name in agent.guardrails if name not in self._guardrails]
        if missing:
            raise ConfigurationError(
                f"agent {agent.name!r} declares guardrails the runner was not given: "
                f"{', '.join(missing)}. Running without a declared check is worse than "
                f"not starting"
            )
        if agent.approval_required_tools and self._approvals is None:
            raise ConfigurationError(
                f"agent {agent.name!r} requires approval for "
                f"({', '.join(agent.approval_required_tools)}) but the runner was given no "
                f"approval gate, so the call would go out unapproved"
            )
        if agent.budget is not None and self._budget is None:
            raise ConfigurationError(
                f"agent {agent.name!r} declares a budget but the runner was given no "
                f"budget policy, so the ceiling would not be enforced"
            )

    def _require(self, capability: Capability, *, model: str) -> None:
        """Check the provider's own record. Raises `CapabilityError` naming all three."""
        self._provider.capabilities.require(capability, provider=self._provider.name, model=model)

    def _refuse_unreadable_prompt(self, messages: Sequence[Message], model: str) -> None:
        """Refuse a prompt past the declared window rather than letting the vendor cut it.

        Raises:
            CapabilityError: If any message carries an image and the model cannot see.
            ContextWindowExceededError: If the provider's own count is over its own window.
        """
        if any(isinstance(part, BinaryPart) for message in messages for part in message.content):
            self._require(Capability.VISION, model=model)
        window = self._provider.capabilities.context_window_tokens
        if window is None:
            return
        counted = self._provider.count_tokens(messages)
        if counted > window:
            raise ContextWindowExceededError(
                f"the prompt counts {counted} tokens against {self._provider.name}:{model}'s "
                f"declared window of {window}. Sending it would have the vendor truncate it "
                f"and answer anyway",
                counted=counted,
                limit=window,
                provider=self._provider.name,
                model=model,
            )

    def _bounds_for(
        self, agent: Agent[Any], token: CancellationToken | None, caller: Deadline | None
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
            retry=agent.retry or self._retry,
            loop=self._loop.narrowed_to(agent.loop),
        )

    def _stop_if_over(self, run: Run[Any], bounds: _Bounds) -> None:
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

    def _cancelled(self, run: Run[Any], abort: _Aborted, *, name: str | None = None) -> _Terminal:
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

    def _declarations_for(self, agent: Agent[Any]) -> tuple[ToolDeclaration, ...]:
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

    async def _terminate(
        self, run: Run[Any], agent: Agent[Any], state: RunState, detail: str | None, bounds: _Bounds
    ) -> Run[Any]:
        run = await self._notify(run, agent, state, bounds)
        recorded = run.record_event(self._event(RunEventKind.TERMINATED, detail=detail))
        return recorded.transition_to(state, at=self._clock.now())

    async def _notify(
        self, run: Run[Any], agent: Agent[Any], state: RunState, bounds: _Bounds
    ) -> Run[Any]:
        """Tell the terminal hooks how it ended.

        The only point where a hook is not fail-closed, because there is nothing left to
        close: the run is already over, and a policy that raised here cannot un-run it.
        The failure is recorded so nobody reads silence as approval.
        """
        for hook in self._hooks.at(HookPoint.ON_TERMINAL):
            subject = HookSubject(
                point=HookPoint.ON_TERMINAL,
                run_id=run.id,
                tenant=run.tenant,
                user=run.user,
                agent_name=agent.name,
                state=state,
            )
            try:
                await self._bounded(
                    hook.on(subject),
                    limit=self._limit(bounds, bounds.deadlines.hook_seconds),
                    bounds=bounds,
                    what=f"hook {hook.name}",
                )
            except (_Aborted, Exception) as failure:
                run = run.record_event(
                    self._event(
                        RunEventKind.HOOK_REFUSAL,
                        name=hook.name,
                        detail=f"could not be evaluated after the run ended: {failure}",
                    )
                )
        return run

    async def _call_model(
        self, run: Run[Any], request: ModelRequest, bounds: _Bounds
    ) -> tuple[Run[Any], ModelResponse]:
        estimate = sum(_length(message) for message in request.messages) // _CHARS_PER_TOKEN
        plan = RetryPlan(bounds.retry, random=self._jitter)
        attempt = 1
        while True:
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
                run, delay = self._after_failure(run, plan, attempt, failure, bounds, request.model)
                if delay is None:
                    raise _Terminal(
                        run, RunState.FAILED, f"{type(failure).__name__}: {failure}"
                    ) from failure
                await self._backoff(run, delay, bounds, name=request.model)
                run = run.record_event(self._event(RunEventKind.MODEL_CALL, name=request.model))
                attempt += 1
            else:
                return run, self._readable(response)

    def _readable(self, payload: object) -> ModelResponse:
        """Refuse an answer that is not one.

        Distinct from a schema violation, which is a well-formed answer in the wrong shape
        and can be repaired: this is a provider implementation fault, and repairing it
        would mean guessing what it meant.

        Raises:
            ModelResponseError: If `payload` is not a `ModelResponse`.
        """
        if not isinstance(payload, ModelResponse):
            raise ModelResponseError(
                f"{self._provider.name} returned {type(payload).__name__}, not a ModelResponse",
                payload=payload,
                provider=self._provider.name,
            )
        return payload

    def _after_failure(
        self,
        run: Run[Any],
        plan: RetryPlan,
        attempt: int,
        failure: Exception,
        bounds: _Bounds,
        name: str,
    ) -> tuple[Run[Any], float | None]:
        """Record the failed attempt, and say how long to wait or why there is no next one."""
        if not plan.retryable(failure):
            delay, why = None, "not retryable, so the same request is not sent again"
        elif (delay := plan.delay_for(attempt, retry_after=_retry_after(failure))) is None:
            why = (
                "the provider asked to wait longer than the policy honours"
                if _retry_after(failure) is not None
                else "no attempts left"
            )
        elif not self._fits(delay, bounds):
            delay, why = None, "no time left in the run to wait for another attempt"
        else:
            why = f"retrying in {delay:.2f}s"
        detail = f"attempt {attempt}: {type(failure).__name__}: {failure} — {why}"
        return (
            run.record_event(self._event(RunEventKind.ATTEMPT_FAILED, name=name, detail=detail)),
            delay,
        )

    def _fits(self, delay: float, bounds: _Bounds) -> bool:
        """A backoff that would land past the deadline is not a backoff, it is a stall."""
        return bounds.deadline is None or delay <= bounds.deadline.remaining(self._clock.now())

    async def _backoff(self, run: Run[Any], delay: float, bounds: _Bounds, *, name: str) -> None:
        try:
            await self._bounded(
                self._clock.sleep(delay), limit=None, bounds=bounds, what="retry backoff"
            )
        except _Aborted as abort:
            raise self._cancelled(run, abort, name=name) from None

    async def _settle(
        self,
        run: Run[Any],
        agent: Agent[Any],
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> Run[Any]:
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
            return await self._terminate(
                run,
                agent,
                RunState.FAILED,
                "provider returned no content and no tool calls",
                bounds,
            )
        if response.content:
            # A field of a structured answer can hold retrieved text; echoed back bare it
            # becomes instruction on the next turn, which is the injection this prevents.
            echoed = (
                response.content
                if agent.free_text
                else wrap_untrusted(response.content, source="model_output")
            )
            messages.append(Message(role="assistant", content=[TextPart(text=echoed)]))
        return _carrying(run, messages)

    async def _check_guardrails(
        self, run: Run[Any], agent: Agent[Any], response: ModelResponse, bounds: _Bounds
    ) -> Run[Any]:
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
        run: Run[Any],
        agent: Agent[Any],
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> tuple[Run[Any], bool]:
        """Dispatch any tool calls, or finish. Returns the run and whether it is over."""
        if response.tool_calls:
            return await self._dispatch(run, agent, response, messages, bounds), False
        return await self._finish(run, agent, response, messages, bounds)

    async def _dispatch(
        self,
        run: Run[Any],
        agent: Agent[Any],
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> Run[Any]:
        calls = deduplicate(list(response.tool_calls))
        self._refuse_a_turn_that_would_break_a_cap(run, agent, calls, bounds)
        messages.append(
            Message(
                role="assistant",
                content=[TextPart(text=response.content)] if response.content else [],
                tool_calls=tuple(calls),
            )
        )
        for call in calls:
            if call.name not in agent.tools:
                raise _Terminal(
                    run.record_event(self._event(RunEventKind.TOOL_REFUSED, name=call.name)),
                    RunState.FAILED,
                    f"model called {call.name!r}, which is not on the agent's allowlist",
                )
            run = await self._cleared_to_dispatch(run, agent, call, bounds)
            run = run.model_copy(update={"tool_calls": [*run.tool_calls, call]})
            run = run.record_event(self._event(RunEventKind.TOOL_CALL, name=call.name))
            run, text, source = await self._invoke(run, agent, call, bounds)
            run, text, decision, _ = await self._ask_hooks(
                run,
                agent,
                HookPoint.AFTER_TOOL_RESULT,
                bounds,
                content=text,
                tool_name=call.name,
                tool_arguments=call.arguments,
            )
            self._stop_on_refusal(run, decision)
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=[TextPart(text=wrap_untrusted(text, source=source))],
                )
            )
            run = _carrying(run, messages)
        return run

    def _refuse_a_turn_that_would_break_a_cap(
        self, run: Run[Any], agent: Agent[Any], calls: Sequence[ToolCall], bounds: _Bounds
    ) -> None:
        """Check the whole turn before dispatching any of it.

        Half a fan-out is a set of side effects nobody chose, so a turn that would break a
        cap is refused entire rather than trimmed to fit.

        Raises:
            _Terminal: If the turn is wider than a cap allows, or repeats a call the run
                has already made past its threshold.
        """
        caps = bounds.loop
        made = len(run.tool_calls)
        too_wide = (
            f"turn asked for {len(calls)} tool calls, over the {caps.max_tool_calls_per_turn} "
            f"allowed in one turn"
            if len(calls) > caps.max_tool_calls_per_turn
            else (
                f"turn would take the run to {made + len(calls)} tool calls, over the "
                f"{caps.max_tool_calls_per_run} allowed in one run"
                if made + len(calls) > caps.max_tool_calls_per_run
                else None
            )
        )
        if too_wide is not None:
            raise _Terminal(
                run.record_event(self._event(RunEventKind.FAN_OUT_REFUSED, detail=too_wide)),
                RunState.LOOP_LIMIT_EXCEEDED,
                _named(FanOutLimitError(too_wide)),
            )

        for position, call in enumerate(calls):
            if call.name in agent.idempotent_tools:
                continue
            repeats = _same_call_count(run.tool_calls, call) + _same_call_count(
                calls[:position], call
            )
            if repeats >= caps.max_repeated_calls:
                detail = (
                    f"{call.name} asked for with the same arguments {repeats + 1} times; the "
                    f"run is going round rather than getting anywhere"
                )
                raise _Terminal(
                    run.record_event(
                        self._event(RunEventKind.REPEAT_DETECTED, name=call.name, detail=detail)
                    ),
                    RunState.LOOP_LIMIT_EXCEEDED,
                    _named(RepeatedCallError(detail)),
                )

    async def _too_deep(
        self,
        agent: Agent[Any],
        depth: int,
        bounds: _Bounds,
        *,
        tenant: str,
        user: str | None,
        run_id: str | None,
    ) -> Run[Any]:
        """End a run that was called from too deep a chain, before it calls anything itself."""
        detail = (
            f"run would sit at depth {depth}, past the ceiling of {bounds.loop.max_depth}; "
            f"agents calling agents have stopped making progress"
        )
        run = Run(
            id=run_id or self._ids(),
            tenant=tenant,
            user=user,
            agent_name=agent.name,
            agent_version=agent.version,
            model=agent.model or "",
            depth=depth,
        ).transition_to(RunState.RUNNING, at=self._clock.now())
        run = run.record_event(self._event(RunEventKind.DEPTH_EXCEEDED, detail=detail))
        return await self._terminate(
            run, agent, RunState.LOOP_LIMIT_EXCEEDED, _named(RecursionLimitError(detail)), bounds
        )

    async def _cleared_to_dispatch(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, bounds: _Bounds
    ) -> Run[Any]:
        """Policy and, where it is required, a human, before anything goes out."""
        run, _, decision, _ = await self._ask_hooks(
            run,
            agent,
            HookPoint.BEFORE_TOOL_DISPATCH,
            bounds,
            tool_name=call.name,
            tool_arguments=call.arguments,
        )
        if decision.action is HookAction.REFUSE:
            raise _Terminal(run, RunState.FAILED, _named(HookRefusedError(decision.reason)))
        declared = call.name in agent.approval_required_tools
        if not declared and decision.action is not HookAction.REQUIRE_APPROVAL:
            return run
        reason = (
            decision.reason
            if decision.action is HookAction.REQUIRE_APPROVAL
            else f"{call.name} is declared to require approval"
        )
        return await self._approved(run, agent, call, reason, bounds)

    async def _approved(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, reason: str, bounds: _Bounds
    ) -> Run[Any]:
        """Hold the call until a human decides, and record what they decided.

        The record carries a digest of the arguments rather than the arguments: an
        approval queue outlives the run and is read by people who are not party to it.
        """
        if self._approvals is None:
            raise ConfigurationError(
                f"a hook required approval for {call.name!r} but the runner was given no "
                f"approval gate, so the call would go out unapproved"
            )
        record = ApprovalRecord.for_call(
            run_id=run.id,
            tenant=run.tenant,
            agent_name=agent.name,
            tool_name=call.name,
            arguments=call.arguments,
            reason=reason,
            requested_at=self._clock.now(),
        )
        run = run.record_event(
            self._event(RunEventKind.APPROVAL_REQUIRED, name=call.name, detail=reason)
        )
        try:
            decision = await self._bounded(
                self._approvals.request(record),
                limit=self._limit(bounds, None),
                bounds=bounds,
                what=f"approval for {call.name}",
            )
        except _Aborted as abort:
            raise self._cancelled(run, abort, name=call.name) from None
        except Exception as failure:
            raise _Terminal(
                run,
                RunState.FAILED,
                _named(ApprovalDeniedError(f"approval for {call.name!r} could not be obtained")),
            ) from failure
        return self._honoured(run, record, decision, call)

    def _honoured(
        self, run: Run[Any], record: ApprovalRecord, decision: ApprovalDecision, call: ToolCall
    ) -> Run[Any]:
        """An answer only clears the call it answers, and only while it is current."""
        if decision.record_id != record.id:
            raise _Terminal(
                self._denied(run, call, "the decision answers a different request"),
                RunState.FAILED,
                _named(ApprovalDeniedError(f"approval for {call.name!r} was never given")),
            )
        if self._approval_ttl is not None and not (
            record.requested_at <= decision.decided_at <= record.requested_at + self._approval_ttl
        ):
            raise _Terminal(
                self._denied(run, call, "the decision arrived outside the request's window"),
                RunState.FAILED,
                _named(
                    ApprovalExpiredError(
                        f"approval for {call.name!r} was decided outside its window; permission "
                        f"at a moment is not a standing licence"
                    )
                ),
            )
        if not decision.granted:
            raise _Terminal(
                self._denied(run, call, decision.reason or f"declined by {decision.decided_by}"),
                RunState.FAILED,
                _named(ApprovalDeniedError(f"approval for {call.name!r} was declined")),
            )
        return run.record_event(
            self._event(
                RunEventKind.APPROVAL_GRANTED, name=call.name, detail=f"by {decision.decided_by}"
            )
        )

    def _denied(self, run: Run[Any], call: ToolCall, why: str) -> Run[Any]:
        return run.record_event(
            self._event(RunEventKind.APPROVAL_DENIED, name=call.name, detail=why)
        )

    async def _invoke(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, bounds: _Bounds
    ) -> tuple[Run[Any], str, str]:
        assert self._tools is not None  # noqa: S101 — guarded by _refuse_incomplete_wiring
        await self._reserve(run, len(_signature(call)) // _CHARS_PER_TOKEN)
        plan = RetryPlan(bounds.retry, random=self._jitter)
        attempt = 1
        while True:
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
                delay = self._tool_backoff(agent, call, plan, attempt, bounds)
                if delay is None:
                    return self._tool_failed(run, agent, call, failure, bounds)
                run = run.record_event(
                    self._event(
                        RunEventKind.ATTEMPT_FAILED,
                        name=call.name,
                        detail=f"attempt {attempt}: {failure} — retrying in {delay:.2f}s",
                    )
                )
                await self._backoff(run, delay, bounds, name=call.name)
                attempt += 1
            else:
                break

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
        if self._budget is not None:
            await self._budget.record(len(text) // _CHARS_PER_TOKEN)
        return (
            run.record_event(self._event(RunEventKind.TOOL_RESULT, name=call.name)),
            text,
            "tool_result",
        )

    async def _reserve(self, run: Run[Any], estimate: int) -> None:
        """Spend is checked before it is incurred; reported after, it is a bill."""
        if self._budget is None:
            return
        try:
            await self._budget.reserve(estimate)
        except BudgetExceededError as exceeded:
            raise _Terminal(run, RunState.BUDGET_EXHAUSTED, str(exceeded)) from exceeded

    def _tool_backoff(
        self, agent: Agent[Any], call: ToolCall, plan: RetryPlan, attempt: int, bounds: _Bounds
    ) -> float | None:
        """A tool is retried on its declaration, not on the shape of its exception.

        A tool's exception says nothing about whether its side effect landed, so the only
        safe gate is the agent naming the tool as safe to call again.
        """
        if call.name not in agent.idempotent_tools:
            return None
        delay = plan.delay_for(attempt)
        return delay if delay is not None and self._fits(delay, bounds) else None

    def _tool_failed(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, failure: Exception, bounds: _Bounds
    ) -> tuple[Run[Any], str, str]:
        wrapped = ToolExecutionError(
            f"tool {call.name!r} failed: {failure}", run_id=run.id, tenant=run.tenant
        )
        unretried = (
            "; not declared idempotent, so it was not tried again"
            if bounds.retry.max_attempts > 1 and call.name not in agent.idempotent_tools
            else ""
        )
        run = run.record_event(
            self._event(RunEventKind.TOOL_ERROR, name=call.name, detail=f"{failure}{unretried}")
        )
        if agent.on_tool_error is ToolFailurePolicy.FAIL_RUN:
            raise _Terminal(run, RunState.FAILED, str(wrapped)) from failure
        return run, f"error: {failure}", "tool_error"

    def _indeterminacy(self, agent: Agent[Any], call: ToolCall) -> RunEvent:
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

    def _contract_for(self, agent: Agent[Any]) -> OutputContract | None:
        """The shape of the answer, and whether this provider enforces it itself.

        A provider that has not declared the capability does not have it: assuming it does
        means discovering the schema was ignored from a run that already completed.
        """
        if agent.output_type is None:
            return None
        native = self._provider.capabilities.supports(Capability.STRUCTURED_OUTPUT)
        if not native and not _emulation_allowed(self._provider):
            self._require(Capability.STRUCTURED_OUTPUT, model=agent.model or "")
        return OutputContract.of(agent.output_type, native=native)

    async def _finish(
        self,
        run: Run[Any],
        agent: Agent[Any],
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> tuple[Run[Any], bool]:
        """Validate the answer. Returns the run and whether it is over, so a repair goes on."""
        run, content, decision, _ = await self._ask_hooks(
            run, agent, HookPoint.BEFORE_OUTPUT_VALIDATION, bounds, content=response.content
        )
        self._stop_on_refusal(run, decision)
        contract = self._contract_for(agent)
        if contract is None:
            return await self._terminate(run, agent, RunState.COMPLETED, None, bounds), True

        answer, unwrapped = unwrap_fenced(content)
        if unwrapped:
            run = run.record_event(
                self._event(
                    RunEventKind.OUTPUT_UNWRAPPED, detail="stripped an enclosing code fence"
                )
            )
        try:
            validated = contract.parse(answer)
        except SchemaViolationError as violation:
            return await self._repair_or_fail(run, agent, contract, violation, messages, bounds)
        finished = run.with_output(validated).record_event(
            self._event(RunEventKind.OUTPUT_VALIDATED, name=contract.output_type.__name__)
        )
        return await self._terminate(finished, agent, RunState.COMPLETED, None, bounds), True

    async def _repair_or_fail(
        self,
        run: Run[Any],
        agent: Agent[Any],
        contract: OutputContract,
        violation: SchemaViolationError,
        messages: list[Message],
        bounds: _Bounds,
    ) -> tuple[Run[Any], bool]:
        """Send the failure back where that is allowed and still worth doing, else stop."""
        recorded = self._violation_event(contract, violation)
        repeated = recorded.detail is not None and recorded.detail == self._last_violation(run)
        run = run.record_event(recorded)
        policy = agent.repair
        attempts = sum(1 for e in run.events if e.kind is RunEventKind.REPAIR_REQUESTED)
        if policy is None or not policy.enabled or attempts >= policy.max_attempts:
            stopped = await self._terminate(run, agent, RunState.FAILED, _named(violation), bounds)
            return stopped, True
        if repeated:
            # Told exactly what was wrong and answering identically: the ask is the defect.
            return await self._terminate(
                run.record_event(
                    self._event(
                        RunEventKind.REPAIR_ABANDONED,
                        name=contract.output_type.__name__,
                        detail=(
                            "the same failure after being told what it was; the declared "
                            "constraint cannot be satisfied as instructed"
                        ),
                    )
                ),
                agent,
                RunState.FAILED,
                _named(
                    ConfigurationError(
                        f"repair made no progress on {contract.output_type.__name__}: the "
                        f"same fields failed after the failure was fed back"
                    )
                ),
                bounds,
            ), True
        messages.append(
            Message(role="user", content=[TextPart(text=contract.repair_prompt(violation))])
        )
        fields = ", ".join(violation.paths) or "whole output"
        run = run.record_event(
            self._event(
                RunEventKind.REPAIR_REQUESTED,
                name=contract.output_type.__name__,
                detail=f"attempt {attempts + 1} of {policy.max_attempts} at {fields}",
            )
        )
        return _carrying(run, messages), False

    def _last_violation(self, run: Run[Any]) -> str | None:
        """The detail of the previous violation, which is what a stalled repair repeats."""
        details = [e.detail for e in run.events if e.kind is RunEventKind.SCHEMA_VIOLATION]
        return details[-1] if details else None

    def _violation_event(
        self, contract: OutputContract, violation: SchemaViolationError
    ) -> RunEvent:
        """What failed and against which schema — never the output, which may be anything."""
        fields = ", ".join(violation.paths) or "whole output"
        return self._event(
            RunEventKind.SCHEMA_VIOLATION,
            name=contract.output_type.__name__,
            detail=f"{violation} at {fields}; schema {contract.hash}",
        )


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

    def __init__(self, run: Run[Any], state: RunState, detail: str | None = None) -> None:
        super().__init__(detail or state)
        self.run = run
        self.state = state
        self.detail = detail


def _named(error: AdkError) -> str:
    """A terminal detail that says which cap or fault ended the run, not just that one did."""
    return f"{type(error).__name__}: {error}"


def _same_call_count(calls: Iterable[ToolCall], call: ToolCall) -> int:
    """How many of `calls` are the same request: same tool, same arguments in any order."""
    signature = _signature(call)
    return sum(1 for made in calls if _signature(made) == signature)


def _signature(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=repr)}"


def _text_of(message: Message) -> str:
    """The text a hook sees and may rewrite; non-text parts are not a hook's business."""
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _retexted(message: Message, text: str) -> Message:
    return message.model_copy(update={"content": [TextPart(text=text)]})


def _short(text: str) -> str:
    """A digest of content, so a rewrite is reproducible without the content in the log."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _carrier_for(output_type: type[BaseModel] | None) -> type[Run[Any]]:
    """The run class to build, parameterised: a bare `Run` serialises a typed answer to `{}`."""
    if output_type is None:
        return Run
    return cast("type[Run[Any]]", Run.__class_getitem__(output_type))


def _carrying[OutputT: BaseModel](run: Run[OutputT], messages: list[Message]) -> Run[OutputT]:
    """The conversation belongs on the run: a checkpoint without it cannot be resumed."""
    return run.model_copy(update={"messages": list(messages)})


def _emulation_allowed(provider: ModelProvider) -> bool:
    """Whether the kit may stand in for what this provider has not declared."""
    return provider.emulates if isinstance(provider, DeclaresEmulation) else True


def _retry_after(failure: Exception) -> float | None:
    """What the provider asked for, where it is a provider that asked."""
    after = getattr(failure, "retry_after", None)
    return after if isinstance(after, int | float) else None


def _length(message: Message) -> int:
    return sum(len(part.text) for part in message.content if isinstance(part, TextPart))


def _render(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
