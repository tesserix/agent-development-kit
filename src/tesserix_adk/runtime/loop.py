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
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Never, TypeVar, cast

from pydantic import BaseModel

from tesserix_adk.core import (
    AdkError,
    AgentDefinition,
    ApprovalBindingError,
    ApprovalDecision,
    ApprovalDenial,
    ApprovalDeniedError,
    ApprovalExpiredError,
    ApprovalGate,
    ApprovalRecord,
    AutonomyRefusedError,
    BinaryPart,
    BudgetExceededError,
    BudgetScope,
    BudgetUnavailableError,
    CancelledError,
    Capability,
    Checkpoint,
    CheckpointBoundary,
    ConcurrencyConfig,
    ConfigurationError,
    ContextWindowExceededError,
    CountSource,
    DeadlineConfig,
    DeclaresEmulation,
    FallbackChain,
    FallbackExhaustedError,
    FallbackUnsafeError,
    FanOutLimitError,
    GrantRevokedError,
    GuardrailEvaluationError,
    GuardrailPipeline,
    GuardrailViolationError,
    GuardStage,
    HookAction,
    HookChain,
    HookDecision,
    HookEvaluationError,
    HookPoint,
    HookRefusedError,
    HookSubject,
    Idempotency,
    IdempotencyStore,
    IndeterminateOutcomeError,
    LoopConfig,
    MaxIterationsError,
    Message,
    ModelProvider,
    ModelRef,
    ModelRequirements,
    ModelResponseError,
    ModelRouter,
    RecursionLimitError,
    RepeatedCallError,
    RetryConfig,
    RoutingDecision,
    Run,
    RunBudget,
    RunContext,
    RunEvent,
    RunEventKind,
    RunGrant,
    RunState,
    SchemaViolationError,
    ScopedLimits,
    ScopeEscalationError,
    StateNotFoundError,
    TaskClass,
    TextPart,
    ToolArgumentValidationError,
    ToolCall,
    ToolError,
    ToolExecutionError,
    ToolFailurePolicy,
    ToolNotFoundError,
    ToolNotPermittedError,
    ToolRefusal,
    ToolTimedOutError,
    TrustBoundaryError,
    Usage,
    deduplicate,
    fallback_eligible,
    idempotency_key,
    most_restrictive,
    resolve_hooks,
    scrub,
    verify_conformance,
)
from tesserix_adk.core.autonomy import AutonomyOutcome, InFlightPolicy
from tesserix_adk.core.provider import ModelRequest, ModelResponse
from tesserix_adk.core.streaming import StreamAccumulator, StreamEnd
from tesserix_adk.core.streaming import TextDelta as _StreamedText
from tesserix_adk.runtime.approvals import ApprovalLedger
from tesserix_adk.runtime.blocking import Ambient, LoopMonitor, carrying, drive
from tesserix_adk.runtime.cancellation import CancellationToken, Deadline
from tesserix_adk.runtime.checkpoint import (
    Checkpointer,
    claim_resume,
    plan_resume,
    refuse_if_undecidable,
)
from tesserix_adk.runtime.fanout import Lanes, Turn, phased
from tesserix_adk.runtime.progress import (
    WATCHING,
    AnswerDelta,
    ApprovalRequired,
    Backpressure,
    GuardrailDecision,
    IterationStarted,
    ProgressEvent,
    RunStarted,
    RunStream,
    StructuredDelta,
    ToolCallFailed,
    ToolCallFinished,
    ToolCallIndeterminate,
    ToolCallStarted,
    UsageUpdated,
)
from tesserix_adk.runtime.prompt import (
    ToolDeclaration,
    assemble_prompt,
    wrap_untrusted,
)
from tesserix_adk.runtime.results import ReturningTool, ToolResult, ToolResultBoundary
from tesserix_adk.runtime.retry import RetryPlan
from tesserix_adk.runtime.structured import OutputContract, unwrap_fenced

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
    from random import Random

    from tesserix_adk.core import (
        Agent,
        BudgetPolicy,
        ClaimTicket,
        Clock,
        Guardrail,
        Hook,
        IdFactory,
        ToolRegistry,
    )
    from tesserix_adk.core.autonomy import AutonomyDecision
    from tesserix_adk.runtime.autonomy import AutonomyGate
    from tesserix_adk.runtime.claim_check import ClaimCheck

__all__ = ["AgentRunner", "ModelRequest", "ModelResponse", "SystemClock"]


_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_MAX_TOOL_RESULT_CHARS = 8_000
_DEFAULT_MAX_TOOL_ATTEMPTS = 12
"""How often one tool may be retried in a run, so one flaky dependency cannot own it."""
_CHARS_PER_TOKEN = 4
_NOTHING = Usage(input_tokens=0, output_tokens=0)
_DEFAULT_IDEMPOTENCY_TTL_SECONDS = 86_400.0
"""How long a side effect stays remembered — a day covers a retry, a redeploy and a replay."""
_EFFECT_POLL_SECONDS = 0.05
_EFFECT_POLLS = 200
_TRUNCATION_MARKER = "\n[truncated]"
# Instrumentation is on unless a deployment turns it off, because a stall nobody measures
# is charged to whichever request happened to be next rather than to what caused it.
_WATCHING = LoopMonitor()

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


_NO_REFUSALS: Mapping[str, ToolRefusal] = MappingProxyType({})


class _GuardProgress:
    """Turns the pipeline's verdicts into progress events, so a watcher sees each one.

    Only the guard, the stage and what it decided: the content a guard objected to is the
    one thing that must not travel to a watcher that may be logging it.
    """

    def __init__(self, emit: Callable[[ProgressEvent], None]) -> None:
        self._emit = emit

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[None]:
        """The pipeline records events rather than spans; this is here for the protocol."""
        del name, attributes
        yield

    def event(self, name: str, **attributes: object) -> None:
        """Report one guard's verdict."""
        if name != "guardrail":
            return
        verdict = str(attributes.get("verdict", ""))
        detail = str(attributes.get("code", ""))
        if verdict == "unevaluated":
            detail = "could not evaluate"
        self._emit(
            GuardrailDecision(
                guardrail=str(attributes.get("guard", "")),
                allowed=verdict in {"allow", "redact"},
                detail=detail,
            )
        )


@dataclass(frozen=True, slots=True)
class _Bounds:
    """What limits one run: the caller's switch, the run's instant, the step ceilings."""

    token: CancellationToken
    deadline: Deadline | None
    deadlines: DeadlineConfig
    retry: RetryConfig
    loop: LoopConfig
    concurrency: ConcurrencyConfig
    provider: ModelProvider
    budget: BudgetPolicy
    approvals: ApprovalLedger = field(default_factory=ApprovalLedger)
    granted: dict[str, ApprovalRecord] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Effect:
    """A side effect this call holds the key to, until it records one or fails closed."""

    key: str
    tenant: str
    kind: Idempotency


@dataclass(slots=True)
class _Ticket:
    """One call's place in the batch, and whether it got as far as being made."""

    call: ToolCall
    dispatched: bool = False


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one call in a batch produced, held until the batch is merged in call order."""

    events: tuple[RunEvent, ...]
    text: str = ""
    source: str = "tool_error"
    failure: _Terminal | None = None
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class _Started:
    """Who is about to run and where in the call graph, before there is a run to record on."""

    agent: Agent[Any]
    depth: int
    path: tuple[str, ...]
    tenant: str
    user: str | None
    run_id: str | None
    revision: str | None = None

    @property
    def trail(self) -> str:
        """The call path as one readable string, which is where a cycle becomes visible."""
        return "→".join(self.path)


@dataclass(frozen=True, slots=True)
class _Answered:
    """One answer and where it came from, which is not always where the run started."""

    run: Run[Any]
    response: ModelResponse
    bounds: _Bounds
    request: ModelRequest


# The kinds that mean a tool did something. A tool that errored or was orphaned may still
# have landed its side effect, so both count against a transparent fallback.
_SIDE_EFFECTS = frozenset(
    {
        RunEventKind.TOOL_RESULT,
        RunEventKind.TOOL_RESULT_TRUNCATED,
        RunEventKind.TOOL_ERROR,
        RunEventKind.TOOL_INDETERMINATE,
    }
)


class AgentRunner:
    """Drives one agent to one terminal state.

    Args:
        provider: Where completions come from when the agent names its own model.
        providers: Every vendor the router may resolve to, by provider name. A routed run
            calls the one the table chose; an agent that names a model outright keeps
            `provider`.
        router: How a `task_class` becomes a model. Required for any agent that declares
            one — guessing a model would attribute the run to one that never ran it.
        tools: The registry backing the agent's allowlist. Required if the agent names
            any tool.
        guardrails: Guardrails by name. Every name the agent declares must appear here.
        budget: The spend policy. A runner given none is not a runner without a ceiling:
            each run gets a `RunBudget` resolved from the agent's limits and the
            conservative defaults. `UnlimitedBudget` is how a deployment says otherwise.
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
        approval_denial: What a denied or expired approval means. By default the call is
            refused and the agent may answer; `FAIL_RUN` stops the run instead.
        approval_ttl_seconds: How long a request stays answerable. A decision outside the
            window is refused, because an approval is permission at a moment rather than
            a standing licence. Unbounded by default.
        idempotency: Where the record of an executed side effect lives. Required by any
            tool declaring `effectful` or `idempotent`: without a store the runtime cannot
            tell a retry from a first attempt, and the call fails closed rather than
            booking a second seat.
        idempotency_ttl_seconds: How long that record is kept. The guarantee is
            at-most-once within this window; a retry arriving after it is a call nobody
            has a record of, and is treated as one.
        jitter: The source the backoff is drawn from. Injected so a test can seed it and
            assert the exact schedule instead of waiting it out.
        clock: Injected time. Defaults to wall-clock.
        monitor: How a tool that blocks the event loop is caught. Defaults to a
            `LoopMonitor` with its own defaults; `None` turns the instrumentation off,
            which is a decision to take the tail latency rather than attribute it.
        max_iterations: How many model calls one run may make before it is capped.
        max_tool_attempts: How many retries one tool may consume across a whole run. A
            dependency failing transiently on every call would otherwise spend the
            iteration budget being asked again.
        max_tool_result_chars: Where an oversized tool result is cut. Truncation is
            recorded as its own event; silently dropping half a result is a wrong answer
            nobody can account for. `claim_check` compacts rather than cuts.
        results: What every tool result is held to before it enters the conversation. The
            default holds each result to the type its tool declared, neutralises what can
            forge a turn, and flags instruction-shaped content; a consumer that needs a
            tool to fail closed on suspicion says so here rather than in a prompt.
        claim_check: Where oversized results go instead of into the conversation. Bound,
            a result above the threshold enters as a head and a handle and the content is
            fetched only if the model asks. Unbound, an oversized result is cut at
            `max_tool_result_chars` and the rest is gone.

    Raises:
        ProtocolConformanceError: If a collaborator is missing a member its protocol
            requires, which is a wiring mistake and fails here rather than mid-run.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        providers: Mapping[str, ModelProvider] | None = None,
        router: ModelRouter | None = None,
        tools: ToolRegistry | None = None,
        guardrails: Mapping[str, Guardrail] | None = None,
        budget: BudgetPolicy | None = None,
        deadlines: DeadlineConfig | None = None,
        retry: RetryConfig | None = None,
        loop: LoopConfig | None = None,
        concurrency: ConcurrencyConfig | None = None,
        hooks: HookChain | Iterable[Hook] | None = None,
        approvals: ApprovalGate | None = None,
        approval_ttl_seconds: float | None = None,
        approval_denial: ApprovalDenial = ApprovalDenial.REFUSE_CALL,
        idempotency: IdempotencyStore | None = None,
        idempotency_ttl_seconds: float = _DEFAULT_IDEMPOTENCY_TTL_SECONDS,
        jitter: Random | None = None,
        clock: Clock | None = None,
        monitor: LoopMonitor | None = _WATCHING,
        ids: IdFactory | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        max_tool_result_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
        max_tool_attempts: int = _DEFAULT_MAX_TOOL_ATTEMPTS,
        results: ToolResultBoundary | None = None,
        claim_check: ClaimCheck | None = None,
        checkpoints: Checkpointer | None = None,
        autonomy: AutonomyGate | None = None,
        revoked_runs: InFlightPolicy = InFlightPolicy.CANCEL,
    ) -> None:
        verify_conformance(provider, ModelProvider)
        self._provider = provider
        for vendor in (providers or {}).values():
            verify_conformance(vendor, ModelProvider)
        self._providers = dict(providers or {})
        self._router = router
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
        self._concurrency = concurrency or ConcurrencyConfig()
        self._lanes = Lanes(self._concurrency)
        self._hooks = (
            hooks.sealed() if isinstance(hooks, HookChain) else HookChain(hooks or ()).sealed()
        )
        self._approvals = approvals
        self._approval_ttl = approval_ttl_seconds
        self._approval_denial = ApprovalDenial(approval_denial)
        if idempotency is not None:
            verify_conformance(idempotency, IdempotencyStore)
        self._idempotency = idempotency
        self._idempotency_ttl = idempotency_ttl_seconds
        self._jitter = jitter
        self._clock: Clock = clock or SystemClock()
        self._monitor = monitor
        self._ids: IdFactory = ids or _random_id
        self._max_iterations = max_iterations
        self._max_tool_result_chars = max_tool_result_chars
        self._max_tool_attempts = max_tool_attempts
        self._results = results or ToolResultBoundary()
        self._claims = claim_check
        self._checkpoints = checkpoints
        self._autonomy = autonomy
        self._revoked_runs = revoked_runs
        self._orphans: set[asyncio.Task[Any]] = set()

    def reload(self, router: ModelRouter) -> None:
        """Route subsequent runs by `router`.

        A run already in flight keeps the model it resolved before its first call, because
        a run whose model changed halfway is two runs in one record.

        Raises:
            ConfigurationError: If this runner was built without a router. Adding routing
                to a running process changes what every agent resolves to, so it is a
                decision made at construction rather than by a reload.
        """
        if self._router is None:
            raise ConfigurationError(
                "this runner has no router to reload; construct it with one rather than "
                "having a reload change what every agent already running resolves to"
            )
        self._router = router

    def run_sync[OutputT: BaseModel](
        self,
        agent: Agent[OutputT] | AgentDefinition[OutputT],
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
        budget: BudgetPolicy | None = None,
    ) -> Run[OutputT]:
        """Run `agent` from a synchronous caller. Arguments are `run`'s.

        A deliberate wrapper, not an afterthought: not every consumer is async. It is the
        same run `run` drives, on a loop of its own, so there is one implementation and
        not a second one that drifts.

        Raises:
            RunningLoopError: If called from inside a running event loop. Await `run`
                there, or call this from a thread that has no loop of its own.
        """
        return drive(
            lambda: self.run(
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
                budget=budget,
            ),
            sync_name="run_sync",
            async_name="AgentRunner.run",
        )

    def stream_sync[OutputT: BaseModel](
        self,
        agent: Agent[OutputT] | AgentDefinition[OutputT],
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
        budget: BudgetPolicy | None = None,
        backpressure: Backpressure | None = None,
    ) -> tuple[ProgressEvent, ...]:
        """Drive `agent` and return every progress event it produced. Arguments are `stream`'s.

        The whole run, collected — a sync caller cannot be handed events as they happen
        without a thread and a queue it did not ask for. Where progress has to be acted on
        while the run is still going, that is `stream`, awaited.

        Raises:
            RunningLoopError: If called from inside a running event loop. Iterate `stream`
                there, or call this from a thread that has no loop of its own.
        """

        async def collected() -> tuple[ProgressEvent, ...]:
            stream = self.stream(
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
                budget=budget,
                backpressure=backpressure,
            )
            return tuple([event async for event in stream])

        return drive(collected, sync_name="stream_sync", async_name="AgentRunner.stream")

    def stream[OutputT: BaseModel](
        self,
        agent: Agent[OutputT] | AgentDefinition[OutputT],
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
        budget: BudgetPolicy | None = None,
        backpressure: Backpressure | None = None,
    ) -> RunStream[OutputT]:
        """Watch `agent` run, event by event. Arguments are `run`'s.

        The same run `run` would have driven, reported as it happens: iterating the stream
        drives it, awaiting it gives the finished record, and `stream.run` is that record
        once either has happened. A consumer that only wants the answer keeps calling `run`.

        The stream is also an async context manager, and leaving the block cancels a run
        nobody is reading any more — through `cancellation` where one was given, so the
        caller's own token and an abandoned consumer end the run by the same path.

        A consumer that reads slowly is answered by `backpressure`: the buffer is bounded,
        text deltas merge under pressure, and a reader that stops reading altogether stops
        the run rather than paying for one nobody is watching.

        Returns:
            The stream. Nothing starts until it is iterated or awaited.
        """
        identity = run_id or self._ids()
        token = cancellation or CancellationToken()

        async def drive() -> Run[OutputT]:
            return await self.run(
                agent,
                user_input,
                tenant=tenant,
                user=user,
                run_id=identity,
                history=history,
                memory=memory,
                cancellation=token,
                deadline=deadline,
                parent=parent,
                budget=budget,
            )

        return RunStream(identity, self._clock, drive, token.cancel, backpressure)

    def _emit(self, event: ProgressEvent) -> None:
        """Tell whoever is watching this run, where anybody is."""
        sink = WATCHING.get()
        if sink is not None:
            sink.emit(event)

    async def run[OutputT: BaseModel](
        self,
        agent: Agent[OutputT] | AgentDefinition[OutputT],
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
        budget: BudgetPolicy | None = None,
    ) -> Run[OutputT]:
        """Drive `agent` until it reaches a terminal state, and return the run.

        Args:
            agent: What to run. An `AgentDefinition` pins its revision to the run and to
                every span, so a past run names the exact artifact that produced it.
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
            budget: The ceiling for this run alone. A parent invoking a sub-agent passes
                `bounds.budget.child()` here so the child spends what the parent has left;
                a fresh allowance would be a way to spend one ceiling twice.

        Returns:
            The run, in a terminal state, carrying every event recorded on the way.

        Raises:
            ConfigurationError: If the agent declares a collaborator the runner was not
                given, or a task class, which needs a router (#53).
            asyncio.CancelledError: If the surrounding task is cancelled. It is never
                swallowed — a cancelled task that returns normally leaves its canceller
                waiting forever.
        """
        revision = agent.revision if isinstance(agent, AgentDefinition) else None
        if isinstance(agent, AgentDefinition):
            agent = agent.agent
        self._refuse_an_unrouted_class(agent)
        decision = self._route(agent, tenant)
        bounds = self._bounds_for(agent, cancellation, deadline, self._vendor_for(decision), budget)
        depth = parent.depth + 1 if parent is not None else 0
        path = (*parent.path, agent.name) if parent is not None else (agent.name,)
        try:
            agent = _inherited(agent, parent, path)
        except ScopeEscalationError as escalation:
            return await self._refused(
                _Started(agent, depth, path, tenant=tenant, user=user, run_id=run_id),
                bounds,
                RunEventKind.SCOPE_REFUSED,
                escalation,
                state=RunState.FAILED,
            )
        self._refuse_incomplete_wiring(agent, bounds.provider)
        started = _Started(
            agent, depth, path, tenant=tenant, user=user, run_id=run_id, revision=revision
        )
        refused = await self._refused_before_it_started(
            started, bounds, delegated=parent is not None
        )
        if refused is not None:
            return refused
        model = agent.model or (decision.chosen.model if decision is not None else "")
        chain = FallbackChain.of(decision) if agent.model is None else FallbackChain()
        run: Run[Any] = _carrier_for(agent.output_type)(
            id=run_id or self._ids(),
            tenant=tenant,
            user=user,
            agent_name=agent.name,
            agent_version=agent.version,
            definition_revision=revision,
            model=model,
            depth=depth,
            path=path,
            grant=_granted(agent),
            budget=bounds.budget.resolved,
        ).transition_to(RunState.RUNNING, at=self._clock.now())
        self._emit(RunStarted(agent=agent.name, model=model, tenant=tenant))
        if decision is not None:
            run = run.model_copy(update={"task_class": str(decision.task_class)}).record_event(
                self._event(
                    RunEventKind.MODEL_ROUTED,
                    name=str(decision.chosen),
                    detail=decision.explain(),
                )
            )

        try:
            run, asked = await self._asked(run, agent, user_input, bounds)
            contract = self._contract_for(agent, bounds.provider)
            prompt = assemble_prompt(
                agent,
                asked,
                history=history,
                retrieved=memory,
                tools=self._declarations_for(agent),
                output=contract,
            )
            run = run.model_copy(update={"prompt_version": prompt.version}).record_event(
                self._event(
                    RunEventKind.PROMPT_ASSEMBLED,
                    name=prompt.version,
                    detail=f"prefix {prompt.fingerprint}, {prompt.prefix_tokens} tokens",
                )
            )
            return await self._forgotten(
                await self._drive(
                    run,
                    agent,
                    bounds,
                    model=model,
                    chain=chain,
                    contract=contract,
                    tools=prompt.tools,
                    messages=list(prompt.messages),
                )
            )
        except _Terminal as stop:
            return await self._forgotten(
                await self._terminate(stop.run, agent, stop.state, stop.detail, bounds)
            )

    async def resume[OutputT: BaseModel](
        self,
        agent: Agent[OutputT] | AgentDefinition[OutputT],
        run_id: str,
        *,
        tenant: str,
        user: str | None = None,
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
        budget: BudgetPolicy | None = None,
    ) -> Run[OutputT]:
        """Carry on a run whose process died, from the frontier its checkpoint holds.

        The conversation is restored rather than replayed: work already done is not paid
        for a second time, and a tool whose result was recorded is replayed from that
        record rather than called again. A call nobody can say ran stops the resume.

        Args:
            agent: What was running. A definition whose revision differs from the one the
                checkpoint pinned is refused — resuming into a changed agent is a
                different run wearing the first one's identity.
            run_id: Which run.
            tenant: The isolation boundary.
            user: The acting principal, where there is one.
            cancellation: The caller's switch.
            deadline: A ceiling from the caller.
            budget: The ceiling for what is left of the run. The ledger already holds what
                was spent before the process died, so this is not a fresh allowance.

        Returns:
            The run, in a terminal state.

        Raises:
            ConfigurationError: If this runner has nowhere to read checkpoints from, or
                the agent's revision is not the one that was checkpointed.
            StateNotFoundError: If nothing was ever checkpointed for `run_id`.
            IndeterminateToolCallError: If a call was dispatched and nothing can say
                whether it happened. Retrying it is the duplicate this exists to prevent.
            ResumeConflictError: If another worker is already carrying the run on.
        """
        if self._checkpoints is None:
            raise ConfigurationError(
                "this runner was built without a checkpointer, so there is no frontier to "
                "carry on from; construct it with one before a run needs resuming"
            )
        revision = agent.revision if isinstance(agent, AgentDefinition) else None
        if isinstance(agent, AgentDefinition):
            agent = agent.agent
        checkpoint = await self._checkpoints.latest(run_id, tenant=tenant)
        if checkpoint is None:
            raise StateNotFoundError(
                f"{run_id} was never checkpointed, so there is nothing to carry on from",
                key=f"{tenant}/{run_id}",
                kind="checkpoint",
            )
        if checkpoint.agent_revision != revision:
            raise ConfigurationError(
                f"{run_id} was checkpointed at revision {checkpoint.agent_revision!r} and "
                f"would be carried on at {revision!r}; that is a different run"
            )
        if self._idempotency is not None:
            await claim_resume(run_id, tenant=tenant, idempotency=self._idempotency)
        plan = await plan_resume(checkpoint, self._idempotency)
        refuse_if_undecidable(plan)

        decision = self._route(agent, tenant)
        bounds = self._bounds_for(agent, cancellation, deadline, self._vendor_for(decision), budget)
        run: Run[Any] = _carrier_for(agent.output_type)(
            id=run_id,
            tenant=tenant,
            user=user,
            agent_name=agent.name,
            agent_version=agent.version,
            definition_revision=revision,
            model=checkpoint.model or agent.model or "resumed",
            prompt_version=checkpoint.prompt_version or None,
            grant=_granted(agent),
            usage=checkpoint.usage,
            budget=bounds.budget.resolved,
        ).transition_to(RunState.RUNNING, at=self._clock.now())
        run = run.record_event(
            self._event(
                RunEventKind.RUN_RESUMED,
                name=str(checkpoint.boundary),
                detail=f"from iteration {checkpoint.iterations}",
            )
        )
        self._emit(RunStarted(agent=agent.name, model=run.model, tenant=tenant))

        messages = list(checkpoint.messages)
        for replayed in plan.completed:
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=replayed.call.id,
                    content=[TextPart(text=replayed.outcome or "")],
                )
            )
        contract = self._contract_for(agent, bounds.provider)
        try:
            if plan.to_dispatch:
                run = await self._dispatch(
                    run,
                    agent,
                    ModelResponse(tool_calls=tuple(one.call for one in plan.to_dispatch)),
                    messages,
                    bounds,
                )
            return await self._forgotten(
                await self._drive(
                    run,
                    agent,
                    bounds,
                    model=run.model,
                    chain=FallbackChain(),
                    contract=contract,
                    tools=self._declarations_for(agent),
                    messages=messages,
                    start=checkpoint.iterations + 1,
                )
            )
        except _Terminal as stop:
            return await self._forgotten(
                await self._terminate(stop.run, agent, stop.state, stop.detail, bounds)
            )

    async def _drive(
        self,
        run: Run[Any],
        agent: Agent[Any],
        bounds: _Bounds,
        *,
        model: str,
        chain: FallbackChain,
        contract: OutputContract | None,
        tools: tuple[ToolDeclaration, ...],
        messages: list[Message],
        start: int = 1,
    ) -> Run[Any]:
        """Go round the loop until the run is over.

        A fresh run enters with the assembled prompt; a resumed one enters with the
        conversation its checkpoint held, already carrying whatever its outstanding calls
        returned. Both then want the same thing, so both get the same loop.
        """
        for iteration in range(start, self._max_iterations + 1):
            self._emit(IterationStarted(iteration=iteration))
            self._stop_if_over(run, bounds)
            await self._spent(run, bounds, _NOTHING, iterations=1)
            run, messages = await self._before_the_call(run, agent, messages, bounds)
            self._refuse_unreadable_prompt(messages, model, bounds.provider)
            request = ModelRequest(
                model=model,
                messages=tuple(messages),
                tools=tools,
                output_schema=contract.schema if contract is not None and contract.native else None,
                output_schema_hash=contract.hash if contract is not None else None,
            )
            run = run.record_event(self._event(RunEventKind.MODEL_CALL, name=model))

            answered = await self._call_model(run, agent, request, bounds, chain)
            run, response, bounds = answered.run, answered.response, answered.bounds
            model = answered.request.model
            run, response = await self._after_the_response(run, agent, response, bounds)
            await self._checkpoint(
                run, agent, messages, model, CheckpointBoundary.AFTER_MODEL_CALL, iteration
            )
            run = await self._settle(run, agent, response, messages, bounds)
            if run.state.is_terminal:
                return run

            run, response = await self._check_guardrails(run, agent, response, messages, bounds)
            run, done = await self._advance(run, agent, response, messages, bounds)
            if done:
                return run
            await self._checkpoint(
                run, agent, messages, model, CheckpointBoundary.AFTER_TOOL_RESULT, iteration
            )

        return await self._terminate(
            run,
            agent,
            RunState.MAX_ITERATIONS_EXCEEDED,
            _named(MaxIterationsError(f"stopped after {self._max_iterations} model calls")),
            bounds,
        )

    async def _checkpoint(
        self,
        run: Run[Any],
        agent: Agent[Any],
        messages: list[Message],
        model: str,
        boundary: CheckpointBoundary,
        iteration: int,
    ) -> None:
        """Write the frontier, where the runner was given somewhere to write it.

        Spend is not carried here: the ledger already survives the process, and a second
        copy of a number that only ever goes up is a second copy to disagree with.
        """
        if self._checkpoints is None:
            return
        await self._checkpoints.record(
            Checkpoint(
                run_id=run.id,
                tenant=run.tenant,
                agent_name=agent.name,
                model=model,
                boundary=boundary,
                messages=tuple(messages),
                usage=run.usage,
                iterations=iteration,
                agent_revision=run.definition_revision,
                prompt_version=run.prompt_version or "",
            )
        )

    async def _forgotten(self, run: Run[Any]) -> Run[Any]:
        """Drop the frontier of a run that has reached the end of itself."""
        if self._checkpoints is not None:
            await self._checkpoints.forget(run.id, tenant=run.tenant)
        return run

    async def _asked(
        self, run: Run[Any], agent: Agent[Any], user_input: str, bounds: _Bounds
    ) -> tuple[Run[Any], str]:
        """What is actually asked, after policy has had its say about it."""
        run, asked, decision, _ = await self._ask_hooks(
            run, agent, HookPoint.BEFORE_PROMPT_ASSEMBLY, bounds, content=user_input
        )
        self._stop_on_refusal(run, decision)
        return await self._guarded(run, agent, asked, bounds, stage=GuardStage.INPUT)

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

    def _route(self, agent: Agent[Any], tenant: str) -> RoutingDecision | None:
        """Resolve the agent's task class, once, before anything is spent.

        Raises:
            NoEligibleModelError: If nothing configured can do the work.
        """
        if not agent.task_class or self._router is None:
            return None
        return self._router.resolve(
            TaskClass(agent.task_class),
            requirements=ModelRequirements(capabilities=agent.requires),
            tenant=tenant,
            agent=agent.name,
        )

    def _vendor_for(self, decision: RoutingDecision | None) -> ModelProvider:
        """The provider that answers this run.

        Raises:
            ConfigurationError: If the table resolved to a vendor the runner was not
                given a client for, which is a wiring gap rather than a routing one.
        """
        if decision is None:
            return self._provider
        vendor = self._providers.get(decision.chosen.provider)
        if vendor is None:
            raise ConfigurationError(
                f"the routing table chose {decision.chosen}, but this runner was given no "
                f"provider named {decision.chosen.provider!r} "
                f"(it has: {', '.join(sorted(self._providers)) or 'none'})"
            )
        return vendor

    def _refuse_an_unrouted_class(self, agent: Agent[Any]) -> None:
        if agent.task_class and self._router is None:
            raise ConfigurationError(
                f"agent {agent.name!r} selects its model by task_class "
                f"{agent.task_class!r}; this runner was given no router, and guessing a "
                f"model would attribute the run to one that never ran it"
            )

    def _refuse_incomplete_wiring(self, agent: Agent[Any], provider: ModelProvider) -> None:
        if agent.tools and self._tools is None:
            raise ConfigurationError(
                f"agent {agent.name!r} declares tools ({', '.join(agent.tools)}) but the "
                f"runner was given no registry"
            )
        if agent.tools:
            self._require(Capability.TOOL_CALLING, model=agent.model or "", provider=provider)
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

    def _require(self, capability: Capability, *, model: str, provider: ModelProvider) -> None:
        """Check the provider's own record. Raises `CapabilityError` naming all three."""
        provider.capabilities.require(capability, provider=provider.name, model=model)

    def _refuse_unreadable_prompt(
        self, messages: Sequence[Message], model: str, provider: ModelProvider
    ) -> None:
        """Refuse a prompt past the declared window rather than letting the vendor cut it.

        Raises:
            CapabilityError: If any message carries an image and the model cannot see.
            ContextWindowExceededError: If the provider's own count is over its own window.
        """
        if any(isinstance(part, BinaryPart) for message in messages for part in message.content):
            self._require(Capability.VISION, model=model, provider=provider)
        window = provider.capabilities.context_window_tokens
        if window is None:
            return
        counted = provider.count_tokens(messages)
        if counted > window:
            raise ContextWindowExceededError(
                f"the prompt counts {counted} tokens against {provider.name}:{model}'s "
                f"declared window of {window}. Sending it would have the vendor truncate it "
                f"and answer anyway",
                counted=counted,
                limit=window,
                provider=provider.name,
                model=model,
            )

    def _bounds_for(
        self,
        agent: Agent[Any],
        token: CancellationToken | None,
        caller: Deadline | None,
        provider: ModelProvider,
        budget: BudgetPolicy | None = None,
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
            concurrency=self._concurrency.narrowed_to(agent.concurrency),
            provider=provider,
            budget=budget or self._budget_for(agent),
        )

    def _budget_for(self, agent: Agent[Any]) -> BudgetPolicy:
        """The ceiling this run is held to. There is no path here that returns nothing.

        A runner given no policy does not run unbounded: it resolves the agent's own
        limits against the conservative defaults, which is a ceiling somebody can read off
        the run afterwards. Removing it takes `UnlimitedBudget` and a stated reason.
        """
        if self._budget is not None:
            return self._budget
        stated = (
            (ScopedLimits(scope=BudgetScope.AGENT, limits=agent.budget),)
            if agent.budget is not None
            else ()
        )
        return RunBudget(resolved=most_restrictive(*stated), clock=self._clock)

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
        try:
            done, _ = await asyncio.wait([task, *watchers], return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # A batch stopping its siblings cancels the waiter, not the work it was waiting
            # on; the step is unwound here rather than left running with nobody holding it.
            for unfinished in (task, *watchers):
                unfinished.cancel()
            await asyncio.gather(task, *watchers, return_exceptions=True)
            raise
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
        bounds.approvals.void()
        run = await self._notify(run, agent, state, bounds)
        run = self._to_compensate(run, agent, state)
        recorded = run.record_event(self._event(RunEventKind.TERMINATED, detail=detail))
        return recorded.transition_to(state, at=self._clock.now())

    def _to_compensate(self, run: Run[Any], agent: Agent[Any], state: RunState) -> Run[Any]:
        """Name the side effects that outlived the run, so somebody can undo them.

        A run that stops after a tool has changed the world leaves that change behind.
        The runtime does not undo it — unwinding by re-dispatching is how one side effect
        becomes two — so it says which tools ran and are not declared safe to repeat.
        """
        if state is RunState.COMPLETED:
            return run
        for name in dict.fromkeys(
            event.name
            for event in run.events
            if event.kind in _SIDE_EFFECTS and event.name is not None
        ):
            if name in agent.idempotent_tools:
                continue
            run = run.record_event(
                self._event(
                    RunEventKind.COMPENSATION_REQUIRED,
                    name=name,
                    detail=f"ran before the run ended in {state}, and is not declared idempotent",
                )
            )
        return run

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
        self,
        run: Run[Any],
        agent: Agent[Any],
        request: ModelRequest,
        bounds: _Bounds,
        chain: FallbackChain,
    ) -> _Answered:
        """Get one answer, from this model or from the next one the chain offers."""
        spent: list[tuple[str, str]] = []
        while True:
            try:
                run, response = await self._attempted(run, request, bounds)
            except _Refused as refusal:
                run, bounds, request = self._fell_back(
                    refusal.run, agent, chain, spent, refusal.failure, bounds, request
                )
            else:
                return _Answered(run=run, response=response, bounds=bounds, request=request)

    async def _attempted(
        self, run: Run[Any], request: ModelRequest, bounds: _Bounds
    ) -> tuple[Run[Any], ModelResponse]:
        """Ask this model, retrying it as its own policy allows.

        Raises:
            _Refused: When this model's attempts are spent. Whether that ends the run is
                the chain's decision, not this one's.
        """
        estimate = sum(_length(message) for message in request.messages) // _CHARS_PER_TOKEN
        # Nothing was reported back for an attempt that failed, so the kit's own estimate
        # of what the vendor read is all there is to charge for it.
        burned = Usage(input_tokens=estimate, output_tokens=0, source=CountSource.HEURISTIC)
        plan = RetryPlan(bounds.retry, random=self._jitter)
        attempt = 1
        while True:
            try:
                await bounds.budget.reserve(estimate)
                response: ModelResponse = await self._bounded(
                    self._answered(bounds, request),
                    limit=self._limit(bounds, bounds.deadlines.model_call_seconds),
                    bounds=bounds,
                    what="model call",
                )
            except _Aborted as abort:
                await self._settle_the_hold(bounds, burned)
                raise self._cancelled(run, abort, name=request.model) from None
            except BudgetExceededError as exceeded:
                raise self._exhausted(run, exceeded) from exceeded
            except BudgetUnavailableError as unavailable:
                raise _Terminal(run, RunState.FAILED, str(unavailable)) from unavailable
            except CancelledError as cancelled:
                raise _Terminal(run, RunState.CANCELLED, str(cancelled)) from cancelled
            except Exception as failure:
                run, delay = self._after_failure(
                    run, plan, attempt, failure, bounds, request.model, burned
                )
                # A retry that spent nothing off the ceiling is a way to spend past it.
                await self._spent(run, bounds, burned, model_calls=1)
                if delay is None:
                    raise _Refused(run, failure) from failure
                await self._backoff(run, delay, bounds, name=request.model)
                run = run.record_event(self._event(RunEventKind.MODEL_CALL, name=request.model))
                attempt += 1
            else:
                answer = self._readable(response, bounds.provider)
                if not _streamable(bounds):
                    self._deltas(answer.content, structured=request.output_schema_hash is not None)
                return run, answer

    async def _answered(self, bounds: _Bounds, request: ModelRequest) -> ModelResponse:
        """One answer, streamed to whoever is watching where both ends can stream.

        The buffered path is not a lesser one: a provider without streaming, or a run
        nobody is watching, still emits the answer as a single delta, so a consumer sees
        the same event sequence either way.

        Raises:
            ModelResponseError: If the stream ended without a final response. Accumulated
                text from a dropped connection is not an answer, and presenting it as one
                is the failure this path exists to prevent.
        """
        if not _streamable(bounds):
            return await bounds.provider.complete(request)

        structured = request.output_schema_hash is not None
        accumulator = StreamAccumulator()
        finished: ModelResponse | None = None
        async for event in await bounds.provider.stream(request):
            if isinstance(event, StreamEnd):
                finished = event.response
                break
            if isinstance(event, _StreamedText):
                self._deltas(event.text, structured=structured)
            accumulator.feed(event)
        if finished is None:
            raise ModelResponseError(
                f"the stream from {bounds.provider.name} ended without a final response"
            )
        return finished

    def _deltas(self, text: str, *, structured: bool) -> None:
        """Hand the answer to whoever is watching, as prose or as JSON."""
        if not text:
            return
        self._emit(StructuredDelta(fragment=text) if structured else AnswerDelta(text=text))

    def _fell_back(
        self,
        run: Run[Any],
        agent: Agent[Any],
        chain: FallbackChain,
        spent: list[tuple[str, str]],
        failure: Exception,
        bounds: _Bounds,
        request: ModelRequest,
    ) -> tuple[Run[Any], _Bounds, ModelRequest]:
        """Move the run to the next model in the chain, or end it saying why it could not.

        Raises:
            _Terminal: If nothing may be tried next — an alternative that does not exist,
                a failure another vendor would give the same answer to, a side effect that
                must not be repeated, a cancelled run, or a chain with nothing left.
        """
        here = f"{bounds.provider.name}:{request.model}"
        spent.append((here, _named(failure) if isinstance(failure, AdkError) else str(failure)))
        nowhere = chain.after(here, failed={ref for ref, _ in spent}) is None
        if nowhere and fallback_eligible(failure):
            try:
                chain.refuse_the_excluded()
            except TrustBoundaryError as refused:
                raise _Terminal(run, RunState.FAILED, _named(refused)) from refused
        if not fallback_eligible(failure) or (nowhere and len(spent) == 1):
            raise _Terminal(run, RunState.FAILED, f"{type(failure).__name__}: {failure}")
        if nowhere:
            raise _Terminal(run, RunState.FAILED, _named(_exhausted(spent)))
        self._refuse_a_repeatable_side_effect(run, agent, chain.after(here) or "")
        self._stop_if_cancelled(run, bounds)
        return self._next_link(run, chain, spent, bounds, request)

    def _next_link(
        self,
        run: Run[Any],
        chain: FallbackChain,
        spent: list[tuple[str, str]],
        bounds: _Bounds,
        request: ModelRequest,
    ) -> tuple[Run[Any], _Bounds, ModelRequest]:
        """The first remaining link this runner can actually call, with the run moved to it."""
        here = f"{bounds.provider.name}:{request.model}"
        tried = {ref for ref, _ in spent}
        while (link := chain.after(here, failed=tried)) is not None:
            ref = ModelRef.parse(link)
            vendor = self._providers.get(ref.provider)
            unusable = self._unusable(vendor, ref, request)
            if unusable:
                spent.append((link, unusable))
                tried.add(link)
                continue
            assert vendor is not None  # noqa: S101 — _unusable answers for a missing vendor
            moved = request.model_copy(update={"model": ref.model})
            run = run.model_copy(update={"model": ref.model}).record_event(
                self._event(
                    RunEventKind.MODEL_FELL_BACK, name=link, detail=f"after {here}: {spent[0][1]}"
                )
            )
            return run, replace(bounds, provider=vendor), moved
        raise _Terminal(run, RunState.FAILED, _named(_exhausted(spent)))

    def _unusable(self, vendor: ModelProvider | None, ref: ModelRef, request: ModelRequest) -> str:
        """Why this link cannot answer, or an empty string where it can."""
        if vendor is None:
            return f"the runner was given no provider named {ref.provider!r}"
        window = vendor.capabilities.context_window_tokens
        needed = sum(_length(message) for message in request.messages) // _CHARS_PER_TOKEN
        if window is not None and window < needed:
            return f"its window holds {window} tokens and the prompt is already {needed}"
        return ""

    def _refuse_a_repeatable_side_effect(self, run: Run[Any], agent: Agent[Any], link: str) -> None:
        """Refuse a fallback that would move a run past a side effect nobody can undo.

        Raises:
            _Terminal: If a tool that ran is not declared idempotent. The recorded results
                are replayed to the next model, which is sound only where being invoked
                once is the whole story.
        """
        ran = [
            event.name
            for event in run.events
            if event.kind in _SIDE_EFFECTS and event.name is not None
        ]
        risky = next((name for name in ran if name not in agent.idempotent_tools), None)
        if risky is None:
            return
        raise _Terminal(
            run,
            RunState.FAILED,
            _named(
                FallbackUnsafeError(
                    f"{risky} already ran and is not declared idempotent, so continuing on "
                    f"{link} could repeat a side effect nobody can undo",
                    tool=risky,
                    ref=link,
                )
            ),
        )

    def _stop_if_cancelled(self, run: Run[Any], bounds: _Bounds) -> None:
        """A chain that runs on after the caller let go bills them for changing their mind.

        Raises:
            _Terminal: If the caller's switch was flipped while this model was failing.
        """
        try:
            bounds.token.raise_if_cancelled()
        except CancelledError as cancelled:
            raise _Terminal(run, RunState.CANCELLED, str(cancelled)) from cancelled

    def _readable(self, payload: object, provider: ModelProvider) -> ModelResponse:
        """Refuse an answer that is not one.

        Distinct from a schema violation, which is a well-formed answer in the wrong shape
        and can be repaired: this is a provider implementation fault, and repairing it
        would mean guessing what it meant.

        Raises:
            ModelResponseError: If `payload` is not a `ModelResponse`.
        """
        if not isinstance(payload, ModelResponse):
            raise ModelResponseError(
                f"{provider.name} returned {type(payload).__name__}, not a ModelResponse",
                payload=payload,
                provider=provider.name,
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
        burned: Usage,
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
            run.record(burned).record_event(
                self._event(RunEventKind.ATTEMPT_FAILED, name=name, detail=detail, usage=burned)
            ),
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
                name=bounds.provider.name,
                usage=response.usage,
            )
        )
        self._emit(UsageUpdated(usage=run.usage))
        await self._spent(run, bounds, response.usage, model_calls=1)
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
            echoed = self._echoed(agent, response.content)
            messages.append(Message(role="assistant", content=[TextPart(text=echoed)]))
        return _carrying(run, messages)

    async def _check_guardrails(
        self,
        run: Run[Any],
        agent: Agent[Any],
        response: ModelResponse,
        messages: list[Message],
        bounds: _Bounds,
    ) -> tuple[Run[Any], ModelResponse]:
        """What came back is checked once, in the agent's declared order, before it is used."""
        if not response.content:
            return run, response
        run, checked = await self._guarded(
            run, agent, response.content, bounds, stage=GuardStage.OUTPUT
        )
        if checked == response.content:
            return run, response
        messages[-1] = _retexted(messages[-1], self._echoed(agent, checked))
        run = _carrying(run, messages)
        return run, response.model_copy(update={"content": checked})

    async def _guarded(
        self, run: Run[Any], agent: Agent[Any], content: str, bounds: _Bounds, *, stage: GuardStage
    ) -> tuple[Run[Any], str]:
        """Guardrails fail closed: a check that did not run is not a check that passed."""
        if not agent.guardrails:
            return run, content
        pipeline = GuardrailPipeline(
            [self._guardrails[name] for name in agent.guardrails],
            timeout_seconds=None,
            tracer=_GuardProgress(self._emit),
        )
        check = pipeline.check_input if stage is GuardStage.INPUT else pipeline.check_output
        try:
            checked = await self._bounded(
                check(content),
                limit=self._limit(bounds, None),
                bounds=bounds,
                what=f"guardrails on the {stage}",
            )
        except _Aborted as abort:
            raise self._cancelled(run, abort, name="guardrails") from None
        except GuardrailViolationError as refused:
            raise _Terminal(
                run.record_event(
                    self._event(
                        RunEventKind.GUARDRAIL_REFUSAL, name=refused.guard, detail=refused.code
                    )
                ),
                RunState.FAILED,
                f"guardrail {refused.guard} refused the {stage}",
            ) from None
        except GuardrailEvaluationError as failure:
            raise _Terminal(
                run.record_event(
                    self._event(
                        RunEventKind.GUARDRAIL_REFUSAL,
                        name=failure.guard,
                        detail=f"could not evaluate: {failure.reason}",
                    )
                ),
                RunState.FAILED,
                f"guardrail {failure.guard} could not be evaluated",
            ) from failure
        if checked != content:
            run = run.record_event(self._event(RunEventKind.GUARDRAIL_REDACTION, name=str(stage)))
        return run, checked

    @staticmethod
    def _echoed(agent: Agent[Any], content: str) -> str:
        """What the model said, as the next turn is allowed to read it."""
        return content if agent.free_text else wrap_untrusted(content, source="model_output")

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
        refused: dict[str, ToolRefusal] = {}
        for call in calls:
            if call.name not in agent.tools:
                self._emit(
                    ToolCallFailed(
                        call_id=call.id,
                        tool=call.name,
                        error="ToolRefused",
                        detail="not on the agent's allowlist",
                    )
                )
                raise _Terminal(
                    run.record_event(self._event(RunEventKind.TOOL_REFUSED, name=call.name)),
                    RunState.FAILED,
                    f"model called {call.name!r}, which is not on the agent's allowlist",
                )
            self._refuse_what_a_flagged_result_may_have_asked_for(run, agent, call)
            try:
                run = await self._cleared_to_dispatch(run, agent, call, bounds)
            except _Declined as declined:
                run, refused[call.id] = declined.run, declined.refusal
                continue
            run = run.model_copy(update={"tool_calls": [*run.tool_calls, call]})
            run = run.record_event(self._event(RunEventKind.TOOL_CALL, name=call.name))
            self._emit(
                ToolCallStarted(call_id=call.id, tool=call.name, arguments=_shown(call.arguments))
            )

        outcomes = await self._fanned_out(run, agent, calls, bounds, refused)
        recorded = [event for call in calls for event in outcomes[call.id].events]
        run = run.model_copy(update={"events": [*run.events, *recorded]})
        for call in calls:
            outcome = outcomes[call.id]
            if outcome.failure is not None:
                raise _Terminal(run, outcome.failure.state, outcome.failure.detail)
            run, text, decision, _ = await self._ask_hooks(
                run,
                agent,
                HookPoint.AFTER_TOOL_RESULT,
                bounds,
                content=outcome.text,
                tool_name=call.name,
                tool_arguments=call.arguments,
            )
            self._stop_on_refusal(run, decision)
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    content=[TextPart(text=_as_data(text, outcome))],
                )
            )
            run = _carrying(run, messages)
        return run

    async def _fanned_out(
        self,
        run: Run[Any],
        agent: Agent[Any],
        calls: Sequence[ToolCall],
        bounds: _Bounds,
        refused: Mapping[str, ToolRefusal] = _NO_REFUSALS,
    ) -> dict[str, _Outcome]:
        """Run the batch inside its lanes, and return one outcome per call.

        Every call is dispatched against the same run, and what each recorded is merged
        back in call order afterwards: a batch that resolved in whatever order the network
        allowed still reads, and replays, as the order the model asked for.
        """
        serial = frozenset(
            declaration.name
            for declaration in self._declarations_for(agent)
            if not declaration.parallel_safe
        )
        turn = Turn(bounds.concurrency)
        outcomes: dict[str, _Outcome] = {
            call.id: self._gate_refused(call, refused[call.id])
            for call in calls
            if call.id in refused
        }
        stopped: str | None = None
        for phase in phased([call for call in calls if call.id not in refused], serial=serial):
            if stopped is not None:
                outcomes |= {call.id: self._never_dispatched(call, stopped) for call in phase}
                continue
            done = await self._one_phase(run, agent, phase, bounds, turn)
            outcomes |= done
            failures = (outcomes[call.id].failure for call in phase)
            stopped = next(
                (failure.detail or "the batch stopped" for failure in failures if failure), None
            )
        return outcomes

    async def _one_phase(
        self,
        run: Run[Any],
        agent: Agent[Any],
        phase: Sequence[ToolCall],
        bounds: _Bounds,
        turn: Turn,
    ) -> dict[str, _Outcome]:
        """Dispatch one phase together, stopping the rest of it as soon as one call ends the run."""
        tickets = {call.id: _Ticket(call=call) for call in phase}
        tasks: dict[asyncio.Future[Any], ToolCall] = {
            asyncio.ensure_future(self._one_call(run, agent, tickets[call.id], bounds, turn)): call
            for call in phase
        }
        watcher = asyncio.ensure_future(bounds.token.wait())
        outcomes: dict[str, _Outcome] = {}
        pending: set[asyncio.Future[Any]] = {*tasks, watcher}
        while pending - {watcher}:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for finished in done - {watcher}:
                outcomes[tasks[finished].id] = finished.result()
            if watcher in done or any(outcome.failure is not None for outcome in outcomes.values()):
                break
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        for unfinished in pending - {watcher}:
            unfinished.cancel()
        await asyncio.gather(*(pending - {watcher}), return_exceptions=True)
        for unfinished in pending - {watcher}:
            call = tasks[unfinished]
            outcomes[call.id] = self._stopped_short(agent, tickets[call.id], bounds)
        return outcomes

    async def _one_call(
        self, run: Run[Any], agent: Agent[Any], ticket: _Ticket, bounds: _Bounds, turn: Turn
    ) -> _Outcome:
        """One call's turn in the lanes, reported as what it recorded rather than raised."""
        async with self._lanes.held(ticket.call.name, tenant=run.tenant, turn=turn):
            ticket.dispatched = True
            try:
                after, text, source, checked = await self._invoke(run, agent, ticket.call, bounds)
            except _Terminal as terminal:
                return _Outcome(
                    events=tuple(terminal.run.events[len(run.events) :]), failure=terminal
                )
            return _Outcome(
                events=tuple(after.events[len(run.events) :]),
                text=text,
                source=source,
                result=checked,
            )

    def _gate_refused(self, call: ToolCall, declined: ToolRefusal) -> _Outcome:
        """A call a human declined: nothing ran, and the agent is told so as data."""
        detail = f"{declined.code}: {declined.message}"
        self._emit(
            ToolCallFailed(call_id=call.id, tool=call.name, error="ToolRefusal", detail=detail)
        )
        return _Outcome(
            events=(self._event(RunEventKind.TOOL_REFUSED, name=call.name, detail=detail),),
            text=detail,
            source="tool_refusal",
        )

    def _never_dispatched(self, call: ToolCall, why: str) -> _Outcome:
        """A call the batch stopped before it was made — undone, not unknown."""
        detail = f"never dispatched: {why}"
        self._emit(
            ToolCallFailed(call_id=call.id, tool=call.name, error="Cancelled", detail=detail)
        )
        return _Outcome(
            events=(self._event(RunEventKind.TOOL_ERROR, name=call.name, detail=detail),),
            text=f"error: {detail}",
        )

    def _stopped_short(self, agent: Agent[Any], ticket: _Ticket, bounds: _Bounds) -> _Outcome:
        """A sibling ended the batch: what was queued is undone, what was running is unknown."""
        why = bounds.token.reason or "a sibling call ended the run"
        if not ticket.dispatched:
            return self._never_dispatched(ticket.call, why)
        stopped = self._indeterminacy(agent, ticket.call)
        return _Outcome(events=(stopped,), text=f"error: {stopped.detail}")

    async def _checked_in(self, checked: ToolResult, run: Run[Any]) -> ClaimTicket | None:
        """Store an oversized result and return its ticket, or `None` if none was bound."""
        if self._claims is None:
            return None
        return await self._claims.stored(checked, tenant=run.tenant, run_id=run.id)

    def _returning(self, name: str) -> ReturningTool:
        """What the boundary needs to know about the tool, for buses that can say."""
        resolve = getattr(self._tools, "resolve", None)
        return _Unannotated(name) if resolve is None else cast("ReturningTool", resolve(name))

    def _refuse_what_a_flagged_result_may_have_asked_for(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall
    ) -> None:
        """Stop a privileged call once anything in this conversation has been flagged.

        A result that tried to issue instructions is in the context from the turn it
        arrived until the run ends, so "the very next call" is a window an attacker waits
        out. Approval exists to put a person between the model and an irreversible action,
        and asking that person to adjudicate a call that a scraped page may have written is
        asking them to launder it. The run fails instead, and a consumer that wants this
        call made says so out of band rather than through the result.

        Raises:
            _Terminal: If the call needs approval and a flagged result reached the model.
        """
        if not self._declares_approval(agent, call):
            return
        flagged = [event for event in run.events if event.kind is RunEventKind.TOOL_RESULT_FLAGGED]
        if not flagged:
            return
        detail = (
            f"{call.name} requires approval, and {flagged[-1].name}'s result was flagged "
            f"earlier in this run. A tool result cannot authorise a privileged call"
        )
        self._emit(
            ToolCallFailed(call_id=call.id, tool=call.name, error="ToolRefused", detail=detail)
        )
        raise _Terminal(
            run.record_event(self._event(RunEventKind.TOOL_REFUSED, name=call.name, detail=detail)),
            RunState.FAILED,
            detail,
        )

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
        caps = bounds.budget.resolved.limits
        made = len(run.tool_calls)
        width, total = caps.max_parallel_tool_calls, caps.max_tool_calls
        too_wide = (
            f"turn asked for {len(calls)} tool calls, over the max_parallel_tool_calls of {width}"
            if width is not None and len(calls) > width
            else (
                f"turn would take the run to {made + len(calls)} tool calls, over the "
                f"max_tool_calls of {total}"
                # Only a fan-out is refused here. A single call that will not fit is spend,
                # and it belongs to the ceiling that charges for it, not to this check.
                if total is not None and len(calls) > 1 and made + len(calls) > total
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
            if repeats >= bounds.loop.max_repeated_calls:
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

    async def _refused_before_it_started(
        self, started: _Started, bounds: _Bounds, *, delegated: bool
    ) -> Run[Any] | None:
        """End a run the call graph has no room for, before it calls anything itself.

        Depth and peer invocations are caps on the tree, not on this run, so they are the
        one thing worth checking before a prompt is even assembled.
        """
        caps = bounds.budget.resolved.limits
        depth = caps.max_delegation_depth
        if depth is not None and started.depth > depth:
            return await self._refused(
                started,
                bounds,
                RunEventKind.DEPTH_EXCEEDED,
                RecursionLimitError(
                    f"max_delegation_depth is {depth} and {started.trail} would sit at "
                    f"{started.depth}; agents calling agents have stopped making progress"
                ),
            )
        if not delegated:
            return None
        peers = caps.max_peer_invocations
        if peers is not None and (bounds.budget.limits().max_peer_invocations or 0) < 1:
            return await self._refused(
                started,
                bounds,
                RunEventKind.DEPTH_EXCEEDED,
                RecursionLimitError(
                    f"max_peer_invocations is {peers} for the whole call graph and it is "
                    f"spent; {started.trail} is one delegation too many"
                ),
            )
        await bounds.budget.record(_NOTHING, peer_invocations=1)
        return None

    async def _refused(
        self,
        started: _Started,
        bounds: _Bounds,
        kind: RunEventKind,
        failure: AdkError,
        state: RunState = RunState.LOOP_LIMIT_EXCEEDED,
    ) -> Run[Any]:
        run = Run(
            id=started.run_id or self._ids(),
            tenant=started.tenant,
            user=started.user,
            agent_name=started.agent.name,
            agent_version=started.agent.version,
            definition_revision=started.revision,
            model=started.agent.model or "",
            depth=started.depth,
            path=started.path,
        ).transition_to(RunState.RUNNING, at=self._clock.now())
        run = run.record_event(self._event(kind, detail=str(failure)))
        return await self._terminate(run, started.agent, state, _named(failure), bounds)

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
        run, escalated = await self._within_autonomy(run, call)
        declared = self._declares_approval(agent, call)
        asked = decision.action is HookAction.REQUIRE_APPROVAL
        if not declared and escalated is None and not asked:
            return run
        reason = self._why_held(call, decision, escalated, declared=declared)
        run = await self._approved(run, agent, call, reason, bounds)
        return self._still_granted(run, call, escalated)

    def _still_granted(
        self, run: Run[Any], call: ToolCall, escalated: AutonomyDecision | None
    ) -> Run[Any]:
        """Re-check the withdrawn grants after a human decided, before anything goes out.

        A run suspended on an approval was asleep while the authority behind it could have
        been withdrawn, and a human approving a call is not the same as the grant that put
        the call in front of them still standing.
        """
        withdrawn = (
            None
            if escalated is None or self._autonomy is None
            else (self._autonomy.withdrawn(escalated))
        )
        if withdrawn is None:
            return run
        run = run.record_event(
            self._event(
                RunEventKind.GRANT_REVOKED,
                name=call.name,
                detail=f"{withdrawn.grant_id} withdrawn by {withdrawn.revoked_by}",
            )
        )
        if self._revoked_runs is InFlightPolicy.CANCEL:
            raise _Terminal(
                run,
                RunState.FAILED,
                _named(
                    GrantRevokedError(
                        f"the grant behind {call.name!r} was withdrawn while the run was under way",
                        grant_id=withdrawn.grant_id or "",
                        revoked_by=withdrawn.revoked_by,
                    )
                ),
            )
        return run

    def _why_held(
        self,
        call: ToolCall,
        decision: HookDecision,
        escalated: AutonomyDecision | None,
        *,
        declared: bool,
    ) -> str:
        """What the human waiting on this call is told it is about."""
        if decision.action is HookAction.REQUIRE_APPROVAL:
            return decision.reason
        if declared or escalated is None:
            return f"{call.name} is declared to require approval"
        return f"beyond this agent's autonomy: {escalated.reason}"

    async def _within_autonomy(
        self, run: Run[Any], call: ToolCall
    ) -> tuple[Run[Any], AutonomyDecision | None]:
        """What the grants say about this call, and why where they say it needs a human.

        Autonomy only ever adds a gate. A grant that permits acting unattended does not
        waive an approval the agent or the tool declared, because the two answer different
        questions: how much this agent may do, and whether this call is one a human sees.
        """
        if self._autonomy is None:
            return run, None
        decided = await self._autonomy.decide(
            tool=call.name,
            tenant=run.tenant,
            arguments=call.arguments,
            run_id=run.id,
            user=run.user,
        )
        if decided.outcome is AutonomyOutcome.REFUSE:
            run = run.record_event(
                self._event(RunEventKind.AUTONOMY_REFUSED, name=call.name, detail=decided.reason)
            )
            raise _Terminal(
                run,
                RunState.FAILED,
                _named(
                    AutonomyRefusedError(
                        decided.reason, tool=call.name, action_class=decided.action_class
                    )
                ),
            )
        if decided.outcome is AutonomyOutcome.ESCALATE:
            run = run.record_event(
                self._event(RunEventKind.AUTONOMY_ESCALATED, name=call.name, detail=decided.reason)
            )
            return run, decided
        return run, None

    def _declares_approval(self, agent: Agent[Any], call: ToolCall) -> bool:
        """Whether a human must decide this call, per the agent or per the tool itself.

        The tool is asked too, because the tool is what knows it moves money: an agent that
        adopts a refund tool and forgets to list it is the common case, and a gate that
        depends on every consumer remembering is a gate that is missing somewhere.
        """
        if call.name in agent.approval_required_tools:
            return True
        asks = getattr(self._resolved(call.name), "requires_approval", None)
        return bool(asks(call.arguments)) if callable(asks) else False

    def _resolved(self, name: str) -> object:
        """The tool behind `name`, where the bus can say, and nothing where it cannot."""
        resolve = getattr(self._tools, "resolve", None)
        if resolve is None:
            return None
        try:
            return resolve(name)
        except Exception:
            return None

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
        self._emit(ApprovalRequired(call_id=call.id, tool=call.name, reason=reason))
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
        run = self._honoured(run, record, decision, call)
        bounds.approvals.bind(record)
        bounds.granted[call.id] = record
        return run

    def _spend_the_approval(self, run: Run[Any], call: ToolCall, bounds: _Bounds) -> None:
        """Use the grant this call is holding, for the payload it was granted over.

        No path in today's loop can alter the arguments between the decision and the
        dispatch, and duplicate call ids are collapsed before either happens. The check is
        here so that a future one has to notice.

        Raises:
            ApprovalBindingError: If the arguments changed after the human saw them, or the
                same decision is being cashed twice. Both fail closed rather than dispatch.
        """
        record = bounds.granted.get(call.id)
        if record is None:
            return
        try:
            bounds.approvals.spend(record, call.arguments)
        except ApprovalBindingError as unbound:  # pragma: no cover - defended, not reachable
            raise _Terminal(
                self._denied(run, call, str(unbound)), RunState.FAILED, _named(unbound)
            ) from unbound

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
            why = "the decision arrived outside the request's window"
            self._refuse_the_call(
                self._denied(run, call, why),
                call,
                "approval_expired",
                f"permission at a moment is not a standing licence: {why}",
                _named(
                    ApprovalExpiredError(
                        f"approval for {call.name!r} was decided outside its window; permission "
                        f"at a moment is not a standing licence"
                    )
                ),
            )
        if not decision.granted:
            why = decision.reason or f"declined by {decision.decided_by}"
            self._refuse_the_call(
                self._denied(run, call, why),
                call,
                "approval_denied",
                why,
                _named(ApprovalDeniedError(f"approval for {call.name!r} was declined")),
            )
        return run.record_event(
            self._event(
                RunEventKind.APPROVAL_GRANTED, name=call.name, detail=f"by {decision.decided_by}"
            )
        )

    def _refuse_the_call(
        self, run: Run[Any], call: ToolCall, code: str, why: str, terminal: str
    ) -> Never:
        """A human said no. By default that is an answer the agent can work with.

        A denial that kills the run leaves the agent unable to say "then let me propose
        something smaller", which is the conversation approval exists to make possible. A
        consumer that would rather stop dead asks for `ApprovalDenial.FAIL_RUN`.

        Raises:
            ToolRefusal: Under the default policy.
            _Terminal: Under `ApprovalDenial.FAIL_RUN`.
        """
        if self._approval_denial is ApprovalDenial.FAIL_RUN:
            raise _Terminal(run, RunState.FAILED, terminal)
        raise _Declined(run, ToolRefusal(call.name, code, why))

    def _denied(self, run: Run[Any], call: ToolCall, why: str) -> Run[Any]:
        return run.record_event(
            self._event(RunEventKind.APPROVAL_DENIED, name=call.name, detail=why)
        )

    async def _dispatched(self, run: Run[Any], call: ToolCall, bounds: _Bounds) -> object:
        """Invoke the tool with the run's identity bound and the loop's lag watched.

        The ambient is bound here rather than around the whole run: this is the hop where
        a body can leave the loop for a thread, and identity that is not carried across it
        is identity a tool has to be trusted to have been passed.
        """
        assert self._tools is not None  # noqa: S101 — guarded by _refuse_incomplete_wiring
        registry = self._tools
        ambient = Ambient(
            run_id=run.id,
            tenant=run.tenant,
            user=run.user,
            cancellation=bounds.token,
            idempotency_key=bounds.keys.get(call.id),
        )
        with carrying(ambient):
            if self._monitor is None:
                return await registry.invoke(call.name, call.arguments)
            return await self._monitor.watching(
                f"tool {call.name}", lambda: registry.invoke(call.name, call.arguments)
            )

    async def _inside_its_own_ceiling(
        self, run: Run[Any], call: ToolCall, bounds: _Bounds
    ) -> object:
        """Hold the call to the ceiling declared for that tool, if one was.

        A per-tool ceiling is the call's own, not the run's: it is reported as that tool's
        failure and leaves its siblings' results standing, where the run-wide
        `tool_call_seconds` stops the run.

        Raises:
            ToolTimedOutError: If the call outran the ceiling declared for its tool.
        """
        ceiling = bounds.concurrency.per_tool_seconds.get(call.name)
        if ceiling is None:
            return await self._dispatched(run, call, bounds)
        work = asyncio.ensure_future(self._dispatched(run, call, bounds))
        timer = asyncio.ensure_future(self._clock.sleep(ceiling))
        done, _ = await asyncio.wait([work, timer], return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
            return work.result()
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise ToolTimedOutError(call.name, ceiling)

    async def _holding(
        self, run: Run[Any], call: ToolCall, bounds: _Bounds
    ) -> tuple[_Effect | None, str | None]:
        """Claim this call's side effect, or return what a previous call already recorded.

        Returns the effect this call now holds and nothing, or nothing and the outcome of
        the call that already ran. A tool that declares nothing holds nothing: undeclared
        is not a claim that repeating is safe, but it is the behaviour every existing tool
        already has, and a store nobody configured must not start refusing calls.

        Raises:
            _Terminal: If the effect cannot be identified or the store cannot be reached.
                Both mean a retry would be a guess, so the call does not go out at all.
        """
        policy = getattr(self._resolved(call.name), "idempotency", None)
        if self._idempotency is None or policy is None or not policy.deduplicated:
            return (None, None)
        key = idempotency_key(
            tenant=run.tenant,
            run_id=run.id,
            tool=call.name,
            arguments=call.arguments,
            key_arguments=policy.key_arguments,
        )
        if key is None:
            raise self._indeterminate(
                run,
                call,
                f"{call.name!r} is {policy.kind} and its key is derived from "
                f"{', '.join(policy.key_arguments)}, which this call does not carry",
            )
        bounds.keys[call.id] = key
        recorded = await self._reserved(run, call, key)
        if recorded is not None:
            return (None, recorded)
        return (_Effect(key=key, tenant=run.tenant, kind=policy.kind), None)

    async def _reserved(self, run: Run[Any], call: ToolCall, key: str) -> str | None:
        """Take the key, waiting out a caller that has it, and report what it recorded.

        Raises:
            _Terminal: If the store cannot be reached, or if the caller holding the key
                never finished — an answer nobody has is not an answer this call may
                invent by running the tool again.
        """
        assert self._idempotency is not None  # noqa: S101 — guarded by the caller
        for _ in range(_EFFECT_POLLS):
            try:
                reserved = await self._idempotency.begin(
                    key, tenant=run.tenant, ttl_seconds=self._idempotency_ttl
                )
            except Exception as unreachable:
                raise self._indeterminate(
                    run, call, f"the idempotency store could not be reached: {unreachable}"
                ) from unreachable
            if reserved.outcome is not None:
                return reserved.outcome
            if not reserved.in_flight:
                return None
            await asyncio.sleep(0)
            await self._clock.sleep(_EFFECT_POLL_SECONDS)
        raise self._indeterminate(
            run,
            call,
            f"another caller has been running {call.name!r} for this key and has not said",
        )

    async def _record(
        self, run: Run[Any], call: ToolCall, held: _Effect | None, outcome: str
    ) -> None:
        """Remember what the effect produced, so a repeat is answered rather than run.

        Raises:
            _Terminal: If the record cannot be written. The effect has landed and nothing
                remembers it, which is the state a second booking comes out of.
        """
        if held is None or self._idempotency is None:
            return
        try:
            await self._idempotency.record(
                held.key, tenant=held.tenant, outcome=outcome, ttl_seconds=self._idempotency_ttl
            )
        except Exception as unreachable:
            raise self._indeterminate(
                run, call, f"{call.name!r} ran and its outcome could not be recorded: {unreachable}"
            ) from unreachable

    async def _release(self, held: _Effect | None) -> None:
        """Let go of a key for a call that never reached the tool body."""
        if held is None or self._idempotency is None:
            return
        with suppress(Exception):
            await self._idempotency.abandon(held.key, tenant=held.tenant)

    def _refuse_to_repeat_an_effect(
        self, run: Run[Any], call: ToolCall, held: _Effect | None, failure: Exception
    ) -> None:
        """Stop an effectful call that failed, rather than retrying it into a second effect.

        The key stays held. A tool that raised may still have committed downstream, and a
        released key is permission for the next attempt to commit again.

        Raises:
            _Terminal: If the call that failed was an effectful one.
        """
        if held is None or held.kind is not Idempotency.EFFECTFUL:
            return
        raise self._indeterminate(
            run,
            call,
            f"{call.name!r} is effectful and failed without saying whether it landed: {failure}",
        )

    def _indeterminate(self, run: Run[Any], call: ToolCall, why: str) -> _Terminal:
        """The run stops, because nobody can say whether the side effect happened."""
        unknown = IndeterminateOutcomeError(f"{call.name}: {why}")
        stopped = run.record_event(
            self._event(RunEventKind.TOOL_INDETERMINATE, name=call.name, detail=_named(unknown))
        )
        self._emit(ToolCallIndeterminate(call_id=call.id, tool=call.name, detail=str(unknown)))
        return _Terminal(stopped, RunState.FAILED, _named(unknown))

    def _already_done(
        self, run: Run[Any], call: ToolCall, recorded: str
    ) -> tuple[Run[Any], str, str, ToolResult | None]:
        """A call whose effect has already landed: the record answers it, the tool does not."""
        self._emit(ToolCallFinished(call_id=call.id, tool=call.name, truncated=False))
        return (
            run.record_event(
                self._event(
                    RunEventKind.TOOL_DEDUPLICATED,
                    name=call.name,
                    detail="already executed for this key; the recorded outcome stands",
                )
            ),
            recorded,
            "tool_result",
            None,
        )

    async def _invoke(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, bounds: _Bounds
    ) -> tuple[Run[Any], str, str, ToolResult | None]:
        assert self._tools is not None  # noqa: S101 — guarded by _refuse_incomplete_wiring
        self._spend_the_approval(run, call, bounds)
        held, recorded = await self._holding(run, call, bounds)
        if recorded is not None:
            return self._already_done(run, call, recorded)
        await self._reserve(run, bounds, len(_signature(call)) // _CHARS_PER_TOKEN)
        # Charged before dispatch: a tool counted afterwards is one whose side effect has
        # already landed by the time the ceiling refuses it.
        await self._spent(run, bounds, _NOTHING, tool_calls=1)
        plan = RetryPlan(bounds.retry, random=self._jitter)
        attempt = 1
        while True:
            try:
                result = await self._bounded(
                    self._inside_its_own_ceiling(run, call, bounds),
                    limit=self._limit(bounds, bounds.deadlines.tool_call_seconds),
                    bounds=bounds,
                    what=f"tool {call.name}",
                )
            except _Aborted as abort:
                stopped = run.record_event(self._indeterminacy(agent, call))
                raise self._cancelled(stopped, abort, name=call.name) from None
            except ToolArgumentValidationError as rejected:
                await self._release(held)
                return self._arguments_rejected(run, agent, call, rejected)
            except ToolRefusal as declined:
                await self._release(held)
                return self._tool_declined(run, call, declined)
            except Exception as failure:
                self._refuse_to_repeat_an_effect(run, call, held, failure)
                delay = self._tool_backoff(run, agent, call, plan, attempt, bounds, failure)
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

        checked = self._results.checked(self._returning(call.name), result, tenant=run.tenant)
        if checked.findings:
            run = run.record_event(
                self._event(
                    RunEventKind.TOOL_RESULT_FLAGGED,
                    name=call.name,
                    detail=(
                        f"matched {', '.join(sorted({f.heuristic for f in checked.findings}))} "
                        f"at {', '.join(sorted({f.path or 'result' for f in checked.findings}))}"
                    ),
                )
            )
        text = checked.text
        claim = await self._checked_in(checked, run)
        if claim is not None:
            run = run.record_event(
                self._event(
                    RunEventKind.TOOL_RESULT_STORED,
                    name=call.name,
                    detail=f"{claim.chars} chars held under {claim.handle}",
                )
            )
            text = claim.rendered()
        truncated = checked.truncated or len(text) > self._max_tool_result_chars
        if truncated:
            run = run.record_event(
                self._event(
                    RunEventKind.TOOL_RESULT_TRUNCATED,
                    name=call.name,
                    detail=f"{len(text)} chars cut to {self._max_tool_result_chars}",
                )
            )
        if len(text) > self._max_tool_result_chars:
            text = text[: self._max_tool_result_chars] + _TRUNCATION_MARKER
        await self._record(run, call, held, text)
        await self._spent(
            run,
            bounds,
            Usage(
                input_tokens=len(text) // _CHARS_PER_TOKEN,
                output_tokens=0,
                source=CountSource.HEURISTIC,
            ),
        )
        self._emit(ToolCallFinished(call_id=call.id, tool=call.name, truncated=truncated))
        return (
            run.record_event(self._event(RunEventKind.TOOL_RESULT, name=call.name)),
            text,
            "tool_result",
            checked,
        )

    async def _spent(
        self,
        run: Run[Any],
        bounds: _Bounds,
        usage: Usage,
        *,
        model_calls: int = 0,
        tool_calls: int = 0,
        iterations: int = 0,
    ) -> None:
        """Put what was consumed on the ledger, and stop the run if that passed a ceiling.

        Raises:
            _Terminal: If the spend broke a ceiling, or if a shared ledger holding one
                could not be reached — a run that cannot check its ceiling stops.
        """
        try:
            await bounds.budget.record(
                usage,
                model_calls=model_calls,
                tool_calls=tool_calls,
                iterations=iterations,
            )
        except BudgetExceededError as exceeded:
            raise self._exhausted(run, exceeded) from exceeded
        except BudgetUnavailableError as unavailable:
            raise _Terminal(run, RunState.FAILED, str(unavailable)) from unavailable

    async def _reserve(self, run: Run[Any], bounds: _Bounds, estimate: int) -> None:
        """Spend is checked before it is incurred; reported after, it is a bill."""
        try:
            await bounds.budget.reserve(estimate)
        except BudgetExceededError as exceeded:
            raise self._exhausted(run, exceeded) from exceeded
        except BudgetUnavailableError as unavailable:
            raise _Terminal(run, RunState.FAILED, str(unavailable)) from unavailable

    def _exhausted(self, run: Run[Any], exceeded: BudgetExceededError) -> _Terminal:
        """End the run on the ceiling it reached, saying by how much it went past.

        A reservation holds the overshoot to zero where the call was refused before it
        was made. Where the answer came back dearer than the estimate that reserved for
        it, the difference is recorded rather than rounded away.
        """
        over = max(exceeded.consumed - (exceeded.limit or exceeded.consumed), Decimal(0))
        recorded = run.record_event(
            self._event(
                RunEventKind.BUDGET_EXCEEDED,
                name=exceeded.breached or None,
                detail=f"{exceeded}; over by {over}",
            )
        )
        return _Terminal(recorded, RunState.BUDGET_EXHAUSTED, str(exceeded))

    async def _settle_the_hold(self, bounds: _Bounds, spent: Usage) -> None:
        """Charge what a stopped call had already sent, whatever that does to the ceiling.

        A cancelled run reporting nothing spent is a bill nobody can reconcile. The
        ceiling may well be passed by this, and it changes nothing: the caller's switch
        decides how the run ended, so a breach discovered while unwinding is recorded on
        the budget and not allowed to rewrite the terminal state.
        """
        with suppress(BudgetExceededError, BudgetUnavailableError):
            await bounds.budget.record(spent, model_calls=1)

    def _tool_backoff(
        self,
        run: Run[Any],
        agent: Agent[Any],
        call: ToolCall,
        plan: RetryPlan,
        attempt: int,
        bounds: _Bounds,
        failure: Exception,
    ) -> float | None:
        """A tool is retried on its declaration, not on the shape of its exception.

        A tool's exception says nothing about whether its side effect landed, so the only
        safe gate is the agent naming the tool as safe to call again. The exception is a
        decision rather than a fault: a name nobody registered and a name this agent may
        not call are the same answer however often they are asked.
        """
        if isinstance(failure, ToolNotFoundError | ToolNotPermittedError | ToolRefusal):
            return None
        if isinstance(failure, ToolError):
            if not failure.retryable:
                return None
        elif call.name not in agent.idempotent_tools:
            return None
        if self._retried_enough(run, call.name):
            return None
        after = failure.retry_after if isinstance(failure, ToolError) else None
        delay = plan.delay_for(attempt, retry_after=after)
        return delay if delay is not None and self._fits(delay, bounds) else None

    def _retried_enough(self, run: Run[Any], tool: str) -> bool:
        """Whether this tool has already had its share of the run's attempts."""
        spent = sum(
            1
            for event in run.events
            if event.kind is RunEventKind.ATTEMPT_FAILED and event.name == tool
        )
        return spent >= self._max_tool_attempts

    def _tool_declined(
        self, run: Run[Any], call: ToolCall, declined: ToolRefusal
    ) -> tuple[Run[Any], str, str, ToolResult | None]:
        """A tool that worked and said no. The answer reaches the model once, as data.

        Not retried and not a run failure: asking again gets the same answer, and a refusal
        the model never sees is a refusal it will work around.
        """
        detail = f"{declined.code}: {declined.message}"
        run = run.record_event(
            self._event(RunEventKind.TOOL_REFUSED, name=call.name, detail=detail)
        )
        self._emit(
            ToolCallFailed(call_id=call.id, tool=call.name, error="ToolRefusal", detail=detail)
        )
        return run, detail, "tool_refusal", None

    def _tool_failed(
        self, run: Run[Any], agent: Agent[Any], call: ToolCall, failure: Exception, bounds: _Bounds
    ) -> tuple[Run[Any], str, str, ToolResult | None]:
        wrapped = ToolExecutionError(
            f"tool {call.name!r} failed: {scrub(str(failure))}", run_id=run.id, tenant=run.tenant
        )
        unretried = (
            "; not declared idempotent, so it was not tried again"
            if bounds.retry.max_attempts > 1
            and not isinstance(failure, ToolError)
            and call.name not in agent.idempotent_tools
            else ""
        )
        said = _what_failed(failure)
        tried = sum(
            1
            for event in run.events
            if event.kind is RunEventKind.ATTEMPT_FAILED and event.name == call.name
        )
        spent = f" after {tried + 1} attempts" if tried else ""
        run = run.record_event(
            self._event(RunEventKind.TOOL_ERROR, name=call.name, detail=f"{said}{spent}{unretried}")
        )
        self._emit(
            ToolCallFailed(
                call_id=call.id,
                tool=call.name,
                error=type(failure).__name__,
                detail=f"{said}{spent}{unretried}",
            )
        )
        if agent.on_tool_error is ToolFailurePolicy.FAIL_RUN:
            raise _Terminal(run, RunState.FAILED, str(wrapped)) from failure
        return run, f"error: {said}", "tool_error", None

    def _arguments_rejected(
        self,
        run: Run[Any],
        agent: Agent[Any],
        call: ToolCall,
        rejected: ToolArgumentValidationError,
    ) -> tuple[Run[Any], str, str, ToolResult | None]:
        """Arguments the tool refused: nothing ran, so this is correctable rather than failed.

        Retrying the same payload would be the same refusal, so what goes back is the
        failure itself, on the repair budget the agent declared. Running out fails the run
        rather than dropping the call: a model that cannot address a tool correctly is not
        a run that should quietly continue without it.

        Raises:
            _Terminal: If repair is off or its attempts are spent.
        """
        fields = ", ".join(rejected.paths) or "the whole payload"
        run = run.record_event(
            self._event(
                RunEventKind.SCHEMA_VIOLATION,
                name=call.name,
                detail=f"{call.name} refused the arguments at {fields}; the tool did not run",
            )
        )
        self._emit(
            ToolCallFailed(
                call_id=call.id,
                tool=call.name,
                error=type(rejected).__name__,
                detail=f"arguments refused at {fields}",
            )
        )
        policy = agent.repair
        attempts = sum(1 for event in run.events if event.kind is RunEventKind.REPAIR_REQUESTED)
        if policy is None or not policy.enabled or attempts >= policy.max_attempts:
            raise _Terminal(run, RunState.FAILED, f"{_named(rejected)} at {fields}")
        return (
            run.record_event(
                self._event(
                    RunEventKind.REPAIR_REQUESTED,
                    name=call.name,
                    detail=f"attempt {attempts + 1} of {policy.max_attempts} at {fields}",
                )
            ),
            rejected.feedback(),
            "tool_error",
            None,
        )

    def _indeterminacy(self, agent: Agent[Any], call: ToolCall) -> RunEvent:
        """A tool stopped mid-flight is unknown, not undone — unless it said it is safe to retry."""
        if call.name in agent.idempotent_tools:
            retryable = "stopped before it returned; declared idempotent, so safe to retry"
            self._emit(
                ToolCallFailed(call_id=call.id, tool=call.name, error="Cancelled", detail=retryable)
            )
            return self._event(RunEventKind.TOOL_ERROR, name=call.name, detail=retryable)
        unknown = "stopped after dispatch; whether its effect landed cannot be known"
        self._emit(ToolCallIndeterminate(call_id=call.id, tool=call.name, detail=unknown))
        return self._event(RunEventKind.TOOL_INDETERMINATE, name=call.name, detail=unknown)

    def _contract_for(self, agent: Agent[Any], provider: ModelProvider) -> OutputContract | None:
        """The shape of the answer, and whether this provider enforces it itself.

        A provider that has not declared the capability does not have it: assuming it does
        means discovering the schema was ignored from a run that already completed.
        """
        if agent.output_type is None:
            return None
        native = provider.capabilities.supports(Capability.STRUCTURED_OUTPUT)
        if not native and not _emulation_allowed(provider):
            self._require(Capability.STRUCTURED_OUTPUT, model=agent.model or "", provider=provider)
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
        contract = self._contract_for(agent, bounds.provider)
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


class _Declined(Exception):  # noqa: N818 — control flow, not an error the caller sees
    """A gate said no before dispatch, carrying what was recorded and what to tell the agent."""

    def __init__(self, run: Run[Any], refusal: ToolRefusal) -> None:
        super().__init__(str(refusal))
        self.run = run
        self.refusal = refusal


class _Refused(Exception):  # noqa: N818 — control flow, not an error the caller sees
    """One model is out of attempts. Whether the run is over is the chain's decision."""

    def __init__(self, run: Run[Any], failure: Exception) -> None:
        super().__init__(str(failure))
        self.run = run
        self.failure = failure


def _exhausted(spent: Sequence[tuple[str, str]]) -> FallbackExhaustedError:
    """The one error that names every model asked, so the last refusal is not the whole story."""
    return FallbackExhaustedError(
        f"every model in the chain refused this run: "
        f"{'; '.join(f'{ref} {reason}' for ref, reason in spent)}",
        attempts=spent,
    )


def _named(error: AdkError) -> str:
    """A terminal detail that says which cap or fault ended the run, not just that one did."""
    return f"{type(error).__name__}: {error}"


def _same_call_count(calls: Iterable[ToolCall], call: ToolCall) -> int:
    """How many of `calls` are the same request: same tool, same arguments in any order."""
    signature = _signature(call)
    return sum(1 for made in calls if _signature(made) == signature)


def _streamable(bounds: _Bounds) -> bool:
    """Whether this answer can arrive in pieces: somebody watching, and a provider that can."""
    return WATCHING.get() is not None and bounds.provider.capabilities.streaming


def _shown(arguments: Mapping[str, Any]) -> str:
    """Tool arguments as a consumer may see them: compact JSON, scrubbed of secret shapes."""
    return scrub(json.dumps(arguments, sort_keys=True, default=repr))


def _signature(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=repr)}"


def _granted(agent: Agent[Any]) -> RunGrant:
    """What this run may do, in the form a run below it inherits."""
    return RunGrant(
        tools=agent.tools,
        approval_required_tools=agent.approval_required_tools,
        guardrails=agent.guardrails,
    )


def _inherited[OutputT: BaseModel](
    agent: Agent[OutputT], parent: RunContext | None, path: tuple[str, ...]
) -> Agent[OutputT]:
    """Hold a delegated run to its caller's grant: every guard of it, and no wider a reach.

    Raises:
        ScopeEscalationError: If the agent declares a tool its caller does not hold. It is
            refused rather than narrowed away, because the difference is a wiring mistake
            and a silent intersection is how nobody finds out about it.
    """
    grant = parent.grant if parent is not None else None
    if grant is None:
        return agent
    beyond = tuple(sorted(set(agent.tools) - set(grant.tools)))
    if beyond:
        caller = "/".join(path[:-1]) or "its caller"
        raise ScopeEscalationError(
            f"{path[-1]!r} declares {', '.join(beyond)}, which {caller} does not hold",
            requested=beyond,
            path=path,
        )
    return agent.model_copy(
        update={
            "guardrails": _once(grant.guardrails, agent.guardrails),
            "approval_required_tools": _once(
                tuple(name for name in grant.approval_required_tools if name in agent.tools),
                agent.approval_required_tools,
            ),
        }
    )


def _once(inherited: tuple[str, ...], own: tuple[str, ...]) -> tuple[str, ...]:
    """Inherited first, then whatever is the agent's own, each name appearing once."""
    return (*inherited, *(name for name in own if name not in inherited))


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


@dataclass(frozen=True, slots=True)
class _Unannotated:
    """A tool a bus cannot describe, held to nothing but the boundary's own ceilings."""

    name: str
    returns_type: Any = None


def _what_failed(failure: Exception) -> str:
    """What may be recorded about a failure: its code where it has one, never a credential."""
    if isinstance(failure, ToolError):
        return f"{failure.code}: {failure.message}" if failure.message else failure.code
    return scrub(str(failure))


def _as_data(text: str, outcome: _Outcome) -> str:
    """Render what a tool produced as data, keeping the envelope's own annotations.

    The text is taken after the hooks rather than before: a hook that rewrote a result
    rewrote what the model reads, and wrapping the original would deliver something nobody
    approved. What the boundary decided about the result — flagged, truncated — travels
    with it, because a warning the reader cannot see is not a warning.
    """
    if outcome.result is None:
        return wrap_untrusted(text, source=outcome.source)
    return replace(outcome.result, text=text).rendered()
