"""Bring a foreign agent behind the kit's typed, contextual execution boundary.

The wrapper is deliberately an adapter, not an assertion that a foreign runtime obeys
kit policy internally. It passes only narrowed authority and no credentials, refuses a
projected over-spend before dispatch, shares the caller's ledger, bounds the call, and
validates whatever comes back before a parent can observe it. A foreign implementation
must use the supplied context for downstream work; an implementation that ignores its
context cannot be made tenant-safe by changing its return type.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tesserix_adk.core import (
    INLINE_REFS,
    AdkModel,
    ApprovalPolicy,
    BudgetExceededError,
    CancelledError,
    ConfigurationError,
    Idempotency,
    IdempotencyPolicy,
    IndeterminateOutcomeError,
    Message,
    ProviderTimeoutError,
    RecursionLimitError,
    Run,
    RunContext,
    RunEvent,
    RunEventKind,
    RunState,
    SchemaViolationError,
    TextPart,
    ToolArgumentValidationError,
    Usage,
    schema_for,
)
from tesserix_adk.tools import Tool, ToolContext

if TYPE_CHECKING:
    from tesserix_adk.core.guards import GuardrailPipeline
    from tesserix_adk.core.protocols import BudgetPolicy
    from tesserix_adk.runtime.cancellation import CancellationToken
    from tesserix_adk.tools.validation import ArgumentValidator

__all__ = [
    "ForeignAgentContext",
    "ForeignAgentReply",
    "WrappedAgentPolicy",
    "WrappedSubagent",
    "wrap_agent_as_subagent",
    "wrap_agent_as_tool",
]

_NOTHING = Usage(input_tokens=0, output_tokens=0)
_TRACE_KEYS = frozenset({"baggage", "traceparent", "tracestate"})


@dataclass(frozen=True, slots=True)
class ForeignAgentContext:
    """Caller-owned context supplied beside, and never inside, foreign input.

    No credential field exists. The foreign boundary receives identity, narrowed scopes,
    the effective tool allowlist, W3C trace propagation, cancellation, depth and the
    shared budget. It may use those to call back into the kit without opening a fresh
    authority or allowance.

    Args:
        run_id: The invocation identity.
        tenant: The isolation boundary; never inferred or defaulted.
        user: The acting principal, where one exists.
        scopes: Effective authority after intersection with wrapper policy.
        tools: Effective tools after intersection with the delegating supervisor.
        trace: W3C trace fields only. Call metadata cannot smuggle credentials here.
        cancellation: The parent's cooperative cancellation switch.
        budget: The parent's shared ledger, not a new allowance.
        depth: This invocation's position in the agent call graph.
        path: Agent lineage, root first, used to refuse cycles.
        idempotency_key: The caller's downstream repeat key, where one was bound.
    """

    run_id: str
    tenant: str
    user: str | None
    scopes: tuple[str, ...]
    tools: tuple[str, ...]
    trace: Mapping[str, str]
    cancellation: CancellationToken | None
    budget: BudgetPolicy
    depth: int
    path: tuple[str, ...]
    idempotency_key: str | None = None

    def raise_if_cancelled(self) -> None:
        """Raise the kit's typed cancellation when the parent has stopped."""
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()


class ForeignAgentReply[OutputT: BaseModel](AdkModel):
    """A foreign answer with whatever metering its runtime can report.

    `output` is still validated independently by the wrapper; annotating this envelope is
    not trusted as proof. Missing usage activates the policy's declared estimate and the
    resulting run event labels it as estimated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: object
    usage: Usage | None = None
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    peer_invocations: int = Field(default=0, ge=0)


class WrappedAgentPolicy(AdkModel):
    """Controls one foreign-agent boundary.

    Args:
        timeout_seconds: Hard kit-side ceiling for one invocation.
        projected_usage: Required preflight estimate and fallback metering.
        scopes: Maximum delegated scopes the foreign agent may receive.
        tools: Maximum tool names the foreign agent may receive.
        requires_approval: Approval metadata on the tool export.
        idempotency: Repeat behaviour advertised on the tool export.
        max_depth: Additional wrapper-local recursion ceiling.
        max_concurrency: Maximum simultaneous calls through the tool surface.
    """

    timeout_seconds: float = Field(gt=0, le=300)
    projected_usage: Usage
    scopes: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    requires_approval: bool = True
    idempotency: Idempotency
    max_depth: int = Field(default=4, ge=1, le=64)
    max_concurrency: int = Field(default=1, ge=1, le=64)

    @model_validator(mode="after")
    def _declarations_are_unambiguous(self) -> WrappedAgentPolicy:
        for field_name in ("scopes", "tools"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} may not contain an empty name")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} may not contain duplicates")
        return self


class _ModelArguments[InputT: BaseModel]:
    """Validate a flat model-facing object and hand the body one typed request."""

    __slots__ = ("_input_type", "_tool")

    def __init__(self, input_type: type[InputT], tool: str) -> None:
        self._input_type = input_type
        self._tool = tool

    def arguments(self, arguments: object) -> Mapping[str, object]:
        """Return the one typed request keyword expected by the wrapper body."""
        try:
            payload = _json_object(arguments)
            request = self._input_type.model_validate(payload, strict=True)
        except (TypeError, ValueError, ValidationError) as mismatch:
            problems = _validation_problems(mismatch)
            raise ToolArgumentValidationError(
                f"{self._tool} was called with arguments that do not match "
                f"{self._input_type.__name__}",
                tool=self._tool,
                paths=tuple(sorted(problems)),
                problems=problems,
                payload=arguments,
            ) from mismatch
        return {"request": request}


class WrappedSubagent[InputT: BaseModel, OutputT: BaseModel]:
    """An addressable foreign agent accepted by :class:`~runtime.Supervisor`.

    Construct one with :func:`wrap_agent_as_subagent`. Calling :meth:`run` requires an
    explicit tenant and shared budget, validates JSON input before dispatch, enforces
    cancellation, timeout and recursion, rolls up metering, then validates guarded output.
    Schema, budget, cancellation and timeout failures are raised as the kit's typed errors.
    """

    __slots__ = (
        "_foreign",
        "_guardrails",
        "_input_type",
        "_origin",
        "_output_type",
        "_policy",
        "name",
        "tools",
    )

    def __init__(
        self,
        foreign: Callable[[InputT, ForeignAgentContext], object],
        *,
        name: str,
        input_type: type[InputT],
        output_type: type[OutputT],
        policy: WrappedAgentPolicy,
        guardrails: GuardrailPipeline | None,
        origin: str,
    ) -> None:
        self._foreign = foreign
        self.name = name
        self.tools = policy.tools
        self._input_type = input_type
        self._output_type = output_type
        self._policy = policy
        self._guardrails = guardrails
        self._origin = origin

    @property
    def policy(self) -> WrappedAgentPolicy:
        """The immutable policy enforced at this boundary."""
        return self._policy

    @property
    def input_type(self) -> type[InputT]:
        """The Pydantic input model checked before foreign work starts."""
        return self._input_type

    @property
    def output_type(self) -> type[OutputT]:
        """The Pydantic output model checked before a parent observes the reply."""
        return self._output_type

    @property
    def origin(self) -> str:
        """Stable implementation provenance used for attribution."""
        return self._origin

    async def run(
        self,
        user_input: str,
        *,
        tenant: str,
        budget: BudgetPolicy,
        user: str | None = None,
        run_id: str | None = None,
        cancellation: CancellationToken | None = None,
        parent: RunContext | None = None,
        scopes: Collection[str] = (),
        trace: Mapping[str, str] | None = None,
        tools: Collection[str] | None = None,
    ) -> Run[OutputT]:
        """Invoke the foreign agent as one typed, budgeted kit run.

        Args:
            user_input: JSON satisfying ``input_type``.
            tenant: Required tenant boundary.
            budget: Shared parent ledger.
            user: Acting principal, where one exists.
            run_id: Invocation identity; generated when absent.
            cancellation: Parent cancellation switch.
            parent: Parent lineage and depth.
            scopes: Caller-held scopes, narrowed against policy.
            trace: W3C trace propagation fields.
            tools: Caller-held tools, narrowed against policy.

        Raises:
            ConfigurationError: If tenant or callable wiring is missing.
            RecursionLimitError: If depth or lineage would recurse.
            BudgetExceededError: If projected or actual usage breaches the shared ceiling.
            CancelledError: If the parent cancels the call.
            ProviderTimeoutError: If the foreign call exceeds its hard timeout.
            IndeterminateOutcomeError: If an effectful call is cancelled or times out
                after dispatch and its outcome cannot be known safely.
            SchemaViolationError: If input or output does not satisfy its declared model.
        """
        if not tenant:
            raise ConfigurationError("a wrapped agent needs an explicit tenant")
        identity = run_id or f"foreign-{uuid.uuid4()}"
        depth, path = _lineage(self.name, parent, self._policy.max_depth, budget)
        try:
            request = self._input_type.model_validate_json(user_input, strict=True)
        except (ValueError, ValidationError) as mismatch:
            raise _schema_error(
                self._input_type, mismatch, user_input, run_id=identity, tenant=tenant
            ) from mismatch
        effective_tools = _intersection(
            self._policy.tools,
            self._policy.tools if tools is None else tuple(tools),
        )
        context = _context(
            run_id=identity,
            tenant=tenant,
            user=user,
            caller_scopes=scopes,
            policy_scopes=self._policy.scopes,
            tools=effective_tools,
            trace=trace or {},
            cancellation=cancellation,
            budget=budget,
            depth=depth,
            path=path,
        )
        started = time.time()
        output, usage, estimated = await _execute(
            self._foreign,
            request,
            context,
            output_type=self._output_type,
            policy=self._policy,
            guardrails=self._guardrails,
            origin=self._origin,
            name=self.name,
        )
        ended = time.time()
        rendered = _render_model(output)
        detail = (
            "usage estimated from the wrapper's declared projected_usage"
            if estimated
            else "usage reported by the foreign agent"
        )
        event = RunEvent(
            kind=RunEventKind.MODEL_RESPONSE,
            name=self.name,
            detail=detail,
            at=ended,
            usage=usage,
        )
        return Run[OutputT](
            id=identity,
            tenant=tenant,
            user=user,
            agent_name=self.name,
            agent_version="wrapped-1",
            model=self._origin,
            depth=depth,
            path=path,
            state=RunState.COMPLETED,
            messages=[Message(role="assistant", content=[TextPart(text=rendered)])],
            events=[event],
            output=output,
            usage=usage,
            budget=budget.resolved,
            started_at=started,
            ended_at=ended,
        )


def wrap_agent_as_tool[InputT: BaseModel, OutputT: BaseModel](
    foreign: Callable[[InputT, ForeignAgentContext], object],
    *,
    name: str,
    input_type: type[InputT],
    output_type: type[OutputT],
    policy: WrappedAgentPolicy,
    guardrails: GuardrailPipeline | None = None,
    description: str | None = None,
    provenance: str | None = None,
) -> Tool[..., OutputT]:
    """Expose a foreign agent as a typed kit tool.

    The returned tool requires :class:`ToolContext`; a call without an explicit or ambient
    tenant and shared budget fails closed. Input and output use flat Pydantic schemas,
    credentials are never propagated, and projected usage is checked before dispatch.

    Raises:
        ConfigurationError: If the callable, name, or declared model boundary is invalid.
    """
    _validate_boundary(foreign, name, input_type, output_type)
    origin = provenance or _origin(foreign)
    validator: ArgumentValidator = _ModelArguments(input_type, name)

    async def invoke(request: InputT, context: ToolContext) -> OutputT:
        if not context.tenant:
            raise ConfigurationError("a wrapped agent tool needs an explicit tenant")
        if context.budget is None:
            raise ConfigurationError(
                "a wrapped agent tool needs the caller's shared budget; a fresh or absent "
                "allowance would make roll-up unenforceable"
            )
        effective_tools = _intersection(policy.tools, policy.tools)
        foreign_context = _context(
            run_id=context.run_id,
            tenant=context.tenant,
            user=context.user,
            caller_scopes=context.scopes,
            policy_scopes=policy.scopes,
            tools=effective_tools,
            trace=context.trace,
            cancellation=context.cancellation,
            budget=context.budget,
            depth=0,
            path=(name,),
            idempotency_key=context.idempotency_key,
        )
        output, _, _ = await _execute(
            foreign,
            request,
            foreign_context,
            output_type=output_type,
            policy=policy,
            guardrails=guardrails,
            origin=origin,
            name=name,
        )
        return output

    return Tool[..., OutputT](
        name=name,
        description=description or f"Delegate to the wrapped foreign agent {name}.",
        parameters_schema=schema_for(input_type, dialect=INLINE_REFS),
        returns_schema=schema_for(output_type, dialect=INLINE_REFS),
        is_async=True,
        function=invoke,
        validator=validator,
        context_parameter="context",
        context_required=True,
        timeout=policy.timeout_seconds,
        parallel_safe=policy.max_concurrency > 1,
        approval=ApprovalPolicy(required=policy.requires_approval),
        idempotency=IdempotencyPolicy(kind=policy.idempotency),
        returns_type=output_type,
        origin=origin,
        max_concurrency=policy.max_concurrency,
    )


def wrap_agent_as_subagent[InputT: BaseModel, OutputT: BaseModel](
    foreign: Callable[[InputT, ForeignAgentContext], object],
    *,
    name: str,
    input_type: type[InputT],
    output_type: type[OutputT],
    policy: WrappedAgentPolicy,
    guardrails: GuardrailPipeline | None = None,
    provenance: str | None = None,
) -> WrappedSubagent[InputT, OutputT]:
    """Make a foreign agent addressable from a kit supervisor roster.

    The wrapper implements the runtime's structural sub-agent protocol. Its declared tools
    are still intersected with the supervisor's held tools, and it shares the allowance,
    cancellation, tenant, lineage and trace of the caller.

    Raises:
        ConfigurationError: If the callable/model boundary is invalid or no tool is
            declared for delegation.
    """
    _validate_boundary(foreign, name, input_type, output_type)
    if not policy.tools:
        raise ConfigurationError(
            "a wrapped sub-agent needs at least one declared tool so delegation cannot "
            "silently widen or route work that could never act"
        )
    return WrappedSubagent(
        foreign,
        name=name,
        input_type=input_type,
        output_type=output_type,
        policy=policy,
        guardrails=guardrails,
        origin=provenance or _origin(foreign),
    )


async def _execute[InputT: BaseModel, OutputT: BaseModel](
    foreign: Callable[[InputT, ForeignAgentContext], object],
    request: InputT,
    context: ForeignAgentContext,
    *,
    output_type: type[OutputT],
    policy: WrappedAgentPolicy,
    guardrails: GuardrailPipeline | None,
    origin: str,
    name: str,
) -> tuple[OutputT, Usage, bool]:
    """Preflight, invoke, account and validate one foreign call."""
    context.raise_if_cancelled()
    _preflight(context.budget, policy.projected_usage, tenant=context.tenant)
    await context.budget.reserve(policy.projected_usage.input_tokens)
    called = False
    try:
        context.raise_if_cancelled()
        called = True
        raw_reply = await _bounded_call(
            foreign,
            request,
            context,
            ceiling_seconds=policy.timeout_seconds,
            origin=origin,
            name=name,
            idempotency=policy.idempotency,
        )
    except (Exception, asyncio.CancelledError):
        usage = policy.projected_usage if called else _NOTHING
        await context.budget.record(usage, peer_invocations=1 if called else 0)
        raise
    raw_output: object = raw_reply
    reply: ForeignAgentReply[BaseModel] | None
    if isinstance(raw_reply, ForeignAgentReply):
        reply = cast("ForeignAgentReply[BaseModel]", raw_reply)
        raw_output = reply.output
    else:
        reply = None
    usage = reply.usage if reply is not None and reply.usage is not None else policy.projected_usage
    estimated = reply is None or reply.usage is None
    model_calls = reply.model_calls if reply is not None else 0
    tool_calls = reply.tool_calls if reply is not None else 0
    iterations = reply.iterations if reply is not None else 0
    nested_peers = reply.peer_invocations if reply is not None else 0
    await context.budget.record(
        usage,
        model_calls=model_calls,
        tool_calls=tool_calls,
        iterations=iterations,
        peer_invocations=1 + nested_peers,
    )
    output = await _validated_output(
        raw_output,
        output_type,
        guardrails,
        run_id=context.run_id,
        tenant=context.tenant,
    )
    return output, usage, estimated


async def _bounded_call[InputT: BaseModel](
    foreign: Callable[[InputT, ForeignAgentContext], object],
    request: InputT,
    context: ForeignAgentContext,
    *,
    ceiling_seconds: float,
    origin: str,
    name: str,
    idempotency: Idempotency,
) -> object:
    """Race work against kit timeout and cooperative parent cancellation."""
    work = asyncio.create_task(_call_foreign(foreign, request, context))
    cancelled = (
        asyncio.create_task(context.cancellation.wait())
        if context.cancellation is not None
        else None
    )
    try:
        waiting = {work, *(() if cancelled is None else (cancelled,))}
        done, _ = await asyncio.wait(
            waiting, timeout=ceiling_seconds, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        if cancelled is not None and cancelled in done:
            if idempotency is Idempotency.EFFECTFUL:
                raise _indeterminate(name, "was cancelled after dispatch", context)
            raise CancelledError(
                cancelled.result(),
                run_id=context.run_id,
                tenant=context.tenant,
            )
        if idempotency is Idempotency.EFFECTFUL:
            raise _indeterminate(name, "timed out after dispatch", context)
        raise ProviderTimeoutError(
            f"foreign agent {name!r} did not answer within {ceiling_seconds:g}s",
            provider=origin,
            run_id=context.run_id,
            tenant=context.tenant,
            details={"agent": name, "timeout_seconds": f"{ceiling_seconds:g}"},
        )
    except asyncio.CancelledError:
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise
    finally:
        if cancelled is not None:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)


def _indeterminate(
    name: str,
    reason: str,
    context: ForeignAgentContext,
) -> IndeterminateOutcomeError:
    return IndeterminateOutcomeError(
        f"foreign agent {name!r} {reason}; its effectful outcome is unknown",
        run_id=context.run_id,
        tenant=context.tenant,
        details={"agent": name, "reason": reason},
    )


async def _call_foreign[InputT: BaseModel](
    foreign: Callable[[InputT, ForeignAgentContext], object],
    request: InputT,
    context: ForeignAgentContext,
) -> object:
    """Keep a synchronous foreign body off the event loop and await hybrid callables."""
    async_body = inspect.iscoroutinefunction(foreign) or inspect.iscoroutinefunction(
        type(foreign).__call__
    )
    produced = (
        foreign(request, context)
        if async_body
        else await asyncio.to_thread(foreign, request, context)
    )
    if inspect.isawaitable(produced):
        return await cast("Awaitable[object]", produced)
    return produced


async def _validated_output[OutputT: BaseModel](
    raw: object,
    output_type: type[OutputT],
    guardrails: GuardrailPipeline | None,
    *,
    run_id: str,
    tenant: str,
) -> OutputT:
    """Guard raw foreign data, then strictly validate it against the declared model."""
    text = raw if isinstance(raw, str) else _json_text(raw)
    checked = await guardrails.check_output(text) if guardrails is not None else text
    try:
        payload = json.loads(checked)
    except (TypeError, ValueError) as mismatch:
        raise _schema_error(output_type, mismatch, raw, run_id=run_id, tenant=tenant) from mismatch
    try:
        return output_type.model_validate(payload, strict=True)
    except ValidationError as mismatch:
        raise _schema_error(output_type, mismatch, raw, run_id=run_id, tenant=tenant) from mismatch


def _preflight(budget: BudgetPolicy, projected: Usage, *, tenant: str) -> None:
    """Refuse projected tokens, cost or peer invocation against the public remainder."""
    remaining = budget.limits()
    checks: tuple[tuple[str, Decimal], ...] = (
        ("max_cost", projected.cost.total if projected.cost is not None else Decimal(0)),
        ("max_input_tokens", Decimal(projected.input_tokens)),
        ("max_output_tokens", Decimal(projected.output_tokens)),
        ("max_peer_invocations", Decimal(1)),
    )
    for name, wanted in checks:
        left_value = getattr(remaining, name)
        if left_value is None or wanted <= Decimal(str(left_value)):
            continue
        stated_value = getattr(budget.resolved.limits, name)
        limit = None if stated_value is None else Decimal(str(stated_value))
        left = max(Decimal(str(left_value)), Decimal(0))
        consumed = Decimal(0) if limit is None else max(limit - left, Decimal(0))
        scope = budget.resolved.sources.get(name)
        raise BudgetExceededError(
            f"{name} has {left} remaining and the wrapped agent projects {wanted}",
            breached=name,
            scope=scope,
            limit=limit,
            consumed=consumed,
            remaining=left,
            tenant=tenant,
            details={"projected": str(wanted)},
        )


def _lineage(
    name: str,
    parent: RunContext | None,
    policy_depth: int,
    budget: BudgetPolicy,
) -> tuple[int, tuple[str, ...]]:
    """Calculate and enforce wrapper, budget and cycle bounds before foreign work."""
    depth = parent.depth + 1 if parent is not None else 0
    path = (*parent.path, name) if parent is not None else (name,)
    if parent is not None and name in parent.path:
        raise RecursionLimitError(
            f"{name!r} is already on {'/'.join(parent.path)}; a foreign callback would cycle"
        )
    budget_depth = budget.resolved.limits.max_delegation_depth
    ceiling = policy_depth if budget_depth is None else min(policy_depth, budget_depth)
    if depth > ceiling:
        raise RecursionLimitError(
            f"wrapped agent {name!r} would sit at depth {depth}, beyond {ceiling}"
        )
    return depth, path


def _context(
    *,
    run_id: str,
    tenant: str,
    user: str | None,
    caller_scopes: Collection[str],
    policy_scopes: Collection[str],
    tools: tuple[str, ...],
    trace: Mapping[str, str],
    cancellation: CancellationToken | None,
    budget: BudgetPolicy,
    depth: int,
    path: tuple[str, ...],
    idempotency_key: str | None = None,
) -> ForeignAgentContext:
    """Build the credential-free context with ordered intersections and trace filtering."""
    return ForeignAgentContext(
        run_id=run_id,
        tenant=tenant,
        user=user,
        scopes=_intersection(policy_scopes, tuple(caller_scopes)),
        tools=tools,
        trace={key: value for key, value in trace.items() if key.lower() in _TRACE_KEYS},
        cancellation=cancellation,
        budget=budget,
        depth=depth,
        path=path,
        idempotency_key=idempotency_key,
    )


def _validate_boundary(
    foreign: object,
    name: str,
    input_type: object,
    output_type: object,
) -> None:
    """Refuse malformed public wiring at construction rather than first invocation."""
    if not callable(foreign):
        raise ConfigurationError("a wrapped foreign agent must be callable")
    if not name.strip():
        raise ConfigurationError("a wrapped foreign agent needs a non-empty name")
    if not isinstance(input_type, type) or not issubclass(input_type, BaseModel):
        raise ConfigurationError("input_type must be a Pydantic BaseModel subclass")
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise ConfigurationError("output_type must be a Pydantic BaseModel subclass")


def _intersection(declared: Collection[str], held: Collection[str]) -> tuple[str, ...]:
    """Keep declaration order while ensuring the result is never wider than either side."""
    allowed = set(held)
    return tuple(value for value in declared if value in allowed)


def _origin(foreign: object) -> str:
    """Stable source attribution without addresses or repr payloads."""
    kind = type(foreign) if not inspect.isfunction(foreign) else foreign
    module = getattr(kind, "__module__", "foreign")
    name = getattr(kind, "__qualname__", type(foreign).__name__)
    return f"foreign-agent:{module}.{name}"


def _render_model(value: BaseModel) -> str:
    """Canonical JSON for the untrusted hand-back envelope and run record."""
    return json.dumps(
        value.model_dump(mode="json", by_alias=True, exclude_none=False, round_trip=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_text(value: object) -> str:
    """Render a foreign value as JSON or retain it as the schema error payload."""
    try:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", by_alias=True, exclude_none=False)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as mismatch:
        raise SchemaViolationError(
            "foreign agent output is not JSON serialisable",
            payload=value,
        ) from mismatch


def _json_object(value: object) -> Mapping[str, object]:
    """Parse one bounded JSON object, refusing duplicated keys."""
    if isinstance(value, BaseModel):
        parsed: object = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    elif isinstance(value, str | bytes):
        if len(value) > 64 * 1024:
            raise ValueError("arguments exceed the 65536-byte ceiling")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            keys = [key for key, _ in pairs]
            duplicated = sorted({key for key in keys if keys.count(key) > 1})
            if duplicated:
                raise ValueError(f"duplicate keys: {', '.join(duplicated)}")
            return dict(pairs)

        parsed = json.loads(value, object_pairs_hook=unique)
    elif isinstance(value, Mapping):
        mapped: dict[str, object] = {}
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise TypeError("tool argument names must be strings")
            mapped[key] = item
        parsed = mapped
    else:
        raise TypeError("tool arguments must be a JSON object")
    if not isinstance(parsed, Mapping):
        raise TypeError("tool arguments must be a JSON object")
    return cast("Mapping[str, object]", parsed)


def _schema_error(
    model: type[BaseModel],
    mismatch: Exception,
    payload: object,
    *,
    run_id: str,
    tenant: str,
) -> SchemaViolationError:
    """Translate Pydantic or JSON mismatch while retaining the raw foreign reply."""
    problems = _validation_problems(mismatch)
    return SchemaViolationError(
        f"{model.__name__} rejected the foreign payload",
        model=model.__name__,
        paths=tuple(sorted(problems)),
        problems=problems,
        payload=payload,
        run_id=run_id,
        tenant=tenant,
    )


def _validation_problems(mismatch: Exception) -> dict[str, str]:
    """Return every safe path and message without echoing field values."""
    if isinstance(mismatch, ValidationError):
        return {
            ".".join(str(part) for part in error["loc"]): str(error["msg"])
            for error in mismatch.errors()
        }
    return {"": str(mismatch)}
