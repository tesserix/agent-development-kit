"""Expose kit agents without surrendering the kit's execution boundaries.

An exported descriptor is metadata, not authority. Every invocation still enters with a
verified principal, an explicit tenant, user and scopes, and runs through the normal
runner. The caller's W3C context becomes ambient context for downstream tools and peers;
budgets, guardrails and structured output remain owned by the runner. Failures leave as a
versioned, non-secret envelope so a foreign orchestrator can retry faults without retrying
budget, policy or schema decisions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from tesserix_adk.adapters.a2a import (
    A2ABearerSecurity,
    A2AInterface,
    A2APrincipalResolver,
    A2ASkill,
    a2a_agent_executor,
    a2a_card_for,
)
from tesserix_adk.adapters.mcp_server import McpServer
from tesserix_adk.core import (
    STRICT_SUBSET,
    AdkError,
    AdkModel,
    AgentDefinition,
    ApprovalPolicy,
    AuthorisationError,
    BudgetExceededError,
    BudgetUnavailableError,
    CancelledError,
    ConfigurationError,
    GuardrailError,
    MissingTenantContextError,
    Principal,
    ProviderError,
    Run,
    RunEventKind,
    RunState,
    SchemaViolationError,
    ToolArgumentValidationError,
    TypedAgent,
    TypedAgentDefinition,
    Usage,
    schema_for,
)
from tesserix_adk.core.identity import ScopeSet, principal_scope
from tesserix_adk.runtime.blocking import Ambient, carrying
from tesserix_adk.tools import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.types import AgentCard
    from pydantic import JsonValue

    from tesserix_adk.core.protocols import BudgetPolicy
    from tesserix_adk.runtime import AgentRunner
    from tesserix_adk.runtime.cancellation import CancellationToken
    from tesserix_adk.tools.validation import ArgumentValidator

__all__ = [
    "ExportDescriptorDriftError",
    "ExportErrorCode",
    "ExportErrorEnvelope",
    "ExportInvocation",
    "ExportedA2AAgent",
    "ExportedAgentResult",
    "ExportedAgentTool",
    "export_as_a2a",
    "export_as_mcp_tool",
    "export_as_tool",
]

EXPORT_ENVELOPE_VERSION: Final[Literal["1.0"]] = "1.0"
"""Semver-governed wire version of :class:`ExportedAgentResult`."""

_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NOTHING = Usage(input_tokens=0, output_tokens=0)
_TRACE_KEYS = frozenset({"baggage", "traceparent", "tracestate"})


class ExportErrorCode(StrEnum):
    """Stable categories a foreign orchestrator may branch on."""

    AUTHORISATION = "authorisation"
    BUDGET_REFUSAL = "budget_refusal"
    CANCELLED = "cancelled"
    EXECUTION_FAILED = "execution_failed"
    GUARDRAIL_BLOCK = "guardrail_block"
    INVALID_ARGUMENTS = "invalid_arguments"
    PROVIDER_OUTAGE = "provider_outage"
    SCHEMA_VIOLATION = "schema_violation"
    TIMEOUT = "timeout"


class ExportDescriptorDriftError(AdkError):
    """Raised when a consumer's pinned function descriptor no longer matches.

    Args:
        expected: Fingerprint the consumer pinned.
        actual: Fingerprint this export currently emits.
    """

    def __init__(self, message: str, *, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(message, details={"expected": expected, "actual": actual})


class ExportErrorEnvelope(AdkModel):
    """A non-secret failure shape shared by direct, MCP and framework callers.

    `message` is chosen by the adapter and never copies model output, tool arguments,
    provider bodies or credentials. `error_type` is diagnostic; callers branch only on
    `code` and `retryable`, whose meanings are stable for this envelope version.
    """

    code: ExportErrorCode
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class ExportInvocation:
    """Authenticated ingress context supplied beside model-controlled arguments.

    Args:
        tenant: Tenant the foreign request claims to act for.
        user: Acting subject attributed to every run record and hook.
        scopes: Authority requested for this call. It must be a subset of ``principal``.
        trace: W3C trace fields propagated inward; no credential metadata is accepted.
        principal: Identity established by the host's authentication layer. The adapter
            compares it with every claim before starting a run.
        run_id: Foreign parent/run correlation identifier, where one exists.
        budget: Optional narrower shared policy. With none, the runner's normal bounded
            policy applies; this never means unbounded.
        cancellation: Foreign parent's cooperative cancellation switch.
    """

    tenant: str
    user: str
    scopes: tuple[str, ...]
    trace: Mapping[str, str]
    principal: Principal
    run_id: str | None = None
    budget: BudgetPolicy | None = None
    cancellation: CancellationToken | None = None

    def with_budget(self, budget: BudgetPolicy) -> ExportInvocation:
        """Return the same authenticated call narrowed by a shared budget policy."""
        return replace(self, budget=budget)


class ExportedAgentResult[OutputT: BaseModel](AdkModel):
    """Versioned success-or-error result serialised to foreign frameworks.

    Successful results contain a runner-validated ``output``. Failed results contain one
    :class:`ExportErrorEnvelope` and never a partial or invented output. Usage and identity
    remain present for terminal runs so cost and attribution need no per-caller wiring.
    """

    schema_version: Literal["1.0"] = EXPORT_ENVELOPE_VERSION
    ok: bool
    run_id: str = ""
    tenant: str = ""
    user: str | None = None
    state: RunState
    output: OutputT | None = None
    usage: Usage = Field(default_factory=lambda: _NOTHING)
    error: ExportErrorEnvelope | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> ExportedAgentResult[OutputT]:
        if self.ok and (self.output is None or self.error is not None):
            raise ValueError("a successful export needs output and no error")
        if not self.ok and (self.output is not None or self.error is None):
            raise ValueError("a failed export needs one error and no output")
        return self


@dataclass(frozen=True, slots=True)
class ExportedA2AAgent:
    """Official A2A card and executor produced by the existing A2A adapter."""

    card: AgentCard
    executor: AgentExecutor


class _ExportArguments[InputT: str | BaseModel]:
    """Hold foreign function arguments to the descriptor the caller saw."""

    __slots__ = ("_input_type", "_tool")

    def __init__(self, input_type: type[InputT], tool: str) -> None:
        self._input_type = input_type
        self._tool = tool

    def arguments(self, arguments: object) -> Mapping[str, object]:
        """Return one typed request or raise a redacted typed validation failure."""
        try:
            payload = _object(arguments)
            if set(payload) == {"request"}:
                validated = payload["request"]
                if self._input_type is str and isinstance(validated, str):
                    return {"request": validated}
                if self._input_type is not str and isinstance(validated, self._input_type):
                    return {"request": validated}
            if self._input_type is str:
                if set(payload) != {"input"} or not isinstance(payload.get("input"), str):
                    raise ValueError("a string-input export takes one string field named input")
                request: object = payload["input"]
            else:
                model_type = cast("type[BaseModel]", self._input_type)
                request = model_type.model_validate(payload, strict=True)
        except (TypeError, ValueError, ValidationError) as mismatch:
            problems = _problems(mismatch)
            raise ToolArgumentValidationError(
                f"{self._tool} arguments do not satisfy its exported input schema",
                tool=self._tool,
                paths=tuple(sorted(problems)),
                problems=problems,
                payload=arguments,
            ) from mismatch
        return {"request": request}


class ExportedAgentTool[InputT: str | BaseModel, OutputT: BaseModel]:
    """An OpenAI descriptor, typed callable and kit tool backed by one agent.

    Use :func:`export_as_tool` rather than constructing this class. The descriptor is
    canonical and fingerprinted. :meth:`invoke` authenticates before parsing arguments,
    runs the normal runner under narrowed principal and W3C context, and always returns a
    stable result envelope. It is deliberately non-streaming; SSE and long-lived work use
    the runtime streaming or workflows surfaces rather than pretending one function call
    has resumable transport semantics.
    """

    __slots__ = (
        "_agent",
        "_descriptor",
        "_fingerprint",
        "_input_type",
        "_output_type",
        "_runner",
        "_timeout",
        "_tool",
        "_validator",
    )

    def __init__(
        self,
        runner: AgentRunner,
        agent: TypedAgent[InputT, OutputT] | TypedAgentDefinition[InputT, OutputT],
        *,
        name: str,
        description: str,
        timeout_seconds: float,
    ) -> None:
        declared = agent.agent if isinstance(agent, AgentDefinition) else agent
        if declared.output_type is None:
            raise ConfigurationError(
                "an agent exported as a function needs structured output; free text has no "
                "payload schema a foreign caller can validate"
            )
        if not _FUNCTION_NAME.fullmatch(name):
            raise ConfigurationError(
                f"{name!r} is not an OpenAI-compatible function name (1-64 letters, "
                "numbers, underscores or hyphens)"
            )
        if timeout_seconds <= 0:
            raise ConfigurationError("an exported call timeout must be positive")
        self._runner = runner
        self._agent = agent
        self._input_type = declared.input_type
        self._output_type = declared.output_type
        self._timeout = timeout_seconds
        self._validator = _ExportArguments(self._input_type, name)
        parameters = _input_schema(self._input_type)
        self._descriptor: dict[str, JsonValue] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "strict": True,
                "parameters": parameters,
            },
        }
        self._fingerprint = _fingerprint(self._descriptor)
        validator: ArgumentValidator = self._validator
        result_type = _result_type(self._output_type)
        self._tool = Tool[..., ExportedAgentResult[OutputT]](
            name=name,
            description=description,
            parameters_schema=parameters,
            returns_schema=schema_for(result_type, dialect=STRICT_SUBSET),
            is_async=True,
            function=self._tool_call,
            validator=validator,
            context_parameter="context",
            context_required=True,
            timeout=timeout_seconds,
            parallel_safe=True,
            approval=ApprovalPolicy(required=False),
            returns_type=result_type,
            origin=f"agent-export:{declared.name}@{declared.version}",
        )

    @property
    def descriptor(self) -> dict[str, JsonValue]:
        """A detached OpenAI-compatible descriptor safe for JSON serialisation."""
        return cast("dict[str, JsonValue]", json.loads(json.dumps(self._descriptor)))

    @property
    def descriptor_fingerprint(self) -> str:
        """Canonical SHA-256 fingerprint consumers can pin against descriptor drift."""
        return self._fingerprint

    @property
    def streaming(self) -> Literal[False]:
        """Whether this function export emits SSE. It never does."""
        return False

    @property
    def tool(self) -> Tool[..., ExportedAgentResult[OutputT]]:
        """The same callable as a kit tool, used by the MCP export helper."""
        return self._tool

    def assert_descriptor(self, expected: str) -> None:
        """Refuse descriptor drift against a caller-pinned fingerprint.

        Raises:
            ExportDescriptorDriftError: If ``expected`` differs from this export.
        """
        if expected != self._fingerprint:
            raise ExportDescriptorDriftError(
                "the exported agent descriptor changed from the caller's pinned shape",
                expected=expected,
                actual=self._fingerprint,
            )

    async def invoke(
        self,
        arguments: Mapping[str, object] | str | bytes,
        context: ExportInvocation | None,
    ) -> ExportedAgentResult[OutputT]:
        """Invoke the export under authenticated context and return a stable envelope.

        Authentication is checked before argument parsing or provider work. All kit errors
        are converted to non-secret envelopes; surrounding task cancellation remains
        ``asyncio.CancelledError`` and is never swallowed.
        """
        try:
            authorised = _authorised(context)
        except AdkError as refused:
            return _exception_result(self._output_type, refused, context)
        try:
            checked = self._validator.arguments(arguments)
            request = cast("InputT", checked["request"])
        except AdkError as refused:
            return _exception_result(self._output_type, refused, authorised)
        return await self._run(request, authorised)

    async def _tool_call(
        self, request: InputT, context: ToolContext
    ) -> ExportedAgentResult[OutputT]:
        """Adapt authenticated MCP ambient context to the same direct call path."""
        if not context.user:
            return _exception_result(
                self._output_type,
                AuthorisationError(
                    "an exported agent requires an authenticated user",
                    agent=self._tool.name,
                    where="agent export",
                ),
                None,
            )
        invocation = ExportInvocation(
            tenant=context.tenant,
            user=context.user,
            scopes=context.scopes,
            trace=context.trace,
            principal=Principal(
                subject=context.user,
                tenant=context.tenant,
                scopes=frozenset(context.scopes),
            ),
            run_id=context.run_id,
            budget=context.budget,
            cancellation=context.cancellation,
        )
        return await self._run(request, invocation)

    async def _run(
        self, request: InputT, context: ExportInvocation
    ) -> ExportedAgentResult[OutputT]:
        """Bind narrowed authority and drive the normal runner inside a hard ceiling."""
        principal = context.principal.model_copy(update={"scopes": frozenset(context.scopes)})
        ambient = Ambient(
            run_id=context.run_id or f"export-{self._tool.name}",
            tenant=context.tenant,
            user=context.user,
            cancellation=context.cancellation,
            scopes=context.scopes,
            trace=_trace(context.trace),
            budget=context.budget,
        )
        try:
            with principal_scope(principal), carrying(ambient):
                async with asyncio.timeout(self._timeout):
                    run = await self._runner.run_typed(
                        self._agent,
                        request,
                        tenant=context.tenant,
                        user=context.user,
                        run_id=context.run_id,
                        cancellation=context.cancellation,
                        budget=context.budget,
                    )
        except TimeoutError:
            return _error_result(
                self._output_type,
                code=ExportErrorCode.TIMEOUT,
                error_type="TimeoutError",
                retryable=True,
                context=context,
            )
        except AdkError as failure:
            return _exception_result(self._output_type, failure, context)
        return _run_result(self._output_type, run)


def export_as_tool[InputT: str | BaseModel, OutputT: BaseModel](
    runner: AgentRunner,
    agent: TypedAgent[InputT, OutputT] | TypedAgentDefinition[InputT, OutputT],
    *,
    description: str,
    name: str | None = None,
    timeout_seconds: float = 30.0,
) -> ExportedAgentTool[InputT, OutputT]:
    """Export one structured kit agent as descriptor, callable and reusable kit tool.

    Raises:
        ConfigurationError: If the export has free-text output, an invalid function name,
            an empty description or a non-positive timeout.
    """
    declared = agent.agent if isinstance(agent, AgentDefinition) else agent
    if not description.strip():
        raise ConfigurationError("an exported function needs a public description")
    return ExportedAgentTool(
        runner,
        agent,
        name=name or declared.name,
        description=description.strip(),
        timeout_seconds=timeout_seconds,
    )


def export_as_mcp_tool[InputT: str | BaseModel, OutputT: BaseModel](
    runner: AgentRunner,
    agent: TypedAgent[InputT, OutputT] | TypedAgentDefinition[InputT, OutputT],
    *,
    description: str,
    name: str | None = None,
    timeout_seconds: float = 30.0,
    per_tenant_calls: int = 8,
    secrets: Sequence[str] = (),
) -> McpServer:
    """Publish an agent through the existing authenticated MCP server helper.

    The server requires ``authenticated=`` when a session connects; request tenant
    metadata is still compared with it by the normal MCP ingress. The generated tool,
    descriptor, validation, timeout and error envelope are the same as direct export.
    """
    exported = export_as_tool(
        runner,
        agent,
        description=description,
        name=name,
        timeout_seconds=timeout_seconds,
    )
    registry = ToolRegistry((exported.tool,))
    declared = agent.agent if isinstance(agent, AgentDefinition) else agent
    view = registry.view(allow=(exported.tool.name,), agent=declared.name)
    return McpServer(
        view,
        exports=(exported.tool.name,),
        name=declared.name,
        version=declared.version,
        per_tenant_calls=per_tenant_calls,
        secrets=secrets,
        require_authenticated_tenant=True,
    )


def export_as_a2a[OutputT: BaseModel](
    runner: AgentRunner,
    definition: AgentDefinition[OutputT],
    *,
    resolve: A2APrincipalResolver,
    description: str,
    provider_url: str,
    interfaces: Iterable[A2AInterface],
    skills: Iterable[A2ASkill],
    documentation_url: str = "",
    default_input_modes: Sequence[str] = ("text/plain",),
    default_output_modes: Sequence[str] = ("text/plain",),
    streaming: bool = False,
    push_notifications: bool = False,
    extended_agent_card: bool = False,
    security: A2ABearerSecurity | None = None,
    max_input_bytes: int = 64 * 1024,
    max_output_bytes: int = 1024 * 1024,
) -> ExportedA2AAgent:
    """Return the official A2A card and authenticated executor for one definition.

    Card schema, request mapping and principal resolution are delegated to the kit's
    official A2A adapter rather than reimplemented here. Missing optional dependencies and
    invalid card metadata therefore raise that adapter's typed errors unchanged.
    """
    card = a2a_card_for(
        definition,
        description=description,
        provider_url=provider_url,
        interfaces=interfaces,
        skills=skills,
        documentation_url=documentation_url,
        default_input_modes=default_input_modes,
        default_output_modes=default_output_modes,
        streaming=streaming,
        push_notifications=push_notifications,
        extended_agent_card=extended_agent_card,
        security=security,
    )
    executor = a2a_agent_executor(
        runner,
        definition,
        resolve=resolve,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )
    return ExportedA2AAgent(card=card, executor=executor)


def _authorised(context: ExportInvocation | None) -> ExportInvocation:
    """Verify every foreign identity claim and narrow scopes before any parsing or work."""
    if context is None or not context.tenant.strip():
        raise MissingTenantContextError("an exported agent call names no authenticated tenant")
    if not context.user.strip():
        raise AuthorisationError(
            "an exported agent call names no authenticated user", where="agent export"
        )
    principal = context.principal
    if context.tenant != principal.tenant:
        raise AuthorisationError(
            "the requested tenant differs from the authenticated principal",
            subject=principal.subject,
            where="agent export",
        )
    if context.user != principal.subject:
        raise AuthorisationError(
            "the requested user differs from the authenticated principal",
            subject=principal.subject,
            where="agent export",
        )
    requested = ScopeSet.of(*context.scopes)
    beyond = requested.names - principal.granted.names
    if beyond:
        raise AuthorisationError(
            "the exported call requests scopes the authenticated principal does not hold",
            scope=sorted(beyond)[0],
            subject=principal.subject,
            where="agent export",
        )
    return context


def _run_result[OutputT: BaseModel](
    output_type: type[OutputT], run: Run[OutputT]
) -> ExportedAgentResult[OutputT]:
    """Map a terminal run to success or a stable decision/fault category."""
    if run.state is RunState.COMPLETED and run.output is not None:
        carrier = _result_type(output_type)
        return carrier(
            ok=True,
            run_id=run.id,
            tenant=run.tenant,
            user=run.user,
            state=run.state,
            output=run.output,
            usage=run.usage,
        )
    code, error_type, retryable = _terminal_error(run)
    return _error_result(
        output_type,
        code=code,
        error_type=error_type,
        retryable=retryable,
        run=run,
    )


def _terminal_error[OutputT: BaseModel](
    run: Run[OutputT],
) -> tuple[ExportErrorCode, str, bool]:
    """Classify a terminal run without parsing or publishing its potentially sensitive prose."""
    kinds = {event.kind for event in run.events}
    terminal = next(
        (
            event.detail or ""
            for event in reversed(run.events)
            if event.kind is RunEventKind.TERMINATED
        ),
        "",
    )
    error_type = terminal.partition(":")[0] or type(run.state).__name__
    if run.state is RunState.BUDGET_EXHAUSTED or RunEventKind.BUDGET_EXCEEDED in kinds:
        return ExportErrorCode.BUDGET_REFUSAL, error_type, False
    if RunEventKind.GUARDRAIL_REFUSAL in kinds:
        return ExportErrorCode.GUARDRAIL_BLOCK, error_type, False
    if RunEventKind.SCHEMA_VIOLATION in kinds:
        return ExportErrorCode.SCHEMA_VIOLATION, error_type, False
    if run.state is RunState.CANCELLED:
        return ExportErrorCode.CANCELLED, error_type, False
    if "Provider" in error_type or "FallbackExhausted" in error_type:
        return ExportErrorCode.PROVIDER_OUTAGE, error_type, True
    return ExportErrorCode.EXECUTION_FAILED, error_type, False


def _exception_result[OutputT: BaseModel](
    output_type: type[OutputT],
    failure: AdkError,
    context: ExportInvocation | None,
) -> ExportedAgentResult[OutputT]:
    """Map a raised kit error to the same wire vocabulary as terminal runs."""
    if isinstance(failure, AuthorisationError | MissingTenantContextError):
        code = ExportErrorCode.AUTHORISATION
    elif isinstance(failure, BudgetExceededError | BudgetUnavailableError):
        code = ExportErrorCode.BUDGET_REFUSAL
    elif isinstance(failure, GuardrailError):
        code = ExportErrorCode.GUARDRAIL_BLOCK
    elif isinstance(failure, SchemaViolationError):
        code = (
            ExportErrorCode.INVALID_ARGUMENTS
            if isinstance(failure, ToolArgumentValidationError)
            else ExportErrorCode.SCHEMA_VIOLATION
        )
    elif isinstance(failure, ProviderError):
        code = ExportErrorCode.PROVIDER_OUTAGE
    elif isinstance(failure, CancelledError):
        code = ExportErrorCode.CANCELLED
    else:
        code = ExportErrorCode.EXECUTION_FAILED
    return _error_result(
        output_type,
        code=code,
        error_type=type(failure).__name__,
        retryable=failure.retryable,
        context=context,
    )


def _error_result[OutputT: BaseModel](
    output_type: type[OutputT],
    *,
    code: ExportErrorCode,
    error_type: str,
    retryable: bool,
    context: ExportInvocation | None = None,
    run: Run[OutputT] | None = None,
) -> ExportedAgentResult[OutputT]:
    """Build one failure envelope with no source-controlled prose."""
    carrier = _result_type(output_type)
    messages = {
        ExportErrorCode.AUTHORISATION: "The caller is not authorised for this agent export.",
        ExportErrorCode.BUDGET_REFUSAL: "The run budget refused this invocation.",
        ExportErrorCode.CANCELLED: "The caller cancelled this invocation.",
        ExportErrorCode.EXECUTION_FAILED: "The agent invocation failed.",
        ExportErrorCode.GUARDRAIL_BLOCK: "A guardrail refused this invocation.",
        ExportErrorCode.INVALID_ARGUMENTS: "The arguments do not match the exported schema.",
        ExportErrorCode.PROVIDER_OUTAGE: "The model provider was unavailable.",
        ExportErrorCode.SCHEMA_VIOLATION: "The agent output did not match its schema.",
        ExportErrorCode.TIMEOUT: "The exported invocation exceeded its timeout.",
    }
    return carrier(
        ok=False,
        run_id=run.id if run is not None else (context.run_id or "" if context else ""),
        tenant=run.tenant if run is not None else (context.tenant if context else ""),
        user=run.user if run is not None else (context.user if context else None),
        state=run.state if run is not None else RunState.FAILED,
        usage=run.usage if run is not None else _NOTHING,
        error=ExportErrorEnvelope(
            code=code,
            error_type=error_type or "UnknownError",
            message=messages[code],
            retryable=retryable,
        ),
    )


def _result_type[OutputT: BaseModel](
    output_type: type[OutputT],
) -> type[ExportedAgentResult[OutputT]]:
    """Parameterise the Pydantic carrier so output never serialises as a bare model."""
    return cast(
        "type[ExportedAgentResult[OutputT]]",
        ExportedAgentResult.__class_getitem__(output_type),
    )


def _input_schema[InputT: str | BaseModel](input_type: type[InputT]) -> dict[str, JsonValue]:
    """Return an OpenAI strict function parameter object for either supported input kind."""
    if input_type is str:
        return {
            "additionalProperties": False,
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "type": "object",
        }
    return schema_for(cast("type[BaseModel]", input_type), dialect=STRICT_SUBSET)


def _object(value: object) -> Mapping[str, object]:
    """Read a bounded JSON object and refuse duplicated keys."""
    if isinstance(value, BaseModel):
        parsed: object = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    elif isinstance(value, str | bytes):
        if len(value) > 64 * 1024:
            raise ValueError("arguments exceed the 65536-byte ceiling")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            keys = [key for key, _ in pairs]
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            if duplicates:
                raise ValueError(f"duplicate keys: {', '.join(duplicates)}")
            return dict(pairs)

        parsed = json.loads(value, object_pairs_hook=unique)
    elif isinstance(value, Mapping):
        mapped: dict[str, object] = {}
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise TypeError("argument names must be strings")
            mapped[key] = item
        parsed = mapped
    else:
        raise TypeError("function arguments must be a JSON object")
    if not isinstance(parsed, Mapping):
        raise TypeError("function arguments must be a JSON object")
    return cast("Mapping[str, object]", parsed)


def _problems(mismatch: Exception) -> dict[str, str]:
    """Return safe validation paths and messages without field values."""
    if isinstance(mismatch, ValidationError):
        return {
            ".".join(str(part) for part in error["loc"]): str(error["msg"])
            for error in mismatch.errors()
        }
    return {"": str(mismatch)}


def _trace(trace: Mapping[str, str]) -> dict[str, str]:
    """Keep W3C propagation only; arbitrary metadata cannot smuggle credentials inward."""
    return {key.lower(): value for key, value in trace.items() if key.lower() in _TRACE_KEYS}


def _fingerprint(descriptor: Mapping[str, JsonValue]) -> str:
    """Hash canonical JSON so mapping order cannot look like descriptor drift."""
    encoded = json.dumps(descriptor, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
