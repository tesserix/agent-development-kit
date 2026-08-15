"""Error hierarchy for the kit.

Every failure the kit raises inherits from `AdkError`, so a consumer can catch this
kit's failures without catching `Exception` and swallowing its own bugs alongside.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "RETRYABLE_STATUS",
    "AdkError",
    "AggregationError",
    "ApprovalBindingError",
    "ApprovalDeliveryError",
    "ApprovalDeniedError",
    "ApprovalExpiredError",
    "ApprovalTokenError",
    "AttributionError",
    "AuditUnavailableError",
    "AuthenticationError",
    "AutonomyRefusedError",
    "BudgetExceededError",
    "BudgetUnavailableError",
    "CancelledError",
    "CapabilityError",
    "CeilingExceededError",
    "CheckpointFormatError",
    "CheckpointTooLargeError",
    "ChunkingError",
    "ClaimUnavailableError",
    "ConfigurationError",
    "ContentFilteredError",
    "ContextBudgetError",
    "ContextWindowExceededError",
    "DelegationError",
    "DelegationLimitError",
    "DependencyCycleError",
    "EmbeddingDimensionError",
    "EmbeddingUnavailableError",
    "EstimateUnavailableError",
    "EvalIncompleteError",
    "EventLoopStalledError",
    "ExtractionError",
    "FallbackExhaustedError",
    "FallbackUnsafeError",
    "FanOutLimitError",
    "GrantRevokedError",
    "GuardrailError",
    "GuardrailEvaluationError",
    "GuardrailViolationError",
    "HandoffContractError",
    "HookEvaluationError",
    "HookRefusedError",
    "HookRegistrationError",
    "IncomparableEvalError",
    "IncompatiblePromptVersionError",
    "IndeterminateToolCallError",
    "InexactAmountError",
    "InvalidRequestError",
    "LeaseLostError",
    "LoopLimitError",
    "MaxIterationsError",
    "MediaIntakeError",
    "MemoryConflictError",
    "MemoryContradictionError",
    "MemoryCorruptionError",
    "MemoryLimitError",
    "MemoryScopeError",
    "MemoryUnavailableError",
    "MissingExtraError",
    "MissingTenantContextError",
    "ModelArtifactError",
    "ModelResponseError",
    "NoEligibleModelError",
    "PartialErasureError",
    "PayloadTooLargeError",
    "PlanValidationError",
    "PoolExhaustedError",
    "PrefixDriftError",
    "PromptNotFoundError",
    "PromptRejectedError",
    "ProtocolConformanceError",
    "ProvenanceLostError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QueueUnavailableError",
    "RateLimitError",
    "RecursionLimitError",
    "RepeatedCallError",
    "ResumeConflictError",
    "RetrievalDegradedError",
    "RunningLoopError",
    "SandboxError",
    "SandboxMemoryError",
    "SandboxTimeoutError",
    "SchemaGenerationError",
    "SchemaViolationError",
    "ScopeEscalationError",
    "StateConflictError",
    "StateInUseError",
    "StateNotFoundError",
    "StatePersistenceError",
    "StreamInterruptedError",
    "TemplateError",
    "TenantContextError",
    "TenantCrossingError",
    "TenantLimitError",
    "TenantRefusal",
    "TenantUnconfiguredError",
    "ToolArgumentValidationError",
    "ToolDefinitionError",
    "ToolError",
    "ToolExecutionError",
    "ToolFailure",
    "ToolNotFoundError",
    "ToolNotPermittedError",
    "ToolRefusal",
    "ToolResultError",
    "ToolTimedOutError",
    "TrustBoundaryError",
    "UncitedClaimError",
    "UngroundedCitationError",
    "UnknownTenantError",
    "WorkItemNotFoundError",
    "WorkersBusyError",
    "WriteQueueFullError",
]

_DISTRIBUTION = "tesserix-adk"

# Faults, not answers: the same request sent later can succeed. 429 is here because a rate
# limit is transient by construction; a quota that is not transient is caught by the
# `Retry-After` ceiling rather than by retrying until it clears.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Which refusal a rejected tenant context is, so consumers branch on a value rather than
# on message text: `missing` and `malformed` are dead-letter cases, `contradicted` is an
# authorization event, `version` is a deploy-skew signal.
type TenantRefusal = Literal["missing", "malformed", "version", "contradicted", "oversized"]


class AdkError(Exception):
    """Base class for every error raised by the kit.

    Args:
        run_id: The run it happened in, where there was one. Configuration fails before
            any run exists, so this is optional rather than a value invented to fill it.
        tenant: The tenant the run belonged to.
        details: A debuggable payload — status codes, offending output, tool name.
            Never credentials, and never message content.
    """

    def __init__(
        self,
        *args: object,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        self.run_id = run_id
        self.tenant = tenant
        self.details: dict[str, str] = dict(details or {})
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """Carries the run and tenant: a bare message in a log is a fact nobody can act on."""
        where = f"run_id={self.run_id!r}, tenant={self.tenant!r}"
        return f"{type(self).__name__}({str(self)!r}, {where})"

    @property
    def retryable(self) -> bool:
        """Whether the same request could succeed on a second attempt.

        False here, and overridden only where a failure is a fault rather than an answer.
        A guardrail refusal, a budget ceiling and a schema violation are decisions: asking
        again spends more to be told the same thing.
        """
        return False


class ConfigurationError(AdkError):
    """Raised when the kit is assembled in a way that cannot work.

    Configuration failures are raised during construction, never on the first call
    that happens to exercise the broken setting.
    """


class TrustBoundaryError(ConfigurationError):
    """Raised when the only model left to try sits outside the run's trust boundary.

    Fails the run closed rather than degrading it. A chain that promotes a hosted vendor
    because the self-hosted endpoint is down has swapped a confidentiality guarantee for an
    availability one, which is a decision nobody made and a breach nobody logged.

    Args:
        excluded: The references the boundary refused, in chain order.
    """

    def __init__(
        self,
        *args: object,
        excluded: Sequence[str] = (),
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.excluded = tuple(excluded)
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class NoEligibleModelError(ConfigurationError):
    """Raised when nothing the routing table offers can do the work asked of it.

    A `ConfigurationError`, because that is what it is: the table is missing a model, not
    the request malformed. Never a downgrade — a model that cannot do the job is not a
    cheaper way of doing it, and quietly substituting one moves the failure to whichever
    step needed the capability.

    Args:
        task_class: What was asked for.
        unsatisfied: The requirement names nothing could meet.
        rejected: Each candidate considered and what it could not do, as
            `(ref, reason)` pairs.
    """

    def __init__(
        self,
        *args: object,
        task_class: str = "",
        unsatisfied: Sequence[str] = (),
        rejected: Sequence[tuple[str, str]] = (),
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.task_class = task_class
        self.unsatisfied = tuple(unsatisfied)
        self.rejected = tuple(_Rejection(ref, reason) for ref, reason in rejected)
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class _Rejection(NamedTuple):
    """One candidate the router passed over, on an error rather than in a decision."""

    ref: str
    reason: str


class MissingExtraError(AdkError, ImportError):
    """Raised when an optional integration is used without installing its extra.

    Also an `ImportError`, so existing `except ImportError` guards around an optional
    import keep working.

    Args:
        extra: The extra that installs the dependency, e.g. `redis`.
        module: The module that could not be imported.
    """

    def __init__(self, extra: str, module: str) -> None:
        self.extra = extra
        self.module = module
        self.install_command = f"uv add '{_DISTRIBUTION}[{extra}]'"
        super().__init__(
            f"{module} needs the optional '{extra}' extra, which is not installed. "
            f"Install it with: {self.install_command}",
            name=module,
        )


class ProtocolConformanceError(ConfigurationError):
    """Raised when an object does not provide every member of a protocol.

    Args:
        protocol: Name of the protocol that was not satisfied.
        missing: Member names absent from the object, sorted.
        obj_type: Name of the offending type.
    """

    def __init__(self, protocol: str, missing: tuple[str, ...], obj_type: str) -> None:
        self.protocol = protocol
        self.missing = missing
        self.obj_type = obj_type
        super().__init__(
            f"{obj_type} does not satisfy {protocol}: missing {', '.join(missing)}. "
            f"An implementation must provide every member before it is used, or the run "
            f"fails partway through instead of at construction."
        )


class CapabilityError(AdkError):
    """Raised when something is asked of a provider that it has not declared it can do.

    Raised before the request goes out: discovering a missing capability from a provider
    error message is discovering it after paying for it.

    Args:
        capability: What was required, named as the kit names it everywhere else.
        provider: Who lacks it. Two providers serve the same model ids, so the model
            alone does not say which record was read.
        model: Which model of that provider's was asked.
    """

    def __init__(
        self,
        *args: object,
        capability: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.capability = capability
        self.provider = provider
        self.model = model
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class ContextWindowExceededError(CapabilityError):
    """Raised when an assembled prompt is longer than the model declares it can read.

    A vendor handed a prompt past its window truncates it and answers anyway, so the
    first sign of the problem is an answer that ignores the beginning of the case. The
    kit refuses instead, against the declared window, before the request goes out.

    Args:
        counted: Tokens in the prompt, as the provider itself counted them. Zero where
            the vendor reported the overflow without saying by how much, which is what
            every vendor's own 400 for it does.
        limit: The window the provider declared. Zero where nothing declares one.
    """

    def __init__(
        self,
        *args: object,
        counted: int = 0,
        limit: int = 0,
        provider: str | None = None,
        model: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.counted = counted
        self.limit = limit
        super().__init__(
            *args, provider=provider, model=model, run_id=run_id, tenant=tenant, details=details
        )


class ModelArtifactError(ConfigurationError):
    """Raised when a model file on disk is not one that can be loaded.

    At load, never at the first query: a corrupt weights file discovered by the query that
    needed it has already told a user their request failed, and the operator finds out from
    them. A `ConfigurationError` because that is what a half-downloaded file is.

    Args:
        path: The file, as it was looked for.
        reason: Which check failed — `missing`, `empty` or `digest`.
    """

    def __init__(self, *args: object, path: str = "", reason: str = "") -> None:
        self.path = path
        self.reason = reason
        super().__init__(*args, details={"path": path, "reason": reason})


class MediaIntakeError(AdkError):
    """Raised when a document or recording could not be read into text.

    A file the kit cannot read must say so. Returning the empty string is the one outcome
    that is indistinguishable from a blank page, and it is the one that ends with an agent
    answering confidently from nothing.

    Args:
        path: The file, as it was given.
        reason: Which check failed — `unsupported`, `missing`, `empty` or `corrupt`.
    """

    def __init__(self, *args: object, path: str = "", reason: str = "") -> None:
        self.path = path
        self.reason = reason
        super().__init__(*args, details={"path": path, "reason": reason})


class ModelResponseError(AdkError):
    """Raised when a provider answered with something the kit cannot read as a response.

    Distinct from `SchemaViolationError`, which is a well-formed answer in the wrong
    shape and can be repaired. This is a payload that is not an answer at all, so it
    carries the raw body and the provider's request id: without them the report is
    "the provider returned something odd" and the trail ends there.

    Args:
        payload: What came back, kept verbatim for a debugger.
        request_id: The provider's own id for the call, which is what a support ticket
            is answered against.
    """

    def __init__(
        self,
        *args: object,
        payload: object = None,
        request_id: str | None = None,
        provider: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.request_id = request_id
        self.provider = provider
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class ProviderError(AdkError):
    """Raised when a model provider fails, in the kit's own vocabulary.

    Provider-specific bodies belong in `details`; the type is what a consumer branches
    on, so it is the same across providers.

    Args:
        status: The HTTP status the provider answered with, where it answered at all.
            `None` means the request never got that far — a reset connection, a DNS
            failure — which is the transient case, because a request the provider
            rejected always comes back with a status.
        retry_after: Seconds the provider asked the caller to wait, from its own
            `Retry-After` header. Believed in preference to any computed backoff, up to
            the policy's ceiling.
        provider: Who failed. A run that routes across vendors cannot tell from the
            message which one this was.
        model: Which of that provider's models was asked.
        request_id: The provider's own id for the call, which is what a support ticket
            is answered against.
    """

    def __init__(
        self,
        *args: object,
        status: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self.retry_after = retry_after
        self.provider = provider
        self.model = model
        self.request_id = request_id
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)

    @property
    def retryable(self) -> bool:
        """Transient statuses and transport faults, never a request the provider rejected."""
        return self.status is None or self.status in RETRYABLE_STATUS


class ProviderTimeoutError(ProviderError):
    """Raised when a provider did not answer inside the deadline.

    A `ProviderError`, so the common case is not missed by `except ProviderError`, and
    its own type, because a timeout is retryable where a 400 is not.
    """

    @property
    def retryable(self) -> bool:
        """Always. A timeout says nothing about the request, only about this attempt."""
        return True


class RateLimitError(ProviderError):
    """Raised when a provider refused the call because too many were made.

    Args:
        quota: Whether the limit is an exhausted allowance rather than a rate. A rate
            clears by waiting and a quota clears when somebody buys more, so retrying an
            exhausted quota is the same rejection at the same price until the attempts
            run out.
    """

    def __init__(
        self,
        *args: object,
        quota: bool = False,
        status: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.quota = quota
        super().__init__(
            *args,
            status=status,
            retry_after=retry_after,
            provider=provider,
            model=model,
            request_id=request_id,
            run_id=run_id,
            tenant=tenant,
            details=details,
        )

    @property
    def retryable(self) -> bool:
        """A rate is worth waiting out. An allowance nobody has topped up is not."""
        return not self.quota


class AuthenticationError(ProviderError):
    """Raised when a provider rejected the credential, or what it is allowed to reach.

    Never retryable: a key that is wrong is wrong on every attempt, and retrying turns
    one broken deployment into a burst that looks like an attack from the vendor's side.
    """

    @property
    def retryable(self) -> bool:
        """No. The next attempt sends the same key."""
        return False


class ContentFilteredError(ProviderError):
    """Raised when a provider refused to process or to return content on policy grounds.

    An answer rather than a fault, so it is not retried — but its own type, because the
    caller's response to it is different from the response to a malformed request.
    """

    @property
    def retryable(self) -> bool:
        """No. The same content is refused the same way."""
        return False


class InvalidRequestError(ProviderError):
    """Raised when a provider rejected the request itself.

    A bad argument, a missing model, a shape it does not accept. Never retryable:
    amplifying a broken configuration into hundreds of identical calls is how a deployment
    mistake becomes a bill.
    """

    @property
    def retryable(self) -> bool:
        """No. Nothing about waiting makes the request valid."""
        return False


class ProviderUnavailableError(ProviderError):
    """Raised when a provider could not be reached, or was reached and was not ready.

    Its own type because the answer to it is to wait rather than to change the request:
    a connection that never landed, a gateway with nothing behind it yet, a self-hosted
    model still loading its weights. Any wait the endpoint asked for is on `retry_after`,
    and it is believed in preference to a computed backoff — retrying a cold model as
    fast as the policy allows is how it never finishes starting.
    """

    @property
    def retryable(self) -> bool:
        """Always. Nothing about the request was refused, only about this moment."""
        return True


class PoolExhaustedError(ProviderError):
    """Raised when every connection to a provider was in use and none came free in time.

    A local condition rather than the provider's, and its own type because the answer to
    it is different: the endpoint is fine and the process is over-subscribed. Retryable,
    since a connection freeing up is the ordinary outcome — but the wait is bounded, so
    the caller finds out inside its own deadline rather than queueing past it.
    """

    @property
    def retryable(self) -> bool:
        """Yes. Nothing about the request was refused, only the moment it was made in."""
        return True


class SchemaViolationError(AdkError):
    """Raised when a payload does not match the type declared for that boundary.

    The kit does not coerce, fill or partially accept: an output that does not validate
    is an error, not a half-populated object handed on to the caller.

    Args:
        model: The model that refused it.
        paths: Every dotted field path that failed, sorted. All of them, not the first:
            one field per round trip is how a five-field config takes five deploys.
        problems: Each of those paths with what was wrong with it.
        payload: What arrived, kept for a debugger. Never logged by the kit.
    """

    def __init__(
        self,
        *args: object,
        model: str = "",
        paths: tuple[str, ...] = (),
        problems: Mapping[str, str] | None = None,
        payload: object = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.paths = paths
        self.problems: dict[str, str] = dict(problems or {})
        self.payload = payload
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class ToolArgumentValidationError(SchemaViolationError):
    """Raised when a model's tool-call arguments do not match the tool's own schema.

    Raised by the provider adapter, before the call reaches the registry: the adapter is
    the last place a malformed call can be stopped for free. Nothing is coerced and no
    absent field is filled in — a tool run with arguments the model did not send is a
    side effect nobody asked for.

    Args:
        tool: The tool that was called.
        call_id: The provider's id for the call, which is what a result is matched back to.
    """

    def __init__(
        self,
        *args: object,
        tool: str = "",
        call_id: str = "",
        paths: tuple[str, ...] = (),
        problems: Mapping[str, str] | None = None,
        payload: object = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.tool = tool
        self.call_id = call_id
        super().__init__(
            *args,
            model=tool,
            paths=paths,
            problems=problems,
            payload=payload,
            run_id=run_id,
            tenant=tenant,
            details=details,
        )

    def feedback(self) -> str:
        """What can be said back to the model: which fields failed, never what they held.

        A rejected argument may be a password, a token or someone's address, and a repair
        prompt that quotes it back has copied it into the next request, the provider's
        logs and the transcript. The field and the reason are enough to correct a call.
        """
        named = "\n".join(f"- {path}: {problem}" for path, problem in sorted(self.problems.items()))
        return (
            f"The call to {self.tool or 'the tool'} did not run: {self}.\n"
            f"{named or '- the arguments as a whole were not usable'}\n"
            f"Send the call again with those arguments corrected. The values you sent are "
            f"not repeated back here."
        )


class StreamInterruptedError(ProviderError):
    """Raised when a stream stopped before the model had finished answering.

    The partial text is carried rather than returned: a truncated answer handed back as a
    complete one is a wrong answer with nothing to show that it is wrong. A caller that
    wants what did arrive reads `partial` deliberately.

    Args:
        partial: The text emitted before the stream stopped.
        received: How many events arrived, which separates a stream that died at once
            from one that died a word from the end.
    """

    def __init__(
        self,
        *args: object,
        partial: str = "",
        received: int = 0,
        status: int | None = None,
        retry_after: float | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.partial = partial
        self.received = received
        super().__init__(
            *args,
            status=status,
            retry_after=retry_after,
            run_id=run_id,
            tenant=tenant,
            details=details,
        )


class FallbackExhaustedError(ProviderError):
    """Raised when every model in a run's chain refused it.

    Carries all of them rather than only the last, because the last is rarely the
    interesting one: a chain that failed on a rate limit, then an outage, then a rate limit
    again is a different incident from one that failed on the same key three times.

    Args:
        attempts: Each model tried and why it did not answer, as `(ref, reason)` pairs, in
            the order they were tried.
    """

    def __init__(
        self,
        *args: object,
        attempts: Sequence[tuple[str, str]] = (),
        status: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.attempts = tuple(_Rejection(ref, reason) for ref, reason in attempts)
        super().__init__(
            *args,
            status=status,
            retry_after=retry_after,
            provider=provider,
            model=model,
            request_id=request_id,
            run_id=run_id,
            tenant=tenant,
            details=details,
        )


class FallbackUnsafeError(AdkError):
    """Raised when another model cannot be tried without risking a repeated side effect.

    A fallback replays the tool results already recorded rather than calling the tools
    again, so the second model sees what the first saw. That is sound only where every tool
    already invoked is safe to have been invoked once. If the run charged a card and the
    model then became unreachable, nothing in the record proves the charge did not land,
    and a second provider finishing the run is how a customer is billed twice. The run
    fails closed and the caller decides.

    Args:
        tool: The tool whose side effect cannot be assumed repeatable.
        ref: The model that would have been tried next.
    """

    def __init__(
        self,
        *args: object,
        tool: str = "",
        ref: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.tool = tool
        self.ref = ref
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class SchemaGenerationError(ConfigurationError):
    """Raised when a type cannot be faithfully described as a schema.

    A configuration failure, raised where the type is declared rather than on the call
    that first sends it: a permissive placeholder would let the model answer with
    something the code cannot validate, and that failure lands in production.

    Args:
        field: The field or parameter that cannot be described.
        annotation: What it was annotated with, as text.
    """

    def __init__(
        self,
        *args: object,
        field: str = "",
        annotation: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.field = field
        self.annotation = annotation
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class ToolDefinitionError(ConfigurationError):
    """Raised when a callable cannot be made into a tool.

    Raised at decoration, which is import time: a tool whose schema is missing an argument
    is a tool the model calls wrongly for as long as the process lives, and the call site
    that suffers is nowhere near the definition that caused it.

    Args:
        tool: The tool being defined, by the name it asked for.
        parameter: The parameter that cannot be described, where one is to blame.
    """

    def __init__(
        self,
        *args: object,
        tool: str = "",
        parameter: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.tool = tool
        self.parameter = parameter
        super().__init__(*args, run_id=run_id, tenant=tenant, details=details)


class ToolError(AdkError):
    """Base of the tool taxonomy: something the run loop can branch on.

    Args:
        tool: What raised it.
        code: A stable, machine-readable name for what happened. Required — a failure
            nobody can name is a failure nobody can write a policy about.
        message: What the model may be told, where anything may be. Free of credentials
            and of whatever the upstream put in its own message.
        retry_after: Seconds the upstream asked the caller to wait, where it said so.

    Raises:
        ValueError: If `code` is empty.
    """

    def __init__(
        self, tool: str, code: str, message: str = "", *, retry_after: float | None = None
    ) -> None:
        if not code:
            raise ValueError("a tool error needs a code: an unnamed failure cannot be acted on")
        self.tool = tool
        self.code = code
        self.message = message
        self.retry_after = retry_after
        super().__init__(
            f"{tool}: {code}" + (f" — {message}" if message else ""),
            details={"tool": tool, "code": code},
        )


class ToolFailure(ToolError):  # noqa: N818 — the taxonomy's name, and it is not always an error
    """The tool tried and could not finish.

    Args:
        tool: What failed.
        code: What went wrong, stably named.
        transient: Whether the same call could succeed on a second attempt. False by
            default: an author who has not thought about it has not established that
            repeating the call does not repeat a side effect.
        retry_after: What the upstream asked for, where it asked.
        detail: What the model may be told about it.
    """

    def __init__(
        self,
        tool: str,
        code: str,
        *,
        transient: bool = False,
        retry_after: float | None = None,
        detail: str = "",
    ) -> None:
        self._transient = transient
        super().__init__(tool, code, detail, retry_after=retry_after)

    @property
    def retryable(self) -> bool:
        """Whether the run loop may try again."""
        return self._transient


class ToolRefusal(ToolError):  # noqa: N818 — a refusal is an answer, not an error
    """The tool worked and declined. An answer, not a fault.

    Reaches the model once, as data, with its reason code — never retried, because asking
    again gets the same answer and spends the budget to hear it. The message travels
    through the untrusted-result envelope like any other tool output: a reason string
    authored to read like an instruction is still only a string.

    Args:
        tool: What declined.
        code: Why, stably named, so a consumer can branch on it.
        message: What the model may be told, in words a user could read.
    """

    def __init__(self, tool: str, code: str, message: str) -> None:
        super().__init__(tool, code, message)


class ToolExecutionError(AdkError):
    """Raised when a tool fails. The tool's own exception is the `__cause__`."""


class ToolNotFoundError(AdkError):
    """Raised when nothing is registered under the name that was asked for.

    Distinct from a refusal on purpose: a name nobody registered is a wiring mistake, and
    a name this agent may not call is a permission decision. Telling them apart is the
    difference between fixing a deployment and widening an allowlist.

    Args:
        tool: The name that was asked for.
        known: What is registered, so the mistake is visible without a second lookup.
    """

    def __init__(self, tool: str, *, known: Sequence[str] = ()) -> None:
        self.tool = tool
        self.known = tuple(known)
        super().__init__(
            f"no tool is registered under {tool!r}"
            + (f"; the registry holds {', '.join(self.known)}" if self.known else ""),
            details={"tool": tool},
        )


class ToolNotPermittedError(AdkError):
    """Raised when a tool exists but this agent's allowlist does not name it.

    Raised instead of calling it. An allowlist enforced after dispatch is a side effect
    that has already landed by the time the decision is recorded.

    Args:
        tool: What was asked for.
        agent: Whose allowlist refused it.
        permitted: What that agent may call, which is what the model can be told.
    """

    def __init__(self, tool: str, *, agent: str = "", permitted: Sequence[str] = ()) -> None:
        self.tool = tool
        self.agent = agent
        self.permitted = tuple(permitted)
        whose = f"{agent}'s" if agent else "this agent's"
        super().__init__(
            f"{tool!r} is not in {whose} allowlist, so it was not called"
            + (f". It may call {', '.join(self.permitted)}" if self.permitted else ""),
            details={"tool": tool, "agent": agent},
        )


class ToolResultError(AdkError):
    """Raised when what a tool returned may not cross into the run as it stands.

    Failing closed is the point. A result that does not match the type the tool declared,
    that outruns a ceiling, or that a policy refuses, is not summarised or repaired into
    something plausible — an invented result is indistinguishable from a real one once it
    is in the conversation.

    Args:
        tool: What returned it.
        violation: What was wrong, in terms of the rule rather than the content. The value
            itself is never quoted: a rejected result may be someone's address, and quoting
            it copies it into the logs the refusal was meant to keep it out of.
    """

    def __init__(self, tool: str, violation: str) -> None:
        self.tool = tool
        self.violation = violation
        super().__init__(
            f"{tool!r} returned something that may not enter the run: {violation}",
            details={"tool": tool},
        )


class GuardrailError(AdkError):
    """Base for what a guard decided and for what it could not decide.

    Both stop the step, and a caller that cannot tell them apart is a caller that will
    eventually retry a refusal. Catch this to stop the run either way; catch the subclass
    to tell a decision from an outage.
    """


class GuardrailViolationError(GuardrailError):
    """Raised when a guardrail rejects content. The step stops; it does not degrade.

    Args:
        code: The machine-readable reason the guard gave, so a caller matches on why
            rather than on a sentence that will be reworded.
        stage: Where it happened — `GuardStage.INPUT` or `GuardStage.OUTPUT`.
        guard: Which guard decided it.
        detail: A short, redacted explanation. Never the offending content, which is the
            one thing an error carrying it would put in every log that catches it.
    """

    def __init__(
        self,
        *args: object,
        code: str = "",
        stage: str = "",
        guard: str = "",
        detail: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.guard = guard
        self.detail = detail
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={"code": code, "stage": str(stage), "guard": guard},
        )


class GuardrailEvaluationError(GuardrailError):
    """Raised when a guard could not reach a verdict, which is not consent.

    Args:
        guard: Which guard was asked.
        stage: Where it was asked.
        reason: What went wrong — `raised`, `timeout` or `unreadable`.
    """

    def __init__(
        self,
        *args: object,
        guard: str = "",
        stage: str = "",
        reason: str = "raised",
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.guard = guard
        self.stage = stage
        self.reason = reason
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={"guard": guard, "stage": str(stage), "reason": reason},
        )


class BudgetExceededError(AdkError):
    """Raised when a run would exceed its ceiling. Raised before the spend, not after.

    Args:
        breached: Which limit stopped it, by field name.
        scope: Where that limit was attached, so it is clear whose ceiling this is and
            who could raise it.
        limit: The ceiling, as a `Decimal` whatever dimension it counts.
        consumed: What had been spent against it.
        remaining: What was left, which is what the caller may still fit into.
    """

    def __init__(
        self,
        message: str,
        *,
        breached: str = "",
        scope: object = None,
        limit: Decimal | None = None,
        consumed: Decimal = Decimal(0),
        remaining: Decimal = Decimal(0),
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id, tenant=tenant, details=details)
        self.breached = breached
        self.scope = scope
        self.limit = limit
        self.consumed = consumed
        self.remaining = remaining


class BudgetUnavailableError(AdkError):
    """Raised when the ledger a shared ceiling lives in cannot be reached.

    Distinct from exceeding a budget: nobody knows whether this run would exceed one. The
    runtime fails closed on it, because carrying on without the ledger is how one outage
    becomes an unbounded bill. Permitting degraded operation is an explicit configuration
    choice and is recorded on the run.
    """


class AuditUnavailableError(AdkError):
    """Raised when a decision about an autonomous action could not be recorded.

    The runtime fails closed on it: an action taken with no durable record that it was
    permitted is exactly the action nobody can defend afterwards, and an audit store that
    is down is the moment somebody would most like it to have been optional.

    Args:
        tool: The call that did not go out.
        decision: What was being recorded when the store could not be reached.
    """

    def __init__(
        self,
        message: str,
        *,
        tool: str = "",
        decision: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id, tenant=tenant, details=details)
        self.tool = tool
        self.decision = decision


class EstimateUnavailableError(AdkError):
    """Raised when a run's cost cannot be estimated on anything better than a guess.

    A caller asking what a run will cost is about to make a decision with the answer, and a
    confident-looking figure with nothing behind it is worse for that decision than no
    figure at all. Proceeding without a price is available, by asking for it by name.

    Args:
        model: The model that could not be priced.
        reason: What was missing.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        reason: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id, tenant=tenant, details=details)
        self.model = model
        self.reason = reason


class CancelledError(AdkError):
    """Raised when a run was cancelled. In-flight work stops; state becomes `cancelled`."""


class LoopLimitError(AdkError):
    """Raised when a run hit one of the caps on its shape: how deep, how wide, how often.

    A cap is a decision, never a fault: the run is over, not unlucky, so it is not
    retryable. Which cap bound is the subclass.
    """


class MaxIterationsError(LoopLimitError):
    """Raised when a run hit its iteration cap — the loop was bounded, and it hit the bound."""


class RecursionLimitError(LoopLimitError):
    """Raised when a run was called from too deep a chain of agents calling agents.

    Failing closed at the bottom is the point: a level that invents a substitute result
    keeps the cycle alive one layer up, where nothing can see it.
    """


class FanOutLimitError(LoopLimitError):
    """Raised when one turn, or one run, asked for more tool calls than its cap allows."""


class RepeatedCallError(LoopLimitError):
    """Raised when the same tool was asked for with the same arguments past the threshold.

    A tool the agent declared idempotent is exempt: polling one status endpoint with the
    same arguments is the design, not a cycle.
    """


class HookRegistrationError(ConfigurationError):
    """Raised when a hook cannot be added to a chain.

    A registration failure is a configuration failure: it happens before any run, because
    a chain assembled wrongly enforces a policy nobody wrote.
    """


class HookEvaluationError(AdkError):
    """Raised when a hook could not be evaluated, which is a denial rather than a pass.

    Hooks fail closed. A policy service that is down is a policy nobody checked, and
    carrying on regardless is the failure this kit exists to stop.
    """


class HookRefusedError(AdkError):
    """Raised when a hook refused the step. The reason is the hook's own."""


class ApprovalDeniedError(AdkError):
    """Raised when a human declined a held tool call."""


class ApprovalDeliveryError(AdkError):
    """Raised when the question could not be put in front of anybody.

    Distinct from a denial: nobody decided, so nothing may proceed on the strength of it.
    """


class ApprovalExpiredError(AdkError):
    """Raised when a decision arrived past the request's time to live.

    An approval is permission at a moment, not a standing licence: honouring a stale one
    runs what nobody currently agrees to.
    """


class ApprovalBindingError(AdkError):
    """Raised when a grant is asked to cover something other than what it was shown.

    An approval is permission for one payload, once. Altered arguments, a second execution
    on one decision, or a grant belonging to a run that has been cancelled all fail closed
    here rather than executing an unapproved variant.
    """


class ApprovalTokenError(AdkError):
    """Raised when a token cannot buy the decision it was presented for.

    Unknown, already spent, or presented as a tenant it was not issued to. All three are
    the shape of a decision being replayed, so none of them resumes anything.

    Args:
        run_id: The run named by the token, where it named a real one.
        presented_by: Who presented it.
    """

    def __init__(self, *args: object, run_id: str = "", presented_by: str = "") -> None:
        super().__init__(*args, run_id=run_id or None, details={"presented_by": presented_by})
        self.presented_by = presented_by


class IndeterminateOutcomeError(AdkError):
    """Raised when nobody can say whether a side effect happened.

    A tool declared effectful whose key cannot be derived, or whose record cannot be
    reached, leaves the runtime with two wrong answers: retry and risk a second booking,
    or report a success it did not see. It raises instead, naming the tool and the
    guarantee it is missing, so a human or an approval path resolves it.
    """


class RunningLoopError(AdkError, RuntimeError):
    """Raised when a synchronous helper is called from inside a running event loop.

    Also a `RuntimeError`, so callers already guarding against 'this event loop is
    already running' keep working. It refuses rather than nesting a second loop or
    blocking the one it is standing on, and the message names the async call to use
    instead: a deadlock says nothing about which line caused it.

    Args:
        sync_name: The helper that was called.
        async_name: What to await in its place.
    """

    def __init__(self, sync_name: str, async_name: str) -> None:
        self.sync_name = sync_name
        self.async_name = async_name
        super().__init__(
            f"{sync_name} cannot be called from a running event loop; "
            f"await {async_name} instead, or call {sync_name} from a thread that has no "
            f"loop of its own"
        )


class EventLoopStalledError(AdkError):
    """Raised when work blocked the event loop instead of awaiting on it.

    A blocking body does not slow its own run: it slows every run sharing the loop, and
    the latency lands on requests that did nothing wrong. So it is attributed to whoever
    caused it rather than left as unexplained tail latency.

    Args:
        tool: What was running while the loop stopped turning.
        blocked_seconds: How far behind the loop fell.
    """

    def __init__(self, tool: str, blocked_seconds: float) -> None:
        self.tool = tool
        self.blocked_seconds = blocked_seconds
        super().__init__(
            f"{tool} stalled the event loop for {blocked_seconds:.3f}s; run a blocking "
            f"body on a worker pool rather than on the loop",
            details={"tool": tool, "blocked_seconds": f"{blocked_seconds:.3f}"},
        )


class WorkersBusyError(AdkError):
    """Raised when a synchronous body could not be given a worker in time.

    Growing the pool instead would trade a bounded wait for unbounded threads, which
    fails later, harder and on someone else's request.
    """


class ToolTimedOutError(AdkError):
    """Raised when one tool call outran the ceiling declared for that tool.

    It is the call's own failure rather than the run's: a batch whose slowest member is
    reported by name leaves its siblings' results standing.
    """

    def __init__(self, tool: str, seconds: float) -> None:
        self.tool = tool
        self.seconds = seconds
        super().__init__(
            f"tool {tool!r} did not return inside its ceiling of {seconds:g}s",
            details={"tool": tool, "seconds": f"{seconds:g}"},
        )


class MemoryAdmissionError(AdkError):
    """Raised when a fact was not allowed to become a durable memory.

    Refused at the write, because a poisoned fact removed a week later has already
    influenced every run in between wearing the costume of something the system knows.

    Args:
        origin: Who or what produced it, per `memory.Origin`.
        source: The tool, document or person behind it.
        reason: Which rule refused it, in the words the audit record carries.
    """

    def __init__(self, *args: object, origin: str = "", source: str = "", reason: str = "") -> None:
        self.origin = origin
        self.source = source
        self.reason = reason
        super().__init__(*args, details={"origin": origin, "source": source, "reason": reason})


class MemoryScopeError(AdkError):
    """Raised when a record is filed somewhere it does not say it belongs.

    A record carries the scope and the kind it was written under, and a call that
    disagrees with either is a bug at the call site rather than a merge to resolve —
    the resolution nobody wants is one user's preference written into another's.

    Args:
        expected: What the record itself says.
        given: What the call said.
    """

    def __init__(self, *args: object, expected: str = "", given: str = "") -> None:
        self.expected = expected
        self.given = given
        super().__init__(*args, details={"expected": expected, "given": given})


class MemoryCorruptionError(AdkError):
    """Raised when a stored record no longer validates as the model it was written as.

    Never swallowed: a recall that drops what it could not read assembles a prompt from
    whatever survived, and nobody can explain the answer afterwards.

    Args:
        record_id: Which record, so the row can be found and fixed.
        payload: What was actually stored, kept for a debugger. Never logged by the kit.
    """

    def __init__(self, *args: object, record_id: str = "", payload: object = None) -> None:
        self.record_id = record_id
        self.payload = payload
        super().__init__(*args, details={"record_id": record_id})


class MemoryLimitError(AdkError):
    """Raised when a value is larger than the adapter declared it can hold.

    Refused at the write, because an adapter that truncates instead returns a profile
    that is subtly wrong on every read afterwards.

    Args:
        limit: The declared ceiling, in bytes.
        size: What was offered.
    """

    def __init__(self, *args: object, limit: int = 0, size: int = 0) -> None:
        self.limit = limit
        self.size = size
        super().__init__(*args, details={"limit": str(limit), "size": str(size)})


class EmbeddingDimensionError(AdkError):
    """Raised when an embedding is not the width the collection was built with.

    Vector stores compare what they are given; a mismatch that reaches one is either an
    error there or, worse, a distance computed over the overlap and returned as a rank.

    Args:
        expected: The collection's width.
        received: The width offered.
    """

    def __init__(self, *args: object, expected: int = 0, received: int = 0) -> None:
        self.expected = expected
        self.received = received
        super().__init__(*args, details={"expected": str(expected), "received": str(received)})


class ChunkingError(AdkError):
    """Raised when a run of text cannot be divided under the chunk token limit.

    Emitting the over-long chunk instead would push the failure to a model call, where it
    reads as a context window error about a document nobody can name. Not retryable: the
    document divides no better on a second attempt.

    Args:
        document: Which document could not be split.
        offset: Where in it the indivisible run begins, in characters.
    """

    def __init__(self, *args: object, document: str = "", offset: int = 0) -> None:
        self.document = document
        self.offset = offset
        super().__init__(*args, details={"document": document, "offset": str(offset)})


class RetrievalDegradedError(AdkError):
    """Raised when retrieval could not run every branch the caller required.

    Returning what the surviving branch found would be a narrower result set that reads
    downstream as a complete answer: the agent says what it found and nothing says the
    keyword branch, and with it every exact identifier match, was missing.

    Args:
        missing: The branches that did not answer.
        answered: The branches that did.
    """

    def __init__(
        self,
        *args: object,
        missing: Sequence[str] = (),
        answered: Sequence[str] = (),
    ) -> None:
        self.missing = tuple(missing)
        self.answered = tuple(answered)
        super().__init__(
            *args,
            details={"missing": ",".join(missing), "answered": ",".join(answered)},
        )


class UngroundedCitationError(AdkError):
    """Raised when an answer cites something the run did not retrieve.

    The citation is neither dropped nor repaired: an answer that looked sourced and was
    not is the failure this whole surface exists to catch, and stripping the offending
    citation would leave a claim standing with nothing behind it.

    Args:
        missing: The citation ids the answer used that nothing retrieved.
        available: The citation ids the retrieval result actually offered.
    """

    def __init__(
        self,
        *args: object,
        missing: Sequence[str] = (),
        available: Sequence[str] = (),
    ) -> None:
        self.missing = tuple(missing)
        self.available = tuple(available)
        super().__init__(
            *args,
            details={"missing": ",".join(missing), "available": ",".join(available)},
        )


class UncitedClaimError(AdkError):
    """Raised when a claim carries no citation at all.

    Where the corpus returned nothing, the answer is a refusal. An uncited claim reads
    exactly like a cited one to the person acting on it.

    Args:
        claims: The claims that cite nothing.
    """

    def __init__(self, *args: object, claims: Sequence[str] = ()) -> None:
        self.claims = tuple(claims)
        super().__init__(*args, details={"claims": str(len(self.claims))})


class ProvenanceLostError(AdkError):
    """Raised when compacting a conversation would drop a source it was carrying.

    Prose may be lost — that is what compaction is. Provenance may not: a summary of turns
    that cited something, carrying no citation, reads as a claim the agent made itself.

    Args:
        lost: The citation ids present before compaction and absent after.
        folded: How many turns the summary was standing for.
    """

    def __init__(self, *args: object, lost: Sequence[str] = (), folded: int = 0) -> None:
        self.lost = tuple(lost)
        self.folded = folded
        super().__init__(*args, details={"lost": ",".join(self.lost), "folded": str(folded)})


class EmbeddingUnavailableError(AdkError):
    """Raised when an ingest cannot embed a batch and stops rather than filling the gap.

    A zero or random vector for the batch that failed would keep the pipeline moving and
    leave a hole in the index that nothing surfaces: the passages are simply never
    retrieved. Carries where to resume from so the ingest is restartable, not restarted.

    Args:
        batch: Which batch failed, counting from zero.
        cursor: The index into the texts of the first one not embedded.
    """

    def __init__(self, *args: object, batch: int = 0, cursor: int = 0) -> None:
        self.batch = batch
        self.cursor = cursor
        super().__init__(*args, details={"batch": str(batch), "cursor": str(cursor)})


class ContextBudgetError(AdkError):
    """Raised when a prompt cannot be made to fit the tokens available for it.

    Assembly fails rather than emitting an over-budget prompt or a fabricated summary:
    a prompt the provider truncates loses whichever part it liked least, which in a long
    conversation is the stated constraint rather than the small talk.

    Args:
        budget_tokens: What there was room for.
        required_tokens: What could not be reduced below.
        section: The section that could not be made to fit, where one is to blame.
    """

    def __init__(
        self,
        *args: object,
        budget_tokens: int = 0,
        required_tokens: int = 0,
        section: str | None = None,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.required_tokens = required_tokens
        self.section = section
        super().__init__(
            *args,
            details={
                "budget_tokens": str(budget_tokens),
                "required_tokens": str(required_tokens),
                "section": section or "",
            },
        )


class MemoryConflictError(AdkError):
    """Raised when a supersession expected a version of a fact that is no longer live.

    Two runs that change the same belief at once must not both succeed: the second
    would either overwrite the first or leave the scope holding two live records for
    one subject. The loser is told which version it was working from, so it can re-read
    and decide again rather than retry blind.

    Args:
        key: The profile key that was contended.
        expected_version: What the caller believed was live.
        actual_version: What is live now.
    """

    def __init__(
        self, *args: object, key: str = "", expected_version: int = 0, actual_version: int = 0
    ) -> None:
        self.key = key
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            *args,
            details={
                "key": key,
                "expected_version": str(expected_version),
                "actual_version": str(actual_version),
            },
        )


class MissingTenantContextError(AdkError):
    """Raised when scoped work is attempted with no tenant context bound.

    Absence is never read as "all tenants" and never filled with a default: an operation
    that cannot say whose data it is about is refused before it reaches a store, because
    the alternative is an unfiltered query that returns and looks like an answer.

    Args:
        where: What was about to happen — the accessor, or the egress point that asked.
    """

    def __init__(self, *args: object, where: str = "") -> None:
        self.where = where
        super().__init__(*args, details={"where": where})


class TenantCrossingError(AdkError):
    """Raised when a scope names a different tenant than the one it was entered from.

    An administrative operation that reaches across tenants is legitimate and has to say
    so; one that does it silently is the incident this refuses to let happen quietly.

    Args:
        into: The tenant the block asked for.
    """

    def __init__(self, *args: object, tenant: str | None = None, into: str = "") -> None:
        self.into = into
        super().__init__(*args, tenant=tenant, details={"into": into})


class TenantContextError(AdkError):
    """Raised when a tenant context arriving from elsewhere cannot be trusted or read.

    The refusal is deliberate in every case. A message with no context is not run under
    the consuming worker's own tenant, because a worker's default is exactly the wrong
    answer. A context contradicting the caller's authenticated claim is not honoured,
    because the payload never outranks the credential. A version this side does not know
    is not read field by field, because that is how a tenant becomes a locale.

    Args:
        reason: Which refusal this is — `missing`, `malformed`, `version`, `contradicted`
            or `oversized` — so a consumer can retry, dead-letter or alert on the right
            ones without matching on message text.
        tenant: The tenant the refusal is about: the authenticated one for `contradicted`,
            the offered one otherwise, where there was a readable one at all.
    """

    def __init__(
        self,
        *args: object,
        reason: TenantRefusal,
        tenant: str | None = None,
    ) -> None:
        self.reason: TenantRefusal = reason
        super().__init__(*args, tenant=tenant, details={"reason": reason})


class UnknownTenantError(ConfigurationError):
    """Raised when work arrives for a tenant nothing is configured for.

    Rejected rather than given the global defaults: a tenant nobody has entitled is a
    tenant nobody has priced, and running them on whatever the deployment happens to
    permit is how an unbounded spend gets attributed to a name in a header.

    Args:
        tenant: Who was asked for.
    """

    def __init__(self, *args: object, tenant: str | None = None) -> None:
        super().__init__(*args, tenant=tenant)


class TenantUnconfiguredError(ConfigurationError):
    """Raised when a tenant's configuration exists somewhere but could not be read.

    Distinct from `UnknownTenantError` because the responses differ: an unknown tenant is
    a request to reject, an unreadable store is an outage to page on. Both fail closed —
    the kit never falls back to permissive defaults, because a limits system that keeps
    running when it cannot read the limits is not enforcing anything.

    Args:
        tenant: Who could not be answered for, where there was one.
    """

    def __init__(self, *args: object, tenant: str | None = None) -> None:
        super().__init__(*args, tenant=tenant)


class TenantLimitError(AdkError):
    """Raised when a tenant's configuration does not permit what was asked for.

    A ceiling catches spend after it happens; this refuses the call. Not retryable — the
    same request will be refused by the same setting until somebody changes the plan.

    Args:
        limit: Which setting refused — `models`, `region` — so a caller can respond to
            the right one without matching on message text.
    """

    def __init__(self, *args: object, tenant: str | None = None, limit: str = "") -> None:
        self.limit = limit
        super().__init__(*args, tenant=tenant, details={"limit": limit})


class MemoryUnavailableError(AdkError):
    """Raised when a memory store could not be reached, after the retries were spent.

    A failover is ordinary and is waited out. A store that is still gone once the budget
    is spent is not: the run fails closed rather than continuing with an empty memory,
    because an agent that silently remembers nothing looks exactly like one whose user
    said nothing. The message never carries the DSN.

    Args:
        store: Which adapter gave up, by class name.
        attempts: How many times it tried.
    """

    def __init__(self, *args: object, store: str = "", attempts: int = 0) -> None:
        self.store = store
        self.attempts = attempts
        super().__init__(*args, details={"store": store, "attempts": str(attempts)})

    @property
    def retryable(self) -> bool:
        """Yes, later. Nothing about the request was refused, only the moment."""
        return True


class MemoryContradictionError(AdkError):
    """Raised when a scope holds contradictory beliefs that nothing may resolve for it.

    Two live records for one subject are not averaged and not ordered by luck. A read
    that would have to choose raises instead, and the caller — or a person — decides.

    Args:
        key: The profile key holding the contradiction.
        subject: What the records disagree about.
        holds: Every live record involved, so the choice can be made on the evidence.
    """

    def __init__(
        self,
        *args: object,
        key: str = "",
        subject: str = "",
        holds: tuple[object, ...] = (),
    ) -> None:
        self.key = key
        self.subject = subject
        self.holds = holds
        super().__init__(*args, details={"key": key, "subject": subject, "holds": str(len(holds))})


class PartialErasureError(AdkError):
    """Raised when an erasure removed the records but could not reach a derived index.

    Half an erasure is worse than none, because the receipt would say the promise was
    kept while a vector built from the erased text is still searchable. The rows stay
    tombstoned and out of reach, the receipt is marked incomplete, and the caller is
    told which adapter to come back for. Re-running the erasure resumes it.

    Args:
        adapter: The index that could not be reached.
        receipt: The incomplete `ErasureReceipt`, so the caller can record what did go.
    """

    def __init__(
        self,
        *args: object,
        adapter: str = "",
        receipt: Any = None,  # noqa: ANN401 — core cannot name a memory type and stay independent
    ) -> None:
        self.adapter = adapter
        self.receipt = receipt
        super().__init__(*args, details={"adapter": adapter})


class ExtractionError(AdkError):
    """Raised when a model's extraction output cannot be trusted as a subgraph.

    Nothing is committed. A subgraph half of which the model invented reads exactly like
    one it derived, and there is no later signal that would tell them apart, so the write
    fails rather than guessing which half was real.

    Args:
        model: Which model produced the output.
        payload: The raw output, kept so the failure can be diagnosed rather than
            re-elicited. It is model output, so it is treated as untrusted text.
        reason: What the payload violated.
    """

    def __init__(
        self,
        *args: object,
        model: str = "",
        payload: str = "",
        reason: str = "",
    ) -> None:
        self.model = model
        self.payload = payload
        self.reason = reason
        super().__init__(*args, details={"model": model, "reason": reason})


class WriteQueueFullError(AdkError):
    """Raised when an asynchronous write cannot be accepted because the queue is full.

    A bounded queue that drops its overflow loses a write nobody will ever look for.
    Refusing is louder and is the caller's decision to make: wait, shed, or write through
    synchronously and pay the latency.

    Args:
        depth: How many writes are waiting, which is also the bound that was reached.
    """

    def __init__(self, *args: object, depth: int = 0) -> None:
        self.depth = depth
        super().__init__(*args, details={"depth": str(depth)})

    @property
    def retryable(self) -> bool:
        """Yes, once the queue has drained. Nothing about the write itself was refused."""
        return True


class ClaimUnavailableError(AdkError):
    """Raised when a checked-in tool result cannot be produced for the handle asked about.

    Expired, erased, never stored, or belonging to another tenant or run — all one answer,
    deliberately. Distinguishing "gone" from "not yours" tells a caller which handles other
    runs hold, and the model can do nothing different with either.

    The alternative to raising is returning something, and the only something available is
    invented. A model handed a plausible substitute for a document it asked to read has no
    way to know it is reasoning about nothing.

    Args:
        handle: What was asked for.
        tenant: Whose run asked.
        run_id: Which run asked.
    """

    def __init__(self, *args: object, handle: str = "", tenant: str = "", run_id: str = "") -> None:
        self.handle = handle
        self.tenant = tenant
        self.run_id = run_id
        super().__init__(*args, details={"handle": handle, "run_id": run_id})


class PrefixDriftError(AdkError):
    """Raised when the frozen part of a prompt is not what was frozen.

    Something rewrote cached bytes, and the next call pays a full prefill for a prompt that
    was supposed to be free. The failure is loud on purpose: a doubling of prefill latency
    that nobody notices for a week is worse than a build that stops today.

    Args:
        layer: Which part of the prompt moved, so the report names a thing rather than an
            index — the instructions, the tool declarations, the retrieved documents.
        position: Where in the frozen region it moved.
    """

    def __init__(self, *args: object, layer: str = "", position: int = 0) -> None:
        self.layer = layer
        self.position = position
        super().__init__(*args, details={"layer": layer, "position": str(position)})


class PromptNotFoundError(ConfigurationError):
    """Raised when a prompt name, version or alias does not exist in the registry.

    Never an empty prompt, a nearest match or a default: an agent that silently ran on
    something other than the prompt it named is a behaviour change nobody can trace.

    Args:
        name: Which prompt was asked for.
        available: The versions that do exist, so the fix is in the message.
    """

    def __init__(self, *args: object, name: str = "", available: tuple[str, ...] = ()) -> None:
        self.name = name
        self.available = available
        super().__init__(*args, details={"prompt": name, "available": ", ".join(available)})


class PromptRejectedError(ConfigurationError):
    """Raised when a stored prompt exists but may not be served.

    An unreadable file, an empty body, text shaped like a credential, or a published
    version edited in place. Each of them fails at load rather than reaching a provider.

    Args:
        name: Which prompt was refused.
    """

    def __init__(self, *args: object, name: str = "") -> None:
        self.name = name
        super().__init__(*args, details={"prompt": name})


class IncompatiblePromptVersionError(ConfigurationError):
    """Raised when an alias may not be moved to a version the call sites cannot render.

    A rollback target that drops a variable current call sites still supply would start
    runs that cannot render, so nothing is repointed and the variables are named.

    Args:
        name: Which prompt.
        version: The version that was refused as a target.
        variables: What it does not declare and the current version does.
    """

    def __init__(
        self,
        *args: object,
        name: str = "",
        version: str = "",
        variables: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.version = version
        self.variables = variables
        super().__init__(
            *args,
            details={"prompt": name, "version": version, "variables": ",".join(variables)},
        )


class TemplateError(ConfigurationError):
    """Raised when a prompt template, or a value for one, may not be rendered.

    Covers both halves: a body and its declarations disagreeing, and a value that is
    missing, null, wrong-typed, undeclared, forging the untrusted envelope, or overrunning
    the window. The message names the variable and never quotes the value.

    Args:
        template: Which template refused.
        variable: Which slot refused, where one is at fault.
        reason: What was wrong, as a code a caller can branch on.
    """

    def __init__(
        self,
        *args: object,
        template: str = "",
        variable: str = "",
        reason: str = "",
    ) -> None:
        self.template = template
        self.variable = variable
        self.reason = reason
        super().__init__(
            *args, details={"template": template, "variable": variable, "reason": reason}
        )


class SandboxError(AdkError):
    """Raised when the sandbox stopped generated code, rather than the code stopping itself.

    Code that raises, exits non-zero or writes nothing is a result, not an error: the
    caller wanted to know what it did and now knows. This is the other case — the sandbox
    took the process away before it could finish, so there is no result to report.
    """


class SandboxTimeoutError(SandboxError):
    """Raised when generated code ran past a time ceiling and was killed.

    Args:
        limit: Which ceiling fired — `"wall"` for elapsed time, `"cpu"` for processor
            time. They are different diagnoses: wall time catches something waiting,
            processor time catches something spinning.
        seconds: The ceiling that fired, so the message names a number the caller set.
    """

    def __init__(self, *args: object, limit: str = "wall", seconds: float = 0.0) -> None:
        self.limit = limit
        self.seconds = seconds
        super().__init__(*args, details={"limit": limit, "seconds": str(seconds)})


class SandboxMemoryError(SandboxError):
    """Raised when generated code asked for more memory than its ceiling allowed.

    The allocation fails inside the sandbox, never on the host: the ceiling is set on the
    child before any generated code runs, so the host cannot be starved by what it ran.

    Args:
        limit_bytes: The ceiling the code was under.
    """

    def __init__(self, *args: object, limit_bytes: int = 0) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(*args, details={"limit_bytes": str(limit_bytes)})


class DelegationLimitError(LoopLimitError):
    """Raised when one agent may not hand work to another, and the run must say why.

    Surfaced to the parent rather than ending the run: an agent that cannot delegate can
    often still answer, and a refusal it can read is a refusal it can reason about. It is
    not retryable — the same call refused for the same reason refuses again — so a parent
    that retries it is looping, not recovering.

    Args:
        reason: Which ceiling bound. `"depth"`, `"fan_out"`, `"run"`, `"cycle"` or
            `"expired"`, so a refusal is attributable to a decision somebody made.
        path: The agents the refused call would have run through, root first, ending with
            the one that was refused.
    """

    def __init__(self, *args: object, reason: str = "depth", path: tuple[str, ...] = ()) -> None:
        self.reason = reason
        self.path = path
        super().__init__(*args, details={"reason": reason, "path": " -> ".join(path)})


class DelegationError(AdkError):
    """Raised when work handed to a worker did not come back as an answer.

    A supervisor that cannot tell a refusal from an empty answer reads one as the other,
    so this says which worker, and why. It is a value by default — the supervisor is
    handed it and can route the task elsewhere, ask a person, or answer without it — and
    is raised only where the delegation was declared fatal, or where the wiring itself is
    wrong and no worker could have run.

    Args:
        specialist: Which worker the task went to, or was going to.
        reason: Why there is no answer. `"no_worker"` (nobody on the roster can do it),
            `"no_tools"` (nothing in common between what it holds and what its caller
            holds), `"blocked"` (a guardrail stopped what came back), `"budget"` (it
            spent its slice), `"cancelled"`, `"conflict"` (a second worker for one memory
            key) or `"failed"`.
        path: The agents this ran through, root first.
    """

    def __init__(
        self,
        *args: object,
        specialist: str = "",
        reason: str = "failed",
        path: tuple[str, ...] = (),
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.specialist = specialist
        self.reason = reason
        self.path = path
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={"worker": specialist, "reason": reason, "path": " -> ".join(path)},
        )


class AggregationError(AdkError):
    """Raised when concurrent branches did not add up to the aggregate that was asked for.

    A partial result presented as a whole one is the failure mode this exists to stop: the
    caller reads an answer built from three branches out of five and nothing on it says so.
    So an aggregate that cannot be formed is a refusal carrying its own provenance —
    which branches contributed, which were left out, and why each was.

    Args:
        strategy: What was asked for: `"all"`, `"first_success"`, `"quorum"` or `"reduce"`.
        reason: Why it could not be formed. `"failed"` (a branch a strategy required did
            not answer), `"quorum"` (fewer branches answered than the quorum needed),
            `"none"` (no branch answered at all) or `"cancelled"` (the fan-out was stopped
            while branches were still running).
        contributed: The branches that did answer, in declared order.
        excluded: Why each of the others is not in the aggregate, by branch name.
    """

    def __init__(
        self,
        *args: object,
        strategy: str = "",
        reason: str = "failed",
        contributed: tuple[str, ...] = (),
        excluded: Mapping[str, str] | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.reason = reason
        self.contributed = contributed
        self.excluded = dict(excluded or {})
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={
                "strategy": strategy,
                "reason": reason,
                "contributed": ", ".join(contributed),
                "excluded": ", ".join(f"{name}: {why}" for name, why in self.excluded.items()),
            },
        )


class AttributionError(AdkError):
    """Raised when spend could not be attributed to the participants that incurred it.

    Every case is one where carrying on would produce a total that reads as complete and
    is not: a tree missing a participant, a participant counted twice, or a span exported
    without the tenant that would let anyone find the money again. A number nobody can
    reconcile is worse than a refusal, because only one of them gets noticed.

    Args:
        reason: What was wrong. `"empty"`, `"no_root"`, `"two_roots"`, `"duplicate"` and
            `"orphan"` are trees that cannot be read as one run; `"no_tenant"` is an export
            refused because the spend would be unattributable once it had left.
    """

    def __init__(
        self,
        *args: object,
        reason: str = "orphan",
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.reason = reason
        super().__init__(*args, run_id=run_id, tenant=tenant, details={"reason": reason})


class HandoffContractError(AdkError):
    """Raised when a conversation could not be handed to the agent it was addressed to.

    A handoff that half-happens is the worst outcome available: the source has moved on,
    the target has a payload it cannot read, and nobody owns the conversation. So every
    check here happens before the target is invoked and nothing is written on the way out.

    Args:
        source: The agent handing over.
        target: Who it was addressed to, whether or not anything answers to that name.
        reason: Why it did not happen. `"contract"` (the payload is not what the target
            declared it accepts), `"unknown_target"`, `"no_tools"` (nothing in common
            between what the target holds and what the source holds) or `"in_flight"`
            (the source run has not settled, so a call would be left hanging).
        violations: The fields the payload got wrong, in the order the model reports them.
        path: The agents the conversation has passed through, root first.
    """

    def __init__(
        self,
        *args: object,
        source: str = "",
        target: str = "",
        reason: str = "contract",
        violations: tuple[str, ...] = (),
        path: tuple[str, ...] = (),
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.reason = reason
        self.violations = violations
        self.path = path
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={
                "source": source,
                "target": target,
                "reason": reason,
                "violations": ", ".join(violations),
                "path": " -> ".join(path),
            },
        )


class PlanValidationError(AdkError):
    """Raised when a plan is not something the runtime could execute as written.

    Every check that produces this happens before the first step runs, because a plan
    refused halfway is a plan that half happened. Nothing here repairs what it refuses: an
    executor that dropped an undeclared argument or trimmed a plan to fit would be deciding
    what the planner meant, which is exactly the decision the planner/executor split exists
    to keep out of the runtime.

    Args:
        step: Which step, where one step is at fault.
        tool: What that step wanted to call.
        reason: Why it was refused. `"empty"` (no steps at all), `"too_long"`,
            `"unknown_tool"`, `"not_allowed"` (outside the agent's allowlist or the
            delegated scope), `"arguments"`, `"dependency"` (waits for a step nobody
            planned), `"cycle"`, or `"replan"` (the planner kept producing invalid plans).
        violations: What was wrong — the argument names, the stray dependencies, or the
            steps in the loop.
        payload: The arguments the planner wrote, as it wrote them, so the plan that
            produced this can be read back in a log rather than reconstructed.
        attempts: How many plans were refused, for `reason="replan"`.
    """

    def __init__(
        self,
        *args: object,
        step: str = "",
        tool: str = "",
        reason: str = "arguments",
        violations: tuple[str, ...] = (),
        payload: Mapping[str, Any] | None = None,
        attempts: int = 0,
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.step = step
        self.tool = tool
        self.reason = reason
        self.violations = violations
        self.payload = payload
        self.attempts = attempts
        super().__init__(
            *args,
            run_id=run_id,
            tenant=tenant,
            details={
                "step": step,
                "tool": tool,
                "reason": reason,
                "violations": ", ".join(violations),
            },
        )


class ScopeEscalationError(AdkError):
    """Raised when a sub-agent asked to hold access the agent it acts for does not hold.

    A child's scope is its parent's, narrowed. Anything else is a privilege escalation
    dressed as a default, and it is refused whether or not the child's own configuration
    would have permitted it.

    Args:
        requested: What was asked for and not held, in the order it was asked for.
        path: The agents the call ran through, so the refusal names who asked.
    """

    def __init__(
        self, *args: object, requested: tuple[str, ...] = (), path: tuple[str, ...] = ()
    ) -> None:
        self.requested = requested
        self.path = path
        super().__init__(
            *args, details={"requested": ", ".join(requested), "path": " -> ".join(path)}
        )


class CeilingExceededError(AdkError):
    """Raised when an action would take more headroom than the ceiling has left.

    Args:
        action_class: Which class ran out.
        headroom: What was left, as an exact string.
        requested: What was asked for, as an exact string.
    """

    def __init__(
        self, *args: object, action_class: str = "", headroom: str = "", requested: str = ""
    ) -> None:
        self.action_class = action_class
        super().__init__(
            *args,
            details={"action_class": action_class, "headroom": headroom, "requested": requested},
        )


class InexactAmountError(AdkError):
    """Raised when an amount cannot be counted against a ceiling without drifting.

    A float, a phrase, or a negative number. Each is refused at the boundary rather than
    coerced, because a ceiling is only enforceable in arithmetic that is exact.

    Args:
        amount: What arrived, as it arrived.
    """

    def __init__(self, *args: object, amount: str = "") -> None:
        self.amount = amount
        super().__init__(*args, details={"amount": amount})


class AutonomyRefusedError(AdkError):
    """Raised when an action is one no grant could ever have permitted.

    Distinct from escalation: escalating puts the call in front of a human, and there is
    no human who can wave this one through from inside the run.

    Args:
        tool: What was attempted.
        action_class: The class it belongs to.
    """

    def __init__(self, *args: object, tool: str = "", action_class: str = "") -> None:
        self.tool = tool
        self.action_class = action_class
        super().__init__(*args, details={"tool": tool, "action_class": action_class})


class GrantRevokedError(AdkError):
    """Raised when the grant a run was acting under was withdrawn while it was under way.

    Args:
        grant_id: Which grant, so an operator can tell one withdrawal from another.
        revoked_by: Who withdrew it.
    """

    def __init__(self, *args: object, grant_id: str = "", revoked_by: str = "") -> None:
        self.grant_id = grant_id
        self.revoked_by = revoked_by
        super().__init__(*args, details={"grant_id": grant_id, "revoked_by": revoked_by})


class DependencyCycleError(ConfigurationError):
    """Raised when declared dependencies close a loop, so nothing in it could ever start.

    Found while the graph is built rather than while it runs: a cycle discovered at
    runtime is a set of nodes waiting on each other, which looks like slow work.

    Args:
        cycle: The nodes that wait on each other, so the message names them rather than
            saying a cycle exists somewhere.
    """

    def __init__(self, *args: object, cycle: tuple[str, ...] = ()) -> None:
        self.cycle = cycle
        super().__init__(*args, details={"cycle": ", ".join(cycle)})


class StateConflictError(AdkError):
    """Raised when a state write named a version that has since moved.

    Two workers holding the same run both write it back and, without this, the second
    silently wins — the first worker's iteration, spend and cursor are gone with nothing
    recording that they happened. The loser is told both numbers so it can re-read and
    decide, rather than retry against the same stale copy.

    Args:
        key: What was contended, as `tenant/id`.
        expected_version: The version the write claimed to have read.
        actual_version: The version that is stored.
    """

    def __init__(
        self, *args: object, key: str = "", expected_version: int = 0, actual_version: int = 0
    ) -> None:
        self.key = key
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            *args,
            details={
                "key": key,
                "expected_version": str(expected_version),
                "actual_version": str(actual_version),
            },
        )


class StateNotFoundError(AdkError):
    """Raised when a patch or a resume named state that is not there.

    A patch adds to what is stored, so there is nothing sensible to do with an absent
    record: creating one would invent a run that never started, and returning quietly
    would report a write that did not happen.

    Args:
        key: What was looked for, as `tenant/id`.
        kind: `session` or `run`.
    """

    def __init__(self, *args: object, key: str = "", kind: str = "") -> None:
        self.key = key
        self.kind = kind
        super().__init__(*args, details={"key": key, "kind": kind})


class StatePersistenceError(AdkError):
    """Raised when a state store could not take a write, or could not be reached.

    A store that is unreachable is not a store that accepted the write, and a run that
    carries on regardless is a run whose recorded spend is fiction. Nothing is partially
    applied: the caller sees a failure or it sees the new version.

    Args:
        store: Which adapter failed, by class name.
        reason: `unavailable` where it could not be reached, `contended` where the write
            lost a serialisation race and the transaction rolled back, `too_large` where
            the record exceeded what the store will hold.
    """

    def __init__(self, *args: object, store: str = "", reason: str = "unavailable") -> None:
        self.store = store
        self.reason = reason
        super().__init__(*args, details={"store": store, "reason": reason})

    @property
    def retryable(self) -> bool:
        """Everything but size: a record too large stays too large, however often it is sent."""
        return self.reason != "too_large"


class StateInUseError(AdkError):
    """Raised when deleting a session would orphan runs that have not finished.

    A live run whose session has gone cannot be resumed and cannot be found by any
    listing that starts from the session, so it becomes work nothing will ever reap.
    Callers that mean it pass `cascade=True`.

    Args:
        key: The session, as `tenant/id`.
        live_runs: The runs that have not reached a terminal state.
    """

    def __init__(self, *args: object, key: str = "", live_runs: tuple[str, ...] = ()) -> None:
        self.key = key
        self.live_runs = live_runs
        super().__init__(*args, details={"key": key, "live_runs": ", ".join(live_runs)})


class CheckpointTooLargeError(AdkError):
    """Raised when a run's frontier exceeds what a checkpoint may carry.

    Truncating it would be worse than not writing it: half a frontier resumes into a
    conversation that never happened, and nothing downstream could tell. The run carries
    on uncheckpointed, and the caller is told what it would have taken.

    Args:
        run_id: Whose frontier.
        size_bytes: What it came to.
        max_bytes: What the policy allows.
    """

    def __init__(
        self, *args: object, run_id: str = "", size_bytes: int = 0, max_bytes: int = 0
    ) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            *args,
            run_id=run_id,
            details={
                "run_id": run_id,
                "size_bytes": str(size_bytes),
                "max_bytes": str(max_bytes),
            },
        )


class CheckpointFormatError(AdkError):
    """Raised when a checkpoint was written by a kit version this one cannot read.

    Reading fields it has to guess at is how a resume replays a call it thought had not
    run. The run is left where it is, for a worker on the newer version to pick up.

    Args:
        run_id: Whose checkpoint.
        format_version: What it was written at.
        readable_version: The newest this reader understands.
    """

    def __init__(
        self,
        *args: object,
        run_id: str = "",
        format_version: int = 0,
        readable_version: int = 0,
    ) -> None:
        self.format_version = format_version
        self.readable_version = readable_version
        super().__init__(
            *args,
            run_id=run_id,
            details={
                "run_id": run_id,
                "format_version": str(format_version),
                "readable_version": str(readable_version),
            },
        )


class IndeterminateToolCallError(AdkError):
    """Raised when a resume cannot say whether an effectful call already happened.

    The process died between dispatching a call and recording its result. Retrying might
    book a second seat; skipping might strand the run having promised something it never
    did. Neither is a guess the kit is entitled to make, so it stops and names the calls,
    for the tool's own status endpoint or a person to resolve.

    Args:
        run_id: Which run cannot be carried on.
        calls: The tools involved, in the order the model asked for them.
    """

    def __init__(self, *args: object, run_id: str = "", calls: tuple[str, ...] = ()) -> None:
        self.calls = calls
        super().__init__(
            *args, run_id=run_id, details={"run_id": run_id, "calls": ", ".join(calls)}
        )

    @property
    def retryable(self) -> bool:
        """No. Retrying is the outcome this exists to prevent."""
        return False


class ResumeConflictError(AdkError):
    """Raised when another worker is already carrying this run on.

    Two workers resuming one run is two runs, spending one budget twice and dispatching
    each outstanding call twice. The second is refused rather than queued: the first is
    already doing the work.

    Args:
        run_id: The run in question.
    """

    def __init__(self, *args: object, run_id: str = "") -> None:
        super().__init__(*args, run_id=run_id, details={"run_id": run_id})


class WorkItemNotFoundError(AdkError):
    """Raised when a queue operation named an item the queue does not hold.

    Completing an item that is not there would report work done that nothing recorded, so
    the caller is told rather than quietly succeeding.

    Args:
        item_id: What was named.
        tenant: Whose queue was looked in. An item is never found across tenants.
    """

    def __init__(self, *args: object, item_id: str = "", tenant: str = "") -> None:
        self.item_id = item_id
        super().__init__(*args, tenant=tenant, details={"item_id": item_id, "tenant": tenant})


class LeaseLostError(AdkError):
    """Raised when a worker acted on a claim it no longer holds.

    The lease lapsed, another worker has the item, or it has been renewed for longer than
    the policy allows. Either way this worker's result is a duplicate of somebody else's
    work, and writing it back would overwrite the outcome that counts.

    Args:
        item_id: The item in question.
        worker: Who thought they held it.
        holder: Who holds it now, where anyone does.
        reason: `expired`, `taken`, or `capped` where renewals ran past the bound.
    """

    def __init__(
        self,
        *args: object,
        item_id: str = "",
        worker: str = "",
        holder: str | None = None,
        reason: str = "expired",
    ) -> None:
        self.item_id = item_id
        self.worker = worker
        self.holder = holder
        self.reason = reason
        super().__init__(
            *args,
            details={
                "item_id": item_id,
                "worker": worker,
                "holder": holder or "",
                "reason": reason,
            },
        )

    @property
    def retryable(self) -> bool:
        """No. The item is somebody else's now, and retrying is the duplicate."""
        return False


class QueueUnavailableError(AdkError):
    """Raised when a work queue could not be reached.

    An enqueue that failed silently is work that nobody is waiting for and nobody will
    reap, because nothing ever recorded that it existed. The caller decides what to do
    with the failure; the kit will not decide by dropping it.

    Args:
        queue: Which queue, by name.
        operation: What was being attempted.
    """

    def __init__(self, *args: object, queue: str = "", operation: str = "") -> None:
        self.queue = queue
        self.operation = operation
        super().__init__(*args, details={"queue": queue, "operation": operation})

    @property
    def retryable(self) -> bool:
        """Yes. A store that is unreachable now may be reachable shortly."""
        return True


class EvalIncompleteError(AdkError):
    """Raised when a quality gate was asked to judge a partly scored dataset.

    A gate that scores what it can and passes on the rest is not a gate: the examples it
    skipped are exactly where a new prompt breaks. Failing closed here costs a rerun;
    guessing costs the regression the suite exists to catch.

    Args:
        prompt: The prompt whose run fell short.
        version: The concrete version measured.
        coverage: The share of the dataset actually scored, between `0.0` and `1.0`.
    """

    def __init__(
        self, *args: object, prompt: str = "", version: str = "", coverage: float = 0.0
    ) -> None:
        self.prompt = prompt
        self.version = version
        self.coverage = coverage
        super().__init__(
            *args,
            details={"prompt": prompt, "version": version, "coverage": f"{coverage:g}"},
        )

    @property
    def retryable(self) -> bool:
        """Yes. Scoring the rest of the dataset makes the same comparison answerable."""
        return True


class IncomparableEvalError(AdkError):
    """Raised when two eval runs cannot be compared to each other.

    A judge that changed between the runs, or a prompt that changed the variables the
    dataset supplies, moves the numbers on its own. The difference measured is then not
    the difference the change made, and reporting it as a pass is worse than reporting
    nothing.

    Args:
        reason: `judge` where the scorer moved, `variables` where the prompt's inputs did.
    """

    def __init__(self, *args: object, reason: str = "judge") -> None:
        self.reason = reason
        super().__init__(*args, details={"reason": reason})

    @property
    def retryable(self) -> bool:
        """No. Re-measuring the baseline is a decision, not a retry."""
        return False


class PayloadTooLargeError(AdkError):
    """Raised when an activity input or result is larger than the transport will carry.

    Truncating it is worse than refusing: a retrieval result cut in half is still a
    plausible-looking answer, and the run continues on evidence nobody chose. The fix is
    to pass a handle to the store rather than the content, which is a decision for the
    caller and not for the kit.

    Args:
        payload: Which side was too large, `input` or `result`.
        step: The workflow step it belonged to.
        size: How many bytes it serialised to.
        limit: What the transport carries.
    """

    def __init__(
        self,
        *args: object,
        payload: str = "result",
        step: str = "",
        size: int = 0,
        limit: int = 0,
    ) -> None:
        self.payload = payload
        self.step = step
        self.size = size
        self.limit = limit
        super().__init__(
            *args,
            details={"payload": payload, "step": step, "size": str(size), "limit": str(limit)},
        )

    @property
    def retryable(self) -> bool:
        """No. The same payload is the same size on the next attempt."""
        return False
