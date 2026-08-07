"""Error hierarchy for the kit.

Every failure the kit raises inherits from `AdkError`, so a consumer can catch this
kit's failures without catching `Exception` and swallowing its own bugs alongside.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "RETRYABLE_STATUS",
    "AdkError",
    "ApprovalDeniedError",
    "ApprovalExpiredError",
    "AuthenticationError",
    "BudgetExceededError",
    "BudgetUnavailableError",
    "CancelledError",
    "CapabilityError",
    "ConfigurationError",
    "ContentFilteredError",
    "ContextWindowExceededError",
    "EstimateUnavailableError",
    "EventLoopStalledError",
    "FallbackExhaustedError",
    "FallbackUnsafeError",
    "FanOutLimitError",
    "GuardrailViolationError",
    "HookEvaluationError",
    "HookRefusedError",
    "HookRegistrationError",
    "InvalidRequestError",
    "LoopLimitError",
    "MaxIterationsError",
    "MissingExtraError",
    "ModelResponseError",
    "NoEligibleModelError",
    "PoolExhaustedError",
    "ProtocolConformanceError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RateLimitError",
    "RecursionLimitError",
    "RepeatedCallError",
    "RunningLoopError",
    "SchemaGenerationError",
    "SchemaViolationError",
    "StreamInterruptedError",
    "ToolArgumentValidationError",
    "ToolDefinitionError",
    "ToolExecutionError",
    "ToolTimedOutError",
    "TrustBoundaryError",
    "WorkersBusyError",
]

_DISTRIBUTION = "tesserix-adk"

# Faults, not answers: the same request sent later can succeed. 429 is here because a rate
# limit is transient by construction; a quota that is not transient is caught by the
# `Retry-After` ceiling rather than by retrying until it clears.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


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


class ToolExecutionError(AdkError):
    """Raised when a tool fails. The tool's own exception is the `__cause__`."""


class GuardrailViolationError(AdkError):
    """Raised when a guardrail rejects content. The step stops; it does not degrade."""


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


class ApprovalExpiredError(AdkError):
    """Raised when a decision arrived past the request's time to live.

    An approval is permission at a moment, not a standing licence: honouring a stale one
    runs what nobody currently agrees to.
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
