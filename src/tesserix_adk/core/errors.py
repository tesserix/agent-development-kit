"""Error hierarchy for the kit.

Every failure the kit raises inherits from `AdkError`, so a consumer can catch this
kit's failures without catching `Exception` and swallowing its own bugs alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "RETRYABLE_STATUS",
    "AdkError",
    "ApprovalDeniedError",
    "ApprovalExpiredError",
    "BudgetExceededError",
    "CancelledError",
    "CapabilityError",
    "ConfigurationError",
    "ContextWindowExceededError",
    "FanOutLimitError",
    "GuardrailViolationError",
    "HookEvaluationError",
    "HookRefusedError",
    "HookRegistrationError",
    "LoopLimitError",
    "MaxIterationsError",
    "MissingExtraError",
    "ModelResponseError",
    "ProtocolConformanceError",
    "ProviderError",
    "ProviderTimeoutError",
    "RecursionLimitError",
    "RepeatedCallError",
    "SchemaGenerationError",
    "SchemaViolationError",
    "StreamInterruptedError",
    "ToolArgumentValidationError",
    "ToolExecutionError",
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
        counted: Tokens in the prompt, as the provider itself counted them.
        limit: The window the provider declared.
    """

    def __init__(
        self,
        *args: object,
        counted: int,
        limit: int,
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
    """

    def __init__(
        self,
        *args: object,
        status: int | None = None,
        retry_after: float | None = None,
        run_id: str | None = None,
        tenant: str | None = None,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self.retry_after = retry_after
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


class ToolExecutionError(AdkError):
    """Raised when a tool fails. The tool's own exception is the `__cause__`."""


class GuardrailViolationError(AdkError):
    """Raised when a guardrail rejects content. The step stops; it does not degrade."""


class BudgetExceededError(AdkError):
    """Raised when a run would exceed its ceiling. Raised before the spend, not after."""


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
