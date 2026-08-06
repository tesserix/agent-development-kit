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
    "FanOutLimitError",
    "GuardrailViolationError",
    "HookEvaluationError",
    "HookRefusedError",
    "HookRegistrationError",
    "LoopLimitError",
    "MaxIterationsError",
    "MissingExtraError",
    "ProtocolConformanceError",
    "ProviderError",
    "ProviderTimeoutError",
    "RecursionLimitError",
    "RepeatedCallError",
    "SchemaViolationError",
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
    """


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
    """Raised when model output does not match the declared type.

    The kit does not coerce, fill or partially accept: an output that does not validate
    is an error, not a half-populated object handed on to the caller.
    """


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
