"""Error hierarchy for the kit.

Every failure the kit raises inherits from `AdkError`, so a consumer can catch this
kit's failures without catching `Exception` and swallowing its own bugs alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AdkError",
    "BudgetExceededError",
    "CancelledError",
    "CapabilityError",
    "ConfigurationError",
    "GuardrailViolationError",
    "MaxIterationsError",
    "MissingExtraError",
    "ProtocolConformanceError",
    "ProviderError",
    "ProviderTimeoutError",
    "SchemaViolationError",
    "ToolExecutionError",
]

_DISTRIBUTION = "tesserix-adk"


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

    Provider-specific status codes and bodies belong in `details`; the type is what a
    consumer branches on, so it is the same across providers.
    """


class ProviderTimeoutError(ProviderError):
    """Raised when a provider did not answer inside the deadline.

    A `ProviderError`, so the common case is not missed by `except ProviderError`, and
    its own type, because a timeout is retryable where a 400 is not.
    """


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


class MaxIterationsError(AdkError):
    """Raised when a run hit its iteration cap — the loop was bounded, and it hit the bound."""
