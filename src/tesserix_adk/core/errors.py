"""Error hierarchy for the kit.

Every failure the kit raises inherits from `AdkError`, so a consumer can catch this
kit's failures without catching `Exception` and swallowing its own bugs alongside.
"""

from __future__ import annotations

__all__ = ["AdkError", "ConfigurationError", "MissingExtraError", "ProtocolConformanceError"]

_DISTRIBUTION = "tesserix-adk"


class AdkError(Exception):
    """Base class for every error raised by the kit."""


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
