"""Frozen primitives, protocols and error types. The vocabulary every other layer speaks."""

from tesserix_adk.core.errors import (
    AdkError,
    ConfigurationError,
    MissingExtraError,
    ProtocolConformanceError,
)
from tesserix_adk.core.extras import require_extra
from tesserix_adk.core.protocols import (
    BudgetPolicy,
    Clock,
    Guardrail,
    MemoryStore,
    ModelProvider,
    ToolRegistry,
    Tracer,
    members_of,
    verify_conformance,
)

__all__ = [
    "AdkError",
    "BudgetPolicy",
    "Clock",
    "ConfigurationError",
    "Guardrail",
    "MemoryStore",
    "MissingExtraError",
    "ModelProvider",
    "ProtocolConformanceError",
    "ToolRegistry",
    "Tracer",
    "members_of",
    "require_extra",
    "verify_conformance",
]
