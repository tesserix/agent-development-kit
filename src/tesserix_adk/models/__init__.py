"""Model adapters and the ModelBus. Provider-specific code, behind a core protocol."""

from tesserix_adk.core.capabilities import Capability, ModelCapabilities, ModelRef, ModelSpec
from tesserix_adk.core.errors import (
    CapabilityError,
    ContextWindowExceededError,
    ModelResponseError,
)
from tesserix_adk.core.protocols import ModelProvider
from tesserix_adk.core.provider import ModelRequest, ModelResponse, ToolDeclaration

__all__ = [
    "Capability",
    "CapabilityError",
    "ContextWindowExceededError",
    "ModelCapabilities",
    "ModelProvider",
    "ModelRef",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseError",
    "ModelSpec",
    "ToolDeclaration",
]
