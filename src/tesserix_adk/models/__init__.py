"""Model adapters and the ModelBus. Provider-specific code, behind a core protocol."""

from tesserix_adk.core.capabilities import Capability, ModelCapabilities, ModelRef, ModelSpec
from tesserix_adk.core.errors import (
    CapabilityError,
    ContextWindowExceededError,
    ModelResponseError,
    StreamInterruptedError,
    ToolArgumentValidationError,
)
from tesserix_adk.core.protocols import ModelProvider, SecretProvider
from tesserix_adk.core.provider import ModelRequest, ModelResponse, StopReason, ToolDeclaration
from tesserix_adk.core.streaming import (
    ReasoningDelta,
    StreamAccumulator,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    UsageDelta,
)
from tesserix_adk.models.catalogue import (
    CATALOGUE_VERSION,
    ModelCard,
    Pricing,
    known_models,
    model_card,
    priced,
)
from tesserix_adk.models.credentials import Credential, EnvironmentSecrets

__all__ = [
    "CATALOGUE_VERSION",
    "Capability",
    "CapabilityError",
    "ContextWindowExceededError",
    "Credential",
    "EnvironmentSecrets",
    "ModelCapabilities",
    "ModelCard",
    "ModelProvider",
    "ModelRef",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseError",
    "ModelSpec",
    "Pricing",
    "ReasoningDelta",
    "SecretProvider",
    "StopReason",
    "StreamAccumulator",
    "StreamEnd",
    "StreamEvent",
    "StreamInterruptedError",
    "TextDelta",
    "ToolArgumentValidationError",
    "ToolCallDelta",
    "ToolDeclaration",
    "UsageDelta",
    "known_models",
    "model_card",
    "priced",
]
