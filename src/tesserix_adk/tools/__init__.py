"""Tool registry, argument schemas, invocation and the ToolBus."""

from tesserix_adk.tools.claim_check import DEFAULT_FETCH_CHARS, claim_check_tool
from tesserix_adk.tools.context import ToolContext
from tesserix_adk.tools.decorator import Tool, tool
from tesserix_adk.tools.errors import (
    ToolError,
    ToolErrorMap,
    ToolErrorRule,
    ToolFailure,
    ToolRefusal,
    permanent,
    refusal,
    transient,
)
from tesserix_adk.tools.intake import (
    DEFAULT_INTAKE_CHARS,
    SUPPORTED_AUDIO,
    SUPPORTED_DOCUMENTS,
    OcrBackend,
    OcrPage,
    Region,
    RegionKind,
    Segment,
    TranscriptionBackend,
    ocr_document,
    ocr_tool,
    transcribe_audio,
    transcribe_tool,
)
from tesserix_adk.tools.registry import AgentToolView, ToolCallSpan, ToolRegistry
from tesserix_adk.tools.sandbox import (
    DEFAULT_LIMITS,
    Sandbox,
    SandboxArtifact,
    SandboxLimits,
    SandboxResult,
    SubprocessSandbox,
    sandbox_tool,
)
from tesserix_adk.tools.validation import (
    LENIENT,
    STRICT,
    ArgumentPolicy,
    ToolArgumentValidator,
)

__all__ = [
    "DEFAULT_FETCH_CHARS",
    "DEFAULT_INTAKE_CHARS",
    "DEFAULT_LIMITS",
    "LENIENT",
    "STRICT",
    "SUPPORTED_AUDIO",
    "SUPPORTED_DOCUMENTS",
    "AgentToolView",
    "ArgumentPolicy",
    "OcrBackend",
    "OcrPage",
    "Region",
    "RegionKind",
    "Sandbox",
    "SandboxArtifact",
    "SandboxLimits",
    "SandboxResult",
    "Segment",
    "SubprocessSandbox",
    "Tool",
    "ToolArgumentValidator",
    "ToolCallSpan",
    "ToolContext",
    "ToolError",
    "ToolErrorMap",
    "ToolErrorRule",
    "ToolFailure",
    "ToolRefusal",
    "ToolRegistry",
    "TranscriptionBackend",
    "claim_check_tool",
    "ocr_document",
    "ocr_tool",
    "permanent",
    "refusal",
    "sandbox_tool",
    "tool",
    "transcribe_audio",
    "transcribe_tool",
    "transient",
]
