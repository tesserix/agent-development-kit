"""Backend-neutral contracts for retrieving structure from a source workspace."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.models import AdkModel

__all__ = [
    "CodeContextArguments",
    "CodeContextBackend",
    "CodeContextOperation",
    "CodeContextRequest",
    "CodeContextResult",
    "CodeIntelligenceError",
    "CodeIntelligenceUnavailableError",
    "CodeWorkspace",
    "CodeWorkspaceNotFoundError",
]


class CodeIntelligenceError(AdkError):
    """A code-intelligence boundary refused or could not answer a request."""


class CodeIntelligenceUnavailableError(CodeIntelligenceError):
    """The configured backend did not answer within its contract."""


class CodeWorkspaceNotFoundError(CodeIntelligenceError):
    """The caller cannot observe the requested workspace."""


class CodeContextOperation(StrEnum):
    """The stable operations an agent can perform against any code backend."""

    FIND = "find"
    FILE_API = "file_api"
    TRACE = "trace"
    FIND_ALL = "find_all"
    REPO_MAP = "repo_map"
    FRESHNESS = "freshness"


class CodeWorkspace(AdkModel):
    """A deployment-bound checkout, including the tenant allowed to observe it."""

    id: str = Field(min_length=1, max_length=256)
    tenant: str = Field(min_length=1, max_length=256)
    root: str = Field(min_length=1, max_length=4096)


class CodeContextArguments(AdkModel):
    """The model-selectable part of a code-intelligence request."""

    query: str = Field(default="", max_length=16_384)
    file: str = Field(default="", max_length=4096)
    symbol: str = Field(default="", max_length=4096)
    pattern: str = Field(default="", max_length=4096)
    path_prefix: str = Field(default="", max_length=4096)
    limit: int = Field(default=5, ge=1, le=20)
    full: bool = False
    direction: Literal["in", "out"] = "in"
    depth: int | Literal["all"] = 1
    ignore_case: bool = False
    fixed: bool = False
    max_dirs: int = Field(default=16, ge=1, le=64)

    @field_validator("file", "path_prefix")
    @classmethod
    def _workspace_relative(cls, value: str) -> str:
        if not value:
            return value
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or PureWindowsPath(value).drive:
            raise ValueError("a code path must be relative to its bound workspace")
        if ".." in path.parts:
            raise ValueError("a code path cannot traverse a parent workspace")
        return str(path)

    @field_validator("depth")
    @classmethod
    def _bounded_depth(cls, value: int | Literal["all"]) -> int | Literal["all"]:
        if isinstance(value, int) and (value < 1 or value > 32):
            raise ValueError("trace depth must be from 1 to 32, or 'all'")
        return value


_REQUIRED_ARGUMENT = {
    CodeContextOperation.FIND: "query",
    CodeContextOperation.FILE_API: "file",
    CodeContextOperation.TRACE: "symbol",
    CodeContextOperation.FIND_ALL: "pattern",
}


class CodeContextRequest(AdkModel):
    """One authorized operation against one workspace."""

    tenant: str = Field(min_length=1, max_length=256)
    workspace: str = Field(min_length=1, max_length=256)
    operation: CodeContextOperation
    arguments: CodeContextArguments = Field(default_factory=CodeContextArguments)

    @model_validator(mode="after")
    def _operation_has_its_argument(self) -> CodeContextRequest:
        required = _REQUIRED_ARGUMENT.get(self.operation)
        if required is not None and not str(getattr(self.arguments, required)).strip():
            raise ValueError(f"{self.operation} requires a non-empty {required}")
        return self


class CodeContextResult(AdkModel):
    """Untrusted context returned by a code-intelligence backend."""

    operation: CodeContextOperation
    content: str = Field(max_length=1_000_000)
    backend: str = Field(min_length=1, max_length=128)
    stale: bool = False


@runtime_checkable
class CodeContextBackend(Protocol):
    """A workspace-bound structural context provider."""

    @property
    def workspace(self) -> CodeWorkspace:
        """The only workspace this backend may answer for."""
        ...

    async def execute(self, request: CodeContextRequest) -> CodeContextResult:
        """Execute one validated operation inside the bound workspace."""
        ...
