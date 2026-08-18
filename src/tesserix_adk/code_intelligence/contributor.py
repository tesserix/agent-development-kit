"""Automatic pointer-pack contribution for the ADK runtime."""

from __future__ import annotations

import hashlib

from tesserix_adk.code_intelligence.contracts import (
    CodeContextArguments,
    CodeContextBackend,
    CodeContextOperation,
    CodeContextRequest,
    CodeWorkspaceNotFoundError,
)
from tesserix_adk.runtime.context import ContextContribution, ContextRequest

__all__ = ["CodeIntelligenceContributor"]


class CodeIntelligenceContributor:
    """Retrieve a compact code pointer pack before each prompt is assembled."""

    name = "code-intelligence"

    def __init__(
        self,
        backend: CodeContextBackend,
        *,
        limit: int = 3,
        path_prefix: str = "",
        required: bool = False,
    ) -> None:
        self._backend = backend
        self._arguments = CodeContextArguments(limit=limit, path_prefix=path_prefix)
        self._required = required

    @property
    def required(self) -> bool:
        """Whether an unavailable code graph stops the run."""
        return self._required

    async def contribute(self, request: ContextRequest) -> ContextContribution:
        """Retrieve code context inside the backend's bound tenant and workspace."""
        workspace = self._backend.workspace
        if request.tenant != workspace.tenant:
            raise CodeWorkspaceNotFoundError("code workspace not found")
        result = await self._backend.execute(
            CodeContextRequest(
                tenant=request.tenant,
                workspace=workspace.id,
                operation=CodeContextOperation.FIND,
                arguments=CodeContextArguments(
                    query=request.query,
                    limit=self._arguments.limit,
                    path_prefix=self._arguments.path_prefix,
                ),
            )
        )
        if not result.content:
            return ContextContribution()
        digest = hashlib.sha256(result.content.encode()).hexdigest()
        return ContextContribution(
            content=(result.content,),
            keys=(f"{workspace.id}:{digest}",),
        )
