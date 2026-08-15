"""The tool that pulls compressed content back, and the only way the original returns.

Built rather than declared, because it closes over whatever a deployment compresses with.
The expander is a protocol rather than a concrete type so that this package does not
import the one compression lives in: what matters is that the thing resolving the handle
enforces the tenant and run scope, not where it was written.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tesserix_adk.core.errors import ClaimUnavailableError
from tesserix_adk.tools.context import (
    ToolContext,  # noqa: TC001 — the decorator reads this annotation at runtime
)
from tesserix_adk.tools.decorator import Tool, tool
from tesserix_adk.tools.errors import ToolRefusal

__all__ = ["ContentExpander", "Expanded", "expand_content_tool"]


@runtime_checkable
class Expanded(Protocol):
    """The part of an expansion this tool returns: the content itself."""

    content: str
    """What the handle resolved to, within whatever ceiling was asked for."""


@runtime_checkable
class ContentExpander(Protocol):
    """Whatever resolves a content handle back to what it stands for."""

    async def expand(
        self,
        handle: str,
        *,
        tenant: str,
        run_id: str,
        budget_tokens: int | None = None,
        user: str | None = None,
    ) -> Expanded:
        """Return the content `handle` names, within the scope the caller stands in."""
        ...


def expand_content_tool(
    expander: ContentExpander,
    *,
    name: str = "expand_content",
    budget_tokens: int | None = None,
) -> Tool[..., str]:
    """Build the tool that reads back what compression left out.

    Args:
        expander: What holds the originals. The same one the run's admissions were made
            through, or the handles it issued resolve to nothing.
        name: What the model calls it. Must match the tool the compressed note names,
            which is `expand_content` unless both are overridden together.
        budget_tokens: A ceiling on one call, where the deployment wants one. The
            expansion comes back windowed rather than refused.

    Returns:
        The tool, holding the name until it is released.
    """

    @tool(name=name, idempotency="read_only")
    async def expand_content(handle: str, context: ToolContext) -> str:
        """Read the content a compressed result left out.

        Args:
            handle: The handle the compressed content named.
            context: The run this call belongs to, which scopes what may be read.
        """
        try:
            expanded = await expander.expand(
                handle,
                tenant=context.tenant,
                run_id=context.run_id,
                budget_tokens=budget_tokens,
            )
        except ClaimUnavailableError as gone:
            raise ToolRefusal(
                name,
                "claim_unavailable",
                f"{handle} holds nothing that can still be read: {gone}",
            ) from gone
        return expanded.content

    return expand_content
