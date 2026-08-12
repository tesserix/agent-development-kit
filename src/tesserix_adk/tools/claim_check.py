"""The tool that redeems a claim-check handle, and the only way content comes back.

Built rather than declared, because it closes over the store a deployment bound. Reading
is windowed: a fetch that returned the whole document would put back into the conversation
exactly what checking it in took out, one tool call later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.claim_check import HANDLE_PREFIX
from tesserix_adk.core.errors import ClaimUnavailableError
from tesserix_adk.tools.context import (
    ToolContext,  # noqa: TC001 — the decorator reads this annotation at runtime
)
from tesserix_adk.tools.decorator import Tool, tool
from tesserix_adk.tools.errors import ToolRefusal

if TYPE_CHECKING:
    from tesserix_adk.core.claim_check import ClaimCheckStore

__all__ = ["DEFAULT_FETCH_CHARS", "claim_check_tool"]

DEFAULT_FETCH_CHARS = 4_096
"""How much one fetch returns. A window, not the document."""


def claim_check_tool(
    store: ClaimCheckStore,
    *,
    name: str = "fetch_result",
    max_chars: int = DEFAULT_FETCH_CHARS,
) -> Tool[..., str]:
    """Build the tool that reads content a claim check held back.

    Args:
        store: Where the content was checked in. The same store the run's `ClaimCheck`
            writes to, or the handles it issues redeem to nothing.
        name: What the model calls it. Must match the name a `ClaimTicket` renders, which
            is `fetch_result` unless both are overridden together.
        max_chars: How much one call returns, from `offset` onward.

    Returns:
        The tool, holding the name until it is released.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.runtime import MemoryClaimCheckStore
        >>> store = MemoryClaimCheckStore()
        >>> fetch = claim_check_tool(store)
        >>> async def read() -> str:
        ...     await store.put("claim:a", "clause 1", tenant="acme", run_id="r", ttl_seconds=60)
        ...     return await fetch.invoke(
        ...         {"handle": "claim:a"}, ToolContext(run_id="r", tenant="acme")
        ...     )
        >>> asyncio.run(read())
        'clause 1'
        >>> fetch.release()
    """

    @tool(name=name, idempotency="read_only")
    async def fetch_result(handle: str, context: ToolContext, offset: int = 0) -> str:
        """Read the content a previous tool result was too large to carry.

        Args:
            handle: The handle the result was replaced by.
            context: The run this call belongs to, which scopes what may be read.
            offset: Where to start reading, for content longer than one window.
        """
        if offset < 0:
            raise _refused(name, "invalid_offset", f"offset {offset} is before the start")
        if not handle.startswith(HANDLE_PREFIX):
            raise _refused(name, "claim_unavailable", f"{handle!r} is not a result handle")
        try:
            content = await store.fetch(handle, tenant=context.tenant, run_id=context.run_id)
        except ClaimUnavailableError as gone:
            raise _refused(
                name,
                "claim_unavailable",
                f"{handle} holds nothing that can still be read: {gone}",
            ) from gone
        return content[offset : offset + max_chars]

    return fetch_result


def _refused(tool_name: str, code: str, message: str) -> ToolRefusal:
    """A refusal rather than a failure: retrying reads the same nothing."""
    return ToolRefusal(tool_name, code, message)
