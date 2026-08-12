"""Substituting an oversized tool result, and an in-process store to substitute it into.

The store here outlives nothing: a restart loses every handle it holds, which for content
scoped to a run in flight is the right shape and for a run that must survive a failover is
not. A deployment that needs the second one binds an adapter; this one exists so the
behaviour can be exercised without a backing service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tesserix_adk.core.claim_check import ClaimCheckPolicy, ClaimTicket, claim_handle
from tesserix_adk.core.errors import ClaimUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core.claim_check import ClaimCheckStore
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.runtime.results import ToolResult

__all__ = ["ClaimCheck", "MemoryClaimCheckStore"]

# Where a head may be cut so it ends on something a reader recognises as an ending.
_BOUNDARIES = ("\n\n", "\n", ". ")


@dataclass(frozen=True, slots=True)
class _Held:
    content: str
    tenant: str
    run_id: str
    expires_at: float


class MemoryClaimCheckStore:
    """A `ClaimCheckStore` in a dict, keyed by handle and checked against its scope.

    Example:
        >>> import asyncio
        >>> store = MemoryClaimCheckStore()
        >>> async def held() -> str:
        ...     await store.put(
        ...         "claim:a", "the document", tenant="acme", run_id="r", ttl_seconds=60
        ...     )
        ...     return await store.fetch("claim:a", tenant="acme", run_id="r")
        >>> asyncio.run(held())
        'the document'
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._held: dict[str, _Held] = {}
        self._clock = clock

    @property
    def held(self) -> int:
        """How many payloads are stored, for a consumer sizing what it bound."""
        return len(self._held)

    async def put(
        self, handle: str, content: str, *, tenant: str, run_id: str, ttl_seconds: float
    ) -> None:
        """Hold `content` under `handle` until its retention window closes."""
        self._held[handle] = _Held(content, tenant, run_id, self._now() + ttl_seconds)

    async def fetch(self, handle: str, *, tenant: str, run_id: str) -> str:
        """Return what was stored under `handle` for this tenant and run.

        Raises:
            ClaimUnavailableError: If nothing is held under that handle for that scope, or
                what was held has passed its retention window.
        """
        held = self._held.get(handle)
        if held is not None and held.expires_at <= self._now():
            del self._held[handle]
            held = None
        if held is None or held.tenant != tenant or held.run_id != run_id:
            raise ClaimUnavailableError(
                f"nothing is held under {handle} for {tenant}/{run_id}",
                handle=handle,
                tenant=tenant,
                run_id=run_id,
            )
        return held.content

    async def forget(self, *, tenant: str, run_id: str = "") -> int:
        """Drop everything held for `tenant`, or for one of its runs, and say how many."""
        doomed = [
            handle
            for handle, held in self._held.items()
            if held.tenant == tenant and run_id in ("", held.run_id)
        ]
        for handle in doomed:
            del self._held[handle]
        return len(doomed)

    def _now(self) -> float:
        return 0.0 if self._clock is None else self._clock.now()


@dataclass(frozen=True, slots=True)
class ClaimCheck:
    """Decides which results are carried and which are checked in, and does the checking.

    Args:
        store: Where checked-in content goes. Any `ClaimCheckStore`; the kit's in-process
            one is the default nobody should ship.
        policy: What every tool is held to unless it is named in `per_tool`.
        per_tool: Thresholds for the tools that need their own. A tool that returns a
            whole PDF and one that returns a row count are not the same decision.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.runtime import ToolResult
        >>> check = ClaimCheck(store=MemoryClaimCheckStore())
        >>> big = ToolResult(tool="read_page", text="x" * 9_000)
        >>> ticket = asyncio.run(check.stored(big, tenant="acme", run_id="run_1"))
        >>> ticket.chars if ticket else 0
        9000
    """

    store: ClaimCheckStore
    policy: ClaimCheckPolicy = field(default_factory=ClaimCheckPolicy)
    per_tool: Mapping[str, ClaimCheckPolicy] = field(default_factory=dict)

    async def stored(self, result: ToolResult, *, tenant: str, run_id: str) -> ClaimTicket | None:
        """Check `result` in and return its ticket, or `None` if it is small enough to carry.

        Args:
            result: What crossed the result boundary, already validated and neutralised.
                Checking in what the boundary refused would store an attack for later.
            tenant: Whose run it belongs to, which scopes the handle.
            run_id: Which run, which scopes it further.
        """
        policy = self.per_tool.get(result.tool, self.policy)
        content = result.text
        if len(content) <= policy.threshold_chars:
            return None
        handle = claim_handle(tenant=tenant, run_id=run_id, content=content)
        await self.store.put(
            handle, content, tenant=tenant, run_id=run_id, ttl_seconds=policy.ttl_seconds
        )
        return ClaimTicket(
            handle=handle,
            tool=result.tool,
            chars=len(content),
            head=_head(content, policy.head_chars),
        )


def _head(content: str, ceiling: int) -> str:
    """The leading extract, cut where the content itself ends something where it can be.

    A head stopping mid-word reads as damage, and a model shown damage tends to fetch the
    rest whether or not it needed to — which is the cost this was meant to avoid.
    """
    window = content[: ceiling + 1]
    ends = [window.rfind(boundary) for boundary in _BOUNDARIES]
    cut = max(ends)
    return window[:ceiling] if cut <= 0 else window[:cut].rstrip()
