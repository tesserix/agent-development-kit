"""Compression the model can undo, so an elided detail is a turn rather than a hole.

Every lossy scheme is a bet that the model will not need the part that was removed. The
bet is usually right and occasionally catastrophically wrong: the one elided log line named
the failing host, and the agent now answers confidently from a gap it cannot see.

The fix is not better compression. It is a handle. The compressed form carries a stable,
content-addressed reference, the original is retained for the run that made it, and a tool
resolves the reference back. That turns an irreversible quality risk into a recoverable
extra turn, which is what makes an aggressive ratio defensible at all — and it makes the
accuracy question testable, because "did the model retrieve when it should" is a behaviour
and "did the summary keep the right sentence" is not.

The handle is the claim check the kit already issues for oversized tool results, so the
scope rules come with it: the tenant and the run are hashed into the handle as well as
checked at the lookup, and a handle from elsewhere is unavailable rather than denied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.audit import AuditDecision, AuditEvent, digest_of_arguments
from tesserix_adk.core.claim_check import HANDLE_PREFIX, claim_handle
from tesserix_adk.core.errors import ClaimUnavailableError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.compression import Compressed, PassThrough
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from tesserix_adk.core.audit import AuditSink
    from tesserix_adk.core.claim_check import ClaimCheckStore
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.memory.compression import ContentRouter

__all__ = ["EXPAND_TOOL", "Expansion", "ReversibleRouter"]

EXPAND_TOOL = "expand_content"
"""What the note in compressed content tells the model to call. The tool must match."""

_ACTION_CLASS = "context.expand"


class Expansion(AdkModel):
    """What a handle resolved to.

    Args:
        handle: What was asked for.
        content: The original, or the leading window of it that fits the budget asked for.
        chars: How large the original is, whether or not all of it came back.
        truncated: Whether `content` stops short of the original.
    """

    handle: str = ""
    content: str = ""
    chars: int = 0
    truncated: bool = False


class ReversibleRouter:
    """A `ContentRouter` that keeps the original and hands back a way to reach it.

    Args:
        router: What decides the compressed form. Anything it passes through is left
            alone entirely — there is nothing to reverse, so there is no handle to issue.
        store: Where originals are retained, scoped to the tenant and run that wrote them.
        ttl_seconds: The retention window. Past it the handle expires rather than dangles,
            because a store outliving its run holds content nobody is going to read.
        audit: Where retrievals are recorded. Every expansion is a deliberate decision to
            put content back into a prompt, and that is worth a record.
        clock: What the audit timestamps come from.
        expand_tool: What the note tells the model to call, where a deployment renamed it.
    """

    def __init__(
        self,
        router: ContentRouter,
        store: ClaimCheckStore,
        *,
        ttl_seconds: float = 3_600.0,
        audit: AuditSink | None = None,
        clock: Clock | None = None,
        expand_tool: str = EXPAND_TOOL,
    ) -> None:
        self._router = router
        self._store = store
        self._ttl = ttl_seconds
        self._audit = audit
        self._clock = clock or SystemClock()
        self._tool = expand_tool
        self._sequence = 0

    async def admit(
        self,
        content: str,
        *,
        budget_tokens: int,
        tenant: str,
        run_id: str,
        untrusted: bool = False,
    ) -> Compressed:
        """Compress `content`, retain the original, and note how to get it back.

        Args:
            content: The text being admitted to context.
            budget_tokens: What the caller would like the compressed form to fit in.
            tenant: The isolation boundary this admission happens in.
            run_id: The run that will be allowed to expand the handle. Only that one.
            untrusted: Whether the content came from outside. Carried through unchanged.

        Returns:
            The compressed admission, with `handle` set and the note appended where
            anything was compressed. Where nothing was, the record is what the underlying
            router returned and `handle` is empty.
        """
        admitted = self._router.admit(content, budget_tokens=budget_tokens, untrusted=untrusted)
        if admitted.compressor == PassThrough.name:
            return admitted
        handle = claim_handle(tenant=tenant, run_id=run_id, content=content)
        await self._store.put(handle, content, tenant=tenant, run_id=run_id, ttl_seconds=self._ttl)
        noted = f"{admitted.content}\n\n{self._note(handle, len(content))}"
        return admitted.model_copy(
            update={
                "content": noted,
                "handle": handle,
                "compressed_tokens": self._router.count(noted),
            }
        )

    async def expand(
        self,
        handle: str,
        *,
        tenant: str,
        run_id: str,
        budget_tokens: int | None = None,
        user: str | None = None,
    ) -> Expansion:
        """Resolve `handle` back to the content it stands for.

        Args:
            handle: What the compressed content named.
            tenant: Who is asking. Hashed into the handle as well as checked here.
            run_id: Which run is asking.
            budget_tokens: What is left of the prompt. Where the original does not fit,
                the leading window that does comes back marked truncated — a retrieval
                that overran the budget would move the problem rather than solve it.
            user: On whose behalf, for the audit record.

        Returns:
            The original, or the part of it the budget allowed.

        Raises:
            ClaimUnavailableError: If the handle is unknown here, belongs to another
                tenant or run, or has passed its retention window. The conditions are
                deliberately indistinguishable to the caller: telling a model which one
                failed tells it what handles other runs hold.
        """
        try:
            content = await self._checked(handle, tenant=tenant, run_id=run_id)
        except ClaimUnavailableError as gone:
            await self._record(
                handle, tenant=tenant, run_id=run_id, user=user, decision=AuditDecision.REFUSED
            )
            raise gone from None
        windowed = self._within(content, budget_tokens)
        await self._record(
            handle,
            tenant=tenant,
            run_id=run_id,
            user=user,
            decision=AuditDecision.EXECUTED,
            chars=len(content),
        )
        return Expansion(
            handle=handle,
            content=windowed,
            chars=len(content),
            truncated=windowed != content,
        )

    def _note(self, handle: str, chars: int) -> str:
        """What tells the model the rest exists and how to ask for it."""
        return (
            f"[compressed from {chars} characters. "
            f"Call {self._tool} with handle={handle!r} for the original.]"
        )

    async def _checked(self, handle: str, *, tenant: str, run_id: str) -> str:
        """The stored content, refusing anything not shaped like a handle we issue."""
        if not handle.startswith(HANDLE_PREFIX):
            raise ClaimUnavailableError(
                f"{handle!r} is not a content handle", handle=handle, tenant=tenant, run_id=run_id
            )
        return await self._store.fetch(handle, tenant=tenant, run_id=run_id)

    def _within(self, content: str, budget_tokens: int | None) -> str:
        """The content, or the leading part of it that fits what is left of the prompt."""
        count = self._router.count
        if budget_tokens is None or count(content) <= budget_tokens:
            return content
        kept = content
        while kept and count(kept) > budget_tokens:
            kept = kept[: max(1, len(kept) * 9 // 10)][:-1]
        return kept

    async def _record(
        self,
        handle: str,
        *,
        tenant: str,
        run_id: str,
        user: str | None,
        decision: AuditDecision,
        chars: int = 0,
    ) -> None:
        """Write down that content went back into a prompt, and how much of it."""
        if self._audit is None:
            return
        self._sequence += 1
        await self._audit.append(
            AuditEvent(
                run_id=run_id,
                sequence=self._sequence,
                tenant=tenant,
                user=user,
                agent_name="",
                tool=self._tool,
                action_class=_ACTION_CLASS,
                decision=decision,
                reason=(
                    f"{handle} resolved to {chars} characters"
                    if decision is AuditDecision.EXECUTED
                    else f"{handle} resolved to nothing readable here"
                ),
                arguments_digest=digest_of_arguments({"handle": handle}),
                idempotency_key=f"{_ACTION_CLASS}:{handle}:{self._sequence}",
                recorded_at=self._clock.now(),
            )
        )
