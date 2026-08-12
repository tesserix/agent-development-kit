"""A tool result large enough to matter is paid for on every turn, not just the one it arrived on.

A tool that returns a contract, a log file or a scraped page puts that content in the
conversation, and the conversation is re-sent on every iteration after it. On a CPU
backend that is prefill, repeatedly, for a document the model read once and then reasoned
past. The size is not the problem — carrying it forward is.

So an oversized result is checked in: the content goes to a store, and what enters the
conversation is an extractive head and a handle. The head is usually the whole answer; the
handle is there for the times it is not, and dereferencing it is a tool call the model
makes deliberately rather than a cost every turn pays silently.

A handle is scoped to the tenant and the run that made it, and the scope is in the digest
as well as in the lookup — a handle from another run is not merely refused, it cannot be
derived. What is stored is content nobody validated, so it never comes back as anything
but text.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tesserix_adk.core.errors import ConfigurationError

__all__ = [
    "HANDLE_PREFIX",
    "ClaimCheckPolicy",
    "ClaimCheckStore",
    "ClaimTicket",
    "claim_handle",
]

HANDLE_PREFIX = "claim:"
"""What every handle starts with, so a handle is recognisable in a transcript."""

_DIGEST_CHARS = 32


@dataclass(frozen=True, slots=True)
class ClaimCheckPolicy:
    """When a result is checked in, how much of it stays, and how long the rest is kept.

    Args:
        threshold_chars: The rendered size at which a result is stored rather than carried.
            Below it nothing happens at all, because a handle costs a tool call and a small
            result is cheaper to just read.
        head_chars: How much of the content stays in the conversation. Enough to answer the
            question outright most of the time, which is the whole saving.
        ttl_seconds: How long the stored content can still be dereferenced. A run outlives
            its own turns; a store that outlives the run is a store holding untrusted
            content nobody is going to read.

    Example:
        >>> ClaimCheckPolicy(threshold_chars=2_000).head_chars
        512
    """

    threshold_chars: int = 4_096
    head_chars: int = 512
    ttl_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        """Refuse a policy that cannot save anything, at the point it is written."""
        if self.threshold_chars <= 0:
            raise ConfigurationError(
                f"a claim-check threshold of {self.threshold_chars} stores every result, "
                f"including the ones a handle costs more than"
            )
        if self.head_chars >= self.threshold_chars:
            raise ConfigurationError(
                f"a head of {self.head_chars} characters against a threshold of "
                f"{self.threshold_chars} saves nothing: the substitution would be as large "
                f"as what it replaced"
            )
        if self.ttl_seconds <= 0:
            raise ConfigurationError(
                "a retention window of zero stores content that can never be fetched"
            )


@dataclass(frozen=True, slots=True)
class ClaimTicket:
    """What replaces an oversized result: the beginning of it, and where the rest is.

    Args:
        handle: What to pass back to fetch the content, scoped to one tenant and one run.
        tool: Which tool returned it, so a transcript still says where the content is from.
        chars: How large the content was, so the model can judge whether to fetch it.
        head: The leading extract that stays in the conversation.
        truncated: Whether the head stops short of the content, which it does whenever a
            ticket exists at all. Present because the rendering reads it.

    Example:
        >>> ticket = ClaimTicket(handle="claim:a1", tool="read_page", chars=9_000, head="C1.")
        >>> ticket.chars
        9000
    """

    handle: str
    tool: str = ""
    chars: int = 0
    head: str = ""
    truncated: bool = True
    fetch_tool: str = "fetch_result"

    def rendered(self) -> str:
        """Return what enters the conversation in place of the content.

        Names the size and the fetching tool, because a handle the model does not know how
        to redeem is a truncation with extra steps.
        """
        return (
            f"{self.head}\n\n"
            f"[{self.chars} characters, {max(self.chars - len(self.head), 0)} not shown. "
            f"Call {self.fetch_tool} with handle={self.handle!r} to read the rest.]"
        )


@runtime_checkable
class ClaimCheckStore(Protocol):
    """Where a checked-in result lives until the run that made it is over.

    Implementations are scoped by tenant and run: a handle is only ever readable by the
    run that stored it, and a fetch outside that scope is unavailable rather than denied,
    because "you may not read this" is itself a statement about what exists.
    """

    async def put(
        self, handle: str, content: str, *, tenant: str, run_id: str, ttl_seconds: float
    ) -> None:
        """Hold `content` under `handle` for this tenant and run, for `ttl_seconds`."""
        ...

    async def fetch(self, handle: str, *, tenant: str, run_id: str) -> str:
        """Return what was stored, or raise `ClaimUnavailableError` if it is not there."""
        ...

    async def forget(self, *, tenant: str, run_id: str = "") -> int:
        """Drop everything held for a tenant, or for one of its runs, and say how many."""
        ...


def claim_handle(*, tenant: str, run_id: str, content: str) -> str:
    """Derive the handle identifying one stored result.

    The scope is hashed with the content, so identical content in two tenants derives two
    handles and a handle cannot be transplanted between runs. Identical content within one
    run derives the same handle, which is how a tool called twice is stored once.

    Example:
        >>> claim_handle(tenant="acme", run_id="run_1", content="x").startswith("claim:")
        True
    """
    digest = hashlib.blake2b(
        b"\x00".join(part.encode("utf-8") for part in (tenant, run_id, content)),
        digest_size=_DIGEST_CHARS // 2,
    )
    return f"{HANDLE_PREFIX}{digest.hexdigest()}"
