"""What may be tried after a vendor will not answer, and what may not.

A fallback is a second bill and, if it happens after a tool ran, possibly a second side
effect. So it is narrow on purpose: only failures another vendor could plausibly answer,
only models the router already found eligible, only while nothing irreversible has
happened, and never silently — every attempt is on the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from tesserix_adk.core.errors import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    StreamInterruptedError,
    TrustBoundaryError,
)
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Collection

    from tesserix_adk.core.routing import RoutingDecision

__all__ = ["FallbackChain", "fallback_eligible"]

# A failure of this vendor rather than of the request: another vendor's allowance, capacity
# and queue are its own, so the same request there is a different question.
_ELSEWHERE = (RateLimitError, ProviderUnavailableError, ProviderTimeoutError)


def fallback_eligible(failure: BaseException) -> bool:
    """Whether another model may be asked the same question after `failure`.

    A spent quota qualifies even though it is not retryable: waiting will not clear an
    allowance, but another vendor's allowance is a different allowance. Everything else is
    terminal, including anything unmapped — opening a second bill on a failure nobody has
    classified is how one broken deployment becomes two.

    A stream that already emitted is the caller's to restart. Restarting transparently
    under a consumer that has seen half an answer shows it two answers, so the partial text
    is carried on `StreamInterruptedError` and the decision is theirs.

    Args:
        failure: What the model call raised.

    Returns:
        Whether the chain may move on.

    Example:
        >>> fallback_eligible(RateLimitError("slow down"))
        True
        >>> fallback_eligible(RuntimeError("who knows"))
        False
    """
    if isinstance(failure, StreamInterruptedError):
        return False
    return isinstance(failure, _ELSEWHERE)


class FallbackChain(AdkModel):
    """The models one run may be attempted against, in order.

    Built from the routing decision rather than configured separately: a fallback order
    invented apart from the routing order is a second opinion on the same question, and the
    two drift. Every link has already passed the run's capability floor, so falling down
    the chain cannot quietly lose structured output or tool calling.

    Args:
        links: Model references as `provider:model`, the chosen one first.
        excluded: Candidates that could have done the work but sit outside the run's trust
            boundary. They are never links; they are kept so exhaustion can say that an
            alternative existed and was refused, rather than that none existed.
    """

    links: tuple[str, ...] = Field(default=())
    excluded: tuple[str, ...] = Field(default=())

    @classmethod
    def of(cls, decision: RoutingDecision | None) -> FallbackChain:
        """The chain a routing decision implies, empty where there was no decision."""
        if decision is None:
            return cls()
        return cls(links=decision.chain, excluded=decision.excluded_by_boundary)

    def refuse_the_excluded(self) -> None:
        """Fail the run closed where the only alternatives left are out of boundary.

        Raises:
            TrustBoundaryError: Naming what would have been tried. Called when the chain is
                spent, so that an unavailable self-hosted model ends the run rather than
                promoting a vendor nobody approved for this data.
        """
        if not self.excluded:
            return
        raise TrustBoundaryError(
            f"every model left sits outside this run's trust boundary: "
            f"{', '.join(self.excluded)}. Falling back to one would trade a data-handling "
            f"guarantee for an availability one",
            excluded=self.excluded,
        )

    def after(self, ref: str, *, failed: Collection[str] = ()) -> str | None:
        """The next link after `ref` that has not already failed, or nothing.

        Args:
            ref: Where the run is now.
            failed: References already tried in this run. A table may offer one model under
                two rules, and a chain that comes back to a model that just refused is a
                loop rather than a fallback.

        Returns:
            The next reference to try, or `None` where the chain is spent.

        Example:
            >>> chain = FallbackChain(links=("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"))
            >>> chain.after("openai:gpt-4o-mini")
            'anthropic:claude-haiku-4-5'
            >>> chain.after("anthropic:claude-haiku-4-5") is None
            True
        """
        if ref not in self.links:
            return None
        rest = self.links[self.links.index(ref) + 1 :]
        return next((link for link in rest if link not in failed), None)
