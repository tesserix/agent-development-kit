"""The guard that applies this tenant's bar, and normalises a provider's own refusal.

Two things are kept apart here. What a passage is about is a classifier's answer, and it is
the same answer for every tenant. What to do about it is the tenant's bar, and a
customer-facing agent and an internal triage agent reading the same abusive transcript
differ by configuration, not by code.

A block on the way in refuses before a model call is spent. A block on the way out never
emits the payload — including into the refusal, which is a caller-safe structure carrying
categories and a severity and nothing that was said.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core.content_policy import ContentSeverity, HeuristicClassifier, Thresholds
from tesserix_adk.core.errors import ContentBlockedError, GuardrailEvaluationError
from tesserix_adk.core.guards import GuardResult
from tesserix_adk.guardrails.base import Guard

if TYPE_CHECKING:
    from tesserix_adk.core.content_policy import Classification, ContentClassifier

__all__ = ["ContentFilterGuard", "refusal_of"]

_TIMEOUT_SECONDS = 5.0


class ContentFilterGuard(Guard):
    """Refuses content that reaches this tenant's bar, on either stage.

    Args:
        thresholds: The severity at which each category is refused here.
        classifier: Who decides what the passage is about. A heuristic by default, so the
            policy path works before a deployment has a model for it.
        timeout_seconds: How long the classifier has. A classifier nobody waited for has
            not classified anything, so the guard blocks rather than passing content
            through unclassified.
        tenant: Recorded on the refusal, never used to change the decision.

    Example:
        >>> ContentFilterGuard().name
        'content_filter'
    """

    name = "content_filter"

    def __init__(
        self,
        *,
        thresholds: Thresholds | None = None,
        classifier: ContentClassifier | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
        tenant: str = "",
    ) -> None:
        self.thresholds = thresholds or Thresholds()
        self.classifier: ContentClassifier = classifier or HeuristicClassifier()
        self.timeout_seconds = timeout_seconds
        self.tenant = tenant

    async def classify(self, content: str) -> Classification:
        """What the classifier says, or a refusal naming it where it could not say.

        Raises:
            GuardrailEvaluationError: Where the classifier timed out or raised. Content is
                never passed through unclassified on the grounds that nobody could answer.
        """
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self.classifier.classify(content)
        except TimeoutError as late:
            raise GuardrailEvaluationError(
                f"classifier {self.classifier.name!r} did not answer in {self.timeout_seconds}s",
                guard=self.name,
                reason="timeout",
            ) from late
        except Exception as broke:
            raise GuardrailEvaluationError(
                f"classifier {self.classifier.name!r} could not evaluate the content",
                guard=self.name,
                reason="raised",
            ) from broke

    async def check_input(self, content: str) -> GuardResult:
        """Refuse before a model call is spent."""
        return await self._verdict(content)

    async def check_output(self, content: str) -> GuardResult:
        """Refuse without emitting what was generated."""
        return await self._verdict(content)

    async def raise_for(self, content: str, *, stage: str = "input") -> None:
        """Refuse the content with the typed violation, carrying no part of it.

        Raises:
            ContentBlockedError: Where any category reached this tenant's bar.
        """
        found = await self.classify(content)
        over = found.breaches(self.thresholds)
        if not over:
            return
        raise ContentBlockedError(
            f"content reached the bar for {', '.join(over)}",
            categories=tuple(category.value for category in over),
            severity=found.worst.name.lower(),
            classifier=found.classifier,
            stage=stage,
            guard=self.name,
            tenant=self.tenant or None,
        )

    async def _verdict(self, content: str) -> GuardResult:
        found = await self.classify(content)
        over = found.breaches(self.thresholds)
        if not over:
            return GuardResult.allow()
        named = ", ".join(category.value for category in over)
        return GuardResult.blocked(
            code="content_blocked",
            detail=f"{named} at {found.worst.name.lower()}",
        )


def refusal_of(
    categories: tuple[str, ...],
    *,
    severity: str = ContentSeverity.HIGH.name.lower(),
    classifier: str = "provider",
    stage: str = "output",
) -> ContentBlockedError:
    """A provider's own safety refusal, as the kit's violation.

    A vendor refusal arrives as an opaque error whose shape differs per vendor. Normalising
    it here is what lets a caller handle one thing, and lets a run change provider without
    changing what a refusal looks like.

    Example:
        >>> refusal_of(("hate",)).code
        'content_blocked'
    """
    return ContentBlockedError(
        f"the provider refused: {', '.join(categories)}",
        categories=categories,
        severity=severity,
        classifier=classifier,
        stage=stage,
    )
