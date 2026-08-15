"""The guard that takes identifiers out before anything downstream can keep them.

It redacts rather than blocks. A message with a card number in it is still the customer's
question, and a guard that refuses the whole turn teaches the product to turn the guard off.
The value is replaced by a placeholder carrying its type and a per-tenant pseudonym, so the
model can still reason about "the same traveller" without ever holding the number.

Redaction is applied on the way in *and* on the way out, because a model that was given a
placeholder can reconstruct the value from context it was given earlier in the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.guards import GuardResult
from tesserix_adk.core.pii import DEFAULT_DETECTORS, redact
from tesserix_adk.guardrails.base import Guard

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tesserix_adk.core.pii import PIIDetector, Redacted

__all__ = ["PIIGuard"]


class PIIGuard(Guard):
    """Replaces personal identifiers with placeholders on both stages of a run.

    Args:
        tenant: Whose data this is, which decides the pseudonym. Two tenants never share
            one, so neither can probe the other by guessing a subject.
        detectors: What to look for. A consumer's own detector goes here beside the
            built-ins rather than replacing them.
        threshold: The confidence a match needs before it is redacted. Raising it is how a
            tenant stops a booking reference being read as a passport number.
        allow: Shapes this tenant has said are not identifiers.

    Example:
        >>> PIIGuard(tenant="acme").name
        'pii'
    """

    name = "pii"

    def __init__(
        self,
        *,
        tenant: str,
        detectors: Sequence[PIIDetector] = DEFAULT_DETECTORS,
        threshold: float = 0.6,
        allow: Iterable[str] = (),
    ) -> None:
        self.tenant = tenant
        self.detectors = tuple(detectors)
        self.threshold = threshold
        self.allow = tuple(allow)

    def apply(self, content: str) -> Redacted:
        """What this content becomes once its identifiers are stood in for.

        Raises:
            GuardrailEvaluationError: Where a detector could not answer. Content is never
                passed through unredacted because the check did not complete.
        """
        return redact(
            content,
            tenant=self.tenant,
            detectors=self.detectors,
            threshold=self.threshold,
            allow=self.allow,
        )

    async def check_input(self, content: str) -> GuardResult:
        """Redact on the way in, so no identifier reaches prompt assembly."""
        return self._verdict(content)

    async def check_output(self, content: str) -> GuardResult:
        """Redact on the way out, so nothing the model reconstructed is emitted."""
        return self._verdict(content)

    def _verdict(self, content: str) -> GuardResult:
        """Allow untouched content, and hand on the rewritten passage otherwise."""
        applied = self.apply(content)
        if not applied.found:
            return GuardResult.allow()
        kinds = ",".join(kind.value for kind in applied.kinds)
        return GuardResult.redacted(
            applied.text,
            code="pii_redacted",
            detail=f"{applied.count} identifier(s): {kinds}",
        )
