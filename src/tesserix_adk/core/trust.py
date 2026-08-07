"""The boundary a run's data may not be moved across, and what that does to fallback.

Resilience and confidentiality pull in opposite directions the moment a model is
unavailable. A chain that promotes a hosted vendor when the self-hosted endpoint is down,
or the standard tier when the sealed one is, has traded a data-handling guarantee for an
availability one — silently, and in the direction nobody would have approved.

So a boundary is declared on the model, and a fallback is legal only between models that
share it. Where the source declares a boundary and the target does not, the target is
refused: an undeclared boundary is an unknown one, and unknown is not equal.
"""

from __future__ import annotations

from tesserix_adk.core.models import AdkModel

__all__ = ["AXES", "TrustBoundary"]

# What must match for two models to be interchangeable. Ordered as an operator reads them.
AXES = ("tier", "hosting", "residency")


class TrustBoundary(AdkModel):
    """Where a model sits, on the axes a fallback may not cross.

    Args:
        tier: The sensitivity tier the model is approved for — `sealed`, `standard`.
        hosting: Who runs the weights — `self-hosted`, `vendor-api`.
        residency: The data-residency class the inference happens in.

    Every axis is a free string rather than an enum: the tiers, hosting arrangements and
    residency classes that matter are a deployment's own vocabulary, and a kit-level enum
    would either be wrong or force a fork to add a value to it.

    Example:
        >>> sealed = TrustBoundary(tier="sealed", hosting="self-hosted", residency="in")
        >>> sealed.admits(TrustBoundary(tier="standard", hosting="self-hosted", residency="in"))
        False
    """

    tier: str = ""
    hosting: str = ""
    residency: str = ""

    @property
    def stated(self) -> bool:
        """Whether this boundary says anything at all.

        A deployment that declares nothing constrains nothing — the kit cannot enforce a
        boundary nobody wrote down, and pretending otherwise would refuse every fallback
        in every existing configuration rather than protect anything.
        """
        return any(getattr(self, axis) for axis in AXES)

    def differs_from(self, other: TrustBoundary) -> tuple[str, ...]:
        """Which axes disagree, in order. Empty means the two are interchangeable."""
        return tuple(axis for axis in AXES if getattr(self, axis) != getattr(other, axis))

    def admits(self, other: TrustBoundary) -> bool:
        """Whether a run inside this boundary may be moved to a model in `other`.

        Fails closed on every axis this boundary states: an unstated target is not a
        matching one. An unstated *source* admits anything, which is the honest reading of
        a deployment that has declared no boundary rather than a licence to leave one.
        """
        return not self.stated or not self.differs_from(other)
