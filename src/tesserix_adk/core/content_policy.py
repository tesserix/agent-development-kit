"""What content is about, how bad it is, and the bar each tenant sets for refusing it.

Content policy is otherwise whatever the model provider happens to refuse, returned as an
opaque error that differs per vendor. A run that changes provider changes its safety
behaviour, and nobody wrote that down. The taxonomy here is declared, so a refusal reads
the same whoever served the turn, and a provider's native refusal is normalised into it.

Thresholds are per tenant. A customer-facing agent and an internal triage agent looking at
the same abusive support transcript should differ by configuration, not by code: the
transcript is the thing being triaged, and blocking it stops the work it exists for.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from enum import IntEnum, StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.models import AdkModel

__all__ = [
    "Classification",
    "ContentCategory",
    "ContentClassifier",
    "ContentSeverity",
    "HeuristicClassifier",
    "Thresholds",
]


class ContentCategory(StrEnum):
    """What a passage is about, in the terms a policy is written in."""

    HATE = "hate"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    WEAPONS = "weapons"
    ILLEGAL = "illegal"


class ContentSeverity(IntEnum):
    """How bad, ordered so a threshold is a comparison rather than a lookup table."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Classification(AdkModel):
    """What a classifier decided about one passage.

    Args:
        severities: A severity per category it had an opinion about. A category absent
            means the classifier said nothing, not that it said `NONE`.
        classifier: Which one answered, so a change in behaviour can be traced to it.

    Example:
        >>> Classification(severities={ContentCategory.HATE: ContentSeverity.HIGH}).worst
        <ContentSeverity.HIGH: 3>
    """

    severities: Mapping[ContentCategory, ContentSeverity] = Field(default_factory=dict)
    classifier: str = ""

    @property
    def worst(self) -> ContentSeverity:
        """The highest severity in the classification, or `NONE` where there is none."""
        return max(self.severities.values(), default=ContentSeverity.NONE)

    def breaches(self, thresholds: Thresholds) -> tuple[ContentCategory, ...]:
        """Every category that reached or passed this tenant's bar, sorted."""
        return tuple(
            sorted(
                category
                for category, severity in self.severities.items()
                if severity >= thresholds.bar_for(category)
            )
        )


class Thresholds(AdkModel):
    """The severity at which each category is refused, for one tenant.

    Args:
        default: The bar for a category not named. `HIGH` rather than `LOW`, so a tenant
            that configures nothing gets a working agent and tightens deliberately.
        per_category: The bar where this tenant differs — an internal triage agent raising
            `harassment` to a level it will never reach, because reading harassment is the
            work it exists for.

    Example:
        >>> Thresholds().bar_for(ContentCategory.HATE)
        <ContentSeverity.HIGH: 3>
    """

    default: ContentSeverity = ContentSeverity.HIGH
    per_category: Mapping[ContentCategory, ContentSeverity] = Field(default_factory=dict)

    def bar_for(self, category: ContentCategory) -> ContentSeverity:
        """The severity at which this tenant refuses this category."""
        return self.per_category.get(category, self.default)


@runtime_checkable
class ContentClassifier(Protocol):
    """Anything that can say what a passage is about.

    A heuristic, a provider-native classifier and a self-hosted model are interchangeable
    here, so a deployment can change how classification is done without changing what a
    refusal means.
    """

    name: str

    async def classify(self, text: str) -> Classification:
        """What the passage is about, per category it has an opinion on."""
        ...


class HeuristicClassifier:
    """A term-list classifier, so a kit consumer has something before they have a model.

    It is deliberately shallow: it exists so the policy path — thresholds, categories,
    refusals — can be wired and tested without a network call, not so a product can ship
    without a real classifier. Its severity is `MEDIUM`, never `HIGH`, because a term list
    cannot tell a slur from a quotation of one.

    Args:
        terms: A term list per category, matched case-insensitively on word boundaries.

    Example:
        >>> HeuristicClassifier().name
        'heuristic'
    """

    name = "heuristic"

    def __init__(self, terms: Mapping[ContentCategory, tuple[str, ...]] | None = None) -> None:
        self.terms = dict(terms or _DEFAULT_TERMS)

    async def classify(self, text: str) -> Classification:
        """Every category with a term in the passage, at `MEDIUM`."""
        folded = text.casefold()
        return Classification(
            severities={
                category: ContentSeverity.MEDIUM
                for category, terms in self.terms.items()
                if any(term in folded for term in terms)
            },
            classifier=self.name,
        )


_DEFAULT_TERMS: Mapping[ContentCategory, tuple[str, ...]] = {
    ContentCategory.SELF_HARM: ("kill myself", "end my life"),
    ContentCategory.VIOLENCE: ("kill him", "beat them up"),
    ContentCategory.WEAPONS: ("build a bomb", "pipe bomb"),
    ContentCategory.ILLEGAL: ("launder the money", "steal the card"),
}
