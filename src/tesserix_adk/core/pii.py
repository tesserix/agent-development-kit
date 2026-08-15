"""Finding personal identifiers, and the placeholder that stands in for one.

A passport number in a prompt is a passport number in the provider's logs, in the
checkpoint, in the memory record and in the span attribute. Products each hand-roll a regex
and none of them agree on what a redacted value looks like, so the same traveller is
`[REDACTED]` in one store and `xxxx1234` in another and nothing can be joined.

The detectors live in `core` because three separate paths need the same ones: the guardrail
chain before prompt assembly, the memory write path, and the telemetry exporter. Redaction
that only happens at the agent boundary has already been written by whatever else emitted.

A placeholder keeps the type and a stable per-tenant pseudonym, so an agent can still reason
about "the same traveller" across turns without ever holding the value. The pseudonym is
per-tenant by construction: one salted digest shared across tenants would let one tenant
confirm the presence of another's subject by guessing it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.audit import pseudonym
from tesserix_adk.core.errors import GuardrailEvaluationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_DETECTORS",
    "PIIDetector",
    "PIIKind",
    "PIIMatch",
    "PatternDetector",
    "Redacted",
    "placeholder",
    "redact",
]

_HIGH = 1.0
_MODERATE = 0.6


class PIIKind(StrEnum):
    """What kind of identifier was found. The placeholder keeps it, so reasoning survives."""

    CARD = "card"
    PASSPORT = "passport"
    NATIONAL_ID = "national_id"
    EMAIL = "email"
    PHONE = "phone"
    IBAN = "iban"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"  # noqa: S105 — a category name, not a credential


class PIIMatch(AdkModel):
    """One identifier, located but not carried.

    Args:
        kind: What it is.
        start: Where it begins in the text scanned.
        end: Where it ends, exclusive.
        confidence: From 0 to 1. A flight number matching an ID shape is a real cost to
            the agent's reasoning, so a low-confidence match is reported and left for a
            threshold to decide rather than redacted on sight.

    Example:
        >>> PIIMatch(kind=PIIKind.EMAIL, start=0, end=5).confidence
        1.0
    """

    kind: PIIKind
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence: float = Field(default=_HIGH, ge=0.0, le=1.0)

    @property
    def length(self) -> int:
        """How much text it covers. Recorded in place of the text itself."""
        return self.end - self.start


@runtime_checkable
class PIIDetector(Protocol):
    """Anything that can find identifiers in a passage.

    A consumer's own detector — a named-entity model, a checksum for a national scheme —
    plugs in beside the built-ins rather than replacing them.
    """

    name: str

    def detect(self, text: str) -> tuple[PIIMatch, ...]:
        """Every identifier in `text`, located. Never the values."""
        ...


class PatternDetector:
    """A detector built from one shape, optionally checked before it is believed.

    Args:
        name: What it is, for the failure message when it raises.
        kind: What it finds.
        pattern: The shape.
        confidence: How sure a match is, before any per-tenant allow pattern applies.
        checksum: An extra test the matched text must pass. A sixteen-digit run is a card
            number only if it satisfies Luhn; without that, every order reference is one.

    Example:
        >>> CARD.detect("pay with 4111 1111 1111 1111")[0].kind
        <PIIKind.CARD: 'card'>
    """

    def __init__(
        self,
        name: str,
        *,
        kind: PIIKind,
        pattern: str,
        confidence: float = _HIGH,
        checksum: str = "",
    ) -> None:
        self.name = name
        self.kind = kind
        self.confidence = confidence
        self.checksum = checksum
        self._pattern = re.compile(pattern)

    def __repr__(self) -> str:
        """Its name, so a default argument in the API snapshot is not a memory address."""
        return f"PatternDetector({self.name!r})"

    def detect(self, text: str) -> tuple[PIIMatch, ...]:
        """Every match of this shape that also passes its checksum."""
        return tuple(
            PIIMatch(
                kind=self.kind,
                start=found.start(),
                end=found.end(),
                confidence=self.confidence,
            )
            for found in self._pattern.finditer(text)
            if self._passes(found.group(0))
        )

    def _passes(self, value: str) -> bool:
        return not self.checksum or _luhn(value)


def _luhn(value: str) -> bool:
    """Whether a digit run is a plausible card number rather than an order reference."""
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        doubled = digit * 2 if position % 2 else digit
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


CARD = PatternDetector(
    "card",
    kind=PIIKind.CARD,
    pattern=r"\b(?:\d[ -]?){12,18}\d\b",
    checksum="luhn",
)
EMAIL = PatternDetector("email", kind=PIIKind.EMAIL, pattern=r"[\w.+-]+@[\w-]+\.[\w.-]+")
IBAN = PatternDetector("iban", kind=PIIKind.IBAN, pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
BEARER = PatternDetector(
    "bearer_token", kind=PIIKind.BEARER_TOKEN, pattern=r"\bBearer\s+[A-Za-z0-9._\-]+"
)
API_KEY = PatternDetector(
    "api_key", kind=PIIKind.API_KEY, pattern=r"\b(?:sk|pk|rk|api)[-_][A-Za-z0-9\-_]{8,}"
)
PHONE = PatternDetector(
    "phone",
    kind=PIIKind.PHONE,
    pattern=r"\+\d[\d\s().-]{7,}\d",
    confidence=_MODERATE,
)
PASSPORT = PatternDetector(
    "passport",
    kind=PIIKind.PASSPORT,
    pattern=r"\b[A-Z]{1,2}\d{6,9}\b",
    confidence=_MODERATE,
)
NATIONAL_ID = PatternDetector(
    "national_id",
    kind=PIIKind.NATIONAL_ID,
    pattern=r"\b\d{3}-\d{2}-\d{4}\b",
)

DEFAULT_DETECTORS: tuple[PIIDetector, ...] = (
    CARD,
    EMAIL,
    IBAN,
    BEARER,
    API_KEY,
    NATIONAL_ID,
    PASSPORT,
    PHONE,
)
"""The built-ins, most specific first, so a card is a card rather than a phone number."""


def placeholder(kind: PIIKind, value: str, *, tenant: str) -> str:
    """The stand-in for one value: its type, and a pseudonym stable within the tenant.

    Args:
        kind: What was removed, kept so the agent knows what it is reasoning about.
        value: The identifier. Digested, never stored.
        tenant: Whose data it is. Part of the digest, so the same value under two tenants
            gives two pseudonyms and neither can be used to probe the other.

    Example:
        >>> placeholder(PIIKind.EMAIL, "ada@example.test", tenant="acme").startswith("[email:")
        True
    """
    return f"[{kind.value}:{pseudonym(value, salt=tenant)}]"


class Redacted(AdkModel):
    """What a passage became, and what was taken out of it.

    Args:
        text: The passage with each identifier replaced by its placeholder.
        kinds: Which kinds were found, sorted. The values are gone by construction.
        count: How many were replaced.

    Example:
        >>> redact("mail ada@example.test", tenant="acme").kinds
        (<PIIKind.EMAIL: 'email'>,)
    """

    text: str
    kinds: tuple[PIIKind, ...] = ()
    count: int = Field(default=0, ge=0)

    @property
    def found(self) -> bool:
        """Whether anything was taken out."""
        return self.count > 0


def redact(
    text: str,
    *,
    tenant: str,
    detectors: Sequence[PIIDetector] = DEFAULT_DETECTORS,
    threshold: float = _MODERATE,
    allow: Iterable[str] = (),
) -> Redacted:
    """Replace every identifier in `text` with a placeholder that stands in for it.

    Args:
        text: What is about to reach a prompt, a memory record or a span.
        tenant: Whose data it is, which decides the pseudonym.
        detectors: What to look for. A consumer's own detectors go here beside the built-ins.
        threshold: The confidence a match needs. A flight number that matches a passport
            shape is a real cost to the agent's reasoning, so the bar is a setting.
        allow: Shapes this tenant has said are not identifiers — a booking reference, an
            internal case number — matched against the text a detector claimed.

    Returns:
        The rewritten passage and what was taken out of it.

    Raises:
        GuardrailEvaluationError: If a detector raises. Content is never passed through
            unredacted on the grounds that the check could not complete.

    Example:
        >>> redact("card 4111 1111 1111 1111", tenant="acme").count
        1
    """
    permitted = tuple(re.compile(shape) for shape in allow)
    found = _located(text, detectors, threshold, permitted)
    rewritten: list[str] = []
    cursor = 0
    for match in found:
        rewritten.append(text[cursor : match.start])
        rewritten.append(placeholder(match.kind, text[match.start : match.end], tenant=tenant))
        cursor = match.end
    rewritten.append(text[cursor:])
    return Redacted(
        text="".join(rewritten),
        kinds=tuple(sorted({match.kind for match in found})),
        count=len(found),
    )


def _located(
    text: str,
    detectors: Sequence[PIIDetector],
    threshold: float,
    permitted: tuple[re.Pattern[str], ...],
) -> tuple[PIIMatch, ...]:
    """Every match worth replacing, in reading order and never overlapping.

    Earlier detectors win an overlap, which is why `DEFAULT_DETECTORS` puts the checksummed
    ones first: a card that is also a phone number is a card.
    """
    kept: list[PIIMatch] = []
    for detector in detectors:
        for match in _asked(detector, text):
            if match.confidence < threshold or _allowed(text[match.start : match.end], permitted):
                continue
            if not any(match.start < seen.end and seen.start < match.end for seen in kept):
                kept.append(match)
    return tuple(sorted(kept, key=lambda match: match.start))


def _asked(detector: PIIDetector, text: str) -> tuple[PIIMatch, ...]:
    """What one detector found, or a refusal naming it where it could not answer."""
    try:
        return detector.detect(text)
    except Exception as broke:
        raise GuardrailEvaluationError(
            f"detector {detector.name!r} could not evaluate the content",
            guard="pii",
        ) from broke


def _allowed(value: str, permitted: tuple[re.Pattern[str], ...]) -> bool:
    """Whether this tenant has said this shape is not an identifier after all."""
    return any(pattern.fullmatch(value) for pattern in permitted)
