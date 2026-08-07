"""What a call cost, in money that adds up.

Two things are kept apart on purpose. `Usage` is what was consumed, which the vendor
counts; `Cost` is what that came to, which a price list decides. Folding them into one
float loses both — the cache saving disappears into a total, and a hundred thousand
fractions of a cent drift away from the invoice.

A cost also carries how much to trust it. A counted usage at a known price is a fact; an
estimated usage at a known price is arithmetic on a guess; a model nobody has priced is
neither, and reporting it as zero would say the call was free.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import field_validator

from tesserix_adk.core.models import AdkModel

__all__ = ["Cost", "CostConfidence", "CountSource", "weaker_source"]

_ISO_4217 = re.compile(r"^[A-Z]{3}$")


class CountSource(StrEnum):
    """Who counted the tokens, which is the difference between a figure and a guess.

    Args:
        PROVIDER: The vendor reported them. Treated as the fact.
        TOKENISER: The kit ran the model's own tokeniser over the request.
        HEURISTIC: The kit divided characters by a constant, because nothing better was
            available. Honest, and not much more than that.
    """

    PROVIDER = "provider"
    TOKENISER = "tokeniser"
    HEURISTIC = "heuristic"

    @property
    def is_estimate(self) -> bool:
        """Whether these counts were worked out rather than reported."""
        return self is not CountSource.PROVIDER


# Weakest wins when two are totalled: a total is only as trustworthy as its worst part.
_WEAKNESS = {CountSource.PROVIDER: 0, CountSource.TOKENISER: 1, CountSource.HEURISTIC: 2}


def weaker_source(one: CountSource, other: CountSource) -> CountSource:
    """The less trustworthy of two counting sources.

    Example:
        >>> weaker_source(CountSource.PROVIDER, CountSource.HEURISTIC)
        <CountSource.HEURISTIC: 'heuristic'>
    """
    return one if _WEAKNESS[one] >= _WEAKNESS[other] else other


class CostConfidence(StrEnum):
    """How much of the number is known.

    Args:
        COUNTED: Vendor-reported usage at a price the list knows.
        ESTIMATED: A known price applied to counts the kit worked out.
        UNKNOWN: No price for this model on this day. The components are zero because
            there is nothing to put in them, not because the call was free.
    """

    COUNTED = "counted"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


_UNCERTAINTY = {CostConfidence.COUNTED: 0, CostConfidence.ESTIMATED: 1, CostConfidence.UNKNOWN: 2}


class Cost(AdkModel):
    """What one call, or one whole run, came to.

    Components are separate so a cache saving is visible rather than folded into a total
    nobody can question. Every one is a `Decimal`: model pricing is quoted in millionths
    of a dollar per token, and a run makes enough calls for binary floating point to
    disagree with the invoice.

    Args:
        input: Fresh prompt tokens, at the uncached rate.
        output: Generated tokens.
        cache_read: Prompt tokens the vendor served from its own cache.
        cache_write: Prompt tokens the vendor charged to put into its cache.
        reasoning: Hidden reasoning tokens, where the vendor prices them apart.
        image: Image and audio units, priced per unit rather than per token.
        currency: ISO 4217 code. Money without one is a number.
        confidence: How much of this is known.
    """

    input: Decimal = Decimal(0)
    output: Decimal = Decimal(0)
    cache_read: Decimal = Decimal(0)
    cache_write: Decimal = Decimal(0)
    reasoning: Decimal = Decimal(0)
    image: Decimal = Decimal(0)
    currency: str = "USD"
    confidence: CostConfidence = CostConfidence.COUNTED

    @field_validator("currency")
    @classmethod
    def _is_a_currency(cls, value: str) -> str:
        if not _ISO_4217.match(value):
            raise ValueError(f"{value!r} is not an ISO 4217 currency code")
        return value

    @classmethod
    def nothing(cls, currency: str = "USD") -> Self:
        """Nothing spent yet — the identity a run starts from and totals against."""
        return cls(currency=currency)

    @classmethod
    def unknown(cls, currency: str = "USD") -> Self:
        """A call at a price nobody knows, which is not a call that was free."""
        return cls(currency=currency, confidence=CostConfidence.UNKNOWN)

    @property
    def total(self) -> Decimal:
        """Every component added up.

        Example:
            >>> Cost(input=Decimal("1.5"), output=Decimal("2.5")).total
            Decimal('4.0')
        """
        return (
            self.input
            + self.output
            + self.cache_read
            + self.cache_write
            + self.reasoning
            + self.image
        )

    @property
    def is_nothing(self) -> bool:
        """Whether this is the empty cost, as opposed to a priced call that came to zero."""
        return self.total == 0 and self.confidence is CostConfidence.COUNTED

    def __add__(self, other: Cost) -> Cost:
        """Total two costs, keeping the weaker confidence of the two.

        Raises:
            ValueError: If both are non-empty and in different currencies.

        Example:
            >>> (Cost(input=Decimal("1")) + Cost.unknown()).confidence
            <CostConfidence.UNKNOWN: 'unknown'>
        """
        if self.is_nothing:
            return other
        if other.is_nothing:
            return self
        if self.currency != other.currency:
            raise ValueError(
                f"cannot total {self.currency} and {other.currency}: the result would be "
                f"a number that is true in neither currency"
            )
        weakest = max(self.confidence, other.confidence, key=lambda one: _UNCERTAINTY[one])
        return Cost(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            reasoning=self.reasoning + other.reasoning,
            image=self.image + other.image,
            currency=self.currency,
            confidence=weakest,
        )

    def quantised(self, places: int = 6) -> Cost:
        """Round every component for presentation, which is the only place to round.

        Args:
            places: Decimal places to keep. Six is a millionth of a dollar, the unit
                vendors quote in.
        """
        step = Decimal(1).scaleb(-places)
        return Cost(
            input=self.input.quantize(step),
            output=self.output.quantize(step),
            cache_read=self.cache_read.quantize(step),
            cache_write=self.cache_write.quantize(step),
            reasoning=self.reasoning.quantize(step),
            image=self.image.quantize(step),
            currency=self.currency,
            confidence=self.confidence,
        )
