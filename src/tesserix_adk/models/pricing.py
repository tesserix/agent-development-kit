"""What each vendor charges, with a date on it.

A price list is not a constant. Vendors change rates without announcing them, publish a
cheaper tier for long prompts and a cheaper one again for batch, and negotiate rates that
appear in no public table. So a card carries the day it took effect, and a deployment can
lay its own cards over the shipped ones without editing the kit.

Nothing here is discovered by convention. A deployment billing against a file nobody named
is one where the answer to "what is this costing" lives on somebody's laptop.
"""

from __future__ import annotations

import os
import tomllib
import warnings
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import Field, ValidationError, model_validator

from tesserix_adk.core.cost import Cost, CostConfidence
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.models.catalogue import CATALOGUE_VERSION, known_models

if TYPE_CHECKING:
    from tesserix_adk.core.capabilities import ModelRef
    from tesserix_adk.core.primitives import Usage
    from tesserix_adk.runtime.estimate import Pricer

__all__ = [
    "PRICE_LIST_ENV",
    "PriceCard",
    "PriceList",
    "Rate",
    "UnknownPricing",
    "cost_of",
    "kit_prices",
    "price_list",
    "pricing_at",
]

PRICE_LIST_ENV = "ADK_PRICE_LIST"

_PER_MILLION = Decimal(1_000_000)

# The shipped rates are what the snapshot saw, so they speak for that day onwards and not
# for history. A deployment costing older runs supplies its own dated cards.
_SNAPSHOT_DAY = date.fromisoformat(CATALOGUE_VERSION)


class UnknownPricing(UserWarning):
    """Raised as a warning when a model has no price on the day it was called.

    A warning rather than an error: a run that cannot be costed is still a run worth
    finishing, and the cost it reports says `UNKNOWN` rather than zero. Deployments that
    require every call to be priced turn this into an error with a warning filter.
    """


class Rate(AdkModel):
    """What one model costs, per million tokens, in one currency.

    Args:
        input_per_mtok: Fresh prompt tokens.
        output_per_mtok: Generated tokens.
        cache_read_per_mtok: Prompt tokens served from the vendor's cache. `None` where
            the vendor publishes no discount, which means billed at the input rate.
        cache_write_per_mtok: Prompt tokens charged to put into that cache. `None` where
            the vendor does not charge for the write.
        reasoning_per_mtok: Hidden reasoning tokens. `None` where the vendor bills them
            as output, which is what most do.
        image_per_unit: One image, tile or audio second. Priced per unit, not per token.
        currency: ISO 4217 code the rates are quoted in.
    """

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal | None = None
    cache_write_per_mtok: Decimal | None = None
    reasoning_per_mtok: Decimal | None = None
    image_per_unit: Decimal = Decimal(0)
    currency: str = "USD"


class PriceCard(AdkModel):
    """One rate, for one model, from one day, under one request shape.

    Args:
        ref: `provider:model`. The provider is part of the identity because an
            OpenAI-compatible proxy serves the same ids at its own prices.
        effective_from: The first day this rate applies. A price change is a new card,
            never an edit — overwriting one rewrites what last week's runs cost.
        rate: What it charges.
        above_input_tokens: The long-context threshold this card is for. A prompt at or
            below it takes the standard card.
        batch: Whether this card is the vendor's asynchronous batch tier.
    """

    ref: str = Field(min_length=1)
    effective_from: date
    rate: Rate
    above_input_tokens: int = Field(default=0, ge=0)
    batch: bool = False

    @property
    def scope(self) -> tuple[str, int, bool]:
        """What request shape this card answers for, which is what may not be duplicated."""
        return (self.ref, self.above_input_tokens, self.batch)


class PriceList(AdkModel):
    """Every card a deployment bills against.

    Args:
        cards: In any order. Selection is by date and request shape, never by position.
    """

    cards: tuple[PriceCard, ...] = ()

    @model_validator(mode="after")
    def _one_price_per_question(self) -> PriceList:
        seen: set[tuple[str, int, bool, date]] = set()
        for card in self.cards:
            key = (*card.scope, card.effective_from)
            if key in seen:
                raise ConfigurationError(
                    f"the price list gives two prices for {card.ref} effective "
                    f"{card.effective_from}: a deployment cannot bill both"
                )
            seen.add(key)
        return self

    def rate_for(
        self, ref: str | ModelRef, *, at: date, input_tokens: int = 0, batch: bool = False
    ) -> Rate | None:
        """The rate in force for this model on this day, or nothing.

        The narrowest card wins: the batch tier over the standard one where a batch call
        asked for it, then the highest long-context threshold the prompt clears. Among
        cards of one shape, the latest one already in force.

        Args:
            ref: `provider:model`, or a parsed `ModelRef`.
            at: The day of the call, not today. A cost recomputed at today's price is a
                different number from the one that was charged.
            input_tokens: How long the prompt was, for tiered pricing.
            batch: Whether the call went to the vendor's batch endpoint.
        """
        wanted = str(ref)
        live = [
            card
            for card in self.cards
            if card.ref == wanted
            and card.effective_from <= at
            and card.above_input_tokens <= input_tokens
            and (card.batch is batch or not card.batch)
        ]
        if not live:
            return None
        best = max(live, key=lambda card: (card.batch is batch, card.above_input_tokens))
        newest = max(
            (card for card in live if card.scope == best.scope),
            key=lambda card: card.effective_from,
        )
        return newest.rate

    def overridden_by(self, other: PriceList) -> Self:
        """This list with `other`'s cards laid over it, for negotiated rates.

        An override replaces every shipped card for the models it names, dates included.
        A negotiated rate that only won until the vendor's next list price landed would be
        no agreement at all. Models it says nothing about keep the shipped price.
        """
        named = {card.ref for card in other.cards}
        kept = tuple(card for card in self.cards if card.ref not in named)
        return type(self)(cards=(*kept, *other.cards))


def kit_prices() -> PriceList:
    """The prices the kit ships, taken from the model catalogue snapshot."""
    return PriceList(
        cards=tuple(
            PriceCard(
                ref=str(card.ref),
                effective_from=_SNAPSHOT_DAY,
                rate=Rate(
                    input_per_mtok=Decimal(str(card.pricing.input_usd_per_mtok)),
                    output_per_mtok=Decimal(str(card.pricing.output_usd_per_mtok)),
                    cache_read_per_mtok=(
                        None
                        if card.pricing.cached_input_usd_per_mtok is None
                        else Decimal(str(card.pricing.cached_input_usd_per_mtok))
                    ),
                ),
            )
            for card in known_models()
            if card.pricing is not None
        )
    )


def price_list(path: str | Path | None = None) -> PriceList:
    """The kit's prices with a deployment's own laid over them.

    Args:
        path: A TOML price list. Falls back to `ADK_PRICE_LIST`, and to the shipped
            prices alone where neither names a file.

    Raises:
        ConfigurationError: If the named file is missing, is not TOML, or is not a price
            list. A deployment that silently bills at list price because its negotiated
            rates failed to parse finds out on the invoice.
    """
    named = path if path is not None else os.environ.get(PRICE_LIST_ENV)
    if named is None:
        return kit_prices()
    return kit_prices().overridden_by(_read(Path(named)))


def _read(path: Path) -> PriceList:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as missing:
        raise ConfigurationError(f"there is no price list at {path}") from missing
    except tomllib.TOMLDecodeError as malformed:
        raise ConfigurationError(f"{path} is not readable TOML: {malformed}") from malformed
    if "cards" not in raw:
        raise ConfigurationError(f"{path} is not a price list: it declares no cards")
    try:
        return PriceList.model_validate({"cards": raw["cards"]}, strict=False)
    except ValidationError as wrong:
        raise ConfigurationError(f"{path} is not a price list: {wrong}") from wrong


def cost_of(
    usage: Usage,
    ref: str | ModelRef,
    *,
    at: date,
    prices: PriceList | None = None,
    batch: bool = False,
) -> Cost:
    """What `usage` came to on `ref`, at the price in force on `at`.

    Args:
        usage: What the call consumed.
        ref: Which model consumed it.
        at: The day of the call.
        prices: The list to bill against. The shipped prices where none is given.
        batch: Whether the call went to the vendor's batch endpoint.

    Returns:
        A `Cost` whose confidence says how much of it is known. A model nobody has priced
        gives zero components and `UNKNOWN`, never a silent free call.

    Warns:
        UnknownPricing: When no card covers this model on this day.
    """
    against = prices if prices is not None else price_list()
    rate = against.rate_for(ref, at=at, input_tokens=usage.input_tokens, batch=batch)
    if rate is None:
        warnings.warn(
            f"no price for {ref} on {at}: this call is recorded as unknown rather than free",
            UnknownPricing,
            stacklevel=2,
        )
        return Cost.unknown()
    cache_read_rate = (
        rate.input_per_mtok if rate.cache_read_per_mtok is None else rate.cache_read_per_mtok
    )
    reasoning_rate = (
        rate.output_per_mtok if rate.reasoning_per_mtok is None else rate.reasoning_per_mtok
    )
    return Cost(
        input=usage.fresh_input_tokens * rate.input_per_mtok / _PER_MILLION,
        output=usage.output_tokens * rate.output_per_mtok / _PER_MILLION,
        cache_read=usage.cached_tokens * cache_read_rate / _PER_MILLION,
        cache_write=usage.cache_write_tokens * (rate.cache_write_per_mtok or 0) / _PER_MILLION,
        reasoning=usage.reasoning_tokens * reasoning_rate / _PER_MILLION,
        image=usage.image_units * rate.image_per_unit,
        currency=rate.currency,
        confidence=CostConfidence.ESTIMATED if usage.estimated else CostConfidence.COUNTED,
    )


def pricing_at(at: date, *, prices: PriceList | None = None, batch: bool = False) -> Pricer:
    """The shipped price list, as the `Pricing` an estimate is built against.

    Estimation holds no opinion about where prices live, so it takes a callable rather than
    a list. This is that callable over `cost_of`, priced as of `at`.
    """

    def priced(usage: Usage, model: str) -> Cost:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnknownPricing)
            return cost_of(usage, model, at=at, prices=prices, batch=batch)

    return priced
