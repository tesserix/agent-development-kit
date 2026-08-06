"""What each vendor model can do and what it costs, as data with a date on it.

The rows below are a snapshot taken on `CATALOGUE_VERSION`. Vendors add models weekly and
change prices without announcing it, so a refresh is a data change and never a surface
change: the names in this module stay put and only the rows move. Nothing in the kit fails
because a model is missing from it — a provider constructed with an unknown model uses
whatever capabilities it was given.

Known limitations: prices are the standard per-token rates only. Tiered long-context
pricing, batch discounts and negotiated rates are not modelled, so a `cost` computed here
is an upper bound for a batch job and wrong for an enterprise agreement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.capabilities import ModelCapabilities, ModelRef
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from tesserix_adk.core.primitives import Usage

__all__ = [
    "CATALOGUE_VERSION",
    "ModelCard",
    "Pricing",
    "known_models",
    "model_card",
    "priced",
]

CATALOGUE_VERSION = "2026-08-07"

_PER_MILLION = 1_000_000


class Pricing(AdkModel):
    """What a vendor charges, in US dollars per million tokens.

    Args:
        input_usd_per_mtok: Prompt tokens, at the uncached rate.
        output_usd_per_mtok: Generated tokens.
        cached_input_usd_per_mtok: Prompt tokens the vendor served from its own cache.
            `None` where the vendor publishes no discount, which is billed in full.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None = None


class ModelCard(AdkModel):
    """One model, as the snapshot recorded it.

    Args:
        ref: Provider and model id. The provider is part of the identity because an
            OpenAI-compatible proxy serves the same ids and is not the same model.
        capabilities: What that model does, which is what the kit checks before calling.
        pricing: `None` where this snapshot does not record a price. Not free: a zero
            written in for a missing price is a false statement about money.
        aliases: Other ids that serve this model, typically the vendor's dated snapshots.
    """

    ref: ModelRef
    capabilities: ModelCapabilities = ModelCapabilities()
    pricing: Pricing | None = None
    aliases: tuple[str, ...] = ()


def _card(
    provider: str,
    model: str,
    *,
    window: int,
    max_output: int,
    vision: bool = True,
    parallel: bool = True,
    pricing: Pricing | None = None,
    aliases: tuple[str, ...] = (),
) -> ModelCard:
    return ModelCard(
        ref=ModelRef(provider=provider, model=model),
        capabilities=ModelCapabilities(
            structured_output=True,
            tool_calling=True,
            parallel_tool_calls=parallel,
            vision=vision,
            streaming=True,
            context_window_tokens=window,
            max_output_tokens=max_output,
        ),
        pricing=pricing,
        aliases=aliases,
    )


_CARDS: tuple[ModelCard, ...] = (
    _card(
        "anthropic",
        "claude-opus-4-1",
        window=200_000,
        max_output=32_000,
        pricing=Pricing(
            input_usd_per_mtok=15.0, output_usd_per_mtok=75.0, cached_input_usd_per_mtok=1.5
        ),
        aliases=("claude-opus-4-1-20250805",),
    ),
    _card(
        "anthropic",
        "claude-sonnet-4-5",
        window=200_000,
        max_output=64_000,
        pricing=Pricing(
            input_usd_per_mtok=3.0, output_usd_per_mtok=15.0, cached_input_usd_per_mtok=0.3
        ),
        aliases=("claude-sonnet-4-5-20250929",),
    ),
    _card(
        "anthropic",
        "claude-haiku-4-5",
        window=200_000,
        max_output=64_000,
        pricing=Pricing(
            input_usd_per_mtok=1.0, output_usd_per_mtok=5.0, cached_input_usd_per_mtok=0.1
        ),
        aliases=("claude-haiku-4-5-20251001",),
    ),
    _card("anthropic", "claude-opus-5", window=200_000, max_output=64_000),
    _card("anthropic", "claude-sonnet-5", window=200_000, max_output=64_000),
    _card(
        "openai",
        "gpt-4o",
        window=128_000,
        max_output=16_384,
        pricing=Pricing(
            input_usd_per_mtok=2.5, output_usd_per_mtok=10.0, cached_input_usd_per_mtok=1.25
        ),
        aliases=("gpt-4o-2024-11-20",),
    ),
    _card(
        "openai",
        "gpt-4o-mini",
        window=128_000,
        max_output=16_384,
        pricing=Pricing(
            input_usd_per_mtok=0.15, output_usd_per_mtok=0.6, cached_input_usd_per_mtok=0.075
        ),
    ),
    _card(
        "openai",
        "gpt-4.1",
        window=1_000_000,
        max_output=32_768,
        pricing=Pricing(
            input_usd_per_mtok=2.0, output_usd_per_mtok=8.0, cached_input_usd_per_mtok=0.5
        ),
    ),
    _card(
        "openai",
        "gpt-4.1-mini",
        window=1_000_000,
        max_output=32_768,
        pricing=Pricing(
            input_usd_per_mtok=0.4, output_usd_per_mtok=1.6, cached_input_usd_per_mtok=0.1
        ),
    ),
    _card(
        "gemini",
        "gemini-2.5-pro",
        window=1_048_576,
        max_output=65_536,
        pricing=Pricing(input_usd_per_mtok=1.25, output_usd_per_mtok=10.0),
    ),
    _card(
        "gemini",
        "gemini-2.5-flash",
        window=1_048_576,
        max_output=65_536,
        pricing=Pricing(input_usd_per_mtok=0.3, output_usd_per_mtok=2.5),
    ),
    _card(
        "gemini",
        "gemini-2.0-flash",
        window=1_048_576,
        max_output=8_192,
        pricing=Pricing(input_usd_per_mtok=0.1, output_usd_per_mtok=0.4),
    ),
)

_BY_REFERENCE: dict[str, ModelCard] = {
    f"{card.ref.provider}:{name}": card
    for card in _CARDS
    for name in (card.ref.model, *card.aliases)
}


def known_models(provider: str | None = None) -> tuple[ModelCard, ...]:
    """Return every card in the snapshot, or every card for one provider.

    A provider the snapshot does not cover returns nothing rather than failing: this is a
    catalogue, and a question about a vendor nobody recorded has an empty answer.
    """
    if provider is None:
        return _CARDS
    return tuple(card for card in _CARDS if card.ref.provider == provider)


def model_card(ref: str | ModelRef) -> ModelCard:
    """Return the card for `ref`, by model id or by one of the vendor's dated aliases.

    Args:
        ref: `provider:model`, or a parsed `ModelRef`.

    Raises:
        ValueError: If a string reference names no provider. Defaulting one is how a
            proxy's traffic ends up recorded against the vendor.
        ConfigurationError: If this snapshot does not know the model. The version is in
            the message, because the usual cause is a model newer than the snapshot.
    """
    parsed = ModelRef.parse(ref) if isinstance(ref, str) else ref
    found = _BY_REFERENCE.get(f"{parsed.provider}:{parsed.model}")
    if found is None:
        raise ConfigurationError(
            f"{parsed} is not in the model catalogue (snapshot {CATALOGUE_VERSION}). "
            f"Construct the provider with explicit capabilities where the model is newer "
            f"than the snapshot."
        )
    return found


def priced(usage: Usage, card: ModelCard) -> Usage:
    """Return `usage` with its cost filled in from `card`, where the card records one.

    A card with no price leaves the cost unset. Recording zero would say the call was
    free, and a self-hosted or newly released model costs something either way.
    """
    if card.pricing is None:
        return usage
    return usage.model_copy(
        update={"cost": _cost(usage, card.pricing), "currency": "USD"},
    )


def _cost(usage: Usage, pricing: Pricing) -> float:
    cached_rate = (
        pricing.input_usd_per_mtok
        if pricing.cached_input_usd_per_mtok is None
        else pricing.cached_input_usd_per_mtok
    )
    fresh = max(usage.input_tokens - usage.cached_tokens, 0)
    return (
        fresh * pricing.input_usd_per_mtok
        + usage.cached_tokens * cached_rate
        + usage.output_tokens * pricing.output_usd_per_mtok
    ) / _PER_MILLION
