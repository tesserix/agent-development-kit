"""What each vendor model can do and what it costs, as data with a date on it.

The catalogue is a snapshot, not a promise. Vendors add models weekly and change prices
without warning, so refreshing the data is a data change and never a surface change: the
names here stay put, the rows behind them move.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from tesserix_adk.core import Capability, ConfigurationError, ModelRef, Usage
from tesserix_adk.models import ModelCard, Pricing
from tesserix_adk.models.catalogue import (
    CATALOGUE_VERSION,
    known_models,
    model_card,
    priced,
)


class TestTheCatalogueIsDatedAndSelfConsistent:
    def test_the_version_is_the_date_the_snapshot_was_taken(self) -> None:
        assert date.fromisoformat(CATALOGUE_VERSION) <= date(2100, 1, 1)

    def test_every_vendor_the_kit_ships_a_provider_for_has_models(self) -> None:
        providers = {card.ref.provider for card in known_models()}
        assert {"anthropic", "openai", "gemini"} <= providers

    def test_models_can_be_listed_for_one_vendor(self) -> None:
        assert all(card.ref.provider == "openai" for card in known_models("openai"))
        assert known_models("openai")

    def test_a_vendor_nobody_ships_lists_nothing_rather_than_failing(self) -> None:
        assert known_models("nobody") == ()

    def test_every_card_declares_a_window_because_the_check_needs_one(self) -> None:
        assert all(card.capabilities.context_window_tokens for card in known_models())


class TestAModelIsLookedUpByItsFullReference:
    def test_a_reference_resolves_to_its_card(self) -> None:
        card = model_card("anthropic:claude-sonnet-4-5")
        assert card.ref.model == "claude-sonnet-4-5"
        assert card.capabilities.supports(Capability.TOOL_CALLING)

    def test_a_parsed_reference_resolves_the_same_way(self) -> None:
        assert model_card(ModelRef(provider="openai", model="gpt-4o")).ref.provider == "openai"

    def test_a_dated_snapshot_id_resolves_to_the_model_it_is(self) -> None:
        """Vendors serve `-20250929` ids alongside the alias, and they are one model."""
        assert model_card("anthropic:claude-sonnet-4-5-20250929") == model_card(
            "anthropic:claude-sonnet-4-5"
        )

    def test_a_model_the_snapshot_does_not_know_names_itself_and_the_version(self) -> None:
        with pytest.raises(ConfigurationError) as refused:
            model_card("anthropic:claude-from-next-year")
        assert "claude-from-next-year" in str(refused.value)
        assert CATALOGUE_VERSION in str(refused.value)

    def test_a_reference_without_a_provider_is_refused_by_the_reference_itself(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            model_card("gpt-4o")


class TestPriceIsRecordedWhereItIsKnownAndNotInventedWhereItIsNot:
    def test_a_priced_model_costs_what_it_used(self) -> None:
        card = ModelCard(
            ref=ModelRef(provider="test", model="m"),
            pricing=Pricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        )
        usage = priced(Usage(input_tokens=1_000_000, output_tokens=100_000), card)
        assert usage.cost is not None
        assert usage.cost.total == Decimal("4.5")
        assert usage.cost.currency == "USD"

    def test_cached_input_is_billed_at_the_cached_rate(self) -> None:
        card = ModelCard(
            ref=ModelRef(provider="test", model="m"),
            pricing=Pricing(
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=15.0,
                cached_input_usd_per_mtok=0.3,
            ),
        )
        usage = priced(
            Usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=1_000_000), card
        )
        assert usage.cost is not None
        assert usage.cost.total == Decimal("0.3")

    def test_cached_input_falls_back_to_the_full_rate_where_no_discount_is_recorded(
        self,
    ) -> None:
        card = ModelCard(
            ref=ModelRef(provider="test", model="m"),
            pricing=Pricing(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
        )
        usage = priced(Usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000), card)
        assert usage.cost is not None
        assert usage.cost.total == Decimal("3.0")

    def test_a_model_with_no_recorded_price_stays_unpriced(self) -> None:
        """Zero is a statement about money. A model whose price nobody recorded is not free."""
        card = ModelCard(ref=ModelRef(provider="test", model="m"))
        usage = priced(Usage(input_tokens=10, output_tokens=10), card)
        assert usage.cost is None

    def test_the_rest_of_the_usage_record_survives_pricing(self) -> None:
        card = ModelCard(
            ref=ModelRef(provider="test", model="m"),
            pricing=Pricing(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0),
        )
        usage = priced(Usage(input_tokens=7, output_tokens=3, extras={"reasoning": 2}), card)
        assert (usage.input_tokens, usage.output_tokens, usage.extras) == (7, 3, {"reasoning": 2})
