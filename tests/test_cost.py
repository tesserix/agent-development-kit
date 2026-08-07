"""One honest number for what a run spent, whichever vendor answered it.

Vendors count differently — cache reads, cache writes, reasoning tokens, images per tile,
some endpoints reporting nothing at all. A ledger that flattens those into one integer and
one float cannot say what it does not know, and a budget enforced against a guess presented
as a count is not enforcement.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    ConfigurationError,
    Cost,
    CostConfidence,
    CountSource,
    InvalidRequestError,
    ModelCapabilities,
    RateLimitError,
    RetryConfig,
    RunEventKind,
    RunState,
    Usage,
)
from tesserix_adk.core.primitives import Message, TextPart
from tesserix_adk.core.provider import ModelRequest
from tesserix_adk.models.pricing import (
    PRICE_LIST_ENV,
    PriceCard,
    PriceList,
    Rate,
    UnknownPricing,
    cost_of,
    kit_prices,
    price_list,
)
from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import (
    FakeClock,
    FakeSecrets,
    HttpCassette,
    HttpExchange,
    HttpReplay,
    ScriptedProvider,
)

if TYPE_CHECKING:
    from pathlib import Path

TODAY = date(2026, 8, 7)


def rate(**overrides: object) -> Rate:
    fields: dict[str, object] = {
        "input_per_mtok": Decimal("3.00"),
        "output_per_mtok": Decimal("15.00"),
        "cache_read_per_mtok": Decimal("0.30"),
        "cache_write_per_mtok": Decimal("3.75"),
    }
    return Rate(**{**fields, **overrides})  # type: ignore[arg-type]


def card(ref: str = "anthropic:claude-sonnet-4-5", **overrides: object) -> PriceCard:
    fields: dict[str, object] = {
        "ref": ref,
        "effective_from": date(2026, 1, 1),
        "rate": rate(),
    }
    return PriceCard(**{**fields, **overrides})  # type: ignore[arg-type]


def prices(*cards: PriceCard) -> PriceList:
    return PriceList(cards=cards)


class TestWhatOneCallConsumed:
    """A vendor's counting is its own; the record has to hold all of it or admit it did not."""

    def test_every_counted_thing_has_somewhere_to_go(self) -> None:
        used = Usage(
            input_tokens=1_000,
            output_tokens=200,
            cached_tokens=800,
            cache_write_tokens=150,
            reasoning_tokens=64,
            image_units=3,
        )
        assert used.cache_write_tokens == 150
        assert used.reasoning_tokens == 64
        assert used.image_units == 3

    def test_counts_are_from_the_provider_unless_said_otherwise(self) -> None:
        assert Usage(input_tokens=1, output_tokens=1).source is CountSource.PROVIDER

    def test_a_count_the_kit_worked_out_says_which_tokeniser_did_it(self) -> None:
        """A ledger that cannot tell a count from a guess presents a guess as a count."""
        used = Usage(input_tokens=1, output_tokens=1, source=CountSource.TOKENISER)
        assert used.estimated
        assert used.source is CountSource.TOKENISER

    def test_a_provider_count_is_not_an_estimate(self) -> None:
        assert not Usage(input_tokens=1, output_tokens=1).estimated

    def test_totalling_a_guess_with_a_count_gives_a_guess(self) -> None:
        counted = Usage(input_tokens=1, output_tokens=1)
        guessed = Usage(input_tokens=1, output_tokens=1, source=CountSource.HEURISTIC)
        assert (counted + guessed).source is CountSource.HEURISTIC

    def test_the_weaker_source_wins_whichever_side_it_is_on(self) -> None:
        counted = Usage(input_tokens=1, output_tokens=1)
        guessed = Usage(input_tokens=1, output_tokens=1, source=CountSource.HEURISTIC)
        assert (guessed + counted).source is CountSource.HEURISTIC

    def test_every_component_totals(self) -> None:
        one = Usage(
            input_tokens=1, output_tokens=2, cache_write_tokens=3, reasoning_tokens=4, image_units=5
        )
        both = one + one
        assert (both.cache_write_tokens, both.reasoning_tokens, both.image_units) == (6, 8, 10)


class TestMoneyIsNotAFloat:
    """Fractions of a cent, summed a hundred thousand times, are the whole disagreement
    between a bill and an invoice."""

    def test_the_components_are_decimals(self) -> None:
        money = Cost(input=Decimal("0.003"), output=Decimal("0.003"), currency="USD")
        assert isinstance(money.total, Decimal)

    def test_a_total_is_the_sum_of_its_parts(self) -> None:
        money = Cost(
            input=Decimal("1.00"),
            output=Decimal("2.00"),
            cache_read=Decimal("0.10"),
            cache_write=Decimal("0.20"),
            reasoning=Decimal("0.30"),
            image=Decimal("0.40"),
            currency="USD",
        )
        assert money.total == Decimal("4.00")

    def test_a_cache_saving_is_visible_rather_than_folded_into_one_number(self) -> None:
        used = Usage(input_tokens=100_000, output_tokens=100, cached_tokens=90_000)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.cache_read > 0
        assert money.input < money.cache_read + money.input

    def test_many_small_amounts_do_not_drift(self) -> None:
        """A float would have lost this by the ten-thousandth call; a run makes millions."""
        one = Cost(input=Decimal("0.0000001"), currency="USD")
        total = Cost.nothing("USD")
        for _ in range(100_000):
            total = total + one
        assert total.input == Decimal("0.0100000")

    def test_two_currencies_do_not_total(self) -> None:
        with pytest.raises(ValueError, match="neither currency"):
            _ = Cost(input=Decimal("1"), currency="USD") + Cost(input=Decimal("1"), currency="EUR")

    def test_nothing_spent_yet_totals_with_anything(self) -> None:
        """A run starts empty; refusing to add that would leave every run unable to report."""
        spent = Cost(input=Decimal("1"), currency="EUR")
        assert (Cost.nothing("USD") + spent).currency == "EUR"

    def test_a_currency_that_is_not_a_currency_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ISO 4217"):
            Cost(input=Decimal("1"), currency="dollars")


class TestWhatItCost:
    def test_each_kind_of_token_is_priced_at_its_own_rate(self) -> None:
        used = Usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached_tokens=0,
            cache_write_tokens=1_000_000,
        )
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.input == Decimal("3.00")
        assert money.output == Decimal("15.00")
        assert money.cache_write == Decimal("3.75")

    def test_cached_input_is_not_charged_twice(self) -> None:
        """`input_tokens` includes the cached ones, so the fresh half is what is left."""
        used = Usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=1_000_000)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.input == Decimal("0.00")
        assert money.cache_read == Decimal("0.30")

    def test_reasoning_tokens_are_priced_where_the_vendor_prices_them_apart(self) -> None:
        used = Usage(input_tokens=0, output_tokens=0, reasoning_tokens=1_000_000)
        priced = card(rate=rate(reasoning_per_mtok=Decimal("60.00")))
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(priced))
        assert money.reasoning == Decimal("60.00")

    def test_reasoning_tokens_bill_as_output_where_the_vendor_does_not_split_them(self) -> None:
        used = Usage(input_tokens=0, output_tokens=0, reasoning_tokens=1_000_000)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.reasoning == Decimal("15.00")

    def test_images_are_priced_per_unit_rather_than_per_token(self) -> None:
        used = Usage(input_tokens=0, output_tokens=0, image_units=4)
        priced = card(rate=rate(image_per_unit=Decimal("0.002")))
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(priced))
        assert money.image == Decimal("0.008")

    def test_a_counted_usage_gives_a_counted_cost(self) -> None:
        used = Usage(input_tokens=10, output_tokens=10)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.confidence is CostConfidence.COUNTED

    def test_an_estimated_usage_gives_an_estimated_cost(self) -> None:
        """The arithmetic is exact; what went into it was not."""
        used = Usage(input_tokens=10, output_tokens=10, source=CountSource.TOKENISER)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=prices(card()))
        assert money.confidence is CostConfidence.ESTIMATED


class TestAPriceNobodyKnows:
    def test_an_unpriced_model_is_unknown_rather_than_free(self) -> None:
        used = Usage(input_tokens=1_000, output_tokens=100)
        with pytest.warns(UnknownPricing):
            money = cost_of(used, "vllm:qwen-3", at=TODAY, prices=prices(card()))
        assert money.confidence is CostConfidence.UNKNOWN
        assert money.total == Decimal(0)

    def test_the_warning_names_the_model_and_the_date(self) -> None:
        with pytest.warns(UnknownPricing, match="vllm:qwen-3"):
            cost_of(
                Usage(input_tokens=1, output_tokens=1), "vllm:qwen-3", at=TODAY, prices=prices()
            )

    def test_an_unknown_cost_makes_the_total_unknown(self) -> None:
        """A total that quietly omits a step understates the bill."""
        known = Cost(input=Decimal("1"), currency="USD")
        unknown = Cost.unknown("USD")
        assert (known + unknown).confidence is CostConfidence.UNKNOWN

    def test_a_model_priced_only_after_this_call_is_unknown_for_it(self) -> None:
        later = card(effective_from=date(2026, 9, 1))
        with pytest.warns(UnknownPricing):
            money = cost_of(
                Usage(input_tokens=1, output_tokens=1),
                "anthropic:claude-sonnet-4-5",
                at=TODAY,
                prices=prices(later),
            )
        assert money.confidence is CostConfidence.UNKNOWN


class TestAPriceThatChanged:
    """Overwriting a price rewrites what last week's runs cost."""

    def test_the_card_in_force_on_the_day_is_the_one_that_applies(self) -> None:
        old = card(effective_from=date(2026, 1, 1), rate=rate(input_per_mtok=Decimal("3.00")))
        new = card(effective_from=date(2026, 8, 1), rate=rate(input_per_mtok=Decimal("2.00")))
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        assert cost_of(used, card().ref, at=TODAY, prices=prices(old, new)).input == Decimal("2.00")

    def test_a_call_before_the_change_still_costs_what_it_cost(self) -> None:
        old = card(effective_from=date(2026, 1, 1), rate=rate(input_per_mtok=Decimal("3.00")))
        new = card(effective_from=date(2026, 8, 1), rate=rate(input_per_mtok=Decimal("2.00")))
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        was = cost_of(used, card().ref, at=date(2026, 7, 31), prices=prices(old, new))
        assert was.input == Decimal("3.00")

    def test_two_cards_for_one_model_on_one_day_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="two prices"):
            prices(card(), card())


class TestATierThatDependsOnTheRequest:
    def test_a_long_prompt_takes_the_long_context_rate(self) -> None:
        short = card(rate=rate(input_per_mtok=Decimal("3.00")))
        long = card(rate=rate(input_per_mtok=Decimal("6.00")), above_input_tokens=200_000)
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        money = cost_of(used, card().ref, at=TODAY, prices=prices(short, long))
        assert money.input == Decimal("6.00")

    def test_a_prompt_under_the_threshold_takes_the_standard_rate(self) -> None:
        short = card(rate=rate(input_per_mtok=Decimal("3.00")))
        long = card(rate=rate(input_per_mtok=Decimal("6.00")), above_input_tokens=200_000)
        used = Usage(input_tokens=1_000, output_tokens=0)
        money = cost_of(used, card().ref, at=TODAY, prices=prices(short, long))
        assert money.input == Decimal("0.003")

    def test_a_batch_call_takes_the_batch_card(self) -> None:
        standard = card(rate=rate(input_per_mtok=Decimal("3.00")))
        batch = card(rate=rate(input_per_mtok=Decimal("1.50")), batch=True)
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        money = cost_of(used, card().ref, at=TODAY, prices=prices(standard, batch), batch=True)
        assert money.input == Decimal("1.50")

    def test_a_batch_call_with_no_batch_card_is_priced_at_the_standard_rate(self) -> None:
        """An upper bound stated as one is better than a refusal nobody can act on."""
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        money = cost_of(used, card().ref, at=TODAY, prices=prices(card()), batch=True)
        assert money.input == Decimal("3.00")


class TestWhereThePricesComeFrom:
    def test_the_kit_ships_prices_for_the_models_it_ships_cards_for(self) -> None:
        assert kit_prices().cards

    def test_a_negotiated_rate_overrides_the_shipped_one(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.toml"
        path.write_text(
            "version = 1\n\n"
            "[[cards]]\n"
            'ref = "anthropic:claude-sonnet-4-5"\n'
            "effective_from = 2026-01-01\n"
            'rate = { input_per_mtok = "0.50", output_per_mtok = "1.00" }\n',
            encoding="utf-8",
        )
        used = Usage(input_tokens=1_000_000, output_tokens=0)
        money = cost_of(used, "anthropic:claude-sonnet-4-5", at=TODAY, prices=price_list(path))
        assert money.input == Decimal("0.50")

    def test_a_model_the_override_says_nothing_about_keeps_its_shipped_price(self) -> None:
        shipped = kit_prices()
        merged = shipped.overridden_by(prices(card(ref="vllm:qwen-3")))
        assert merged.rate_for("vllm:qwen-3", at=TODAY) is not None
        assert len(merged.cards) == len(shipped.cards) + 1

    def test_the_price_list_is_read_from_the_environment_where_no_path_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "prices.toml"
        path.write_text(
            'version = 1\n\n[[cards]]\nref = "vllm:qwen-3"\neffective_from = 2026-01-01\n'
            'rate = { input_per_mtok = "0.10", output_per_mtok = "0.20" }\n',
            encoding="utf-8",
        )
        monkeypatch.setenv(PRICE_LIST_ENV, str(path))
        assert price_list().rate_for("vllm:qwen-3", at=TODAY) is not None

    def test_nothing_is_discovered_by_convention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment billing against a file nobody named cannot say what it is billing."""
        monkeypatch.delenv(PRICE_LIST_ENV, raising=False)
        assert price_list().cards == kit_prices().cards

    def test_a_file_that_is_not_toml_names_the_file_rather_than_the_parser(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "prices.toml"
        path.write_text("ref = anthropic\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not readable TOML"):
            price_list(path)

    def test_a_card_of_the_wrong_shape_names_the_file_rather_than_the_model(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "prices.toml"
        path.write_text(
            '[[cards]]\nref = "vllm:qwen-3"\neffective_from = 2026-01-01\n'
            'rate = { input_per_mtok = "0.10" }\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="is not a price list"):
            price_list(path)

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="no price list at"):
            price_list(tmp_path / "absent.toml")

    def test_readable_toml_of_the_wrong_shape_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "prices.toml"
        path.write_text('version = "one"\n', encoding="utf-8")
        with pytest.raises(ConfigurationError, match="is not a price list"):
            price_list(path)


class TestTwoCurrenciesDoNotAddUp:
    """A number that is true in neither currency is worse than no number."""

    def test_totalling_two_currencies_names_both(self) -> None:
        with pytest.raises(ValueError, match="cannot total USD and EUR"):
            _ = Cost(input=Decimal("1")) + Cost(input=Decimal("1"), currency="EUR")

    def test_a_usage_priced_in_another_currency_cannot_be_added_either(self) -> None:
        one = Usage(input_tokens=1, output_tokens=1, cost=Cost(input=Decimal("1")))
        other = Usage(
            input_tokens=1, output_tokens=1, cost=Cost(input=Decimal("1"), currency="INR")
        )
        with pytest.raises(ValueError, match="cannot total USD and INR"):
            _ = one + other

    def test_nothing_spent_yet_takes_the_currency_of_whatever_is_added_to_it(self) -> None:
        """A run starts on an empty cost, which is not a claim about currency."""
        assert (Cost.nothing() + Cost(input=Decimal("1"), currency="EUR")).currency == "EUR"

    def test_adding_nothing_to_a_cost_leaves_it_alone(self) -> None:
        spent = Cost(input=Decimal("1"), currency="EUR")
        assert spent + Cost.nothing() == spent


class TestATokenBurnedOnAFailedAttempt:
    """A vendor charges for a prompt it read and then refused. A ledger that only counts
    answers understates a run that was rate-limited four times before it got one."""

    @staticmethod
    def _runner(*script: object, **overrides: object) -> AgentRunner:
        fields: dict[str, object] = {
            "provider": ScriptedProvider(
                *script,  # type: ignore[arg-type]
                name="openai",
                capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
            ),
            "retry": RetryConfig(max_attempts=1),
            "clock": FakeClock(),
        }
        return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]

    @staticmethod
    def _agent() -> Agent:
        return Agent(name="planner", instructions="Plan trips.", free_text=True, model="gpt-4o")

    @pytest.mark.anyio
    async def test_a_prompt_the_vendor_read_and_refused_is_still_on_the_ledger(self) -> None:
        run = await self._runner(
            RateLimitError("slow down", provider="openai", model="gpt-4o"),
            ModelResponse(content="Kyoto.", usage=Usage(input_tokens=10, output_tokens=5)),
            retry=RetryConfig(max_attempts=2),
        ).run(self._agent(), "Where should I go?", tenant="acme")
        assert run.usage.input_tokens > 10

    @pytest.mark.anyio
    async def test_the_burn_is_marked_a_guess_because_the_vendor_reported_nothing(self) -> None:
        run = await self._runner(
            RateLimitError("slow down", provider="openai", model="gpt-4o"),
            ModelResponse(content="Kyoto.", usage=Usage(input_tokens=10, output_tokens=5)),
            retry=RetryConfig(max_attempts=2),
        ).run(self._agent(), "Where should I go?", tenant="acme")
        assert run.usage.source is CountSource.HEURISTIC

    @pytest.mark.anyio
    async def test_a_run_that_never_got_an_answer_still_says_what_it_burned(self) -> None:
        run = await self._runner(
            InvalidRequestError("no such tool", provider="openai", model="gpt-4o")
        ).run(self._agent(), "Where should I go?", tenant="acme")
        assert run.state is RunState.FAILED
        assert run.usage.input_tokens > 0

    @pytest.mark.anyio
    async def test_the_failed_attempt_carries_its_burn_on_the_event(self) -> None:
        run = await self._runner(
            InvalidRequestError("no such tool", provider="openai", model="gpt-4o")
        ).run(self._agent(), "Where should I go?", tenant="acme")
        failed = next(event for event in run.events if event.kind is RunEventKind.ATTEMPT_FAILED)
        assert failed.usage is not None
        assert failed.usage.source is CountSource.HEURISTIC

    @pytest.mark.anyio
    async def test_an_answered_call_is_counted_once_and_by_the_vendor(self) -> None:
        """No phantom burn on a call that worked: the vendor's own count is the whole story."""
        run = await self._runner(
            ModelResponse(content="Kyoto.", usage=Usage(input_tokens=10, output_tokens=5))
        ).run(self._agent(), "Where should I go?", tenant="acme")
        assert run.usage.input_tokens == 10
        assert run.usage.source is CountSource.PROVIDER


WORKLOAD = {
    "anthropic": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 600,
            "output_tokens": 500,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 200,
        },
    },
    "openai": {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 800,
            "prompt_tokens_details": {"cached_tokens": 400},
            "completion_tokens_details": {"reasoning_tokens": 300},
        },
    },
    "gemini": {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 500,
            "cachedContentTokenCount": 400,
            "thoughtsTokenCount": 300,
        },
    },
}


def _replayed(vendor: str, path: str, model: str) -> Any:
    replay = HttpReplay(
        HttpCassette(provider=vendor, exchanges=(HttpExchange(path=path, body=WORKLOAD[vendor]),))
    )
    kinds = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    return kinds[vendor](
        model,
        secrets=FakeSecrets({f"{vendor.upper()}_API_KEY": "test-key"}),
        transport=replay.transport,
    )


VENDORS = [
    ("anthropic", "/v1/messages", "claude-sonnet-4-5"),
    ("openai", "/v1/chat/completions", "gpt-4o"),
    ("gemini", "/v1beta/models/gemini-2.5-flash:generateContent", "gemini-2.5-flash"),
]


class TestOneWorkloadCountedTheSameWayEverywhere:
    """Three vendors, one workload, one shape of answer. A field that lands in `extras`
    under one vendor and in a column under another cannot be budgeted against."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(("vendor", "path", "model"), VENDORS)
    async def test_the_prompt_and_the_answer_are_counted_alike(
        self, vendor: str, path: str, model: str
    ) -> None:
        response = await _replayed(vendor, path, model).complete(
            ModelRequest(
                model=model, messages=(Message(role="user", content=[TextPart(text="?")]),)
            )
        )
        assert (response.usage.input_tokens, response.usage.cached_tokens) == (1000, 400)
        assert response.usage.output_tokens == 500

    @pytest.mark.anyio
    async def test_a_cache_write_is_a_field_and_not_a_vendor_specific_extra(self) -> None:
        response = await _replayed(*VENDORS[0]).complete(
            ModelRequest(
                model="claude-sonnet-4-5",
                messages=(Message(role="user", content=[TextPart(text="?")]),),
            )
        )
        assert response.usage.cache_write_tokens == 200
        assert response.usage.extras == {}

    @pytest.mark.anyio
    @pytest.mark.parametrize(("vendor", "path", "model"), VENDORS[1:])
    async def test_reasoning_is_a_field_and_not_a_vendor_specific_extra(
        self, vendor: str, path: str, model: str
    ) -> None:
        response = await _replayed(vendor, path, model).complete(
            ModelRequest(
                model=model, messages=(Message(role="user", content=[TextPart(text="?")]),)
            )
        )
        assert response.usage.reasoning_tokens == 300
        assert response.usage.extras == {}

    @pytest.mark.anyio
    @pytest.mark.parametrize(("vendor", "path", "model"), VENDORS)
    async def test_the_same_workload_costs_the_same_at_the_same_rate(
        self, vendor: str, path: str, model: str
    ) -> None:
        """Reasoning aside, which two of the three report, one workload is one bill."""
        response = await _replayed(vendor, path, model).complete(
            ModelRequest(
                model=model, messages=(Message(role="user", content=[TextPart(text="?")]),)
            )
        )
        money = cost_of(
            response.usage,
            f"{vendor}:{model}",
            at=TODAY,
            prices=prices(card(ref=f"{vendor}:{model}")),
        )
        assert money.input == Decimal("0.0018")
        assert money.output == Decimal("0.0075")
        assert money.cache_read == Decimal("0.00012")
