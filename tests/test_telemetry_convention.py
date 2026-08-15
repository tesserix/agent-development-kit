"""One set of attribute names, so the same query answers the same question everywhere."""

from __future__ import annotations

from decimal import Decimal

import pytest

from tesserix_adk.core import (
    AttributionError,
    Cost,
    CostConfidence,
    tenant_scope,
)
from tesserix_adk.observability import (
    CARDINALITY,
    CONVENTION_VERSION,
    MANDATORY,
    MAX_VALUE_LENGTH,
    RESERVED_PREFIX,
    AttributeSet,
    CacheStatus,
    Cardinality,
    Measured,
    Outcome,
    Unavailability,
    conforms,
)


def _set(**overrides: object) -> AttributeSet:
    defaults: dict[str, object] = {
        "tenant": "acme",
        "agent": "refunds",
        "agent_version": "3.1.0",
        "model": "gpt-5",
        "provider": "openai",
        "run_id": "run-1",
        "outcome": Outcome.ANSWERED,
        "cache": CacheStatus.MISS,
        "input_tokens": Measured.of(1200),
        "output_tokens": Measured.of(340),
        "latency_seconds": Measured.of(2.5),
        "cost": Cost(input=Decimal("0.0012"), output=Decimal("0.0044")),
        "pricing_version": "2026-08-01",
    }
    return AttributeSet(**(defaults | overrides))  # type: ignore[arg-type]


class TestTheContract:
    def test_every_attribute_lives_in_the_reserved_namespace(self) -> None:
        rendered = _set().rendered()
        assert all(name.startswith(RESERVED_PREFIX) for name in rendered)

    def test_the_convention_version_travels_with_the_attributes(self) -> None:
        """A dashboard pinned to a version needs the span to say which one it speaks."""
        assert _set().rendered()[f"{RESERVED_PREFIX}convention"] == CONVENTION_VERSION

    def test_the_mandatory_subset_is_carried_by_a_complete_set(self) -> None:
        conforms(_set().rendered())

    def test_a_span_missing_a_mandatory_attribute_is_refused(self) -> None:
        rendered = _set().rendered()
        del rendered[f"{RESERVED_PREFIX}tenant"]
        with pytest.raises(AttributionError, match="tenant"):
            conforms(rendered)

    def test_an_ad_hoc_name_in_the_reserved_namespace_is_refused(self) -> None:
        """Squatting a name the convention has not defined yet is how a version breaks."""
        rendered = _set().rendered() | {f"{RESERVED_PREFIX}my_own_field": "value"}
        with pytest.raises(AttributionError, match="my_own_field"):
            conforms(rendered)

    def test_a_product_attribute_outside_the_namespace_is_allowed(self) -> None:
        conforms(_set().rendered() | {"checkout.step": "review"})

    def test_the_names_do_not_collide_with_the_open_telemetry_ai_conventions(self) -> None:
        assert not any(name.startswith("gen_ai.") for name in _set().rendered())


class TestValuesThatWereNotMeasured:
    def test_a_provider_that_reported_no_usage_says_so_rather_than_guessing(self) -> None:
        rendered = _set(input_tokens=Measured.missing(Unavailability.NOT_REPORTED)).rendered()
        assert f"{RESERVED_PREFIX}input_tokens" not in rendered
        assert rendered[f"{RESERVED_PREFIX}input_tokens.unavailable"] == "not reported"

    def test_an_unpriced_model_records_no_cost_rather_than_zero(self) -> None:
        """A self-hosted model has no vendor price, and zero would read as free."""
        rendered = _set(cost=Cost(confidence=CostConfidence.UNKNOWN)).rendered()
        assert f"{RESERVED_PREFIX}cost" not in rendered
        assert rendered[f"{RESERVED_PREFIX}cost.unavailable"] == "not priced"

    def test_a_cost_that_is_known_carries_its_currency_and_price_list(self) -> None:
        rendered = _set().rendered()
        assert rendered[f"{RESERVED_PREFIX}cost"] == "0.0056"
        assert rendered[f"{RESERVED_PREFIX}currency"] == "USD"
        assert rendered[f"{RESERVED_PREFIX}pricing_version"] == "2026-08-01"

    def test_an_estimated_cost_is_labelled_as_one(self) -> None:
        rendered = _set(cost=Cost(input=Decimal("0.01"), confidence=CostConfidence.ESTIMATED))
        assert rendered.rendered()[f"{RESERVED_PREFIX}cost.confidence"] == "estimated"

    def test_a_measurement_cannot_be_both_a_value_and_unavailable(self) -> None:
        with pytest.raises(ValueError, match="either"):
            Measured(value=1.0, unavailable=Unavailability.NOT_REPORTED)

    def test_a_measurement_must_be_one_or_the_other(self) -> None:
        with pytest.raises(ValueError, match="either"):
            Measured()

    def test_a_whole_number_is_rendered_without_a_decimal_point(self) -> None:
        assert Measured.of(1200).rendered() == "1200"


class TestCache:
    def test_the_cache_status_is_on_every_span_so_effectiveness_is_measurable(self) -> None:
        assert _set(cache=CacheStatus.HIT).rendered()[f"{RESERVED_PREFIX}cache"] == "hit"

    def test_a_run_with_no_cache_in_play_says_so_rather_than_reporting_a_miss(self) -> None:
        """A miss and no cache at all are different numbers on a hit-rate dashboard."""
        assert _set(cache=CacheStatus.NONE).rendered()[f"{RESERVED_PREFIX}cache"] == "none"


class TestCardinality:
    def test_the_high_cardinality_attributes_are_declared(self) -> None:
        assert CARDINALITY[f"{RESERVED_PREFIX}user"] is Cardinality.HIGH
        assert CARDINALITY[f"{RESERVED_PREFIX}tenant"] is Cardinality.LOW

    def test_metric_dimensions_leave_out_what_would_overwhelm_a_backend(self) -> None:
        dimensions = _set(user="ada").metric_dimensions()
        assert f"{RESERVED_PREFIX}user" not in dimensions
        assert f"{RESERVED_PREFIX}run_id" not in dimensions
        assert dimensions[f"{RESERVED_PREFIX}tenant"] == "acme"

    def test_every_rendered_name_has_a_declared_cardinality(self) -> None:
        rendered = _set(input_tokens=Measured.missing(Unavailability.NOT_REPORTED)).rendered()
        assert all(name in CARDINALITY for name in rendered)


class TestLongValues:
    def test_a_long_value_is_truncated_visibly_rather_than_dropped(self) -> None:
        rendered = _set(task_class="x" * (MAX_VALUE_LENGTH * 2)).rendered()
        value = rendered[f"{RESERVED_PREFIX}task_class"]
        assert len(value) <= MAX_VALUE_LENGTH
        assert value.endswith("(truncated)")

    def test_a_value_within_the_limit_is_left_alone(self) -> None:
        assert _set().rendered()[f"{RESERVED_PREFIX}agent"] == "refunds"


class TestExtras:
    def test_a_product_may_add_its_own_attributes(self) -> None:
        rendered = _set(extra={"checkout.step": "review"}).rendered()
        assert rendered["checkout.step"] == "review"

    def test_a_product_may_not_squat_a_name_the_convention_has_not_defined_yet(self) -> None:
        with pytest.raises(ValueError, match=RESERVED_PREFIX):
            _set(extra={f"{RESERVED_PREFIX}tenant": "rival"})


class TestAutomaticPopulation:
    def test_the_tenant_and_user_come_from_the_bound_context(self) -> None:
        with tenant_scope("acme", user="ada"):
            built = AttributeSet.here(
                agent="refunds",
                agent_version="3.1.0",
                model="gpt-5",
                provider="openai",
                run_id="run-1",
                outcome=Outcome.ANSWERED,
                cache=CacheStatus.MISS,
                input_tokens=Measured.of(1200),
                output_tokens=Measured.of(340),
                latency_seconds=Measured.of(2.5),
                cost=Cost(input=Decimal("0.01")),
                pricing_version="2026-08-01",
            )
        assert built.tenant == "acme"
        assert built.rendered()[f"{RESERVED_PREFIX}user"] == "ada"

    def test_the_same_run_in_two_products_renders_identical_names(self) -> None:
        """The point of the convention: one query, no per-product field mapping."""
        one = _set(extra={"checkout.step": "review"}).rendered()
        other = _set(extra={"support.queue": "tier-2"}).rendered()
        reserved = {name for name in one if name.startswith(RESERVED_PREFIX)}
        assert reserved == {name for name in other if name.startswith(RESERVED_PREFIX)}


class TestMandatory:
    def test_the_mandatory_names_are_a_subset_of_what_a_complete_set_renders(self) -> None:
        assert set(_set().rendered()) >= MANDATORY
