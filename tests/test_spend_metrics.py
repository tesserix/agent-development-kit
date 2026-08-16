"""Spend visible in near real time, with the gaps visible as gaps."""

from __future__ import annotations

from decimal import Decimal

import pytest

from tesserix_adk.core import Cost, CostConfidence, CountSource, Usage
from tesserix_adk.observability import (
    BUDGET_BREACHES,
    COST,
    DROPPED,
    ITERATIONS,
    LATENCY,
    TOKENS,
    TOOL_CALLS,
    UNKNOWN_USAGE,
    UNPRICED,
    Attribution,
    Dimensions,
    ModelRate,
    Outcome,
    PricingTable,
    SpendMeter,
    SpendRecord,
    Step,
)


class _Collector:
    """A metric store that records what it was told."""

    def __init__(self) -> None:
        self.counted: list[tuple[str, float, dict[str, str]]] = []

    def count(self, name: str, value: float, **dimensions: str) -> None:
        self.counted.append((name, value, dimensions))

    def series(self, name: str) -> list[tuple[str, float, dict[str, str]]]:
        return [entry for entry in self.counted if entry[0] == name]

    def total(self, name: str) -> float:
        return sum(entry[1] for entry in self.series(name))


class _Falling:
    """A metric store whose collector is down, which is how one usually behaves."""

    def count(self, name: str, value: float, **dimensions: str) -> None:  # noqa: ARG002
        message = f"collector unreachable for {name}"
        raise RuntimeError(message)


def _table() -> PricingTable:
    return PricingTable(
        version="2026-08-01",
        rates={
            "gpt-5": ModelRate(input_per_million=Decimal("1.25"), output_per_million=Decimal("10")),
            "local-8b": ModelRate(
                input_per_million=Decimal("0.02"),
                output_per_million=Decimal("0.02"),
                source="self-hosted",
            ),
        },
    )


def _attribution(tenant: str = "acme", model: str = "gpt-5") -> Attribution:
    return Attribution(
        tenant=tenant,
        user="ada",
        agent="refunds",
        agent_version="3",
        definition="refunds@3",
        model=model,
        prompt_version="7",
        task_class="support",
        run_id=f"run-{tenant}",
    )


def _usage(*, cost: Cost | None = None, source: CountSource = CountSource.PROVIDER) -> Usage:
    return Usage(
        input_tokens=1000,
        output_tokens=200,
        extras={},
        source=source,
        cost=cost if cost is not None else Cost(input=Decimal("0.00125"), output=Decimal("0.002")),
    )


def _record(
    *, tenant: str = "acme", model: str = "gpt-5", usage: Usage | None = None
) -> SpendRecord:
    return SpendRecord(
        attribution=_attribution(tenant, model),
        step=Step.MODEL,
        outcome=Outcome.ANSWERED,
        usage=usage if usage is not None else _usage(),
    )


def _meter(collector: _Collector | _Falling, **kwargs: object) -> SpendMeter:
    return SpendMeter(collector, pricing=_table(), **kwargs)  # type: ignore[arg-type]


class TestWhatIsCounted:
    def test_tokens_are_counted(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record())
        assert collector.total(TOKENS) == 1200

    def test_cost_is_counted_in_the_currency_it_was_priced_in(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record())
        _, value, dimensions = collector.series(COST)[0]
        assert value == pytest.approx(0.00325)
        assert dimensions["currency"] == "USD"

    def test_latency_is_counted_where_a_step_was_timed(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record(), seconds=1.5)
        assert collector.total(LATENCY) == pytest.approx(1.5)

    def test_iterations_and_tool_fan_out_are_counted(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record(), iterations=3, tool_calls=7)
        assert collector.total(ITERATIONS) == 3
        assert collector.total(TOOL_CALLS) == 7

    def test_a_budget_breach_is_its_own_series_to_alert_on(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record(), breached=True)
        assert collector.total(BUDGET_BREACHES) == 1

    def test_a_run_that_stayed_inside_its_budget_counts_no_breach(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record())
        assert collector.series(BUDGET_BREACHES) == []


class TestDimensions:
    def test_series_are_split_by_tenant_agent_and_model(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record())
        dimensions = collector.series(TOKENS)[0][2]
        assert dimensions["tenant"] == "acme"
        assert dimensions["agent"] == "refunds"
        assert dimensions["model"] == "gpt-5"

    def test_the_agent_version_is_a_dimension_so_a_regression_is_attributable(self) -> None:
        collector = _Collector()
        _meter(collector).record(_record())
        assert collector.series(TOKENS)[0][2]["agent_version"] == "3"

    def test_the_pricing_version_travels_with_the_cost_series(self) -> None:
        """A mid-window price change must not silently rewrite what was recorded."""
        collector = _Collector()
        _meter(collector).record(_record())
        assert collector.series(COST)[0][2]["pricing_version"] == "2026-08-01"

    def test_a_tenant_the_deployment_did_not_list_lands_in_the_shared_bucket(self) -> None:
        collector = _Collector()
        meter = SpendMeter(
            collector, pricing=_table(), dimensions=Dimensions(tenants=frozenset({"acme"}))
        )
        meter.record(_record(tenant="globex"))
        assert collector.series(TOKENS)[0][2]["tenant"] == "other"

    def test_no_free_form_value_reaches_a_dimension(self) -> None:
        """A user id or a prompt in a dimension is the cardinality blow-up."""
        collector = _Collector()
        _meter(collector).record(_record())
        assert "user" not in collector.series(TOKENS)[0][2]
        assert "run_id" not in collector.series(TOKENS)[0][2]


class TestWhatIsNotKnown:
    def test_a_call_with_no_reported_usage_is_counted_as_unknown_not_as_zero(self) -> None:
        collector = _Collector()
        unpriced = Usage(input_tokens=0, output_tokens=0, extras={}, source=CountSource.HEURISTIC)
        _meter(collector).record(_record(usage=unpriced))
        assert collector.total(UNKNOWN_USAGE) == 1

    def test_an_unreported_call_contributes_nothing_to_the_cost_series(self) -> None:
        collector = _Collector()
        unpriced = Usage(input_tokens=0, output_tokens=0, extras={}, source=CountSource.HEURISTIC)
        _meter(collector).record(_record(usage=unpriced))
        assert collector.series(COST) == []

    def test_a_model_the_table_does_not_price_is_counted_as_unpriced(self) -> None:
        collector = _Collector()
        usage = Usage(input_tokens=10, output_tokens=10, extras={}, cost=None)
        _meter(collector).record(_record(model="mystery", usage=usage))
        assert collector.total(UNPRICED) == 1

    def test_an_unpriced_call_still_has_its_tokens_counted(self) -> None:
        """The tokens were spent whether or not anybody can price them."""
        collector = _Collector()
        usage = Usage(input_tokens=10, output_tokens=10, extras={}, cost=None)
        _meter(collector).record(_record(model="mystery", usage=usage))
        assert collector.total(TOKENS) == 20

    def test_a_self_hosted_model_is_priced_from_its_configured_rate(self) -> None:
        collector = _Collector()
        usage = Usage(input_tokens=1_000_000, output_tokens=0, extras={}, cost=None)
        _meter(collector).record(_record(model="local-8b", usage=usage))
        assert collector.total(COST) == pytest.approx(0.02)

    def test_a_priced_call_the_provider_did_not_count_is_marked_estimated(self) -> None:
        collector = _Collector()
        estimated = Cost(input=Decimal("0.001"), confidence=CostConfidence.ESTIMATED)
        _meter(collector).record(_record(usage=_usage(cost=estimated)))
        assert collector.series(COST)[0][2]["cost_confidence"] == "estimated"

    def test_the_kit_never_presents_a_computed_figure_as_measured(self) -> None:
        collector = _Collector()
        usage = Usage(input_tokens=1_000_000, output_tokens=0, extras={}, cost=None)
        _meter(collector).record(_record(model="local-8b", usage=usage))
        assert collector.series(COST)[0][2]["cost_confidence"] == "estimated"


class TestNotCountingTwice:
    def test_the_same_step_recorded_twice_is_counted_once(self) -> None:
        """A worker restart replays a run; the invoice does not."""
        collector = _Collector()
        meter = _meter(collector)
        meter.record(_record(), key="run-1/model/1")
        meter.record(_record(), key="run-1/model/1")
        assert collector.total(TOKENS) == 1200

    def test_a_duplicate_is_counted_so_the_operator_can_see_it_happened(self) -> None:
        collector = _Collector()
        meter = _meter(collector)
        meter.record(_record(), key="run-1/model/1")
        meter.record(_record(), key="run-1/model/1")
        assert meter.stats.duplicates == 1

    def test_a_retry_of_a_step_is_counted_because_it_really_spent_tokens(self) -> None:
        collector = _Collector()
        meter = _meter(collector)
        meter.record(_record(), key="run-1/model/1")
        meter.record(_record(), key="run-1/model/2")
        assert collector.total(TOKENS) == 2400

    def test_a_run_that_stopped_part_way_still_counts_the_tokens_it_spent(self) -> None:
        """A cancelled or streamed-then-abandoned call spent them all the same."""
        collector = _Collector()
        stopped = SpendRecord(
            attribution=_attribution(), step=Step.MODEL, outcome=Outcome.FAILED, usage=_usage()
        )
        _meter(collector).record(stopped)
        assert collector.total(TOKENS) == 1200
        assert collector.series(TOKENS)[0][2]["outcome"] == "failed"


class TestACollectorThatIsDown:
    def test_a_collector_outage_does_not_reach_the_run(self) -> None:
        meter = _meter(_Falling())
        meter.record(_record())
        assert meter.stats.recorded == 1

    def test_dropped_metrics_are_counted_locally(self) -> None:
        meter = _meter(_Falling())
        meter.record(_record())
        assert meter.stats.dropped > 0

    def test_the_local_drop_counter_has_its_own_series_name(self) -> None:
        assert DROPPED.startswith("adk.")

    def test_a_partial_failure_still_records_what_it_could(self) -> None:
        class _Fussy(_Collector):
            def count(self, name: str, value: float, **dimensions: str) -> None:
                if name == COST:
                    message = "cost series rejected"
                    raise RuntimeError(message)
                super().count(name, value, **dimensions)

        collector = _Fussy()
        meter = _meter(collector)
        meter.record(_record())
        assert collector.total(TOKENS) == 1200
        assert meter.stats.dropped == 1


class TestReconciliation:
    def test_the_counted_total_matches_the_records_it_was_given(self) -> None:
        """Metrics and traces disagreeing about a run is worse than having neither."""
        collector = _Collector()
        meter = _meter(collector)
        records = [_record(), _record(tenant="globex")]
        for index, record in enumerate(records):
            meter.record(record, key=f"k{index}")
        spent = sum(record.usage.input_tokens + record.usage.output_tokens for record in records)
        assert collector.total(TOKENS) == spent

    def test_a_currency_is_never_converted_on_the_kit_s_own_authority(self) -> None:
        collector = _Collector()
        priced = Cost(input=Decimal("1"), currency="EUR")
        _meter(collector).record(_record(usage=_usage(cost=priced)))
        assert collector.series(COST)[0][2]["currency"] == "EUR"


class TestThePricingTable:
    def test_a_table_must_state_its_version(self) -> None:
        with pytest.raises(ValueError, match="version"):
            PricingTable(version="", rates={})

    def test_a_rate_prices_a_million_tokens(self) -> None:
        table = _table()
        rate = table.rate_of("gpt-5")
        assert rate is not None
        assert rate.of(input_tokens=1_000_000, output_tokens=0).input == Decimal("1.25")

    def test_a_model_the_table_does_not_know_has_no_rate(self) -> None:
        assert _table().rate_of("mystery") is None

    def test_a_self_hosted_rate_says_where_it_came_from(self) -> None:
        rate = _table().rate_of("local-8b")
        assert rate is not None
        assert rate.source == "self-hosted"
