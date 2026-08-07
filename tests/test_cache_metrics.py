"""Whether prompt caching is actually working, as a number rather than a belief.

Prefix stability is unfalsifiable without a hit ratio: every context-engineering change
reads as an improvement if nothing counts what the server re-evaluated.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from tesserix_adk.core import (
    Cost,
    CostConfidence,
    CountSource,
    Run,
    RunEvent,
    RunEventKind,
    RunState,
    Usage,
)
from tesserix_adk.observability import (
    CACHED_TOKENS,
    INPUT_TOKENS,
    Attribution,
    Outcome,
    SpendRecord,
    Step,
    record_spend,
    totals_by,
)
from tesserix_adk.testing import FakeMeter


def spent(*, sent: int = 0, cached: int = 0, written: int = 0, out: int = 0, **rest: Any) -> Usage:
    return Usage(
        input_tokens=sent,
        cached_tokens=cached,
        cache_write_tokens=written,
        output_tokens=out,
        cost=rest.pop("cost", Cost(input=Decimal("0.1"))),
        **rest,
    )


def record(usage: Usage, *, model: str = "llama-3.1-8b") -> SpendRecord:
    return SpendRecord(
        attribution=Attribution(
            tenant="acme",
            user="ada",
            agent="clerk",
            agent_version="1",
            definition="rev0",
            model=model,
            prompt_version="p1",
            task_class="cheap",
            run_id="r1",
        ),
        step=Step.MODEL,
        outcome=Outcome.ANSWERED,
        usage=usage,
    )


def run(usage: Usage) -> Run[Any]:
    return Run(
        id="run_1",
        tenant="acme",
        user="ada",
        agent_name="clerk",
        agent_version="1.0.0",
        model="llama-3.1-8b",
        prompt_version="p1",
        state=RunState.PENDING,
        events=[
            RunEvent(kind=RunEventKind.MODEL_CALL, name="llama-3.1-8b"),
            RunEvent(kind=RunEventKind.MODEL_RESPONSE, name="scripted", usage=usage),
        ],
    )


class TestTheRatioOnOneStep:
    def test_cached_input_is_reported_apart_from_fresh_input(self) -> None:
        """A total that folds the two together cannot say whether caching did anything."""
        counted = spent(sent=1_000, cached=800, out=50)
        assert counted.fresh_input_tokens == 200
        assert counted.cache_hit_ratio == 0.8

    def test_nothing_read_is_a_ratio_of_zero_not_a_division_error(self) -> None:
        assert spent().cache_hit_ratio == 0.0

    def test_a_provider_reporting_more_cache_than_input_does_not_go_negative(self) -> None:
        """Vendors disagree about whether cache reads sit inside the input count."""
        assert spent(sent=100, cached=400).fresh_input_tokens == 0

    def test_the_ratio_is_measured_only_where_something_was_sent(self) -> None:
        """Zero over zero is 'nobody looked', which is not the same as 'no hits'."""
        assert spent(sent=10).measured is True
        assert spent().measured is False


class TestAggregationUpTheTree:
    def test_cached_tokens_total_component_wise(self) -> None:
        totals = totals_by(
            [record(spent(sent=1_000, cached=900, out=10)), record(spent(sent=1_000, out=10))],
            "tenant",
        )
        assert totals[("acme",)].cached_tokens == 900
        assert totals[("acme",)].input_tokens == 2_000
        assert totals[("acme",)].hit_ratio == 0.45

    def test_cache_writes_are_totalled_apart_because_they_are_priced_apart(self) -> None:
        totals = totals_by([record(spent(sent=1_000, written=1_000, out=10))], "tenant")
        assert totals[("acme",)].cache_write_tokens == 1_000
        assert totals[("acme",)].hit_ratio == 0.0

    def test_components_are_separated_by_whatever_was_grouped_by(self) -> None:
        totals = totals_by(
            [
                record(spent(sent=100, cached=100, out=1), model="llama-3.1-8b"),
                record(spent(sent=100, out=1), model="qwen2.5-14b"),
            ],
            "model",
        )
        assert totals[("llama-3.1-8b",)].hit_ratio == 1.0
        assert totals[("qwen2.5-14b",)].hit_ratio == 0.0

    def test_a_group_that_sent_nothing_says_so_rather_than_reporting_no_hits(self) -> None:
        """Zero percent and no data are different findings; a dashboard must tell them apart."""
        totals = totals_by([record(spent(out=5))], "tenant")
        assert totals[("acme",)].measured is False
        assert totals[("acme",)].hit_ratio == 0.0


class TestWhenTheProviderReportsNothing:
    def test_no_usage_is_no_record_rather_than_a_zero_cost_one(self) -> None:
        assert totals_by([], "tenant") == {}

    def test_an_unpriced_step_keeps_its_token_counts(self) -> None:
        """The cache question stays answerable where the money question is not."""
        totals = totals_by([record(spent(sent=100, cached=60, out=1, cost=None))], "tenant")
        assert totals[("acme",)].hit_ratio == 0.6
        assert totals[("acme",)].cost.confidence is CostConfidence.UNKNOWN

    def test_an_estimated_count_is_flagged_so_the_ratio_reads_as_a_guess(self) -> None:
        guessed = spent(sent=100, cached=50, out=1, source=CountSource.HEURISTIC)
        assert totals_by([record(guessed)], "tenant")[("acme",)].estimated is True


class TestTheCounters:
    def test_input_and_cached_tokens_are_counted_separately(self) -> None:
        """A ratio cannot be aggregated by a metric store; two counters can be divided."""
        meter = FakeMeter()
        record_spend(run(spent(sent=1_000, cached=800, out=50)), meter=meter)
        assert meter.total(INPUT_TOKENS) == 1_000
        assert meter.total(CACHED_TOKENS) == 800

    def test_a_run_that_read_no_cache_still_counts_its_input(self) -> None:
        """A missing series and a zero series look identical on a dashboard."""
        meter = FakeMeter()
        record_spend(run(spent(sent=1_000, out=50)), meter=meter)
        assert meter.total(INPUT_TOKENS) == 1_000
        assert meter.total(CACHED_TOKENS) == 0

    def test_the_counters_carry_the_same_dimensions_as_the_cost(self) -> None:
        meter = FakeMeter()
        record_spend(run(spent(sent=1_000, cached=800, out=50)), meter=meter)
        assert meter.total(CACHED_TOKENS, model="llama-3.1-8b", tenant="acme") == 800
