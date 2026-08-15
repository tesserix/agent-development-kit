"""Spend attributed to who spent it, from the run rather than from the caller.

A vendor invoice is one number per API key. Everything that makes it answerable — which
tenant, which agent, which version of that agent, which model — is known inside the run and
lost by the time the bill arrives. These are the tests that it is not lost.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from tesserix_adk.core import (
    Agent,
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
    UNKNOWN,
    Attribution,
    Dimensions,
    Outcome,
    Redactor,
    Step,
    attributes_of,
    record_spend,
    spend_of,
    totals_by,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, FakeMeter, FakeTracer, ScriptedProvider


def money(amount: str, confidence: CostConfidence = CostConfidence.COUNTED) -> Cost:
    return Cost(input=Decimal(amount), currency="USD", confidence=confidence)


def usage(cost: Cost | None = None, source: CountSource = CountSource.PROVIDER) -> Usage:
    return Usage(input_tokens=100, output_tokens=20, cost=cost, source=source)


def event(kind: RunEventKind, name: str | None = None, spent: Usage | None = None) -> RunEvent:
    return RunEvent(kind=kind, name=name, usage=spent)


def run(
    *events: RunEvent,
    tenant: str = "acme",
    agent: str = "planner",
    model: str = "scripted-1",
    **overrides: Any,
) -> Run[Any]:
    fields: dict[str, Any] = {
        "id": "run_1",
        "tenant": tenant,
        "user": "ada",
        "agent_name": agent,
        "agent_version": "1.0.0",
        "model": model,
        "prompt_version": "p1",
        "state": RunState.PENDING,
        "events": list(events),
        **overrides,
    }
    return Run(**fields)


def answered(model: str = "scripted-1", cost: Cost | None = None) -> tuple[RunEvent, RunEvent]:
    return (
        event(RunEventKind.MODEL_CALL, name=model),
        event(RunEventKind.MODEL_RESPONSE, name="scripted", spent=usage(cost)),
    )


class TestAttributionComesFromTheRun:
    def test_every_dimension_of_the_bill_is_read_off_the_run(self) -> None:
        """A consumer that has to tag its own spend is a consumer that will mis-tag it."""
        (record,) = spend_of(run(*answered(cost=money("0.20"))))
        assert record.attribution == Attribution(
            tenant="acme",
            user="ada",
            agent="planner",
            agent_version="1.0.0",
            definition=UNKNOWN,
            model="scripted-1",
            prompt_version="p1",
            task_class=UNKNOWN,
            run_id="run_1",
        )

    def test_a_run_acting_for_one_tenant_is_never_billed_to_another(self) -> None:
        """On-behalf-of is a request header, not a licence to move somebody else's spend."""
        one = spend_of(run(*answered(cost=money("0.20")), tenant="acme"))
        other = spend_of(run(*answered(cost=money("0.20")), tenant="globex"))
        assert {r.attribution.tenant for r in (*one, *other)} == {"acme", "globex"}

    def test_what_the_run_could_not_say_is_bucketed_and_flagged(self) -> None:
        """A blank where a tenant should be is a number nobody can chase, so it is named."""
        (record,) = spend_of(run(*answered(cost=money("0.20")), user=None, prompt_version=None))
        assert record.attribution.user == UNKNOWN
        assert record.attribution.unknowns == (
            "definition",
            "prompt",
            "prompt_digest",
            "prompt_version",
            "task_class",
            "user",
        )

    def test_the_task_class_is_taken_from_the_routing_decision(self) -> None:
        (record,) = spend_of(
            run(
                event(RunEventKind.MODEL_ROUTED, name="scripted-1"),
                *answered(cost=money("0.20")),
                task_class="reasoning",
            )
        )
        assert record.attribution.task_class == "reasoning"


class TestSpendThatWouldOtherwiseBeLost:
    def test_a_failed_attempt_is_attributed_rather_than_discarded(self) -> None:
        """Tokens the vendor read before it failed are on the invoice either way."""
        records = spend_of(
            run(
                event(RunEventKind.MODEL_CALL, name="scripted-1"),
                event(RunEventKind.ATTEMPT_FAILED, name="scripted", spent=usage(money("0.05"))),
                *answered(cost=money("0.20")),
            )
        )
        assert [r.outcome for r in records] == [Outcome.FAILED, Outcome.ANSWERED]
        assert sum(r.cost.total for r in records) == Decimal("0.25")

    def test_spend_after_a_fallback_belongs_to_the_model_that_burned_it(self) -> None:
        """One run, two vendors: totalling both against the first is a made-up number."""
        records = spend_of(
            run(
                event(RunEventKind.MODEL_CALL, name="first"),
                event(RunEventKind.ATTEMPT_FAILED, name="scripted", spent=usage(money("0.05"))),
                *answered(model="second", cost=money("0.20")),
            )
        )
        assert [r.attribution.model for r in records] == ["first", "second"]

    def test_a_tool_call_is_a_step_with_no_price_rather_than_a_missing_step(self) -> None:
        records = spend_of(
            run(*answered(cost=money("0.20")), event(RunEventKind.TOOL_RESULT, name="lookup"))
        )
        assert [r.step for r in records] == [Step.MODEL, Step.TOOL]
        assert records[1].cost.total == Decimal(0)

    def test_the_records_total_what_the_run_says_it_spent(self) -> None:
        """Attribution that does not reconcile with the run is a second set of books."""
        spent = run(
            event(RunEventKind.MODEL_CALL, name="scripted-1"),
            event(RunEventKind.ATTEMPT_FAILED, name="scripted", spent=usage(money("0.05"))),
            *answered(cost=money("0.20")),
        ).model_copy(update={"usage": usage(money("0.05")) + usage(money("0.20"))})
        records = spend_of(spent)
        assert sum(r.cost.total for r in records) == spent.usage.cost.total  # type: ignore[union-attr]


class TestChargeback:
    def test_spend_breaks_down_by_whichever_dimensions_are_asked_for(self) -> None:
        records = (
            *spend_of(run(*answered(cost=money("0.20")), tenant="acme")),
            *spend_of(run(*answered(model="other", cost=money("0.30")), tenant="acme")),
            *spend_of(run(*answered(cost=money("0.50")), tenant="globex")),
        )
        by_tenant = totals_by(records, "tenant")
        assert by_tenant[("acme",)].cost.total == Decimal("0.50")
        assert by_tenant[("globex",)].cost.total == Decimal("0.50")
        assert totals_by(records, "tenant", "model")[("acme", "other")].cost.total == Decimal(
            "0.30"
        )

    def test_a_total_carries_the_tokens_and_the_call_count_too(self) -> None:
        records = spend_of(run(*answered(cost=money("0.20"))))
        (total,) = totals_by(records, "agent").values()
        assert (total.input_tokens, total.output_tokens, total.calls) == (100, 20, 1)

    def test_an_estimated_total_says_so_rather_than_reading_as_metered(self) -> None:
        """Reconciling against an invoice needs to know which rows will never appear on it."""
        metered = spend_of(run(*answered(cost=money("0.20"))))
        guessed = spend_of(
            run(*answered(cost=money("0.20", CostConfidence.ESTIMATED)), tenant="globex")
        )
        totals = totals_by((*metered, *guessed), "tenant")
        assert totals[("acme",)].estimated is False
        assert totals[("globex",)].estimated is True

    def test_a_dimension_nothing_was_grouped_by_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not an attribution dimension"):
            totals_by(spend_of(run(*answered())), "department")

    def test_two_currencies_are_not_added_into_one_number(self) -> None:
        records = (
            *spend_of(run(*answered(cost=Cost(input=Decimal("1"), currency="USD")))),
            *spend_of(run(*answered(cost=Cost(input=Decimal("1"), currency="EUR")))),
        )
        with pytest.raises(ValueError, match="currency"):
            totals_by(records, "agent")


class TestWhatIsExported:
    def test_a_span_carries_the_attribution_and_the_spend(self) -> None:
        tracer = FakeTracer()
        record_spend(run(*answered(cost=money("0.20"))), tracer=tracer, meter=FakeMeter())
        (span,) = [r for r in tracer.recorded if r.kind == "span"]
        assert span.attributes["adk.tenant"] == "acme"
        assert span.attributes["adk.cost"] == "0.20"
        assert span.attributes["adk.currency"] == "USD"
        assert span.attributes["adk.estimated"] == "false"

    def test_the_attribute_names_are_one_set_rather_than_one_per_product(self) -> None:
        keys = set(attributes_of(spend_of(run(*answered(cost=money("0.20"))))[0]))
        assert keys == {
            "adk.agent",
            "adk.agent_version",
            "adk.cost",
            "adk.currency",
            "adk.definition",
            "adk.estimated",
            "adk.input_tokens",
            "adk.model",
            "adk.output_tokens",
            "adk.outcome",
            "adk.prompt",
            "adk.prompt_digest",
            "adk.prompt_version",
            "adk.run_id",
            "adk.step",
            "adk.task_class",
            "adk.tenant",
            "adk.user",
        }

    def test_an_email_and_a_key_a_consumer_attached_never_leave_the_process(self) -> None:
        """Cost data is queried by people who were never cleared to read prompts."""
        tracer, meter = FakeTracer(), FakeMeter()
        record_spend(
            run(*answered(cost=money("0.20"))),
            tracer=tracer,
            meter=meter,
            # a fixture, not a credential; gitleaks:allow
            extra={"requested_by": "ada@example.com", "key": "sk-live-4Kx9pQ2mZ7"},
        )
        (span,) = [r for r in tracer.recorded if r.kind == "span"]
        assert "ada@example.com" not in str(span.attributes)
        assert "sk-live-4Kx9pQ2mZ7" not in str(span.attributes)
        assert span.attributes["adk.cost"] == "0.20"

    def test_a_redaction_is_recorded_so_the_drop_is_not_silent(self) -> None:
        tracer = FakeTracer()
        record_spend(
            run(*answered(cost=money("0.20"))),
            tracer=tracer,
            meter=FakeMeter(),
            extra={"requested_by": "ada@example.com"},
        )
        (dropped,) = [r for r in tracer.recorded if r.name == "adk.redacted"]
        assert dropped.attributes["adk.redacted_keys"] == "requested_by"

    def test_a_pattern_a_deployment_knows_about_is_redacted_too(self) -> None:
        scrubbed, redaction = Redactor(extra_patterns=(r"CASE-\d+",)).scrub(
            {"note": "see CASE-4471", "adk.tenant": "acme"}
        )
        assert scrubbed["note"] == "[redacted]"
        assert scrubbed["adk.tenant"] == "acme"
        assert redaction.dropped == ("note",)


class TestMetricsAreNotTraces:
    def test_spend_is_counted_even_when_the_trace_is_not_sampled(self) -> None:
        """A sampled-away trace that takes the spend with it is how a bill goes missing."""
        tracer, meter = FakeTracer(), FakeMeter()
        record_spend(run(*answered(cost=money("0.20"))), tracer=tracer, meter=meter, sampled=False)
        assert tracer.recorded == []
        assert meter.total("adk.cost") == pytest.approx(0.20)

    def test_tokens_and_calls_are_counted_beside_the_money(self) -> None:
        meter = FakeMeter()
        record_spend(run(*answered(cost=money("0.20"))), meter=meter)
        assert meter.total("adk.tokens") == pytest.approx(120.0)
        assert meter.total("adk.calls") == pytest.approx(1.0)

    def test_the_metric_dimensions_are_a_stated_low_cardinality_set(self) -> None:
        """A tenant id per series is a metric store that falls over at the worst moment."""
        meter = FakeMeter()
        record_spend(run(*answered(cost=money("0.20"))), meter=meter)
        assert set(meter.points[0].dimensions) == {
            "tenant",
            "agent",
            "model",
            "task_class",
            "outcome",
            "estimated",
            "currency",
        }

    def test_a_tenant_outside_the_allow_list_is_bucketed_rather_than_dropped(self) -> None:
        meter = FakeMeter()
        record_spend(
            run(*answered(cost=money("0.20")), tenant="globex"),
            meter=meter,
            dimensions=Dimensions(tenants=frozenset({"acme"})),
        )
        assert meter.points[0].dimensions["tenant"] == "other"
        assert meter.total("adk.cost") == pytest.approx(0.20)

    def test_the_span_still_names_the_tenant_the_metric_bucketed(self) -> None:
        """Bucketing protects the metric store; an investigation still needs the id."""
        tracer = FakeTracer()
        record_spend(
            run(*answered(cost=money("0.20")), tenant="globex"),
            tracer=tracer,
            meter=FakeMeter(),
            dimensions=Dimensions(tenants=frozenset({"acme"})),
        )
        (span,) = [r for r in tracer.recorded if r.kind == "span"]
        assert span.attributes["adk.tenant"] == "globex"

    def test_tracing_without_a_meter_exports_the_span_and_counts_nothing(self) -> None:
        """A deployment tracing before it has a metrics pipeline is not a broken one."""
        tracer = FakeTracer()
        records = record_spend(run(*answered(cost=money("0.20"))), tracer=tracer)
        assert len(records) == 1
        assert [r for r in tracer.recorded if r.kind == "span"] != []

    def test_a_run_that_spent_nothing_emits_nothing(self) -> None:
        meter = FakeMeter()
        assert record_spend(run(), meter=meter) == ()
        assert meter.points == []


class TestAConsumerWiresNothing:
    async def test_a_real_run_is_attributed_without_the_caller_tagging_anything(self) -> None:
        """The point of deriving it: a product that forgot to tag its spend still has it."""
        answer = ModelResponse(
            content="Kyoto, four nights.",
            usage=Usage(input_tokens=40, output_tokens=8, cost=money("0.11")),
        )
        finished = await AgentRunner(
            provider=ScriptedProvider(answer, name="scripted", capabilities=CAPABLE),
            clock=FakeClock(),
        ).run(
            Agent(name="planner", instructions="Plan trips.", free_text=True, model="scripted-1"),
            "where to?",
            tenant="acme",
            user="ada",
        )
        meter = FakeMeter()
        (record,) = record_spend(finished, meter=meter)
        assert record.attribution.tenant == "acme"
        assert record.attribution.agent == "planner"
        assert record.attribution.model == "scripted-1"
        assert record.attribution.prompt_version == finished.prompt_version
        assert record.attribution.unknowns == (
            "definition",
            "prompt",
            "prompt_digest",
            "task_class",
        )
        assert meter.total("adk.cost", tenant="acme") == pytest.approx(0.11)
