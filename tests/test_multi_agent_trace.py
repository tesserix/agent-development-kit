"""One trace over every participant, and a total that says what is missing from it.

The failure these are about is a cost figure that covers whichever process held the ledger
and reads as though it covered the run. So: every participant is on the tree, an
unreported one is named rather than counted as zero, a foreign currency is converted
through a recorded rate or held out, and a span nobody could attribute is refused before it
leaves.
"""

from __future__ import annotations

import asyncio
import io
from decimal import Decimal
from typing import Any

import pytest

from tesserix_adk.cli import inspect as inspect_cli
from tesserix_adk.core import (
    AttributionError,
    Cost,
    CostConfidence,
    Run,
    RunEvent,
    RunEventKind,
    Usage,
)
from tesserix_adk.observability import (
    ATTRIBUTE_PREFIX,
    COST,
    TOKENS,
    UNKNOWN,
    Dimensions,
    Pattern,
    Rate,
    RunTree,
    Step,
    TraceContext,
    attributes_of_context,
    node_of,
    peer_node,
    record_tree,
    render,
    tree,
)
from tesserix_adk.observability.trace import HEADER
from tesserix_adk.testing import FakeMeter, FakeTracer


def money(amount: str, currency: str = "USD") -> Cost:
    return Cost(input=Decimal(amount), currency=currency)


def spent(cost: Cost | None = None, tokens: int = 100) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=20, cost=cost)


def run(
    run_id: str,
    *,
    agent: str = "planner",
    tenant: str = "acme",
    cost: str = "0.20",
    started: float | None = 1000.0,
    ended: float | None = 1002.0,
) -> Run[Any]:
    """A finished run with one metered model call, so it has records as well as a total."""
    return Run(
        id=run_id,
        tenant=tenant,
        user="ada",
        agent_name=agent,
        agent_version="1.0.0",
        model="scripted-1",
        events=[
            RunEvent(kind=RunEventKind.MODEL_CALL, name="scripted-1"),
            RunEvent(kind=RunEventKind.MODEL_RESPONSE, usage=spent(money(cost))),
        ],
        usage=spent(money(cost)),
        started_at=started,
        ended_at=ended,
    )


def rooted(run_id: str = "run_1", agent: str = "planner") -> TraceContext:
    return TraceContext.root_of(run(run_id, agent=agent))


def five() -> RunTree:
    """A supervisor, three workers and one peer — the shape the story is written about."""
    root = rooted()
    workers = [
        node_of(
            run(f"w{index}", agent=f"worker{index}", cost="0.10", started=1000.0, ended=1001.0),
            root.child(
                run_id=f"w{index}",
                agent=f"worker{index}",
                pattern=Pattern.FAN_OUT,
                branch=f"leg{index}",
            ),
        )
        for index in range(3)
    ]
    peer = peer_node(
        root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
        usage=spent(money("0.05"), tokens=40),
        cost=money("0.05"),
        started_at=1000.0,
        ended_at=1003.0,
    )
    return tree([node_of(run("run_1"), root), *workers, peer])


class TestOneTraceOverEveryParticipant:
    def test_every_participant_shares_the_root_the_caller_asked_for(self) -> None:
        """Five traces that share nothing is five questions nobody can join up."""
        assembled = five()
        assert {one.context.root_run_id for one in assembled.nodes} == {"run_1"}
        assert len(assembled.nodes) == 5

    def test_the_tree_says_who_called_whom(self) -> None:
        assembled = five()
        assert [one.context.run_id for one in assembled.children("run_1")] == [
            "w0",
            "w1",
            "w2",
            "p1",
        ]
        assert assembled.node("w0") is not None
        assert assembled.node("nobody") is None

    def test_a_child_carries_its_parent_and_its_depth(self) -> None:
        child = rooted().child(run_id="w0", agent="worker0", pattern=Pattern.DELEGATION)
        assert child.parent_run_id == "run_1"
        assert child.parent_agent == "planner"
        assert child.depth == 1
        assert child.child(run_id="w1", agent="deeper").depth == 2

    def test_a_child_may_run_for_another_tenant_without_widening_the_parents(self) -> None:
        """A run acting on another tenant's behalf bills the tenant it ran as."""
        child = rooted().child(run_id="w0", agent="worker0", tenant="globex")
        assert child.tenant == "globex"
        assert child.root_run_id == "run_1"

    def test_the_span_attributes_say_where_in_the_run_this_happened(self) -> None:
        """Depth, parent, pattern and branch on top of the tenant/agent/model set."""
        attributes = attributes_of_context(
            rooted().child(run_id="w0", agent="worker0", pattern=Pattern.FAN_OUT, branch="leg0")
        )
        assert attributes[f"{ATTRIBUTE_PREFIX}delegation_depth"] == "1"
        assert attributes[f"{ATTRIBUTE_PREFIX}parent_agent"] == "planner"
        assert attributes[f"{ATTRIBUTE_PREFIX}pattern"] == "fan_out"
        assert attributes[f"{ATTRIBUTE_PREFIX}branch"] == "leg0"
        assert attributes[f"{ATTRIBUTE_PREFIX}trace_root"] == "run_1"

    def test_what_the_context_could_not_say_is_bucketed_rather_than_blank(self) -> None:
        attributes = attributes_of_context(rooted())
        assert attributes[f"{ATTRIBUTE_PREFIX}parent_run_id"] == UNKNOWN
        assert attributes[f"{ATTRIBUTE_PREFIX}branch"] == UNKNOWN


class TestTheTraceSurvivesALeavingTheProcess:
    def test_a_context_round_trips_through_a_message_header(self) -> None:
        """A NATS hop carries a header or it carries nothing."""
        sender = rooted().child(
            run_id="w0", agent="worker0", pattern=Pattern.FAN_OUT, branch="leg0"
        )
        restored = TraceContext.restored(sender.carried(), run_id="a1", agent="activity")
        assert restored.root_run_id == "run_1"
        assert restored.parent_run_id == "w0"
        assert restored.parent_agent == "worker0"
        assert restored.depth == 2
        assert restored.broken is False

    def test_a_missing_header_is_recorded_as_a_break_rather_than_dropped(self) -> None:
        """Losing the trace must not lose the work, and must not be silent either."""
        restored = TraceContext.restored({}, run_id="a1", agent="activity")
        assert restored.broken is True
        assert restored.root_run_id == "a1"
        assert attributes_of_context(restored)[f"{ATTRIBUTE_PREFIX}trace_broken"] == "true"

    @pytest.mark.parametrize(
        "value",
        ["", "adk/2 root=r1;run=r1", "nonsense", "adk/1 ", "adk/1 root=;run=r1", "adk/1 depth=1"],
    )
    def test_a_header_this_version_cannot_read_is_a_break_not_a_guess(self, value: str) -> None:
        restored = TraceContext.restored({HEADER: value}, run_id="a1", agent="activity")
        assert restored.broken is True

    def test_a_broker_that_normalises_header_case_is_not_a_broken_trace(self) -> None:
        carried = {HEADER.upper(): rooted().header()}
        assert TraceContext.restored(carried, run_id="a1", agent="activity").broken is False

    def test_a_separator_in_a_name_cannot_rewrite_another_field(self) -> None:
        """Percent-encoded, so an agent called `x;tenant=other` moves nobody's money."""
        sender = rooted().child(run_id="w0", agent="x;tenant=globex")
        restored = TraceContext.restored(sender.carried(), run_id="a1", agent="activity")
        assert restored.parent_agent == "x;tenant=globex"
        assert restored.tenant == "acme"

    def test_a_pattern_a_later_version_added_reads_as_an_activity(self) -> None:
        """An unknown shape is still one hop; refusing to parse it would break the chain."""
        carried = {HEADER: rooted().header().replace("pattern=root", "pattern=telepathy")}
        assert TraceContext.restored(carried, run_id="a1", agent="a").pattern is Pattern.ACTIVITY

    def test_a_break_carries_forward_so_a_later_node_is_not_read_as_whole(self) -> None:
        broken = TraceContext.restored({}, run_id="a1", agent="activity")
        assert broken.child(run_id="a2", agent="deeper").broken is True


class TestTheTotalNeverTreatsUnknownSpendAsZero:
    def test_the_run_total_is_the_sum_of_the_nodes(self) -> None:
        totals = five().totals
        assert totals.cost.total == Decimal("0.55")
        assert totals.nodes == 5
        assert totals.attributed == ("run_1", "w0", "w1", "w2", "p1")
        assert totals.lower_bound is False

    def test_a_worker_that_crashed_before_reporting_is_named_not_counted(self) -> None:
        """Unknown spend counted as zero is a budget ceiling that stops meaning anything."""
        root = rooted()
        crashed = peer_node(root.child(run_id="w0", agent="worker0"))
        totals = tree([node_of(run("run_1"), root), crashed]).totals
        assert totals.unattributed == ("w0",)
        assert totals.lower_bound is True
        assert totals.cost.total == Decimal("0.20")

    def test_tokens_are_totalled_only_from_what_was_reported(self) -> None:
        totals = five().totals
        assert totals.input_tokens == 440
        assert totals.output_tokens == 100

    def test_a_peer_billing_in_another_currency_is_converted_at_a_recorded_rate(self) -> None:
        root = rooted()
        peer = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=spent(money("2.00", "EUR")),
            cost=money("2.00", "EUR"),
            rate=Rate(
                source="treasury",
                recorded_at=1000.0,
                multiplier=Decimal("1.10"),
                of_currency="EUR",
                to_currency="USD",
            ),
        )
        totals = tree([node_of(run("run_1"), root), peer]).totals
        assert totals.cost.total == Decimal("2.40")
        assert totals.converted == ("p1",)
        assert totals.cost.confidence is CostConfidence.ESTIMATED

    def test_a_foreign_figure_with_no_rate_is_held_out_rather_than_summed(self) -> None:
        """A naive sum is a number that is true in neither currency."""
        root = rooted()
        peer = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=spent(money("2.00", "EUR")),
            cost=money("2.00", "EUR"),
        )
        totals = tree([node_of(run("run_1"), root), peer]).totals
        assert totals.unattributed == ("p1",)
        assert totals.cost.total == Decimal("0.20")

    def test_a_rate_for_the_wrong_currency_pair_converts_nothing(self) -> None:
        rate = Rate(
            source="treasury",
            recorded_at=1000.0,
            multiplier=Decimal("1.10"),
            of_currency="GBP",
            to_currency="USD",
        )
        assert rate.applied(money("2.00", "EUR")) is None
        root = rooted()
        peer = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=spent(money("2.00", "EUR")),
            cost=money("2.00", "EUR"),
            rate=rate,
        )
        assert tree([node_of(run("run_1"), root), peer]).totals.unattributed == ("p1",)

    def test_a_rate_that_lands_in_a_third_currency_is_not_used(self) -> None:
        root = rooted()
        peer = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=spent(money("2.00", "EUR")),
            cost=money("2.00", "EUR"),
            rate=Rate(
                source="treasury",
                recorded_at=1000.0,
                multiplier=Decimal("1.10"),
                of_currency="EUR",
                to_currency="GBP",
            ),
        )
        assert tree([node_of(run("run_1"), root), peer]).totals.unattributed == ("p1",)

    def test_a_participant_that_reported_usage_but_no_price_still_counts_its_tokens(self) -> None:
        root = rooted()
        unpriced = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER), usage=spent(None)
        )
        totals = tree([node_of(run("run_1"), root), unpriced]).totals
        assert totals.unattributed == ("p1",)


class TestRollUps:
    def test_spend_rolls_up_per_agent(self) -> None:
        totals = five().totals_by("agent")
        assert totals[("planner",)].cost.total == Decimal("0.20")
        assert totals[("worker0",)].cost.total == Decimal("0.10")
        assert ("peer",) not in totals

    def test_spend_rolls_up_per_model(self) -> None:
        assert five().totals_by("model")[("scripted-1",)].calls == 4

    def test_spend_rolls_up_per_step(self) -> None:
        assert five().by_step()[Step.MODEL].calls == 4

    def test_a_peer_contributes_a_total_rather_than_invented_steps(self) -> None:
        """Steps nobody measured would put a shape on the bill that nobody measured."""
        assert len(five().records()) == 4


class TestATreeThatCannotBeReadIsRefused:
    def test_nothing_to_assemble_is_refused(self) -> None:
        with pytest.raises(AttributionError) as refused:
            tree([])
        assert refused.value.reason == "empty"

    def test_a_first_node_with_a_parent_is_not_a_root(self) -> None:
        root = rooted()
        with pytest.raises(AttributionError) as refused:
            tree([peer_node(root.child(run_id="w0", agent="worker0"))])
        assert refused.value.reason == "no_root"

    def test_a_second_root_is_two_trees_rather_than_one_run(self) -> None:
        with pytest.raises(AttributionError) as refused:
            tree([node_of(run("run_1"), rooted()), node_of(run("run_2"), rooted("run_2"))])
        assert refused.value.reason == "two_roots"

    def test_one_participant_twice_is_a_bill_nobody_can_reconcile(self) -> None:
        root = rooted()
        child = node_of(run("w0", agent="worker0"), root.child(run_id="w0", agent="worker0"))
        with pytest.raises(AttributionError) as refused:
            tree([node_of(run("run_1"), root), child, child])
        assert refused.value.reason == "duplicate"

    def test_a_participant_whose_caller_is_missing_is_refused(self) -> None:
        """A dropped participant is dropped spend, so it is never dropped quietly."""
        orphan = TraceContext(
            root_run_id="run_1", run_id="w9", parent_run_id="gone", agent="worker9", tenant="acme"
        )
        with pytest.raises(AttributionError) as refused:
            tree([node_of(run("run_1"), rooted()), peer_node(orphan)])
        assert refused.value.reason == "orphan"


class TestClocksThatDisagree:
    def test_a_negative_duration_is_recorded_as_skew_and_never_shown(self) -> None:
        """Two processes do not share a clock; a child cannot finish before it started."""
        root = rooted()
        skewed = node_of(
            run("w0", agent="worker0", started=1005.0, ended=1001.0),
            root.child(run_id="w0", agent="worker0"),
        )
        assert skewed.skewed is True
        assert skewed.latency_ms == 0.0
        assert "[clock skew]" in render(tree([node_of(run("run_1"), root), skewed]))

    def test_a_participant_that_reported_no_timings_has_no_latency(self) -> None:
        root = rooted()
        untimed = node_of(
            run("w0", agent="worker0", started=None, ended=None),
            root.child(run_id="w0", agent="worker0"),
        )
        assert untimed.latency_ms == 0.0
        assert untimed.skewed is False

    def test_latency_is_reported_in_milliseconds(self) -> None:
        assert five().root.latency_ms == pytest.approx(2000.0)


class TestExport:
    def test_one_span_per_participant_carries_the_multi_agent_attributes(self) -> None:
        tracer = FakeTracer()
        record_tree(five(), tracer=tracer)
        assert tracer.names() == ["adk.participant"] * 5
        assert tracer.recorded[1].attributes[f"{ATTRIBUTE_PREFIX}branch"] == "leg0"
        assert tracer.recorded[1].attributes[f"{ATTRIBUTE_PREFIX}delegation_depth"] == "1"

    def test_a_participant_that_reported_nothing_exports_as_unknown_not_zero(self) -> None:
        tracer = FakeTracer()
        root = rooted()
        crashed = peer_node(root.child(run_id="w0", agent="worker0"))
        record_tree(tree([node_of(run("run_1"), root), crashed]), tracer=tracer)
        attributes = tracer.recorded[1].attributes
        assert attributes[f"{ATTRIBUTE_PREFIX}cost"] == UNKNOWN
        assert attributes[f"{ATTRIBUTE_PREFIX}input_tokens"] == UNKNOWN
        assert attributes[f"{ATTRIBUTE_PREFIX}reported"] == "false"

    def test_a_wide_fan_out_keeps_its_cost_when_the_trace_is_sampled_away(self) -> None:
        """The money never travels on a span, so a sampler cannot drop it."""
        root = rooted()
        branches = [
            node_of(
                run(f"w{index}", agent="worker", cost="0.01"),
                root.child(
                    run_id=f"w{index}",
                    agent="worker",
                    pattern=Pattern.FAN_OUT,
                    branch=f"leg{index}",
                ),
            )
            for index in range(64)
        ]
        meter, tracer = FakeMeter(), FakeTracer()
        record_tree(
            tree([node_of(run("run_1"), root), *branches]),
            meter=meter,
            tracer=tracer,
            sampled=False,
        )
        assert tracer.recorded == []
        counted = [point for point in meter.points if point.name == COST]
        assert len(counted) == 65
        assert sum(point.value for point in counted) == pytest.approx(0.84)
        assert {point.dimensions["tenant"] for point in counted} == {"acme"}

    def test_counters_say_which_participants_are_outside_the_total(self) -> None:
        meter = FakeMeter()
        root = rooted()
        crashed = peer_node(root.child(run_id="w0", agent="worker0"))
        record_tree(tree([node_of(run("run_1"), root), crashed]), meter=meter)
        outside = [
            point
            for point in meter.points
            if point.name == TOKENS and point.dimensions["attributed"] == "false"
        ]
        assert [point.value for point in outside] == [0.0]

    def test_a_tenant_kept_out_of_the_dimension_set_still_has_its_money_counted(self) -> None:
        meter = FakeMeter()
        record_tree(five(), meter=meter, dimensions=Dimensions(tenants=frozenset({"globex"})))
        counted = [point for point in meter.points if point.name == COST]
        assert {point.dimensions["tenant"] for point in counted} == {"other"}
        assert sum(point.value for point in counted) == pytest.approx(0.55)

    def test_a_participant_without_a_tenant_fails_export_closed(self) -> None:
        """Spend exported with no owner is spend whose owner is gone for good."""
        tracer = FakeTracer()
        root = rooted()
        anonymous = peer_node(
            TraceContext(root_run_id="run_1", run_id="p1", parent_run_id="run_1", agent="peer"),
            usage=spent(money("0.05")),
            cost=money("0.05"),
        )
        with pytest.raises(AttributionError) as refused:
            record_tree(tree([node_of(run("run_1"), root), anonymous]), tracer=tracer)
        assert refused.value.reason == "no_tenant"
        assert tracer.recorded == []

    def test_transferred_context_is_scrubbed_before_it_leaves(self) -> None:
        tracer = FakeTracer()
        record_tree(five(), tracer=tracer, extra={"handoff.note": "card 4111111111111111"})
        assert tracer.names()[0] == "adk.redacted"
        assert "4111111111111111" not in str(tracer.recorded[1].attributes)

    def test_exporting_nowhere_still_returns_what_would_have_gone(self) -> None:
        assert len(record_tree(five())) == 5


class TestTheInspectCommand:
    def _drawn(self, argv: list[str], assembled: RunTree | None) -> tuple[int, str]:
        out = io.StringIO()

        async def lookup(run_id: str) -> RunTree | None:
            return assembled if run_id == "run_1" else None

        code = asyncio.run(inspect_cli.main(argv, lookup=lookup, out=out))
        return code, out.getvalue()

    def test_it_draws_every_participant_with_cost_tokens_and_latency(self) -> None:
        code, drawn = self._drawn(["run_1"], five())
        assert code == 0
        assert "planner (run_1)  root  0.200000 USD, 120 tokens, 2000 ms" in drawn
        assert "  worker0 (w0)  fan_out/leg0  0.100000 USD, 120 tokens, 1000 ms" in drawn
        assert "total 0.550000 USD over 5 participants, 540 tokens" in drawn

    def test_it_says_when_the_total_is_a_lower_bound(self) -> None:
        root = rooted()
        crashed = peer_node(root.child(run_id="w0", agent="worker0"))
        _, drawn = self._drawn(["run_1"], tree([node_of(run("run_1"), root), crashed]))
        assert "unreported" in drawn
        assert "lower bound: w0 reported nothing" in drawn

    def test_it_says_what_reached_the_total_through_a_rate(self) -> None:
        root = rooted()
        peer = peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=spent(money("2.00", "EUR")),
            cost=money("2.00", "EUR"),
            rate=Rate(
                source="treasury",
                recorded_at=1000.0,
                multiplier=Decimal("1.10"),
                of_currency="EUR",
                to_currency="USD",
            ),
        )
        _, drawn = self._drawn(["run_1"], tree([node_of(run("run_1"), root), peer]))
        assert "converted at a recorded rate: p1" in drawn

    def test_it_marks_a_participant_whose_trace_was_broken(self) -> None:
        root = rooted()
        picked_up = TraceContext.restored({}, run_id="a1", agent="activity").model_copy(
            update={"root_run_id": "run_1", "parent_run_id": "run_1"}
        )
        _, drawn = self._drawn(["run_1"], tree([node_of(run("run_1"), root), peer_node(picked_up)]))
        assert "[trace broken]" in drawn

    def test_it_rolls_up_on_request(self) -> None:
        _, drawn = self._drawn(["run_1", "--by", "agent"], five())
        assert "worker0  0.100000 USD  1 calls" in drawn

    def test_an_unknown_run_is_not_a_run_that_spent_nothing(self) -> None:
        code, said = self._drawn(["run_9"], five())
        assert code == 1
        assert "no run is kept under 'run_9'" in said

    def test_a_command_line_it_cannot_read_is_a_misuse(self) -> None:
        code, _ = self._drawn([], five())
        assert code == 2
