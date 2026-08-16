"""Reading a failed run without a collector, and sharing the file it came from."""

from __future__ import annotations

import json

import pytest

from tesserix_adk.core import MASK
from tesserix_adk.observability import (
    FILE_VERSION,
    RecordedSpan,
    TraceFile,
    assembled,
    machine_readable,
    rendered,
)


def _span(
    span_id: str,
    *,
    parent: str | None = None,
    name: str = "adk.tool",
    started: float = 0.0,
    ended: float = 1.0,
    attributes: dict[str, str] | None = None,
) -> RecordedSpan:
    return RecordedSpan(
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        started=started,
        ended=ended,
        attributes={"adk.tenant": "acme", **(attributes or {})},
    )


def _timed_out() -> tuple[RecordedSpan, ...]:
    """A tool that timed out after two retries, which is the run somebody debugs."""
    return (
        _span("root", name="adk.run", ended=9.0),
        _span("call-1", parent="root", started=0.0, ended=3.0, attributes={"adk.attempt": "1"}),
        _span("call-2", parent="root", started=3.0, ended=6.0, attributes={"adk.attempt": "2"}),
        _span(
            "call-3",
            parent="root",
            started=6.0,
            ended=9.0,
            attributes={
                "adk.attempt": "3",
                "adk.error.type": "ToolTimeout",
                "adk.outcome": "failed",
            },
        ),
    )


class TestTheTree:
    def test_a_run_renders_as_a_tree_of_the_steps_it_took(self) -> None:
        drawn = rendered(assembled(_timed_out()))
        assert "adk.run" in drawn
        assert drawn.count("adk.tool") == 3

    def test_a_child_is_drawn_under_the_step_that_called_it(self) -> None:
        drawn = rendered(assembled(_timed_out())).splitlines()
        assert drawn[0].startswith("adk.run")
        assert drawn[1].startswith(" ")

    def test_steps_are_drawn_in_the_order_they_ran(self) -> None:
        spans = (_span("root", name="adk.run"), _span("b", parent="root", started=5.0))
        late = _span("a", parent="root", started=1.0)
        drawn = json.loads(machine_readable(assembled((*spans, late))))
        assert [child["span_id"] for child in drawn[0]["children"]] == ["a", "b"]

    def test_each_step_carries_how_long_it_took(self) -> None:
        assert "3.000s" in rendered(assembled(_timed_out()))

    def test_tokens_and_cost_are_shown_where_a_step_reported_them(self) -> None:
        span = _span(
            "root", name="adk.model", attributes={"adk.input_tokens": "120", "adk.cost": "0.004"}
        )
        drawn = rendered(assembled((span,)))
        assert "120" in drawn
        assert "0.004" in drawn

    def test_a_step_that_reported_no_cost_is_not_drawn_as_free(self) -> None:
        """A zero somebody sums is worse than a gap somebody notices."""
        drawn = rendered(assembled((_span("root", name="adk.model"),)))
        assert "0.00" not in drawn


class TestAFailedRun:
    def test_the_failure_is_visible_rather_than_a_tidy_successful_tree(self) -> None:
        drawn = rendered(assembled(_timed_out()))
        assert "ToolTimeout" in drawn
        assert "failed" in drawn

    def test_every_retry_attempt_is_drawn_not_only_the_last(self) -> None:
        drawn = rendered(assembled(_timed_out()))
        assert drawn.count("attempt") == 3

    def test_the_step_the_run_stopped_at_is_marked(self) -> None:
        drawn = rendered(assembled(_timed_out()))
        stopped = next(line for line in drawn.splitlines() if "ToolTimeout" in line)
        assert stopped.strip().startswith("!")

    def test_a_guard_verdict_is_shown_where_one_was_recorded(self) -> None:
        span = _span("root", name="adk.guard", attributes={"adk.verdict": "blocked"})
        assert "blocked" in rendered(assembled((span,)))

    def test_a_cancelled_run_says_where_the_budget_stopped_it(self) -> None:
        span = _span(
            "root", name="adk.run", attributes={"adk.outcome": "cancelled", "adk.budget": "spent"}
        )
        drawn = rendered(assembled((span,)))
        assert "cancelled" in drawn
        assert "spent" in drawn


class TestABrokenTree:
    def test_a_span_whose_parent_is_missing_is_still_drawn(self) -> None:
        """Dropping it hides the step somebody is looking for."""
        drawn = rendered(assembled((_span("orphan", parent="never-arrived"),)))
        assert "orphan" in drawn or "adk.tool" in drawn

    def test_an_orphan_is_marked_rather_than_silently_reparented(self) -> None:
        assert "?" in rendered(assembled((_span("orphan", parent="never-arrived"),)))

    def test_a_cycle_does_not_hang_the_renderer(self) -> None:
        spans = (_span("a", parent="b"), _span("b", parent="a"))
        assert rendered(assembled(spans))

    def test_a_cycle_draws_each_step_once_rather_than_from_both_ends(self) -> None:
        spans = (_span("a", parent="b"), _span("b", parent="a"))
        assert len(json.loads(machine_readable(assembled(spans)))) == 1

    def test_an_empty_trace_says_so_rather_than_rendering_nothing(self) -> None:
        assert "no spans" in rendered(assembled(()))


class TestFilters:
    def test_depth_can_be_capped_for_a_wide_trace(self) -> None:
        spans = (
            _span("root", name="adk.run"),
            _span("mid", parent="root", name="adk.tool"),
            _span("leaf", parent="mid", name="adk.memory"),
        )
        assert "adk.memory" not in rendered(assembled(spans), depth=2)

    def test_a_truncated_branch_says_what_was_hidden(self) -> None:
        spans = (
            _span("root", name="adk.run"),
            _span("mid", parent="root", name="adk.tool"),
            _span("leaf", parent="mid", name="adk.memory"),
        )
        assert "1 hidden" in rendered(assembled(spans), depth=2)

    def test_a_trace_can_be_narrowed_to_one_kind_of_step(self) -> None:
        spans = (_span("root", name="adk.run"), _span("tool", parent="root", name="adk.tool"))
        drawn = rendered(assembled(spans), only=("adk.tool",))
        assert "adk.tool" in drawn

    def test_a_failure_survives_a_filter_that_would_have_hidden_it(self) -> None:
        """A filtered view that looks clean is how a failure gets missed."""
        drawn = rendered(assembled(_timed_out()), only=("adk.model",))
        assert "ToolTimeout" in drawn


class TestMachineReadable:
    def test_a_trace_can_be_read_by_a_test_assertion_rather_than_an_eye(self) -> None:
        document = json.loads(machine_readable(assembled(_timed_out())))
        assert document[0]["name"] == "adk.run"

    def test_the_machine_form_keeps_the_shape_of_the_tree(self) -> None:
        document = json.loads(machine_readable(assembled(_timed_out())))
        assert len(document[0]["children"]) == 3

    def test_the_machine_form_carries_the_failure_too(self) -> None:
        document = json.loads(machine_readable(assembled(_timed_out())))
        assert document[0]["children"][2]["attributes"]["adk.error.type"] == "ToolTimeout"


class TestASavedFile:
    def test_a_saved_file_states_the_version_that_produced_it(self) -> None:
        saved = TraceFile.of(_timed_out())
        assert saved.version == FILE_VERSION
        assert saved.redaction.dropped == ()

    def test_a_file_round_trips_through_json(self) -> None:
        saved = TraceFile.of(_timed_out())
        assert TraceFile.model_validate_json(saved.model_dump_json()) == saved

    def test_a_secret_does_not_survive_into_a_file_meant_to_be_shared(self) -> None:
        leaky = _span("root", attributes={"http.authorization": "Bearer opaque"})
        saved = TraceFile.of((leaky,))
        assert "opaque" not in saved.model_dump_json()

    def test_an_email_in_a_step_is_redacted_before_the_file_is_written(self) -> None:
        leaky = _span("root", attributes={"tool.note": "chase ada@example.com"})
        saved = TraceFile.of((leaky,))
        assert MASK in saved.spans[0].attributes["tool.note"]

    def test_a_file_says_which_attributes_it_dropped(self) -> None:
        leaky = _span("root", attributes={"http.authorization": "Bearer opaque"})
        assert TraceFile.of((leaky,)).redaction.dropped == ("http.authorization",)

    def test_a_file_from_a_colleague_renders_without_the_run_being_present(self) -> None:
        shared = TraceFile.model_validate_json(TraceFile.of(_timed_out()).model_dump_json())
        assert "ToolTimeout" in rendered(assembled(shared.spans))

    def test_a_file_written_by_a_newer_version_is_refused_rather_than_misread(self) -> None:
        document = json.loads(TraceFile.of(_timed_out()).model_dump_json())
        document["version"] = "adk-trace/99"
        with pytest.raises(ValueError, match="version"):
            TraceFile.model_validate(document)


class TestOneSourceOfTruth:
    def test_the_local_view_reads_the_attributes_production_exports(self) -> None:
        """A local renderer with its own attribute set drifts and then lies."""
        span = _span("root", name="adk.run")
        assert "adk.tenant" in machine_readable(assembled((span,)))

    def test_a_span_that_went_through_the_export_processor_still_renders(self) -> None:
        saved = TraceFile.of(_timed_out())
        assert rendered(assembled(saved.spans))
