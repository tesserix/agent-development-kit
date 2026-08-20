"""Comparing a change against the numbers already in production, and blocking on a regression."""

from __future__ import annotations

import io
import json
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.cli import evals_main
from tesserix_adk.core import Message, NoOutput, Run, RunState, TextPart, Usage
from tesserix_adk.core.cost import Cost
from tesserix_adk.core.errors import BaselineUnusableError, ConfigurationError
from tesserix_adk.core.tenancy import current_tenant
from tesserix_adk.evals import (
    BASELINE_FORMAT,
    Baseline,
    BaselinePolicy,
    Bypass,
    CostPerCase,
    EvalCase,
    EvalSuite,
    ExactMatch,
    MetricReport,
    MetricSnapshot,
    Provenance,
    SuiteRunner,
    Tolerance,
    compare,
    measure,
    promote,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

WAS = Provenance(
    suite="refunds",
    dataset_version="2026-08-01",
    agent_version="1.0.0",
    prompt_version="7",
    model="recorded",
    cassettes="abc123",
)
NOW = WAS.model_copy(update={"agent_version": "1.1.0", "prompt_version": "8"})

CORRECTNESS = Tolerance(metric="exact_match", tolerance=0.02, noise=0.01)
SPEND = Tolerance(metric="cost_per_case", direction="lower_is_better", tolerance=0.001)


def _values(scores: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per-case values for one metric, as a baseline stores them."""
    return {"exact_match": dict(scores)}


def _baseline(
    scores: dict[str, float],
    *,
    provenance: Provenance = WAS,
    quarantined: tuple[str, ...] = (),
) -> Baseline:
    """A baseline whose aggregate is the mean of the per-case values it holds."""
    mean = sum(scores.values()) / len(scores)
    return Baseline(
        provenance=provenance,
        metrics=(
            MetricSnapshot(
                metric="exact_match",
                mean=mean,
                p50=mean,
                p95=mean,
                n=len(scores),
                half_width=0.005,
            ),
        ),
        values=_values(scores),
        quarantined=quarantined,
    )


def _fifty(correct: int) -> dict[str, float]:
    """Fifty cases, the first `correct` of them answered right."""
    return {f"case-{index:02d}": (1.0 if index < correct else 0.0) for index in range(50)}


def _policy(**overrides: object) -> BaselinePolicy:
    """The project's policy: correctness declared, everything else reported only."""
    settings: dict[str, object] = {"tolerances": (CORRECTNESS,), "case_tolerance": 0.0}
    settings.update(overrides)
    return BaselinePolicy(**settings)  # type: ignore[arg-type]


class TestFreezingAMeasuredRun:
    """A report becomes an artefact, provenance and per-case values included."""

    async def test_the_artefact_keeps_every_case_not_only_the_mean(self) -> None:
        report = await _measured()
        frozen = Baseline.of(report, provenance=WAS)
        assert frozen.snapshot("exact_match") is not None
        assert frozen.values["exact_match"]["late"] == 1.0
        assert frozen.values["exact_match"]["wrong"] == 0.0

    async def test_a_provenance_naming_another_dataset_is_refused(self) -> None:
        report = await _measured()
        with pytest.raises(ConfigurationError, match="2026-01-01"):
            Baseline.of(report, provenance=WAS.model_copy(update={"dataset_version": "2026-01-01"}))

    async def test_a_measured_interval_becomes_the_noise_band(self) -> None:
        suite = EvalSuite(
            name="refunds",
            version="2026-08-01",
            cases=tuple(
                EvalCase(id=f"c{index}", input="never arrived", tenant="acme", expected="refunded")
                for index in range(6)
            ),
        )

        async def answer(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            """Right on all but one, so the sample has something to spread over."""
            return _run(run_id, "refunded" if case.id != "c5" else "declined")

        report = measure(suite, await SuiteRunner(answer).run(suite), (ExactMatch(),))
        frozen = Baseline.of(report, provenance=WAS)
        snapshot = frozen.snapshot("exact_match")
        assert snapshot is not None
        assert snapshot.half_width > 0.0

    async def test_an_unpriced_case_stays_unknown_rather_than_free(self) -> None:
        report = await _measured()
        frozen = Baseline.of(report, provenance=WAS)
        spend = frozen.snapshot("cost_per_case")
        assert spend is not None
        assert spend.unknown == 1
        assert spend.currency == "USD"

    def test_it_survives_a_round_trip_through_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "baseline.json"
        _baseline({"a": 1.0, "b": 0.0}).write(path)
        assert Baseline.read(path).values["exact_match"]["a"] == 1.0

    def test_two_snapshots_for_one_metric_are_refused(self) -> None:
        row = MetricSnapshot(metric="exact_match", mean=1.0)
        with pytest.raises(ConfigurationError, match="one snapshot per metric"):
            Baseline(provenance=WAS, metrics=(row, row))


class TestReadingSomethingThatIsNotABaseline:
    """The gate fails closed on the artefact itself, and says how to make one."""

    def test_a_missing_baseline_refuses_and_names_the_bootstrap(self, tmp_path: Path) -> None:
        with pytest.raises(BaselineUnusableError) as refused:
            Baseline.read(tmp_path / "absent.json")
        assert refused.value.reason == "missing"
        assert "bootstrap" in refused.value.remedy
        assert refused.value.retryable is False

    def test_a_file_that_is_not_json_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(BaselineUnusableError) as refused:
            Baseline.read(path)
        assert refused.value.reason == "format"

    def test_another_tools_json_is_not_mistaken_for_a_baseline(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"metrics": {"exact_match": 0.9}}), encoding="utf-8")
        with pytest.raises(BaselineUnusableError, match=f"format {BASELINE_FORMAT}"):
            Baseline.read(path)

    def test_a_baseline_from_a_newer_kit_is_not_guessed_at(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        document = json.loads(_baseline({"a": 1.0}).model_dump_json())
        document["format"] = BASELINE_FORMAT + 1
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BaselineUnusableError) as refused:
            Baseline.read(path)
        assert refused.value.reason == "format"

    def test_a_baseline_missing_its_provenance_refuses(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"format": BASELINE_FORMAT}), encoding="utf-8")
        with pytest.raises(BaselineUnusableError) as refused:
            Baseline.read(path)
        assert refused.value.reason == "format"


class TestTheGate:
    """Six cases out of fifty get worse, and the pull request says which six."""

    def test_a_real_regression_fails(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
        )
        assert report.verdict == "fail"
        assert report.exit_code == 1
        assert report.ok is False

    def test_it_names_the_cases_a_reviewer_has_to_open(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
        )
        worse = [case.case_id for case in report.failing()]
        assert worse == [f"case-{index:02d}" for index in range(44, 50)]

    def test_each_delta_carries_the_band_it_was_judged_against(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
        )
        delta = next(each for each in report.deltas if each.metric == "exact_match")
        assert delta.regression == pytest.approx(0.12)
        assert delta.tolerance == 0.02
        assert delta.band == pytest.approx(0.01)
        assert "beyond" in delta.reason

    def test_the_comment_links_each_failing_case_to_its_artefacts(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
        )
        comment = report.comment(artefacts="https://ci.example/run/12/")
        assert "FAIL" in comment
        assert "| exact_match |" in comment
        assert "https://ci.example/run/12/refunds/case-44" in comment
        assert "prompt_version" in comment

    def test_an_improvement_passes(self) -> None:
        report = compare(
            _baseline(_fifty(44)),
            _baseline(_fifty(50), provenance=NOW),
            policy=_policy(),
        )
        assert report.verdict == "pass"
        assert report.exit_code == 0
        assert report.failing() == ()

    def test_a_clean_pass_lists_no_cases_to_open(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(50), provenance=NOW),
            policy=_policy(),
        )
        assert "Cases that got worse" not in report.comment()
        assert "Override" not in report.comment()

    def test_the_summary_says_what_changed_about_the_run(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
        )
        assert "agent_version" in report.summary()
        assert "6 case(s) worse" in report.summary()


class TestOrdinaryVariance:
    """A number that moved inside the noise is not reported as a regression."""

    def test_a_move_inside_the_band_warns_rather_than_fails(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(48), provenance=NOW),
            policy=_policy(
                tolerances=(Tolerance(metric="exact_match", tolerance=0.02, noise=0.05),)
            ),
        )
        assert report.verdict == "warn"
        assert report.exit_code == 0
        assert "noise" in report.deltas[0].reason

    def test_a_move_inside_the_tolerance_passes_clean(self) -> None:
        report = compare(
            _baseline({"a": 1.0, "b": 1.0}),
            _baseline({"a": 1.0, "b": 1.0}, provenance=NOW),
            policy=_policy(),
        )
        assert report.verdict == "pass"
        assert report.deltas[0].reason == ""

    def test_the_baselines_own_interval_widens_the_band(self) -> None:
        wide = _baseline(_fifty(50))
        wide = wide.model_copy(
            update={
                "metrics": (wide.metrics[0].model_copy(update={"half_width": 0.2}),),
            }
        )
        report = compare(wide, _baseline(_fifty(44), provenance=NOW), policy=_policy())
        assert report.verdict == "warn"


class TestComparisonsThatCannotMeanAnything:
    """Two runs that are not about the same thing refuse rather than average."""

    def test_a_dataset_edited_in_the_same_change_is_refused(self) -> None:
        moved = NOW.model_copy(update={"dataset_version": "2026-08-02"})
        with pytest.raises(BaselineUnusableError) as refused:
            compare(
                _baseline(_fifty(50)),
                _baseline(_fifty(44), provenance=moved),
                policy=_policy(),
            )
        assert refused.value.reason == "dataset"
        assert "cannot be told apart" in str(refused.value)

    def test_another_suite_is_refused(self) -> None:
        other = NOW.model_copy(update={"suite": "billing"})
        with pytest.raises(BaselineUnusableError) as refused:
            compare(
                _baseline(_fifty(50)),
                _baseline(_fifty(50), provenance=other),
                policy=_policy(),
            )
        assert refused.value.reason == "suite"

    def test_a_declared_metric_nobody_measured_fails_the_gate(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(50), provenance=NOW),
            policy=BaselinePolicy(tolerances=(SPEND,)),
        )
        assert report.verdict == "fail"
        assert "cannot clear it" in report.deltas[0].reason

    def test_a_case_that_stopped_being_measured_is_a_regression(self) -> None:
        gone = _baseline({"a": 1.0, "b": 1.0}, provenance=NOW)
        gone = gone.model_copy(update={"values": {"exact_match": {"a": 1.0}}})
        report = compare(_baseline({"a": 1.0, "b": 1.0}), gone, policy=_policy())
        assert [case.case_id for case in report.failing()] == ["b"]
        assert report.failing()[0].after is None

    def test_two_tolerances_for_one_metric_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="only one tolerance"):
            BaselinePolicy(tolerances=(CORRECTNESS, CORRECTNESS))


class TestFlakyCases:
    """A case known to flake is reported every time and blocks nothing."""

    def test_a_quarantined_case_does_not_fail_the_gate(self) -> None:
        report = compare(
            _baseline({"steady": 1.0, "flaky": 1.0}),
            _baseline({"steady": 1.0, "flaky": 0.0}, provenance=NOW),
            policy=_policy(quarantined=("flaky",)),
        )
        assert report.verdict == "pass"
        assert report.exit_code == 0

    def test_it_is_still_reported_so_the_flake_can_be_fixed(self) -> None:
        report = compare(
            _baseline({"steady": 1.0, "flaky": 1.0}),
            _baseline({"steady": 1.0, "flaky": 0.0}, provenance=NOW),
            policy=_policy(quarantined=("flaky",)),
        )
        assert [case.case_id for case in report.regressions] == ["flaky"]
        assert report.regressions[0].quarantined is True
        assert report.failing() == ()
        assert "quarantined" in report.comment()

    def test_the_baseline_may_carry_the_quarantine_itself(self) -> None:
        report = compare(
            _baseline({"steady": 1.0, "flaky": 1.0}, quarantined=("flaky",)),
            _baseline({"steady": 1.0, "flaky": 0.0}, provenance=NOW),
            policy=_policy(),
        )
        assert report.verdict == "pass"
        assert report.regressions[0].quarantined is True

    def test_a_real_regression_beside_a_flake_still_fails(self) -> None:
        report = compare(
            _baseline({"steady": 1.0, "flaky": 1.0}),
            _baseline({"steady": 0.0, "flaky": 0.0}, provenance=NOW),
            policy=_policy(quarantined=("flaky",)),
        )
        assert report.verdict == "fail"
        assert [case.case_id for case in report.failing()] == ["steady"]


class TestExploratoryMetrics:
    """A metric on the warn-only list reports and never blocks."""

    def test_it_warns_instead_of_failing(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(30), provenance=NOW),
            policy=_policy(warn_only=("exact_match",)),
        )
        assert report.verdict == "warn"
        assert report.exit_code == 0
        assert report.deltas[0].regression == pytest.approx(0.4)


class TestAPriceChangeWithNoAgentChange:
    """Nothing about the run moved, so the number moved somewhere else."""

    def test_the_reason_points_away_from_the_pull_request(self) -> None:
        before = Baseline(
            provenance=WAS,
            metrics=(MetricSnapshot(metric="cost_per_case", mean=0.01, currency="USD"),),
        )
        after = Baseline(
            provenance=WAS,
            metrics=(MetricSnapshot(metric="cost_per_case", mean=0.05, currency="USD"),),
        )
        report = compare(before, after, policy=BaselinePolicy(tolerances=(SPEND,)))
        assert report.verdict == "fail"
        assert report.moved == ()
        assert "price list" in report.deltas[0].reason


class TestARecordedOverride:
    """An exception is taken in the pull request, with a name on it."""

    def test_it_turns_a_failure_into_a_warning(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
            override=Bypass(
                metrics=("exact_match",), by="sam", reason="upstream provider incident"
            ),
        )
        assert report.verdict == "warn"
        assert report.exit_code == 0

    def test_it_is_visible_in_the_comment_rather_than_in_config(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(),
            override=Bypass(
                metrics=("exact_match",),
                by="sam",
                reason="upstream provider incident",
                incident="INC-12",
            ),
        )
        comment = report.comment()
        assert "Override" in comment
        assert "@sam" in comment
        assert "INC-12" in comment

    def test_an_override_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="who took it and why"):
            Bypass(metrics=("exact_match",), by="sam")

    def test_it_excuses_only_the_metrics_it_names(self) -> None:
        report = compare(
            _baseline(_fifty(50)),
            _baseline(_fifty(44), provenance=NOW),
            policy=_policy(tolerances=(CORRECTNESS,)),
            override=Bypass(metrics=("cost_per_case",), by="sam", reason="unrelated"),
        )
        assert report.verdict == "fail"


class TestPromotion:
    """The merged run becomes the baseline, and the one it replaced is kept."""

    def test_the_first_promotion_has_nothing_to_keep(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        assert promote(_baseline({"a": 1.0}), path) is None
        assert Baseline.read(path).provenance.agent_version == "1.0.0"

    def test_the_previous_baseline_is_kept_for_a_rollback(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        promote(_baseline({"a": 1.0}), path)
        kept = promote(_baseline({"a": 0.5}, provenance=NOW), path)
        assert kept is not None
        assert Baseline.read(kept).provenance.agent_version == "1.0.0"
        assert Baseline.read(path).provenance.agent_version == "1.1.0"


class TestTheCommandLine:
    """What CI actually runs, and the exit codes it reads."""

    def test_bootstrap_records_the_first_baseline(self, tmp_path: Path) -> None:
        candidate = tmp_path / "candidate.json"
        _baseline({"a": 1.0}).write(candidate)
        path = tmp_path / "baseline.json"
        out = io.StringIO()
        command = ["bootstrap", "--candidate", str(candidate), "--baseline", str(path)]
        assert evals_main(command, out=out) == 0
        assert Baseline.read(path).provenance.suite == "refunds"
        assert "recorded" in out.getvalue()

    def test_compare_exits_one_on_a_regression(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        code = evals_main(
            ["compare", "--baseline", paths[0], "--candidate", paths[1], "--policy", paths[2]],
            out=out,
        )
        assert code == 1
        assert "case-44" in out.getvalue()

    def test_compare_writes_the_comment_a_workflow_posts(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        comment = tmp_path / "comment.md"
        evals_main(
            [
                "compare",
                "--baseline",
                paths[0],
                "--candidate",
                paths[1],
                "--policy",
                paths[2],
                "--comment",
                str(comment),
                "--artefacts",
                "https://ci.example/run/12/",
            ],
            out=io.StringIO(),
        )
        assert "https://ci.example/run/12/refunds/case-44" in comment.read_text(encoding="utf-8")

    def test_a_missing_baseline_exits_with_its_own_code(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(50))
        out = io.StringIO()
        code = evals_main(
            [
                "compare",
                "--baseline",
                str(tmp_path / "absent.json"),
                "--candidate",
                paths[1],
                "--policy",
                paths[2],
            ],
            out=out,
        )
        assert code == 3
        assert "bootstrap" in out.getvalue()

    def test_promote_keeps_the_previous_baseline(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        command = ["promote", "--candidate", paths[1], "--baseline", paths[0]]
        assert evals_main(command, out=out) == 0
        assert "previous" in out.getvalue()

    def test_bootstrap_refuses_to_overwrite_without_being_told_to(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        command = ["bootstrap", "--candidate", paths[1], "--baseline", paths[0]]
        assert evals_main(command, out=out) == 3
        assert "--force" in out.getvalue()
        assert evals_main([*command, "--force"], out=io.StringIO()) == 0

    def test_a_policy_that_declares_nothing_blocks_nothing(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        command = ["compare", "--baseline", paths[0], "--candidate", paths[1]]
        assert evals_main(command, out=out) == 0

    def test_a_missing_policy_refuses_rather_than_judging_nothing(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        code = evals_main(
            [
                "compare",
                "--baseline",
                paths[0],
                "--candidate",
                paths[1],
                "--policy",
                str(tmp_path / "absent.json"),
            ],
            out=out,
        )
        assert code == 3
        assert "no policy" in out.getvalue()

    def test_an_override_is_recorded_on_the_command_line(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        code = evals_main(
            [
                "compare",
                "--baseline",
                paths[0],
                "--candidate",
                paths[1],
                "--policy",
                paths[2],
                "--override",
                "exact_match",
                "--override-by",
                "sam",
                "--override-reason",
                "provider incident",
            ],
            out=out,
        )
        assert code == 0
        assert "@sam" in out.getvalue()

    def test_an_override_without_a_reason_refuses(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        code = evals_main(
            [
                "compare",
                "--baseline",
                paths[0],
                "--candidate",
                paths[1],
                "--policy",
                paths[2],
                "--override",
                "exact_match",
            ],
            out=out,
        )
        assert code == 3
        assert "who took it and why" in out.getvalue()

    def test_an_unreadable_command_line_is_not_a_pass(self) -> None:
        assert evals_main(["compare"], out=io.StringIO()) == 2

    def test_json_output_is_what_a_workflow_reads(self, tmp_path: Path) -> None:
        paths = _pair(tmp_path, _fifty(50), _fifty(44))
        out = io.StringIO()
        evals_main(
            [
                "compare",
                "--baseline",
                paths[0],
                "--candidate",
                paths[1],
                "--policy",
                paths[2],
                "--json",
            ],
            out=out,
        )
        document = json.loads(out.getvalue())
        assert document["verdict"] == "fail"
        assert document["regressions"][0]["case_id"] == "case-44"


class TestTheReusableWorkflow:
    """The workflow a consumer calls, checked in rather than described in prose."""

    def test_it_calls_the_gate_and_can_comment_on_the_pull_request(self) -> None:
        from pathlib import Path as _Path

        workflow = _Path(".github/workflows/eval-gate.yml").read_text(encoding="utf-8")
        assert "workflow_call" in workflow
        assert "pull-requests: write" in workflow
        assert "python -m tesserix_adk.cli.evals" in workflow


class TestTheMeasuredReport:
    """`measure` keeps the per-case values a baseline needs to name a failing case."""

    async def test_it_reports_every_case_not_only_the_mean(self) -> None:
        report = await _measured()
        assert report.values["exact_match"]["late"] == 1.0
        assert report.as_dict()["values"]["exact_match"]["wrong"] == 0.0

    async def test_an_unknown_value_is_absent_rather_than_zero(self) -> None:
        report = await _measured()
        assert "unpriced" not in report.values["cost_per_case"]


SUITE = EvalSuite(
    name="refunds",
    version="2026-08-01",
    cases=(
        EvalCase(id="late", input="never arrived", tenant="acme", expected="refunded"),
        EvalCase(id="wrong", input="wrong item", tenant="acme", expected="refunded"),
        EvalCase(id="unpriced", input="where is it", tenant="beta", expected="tomorrow"),
    ),
)

ANSWERS = {"late": "refunded", "wrong": "declined", "unpriced": "tomorrow"}


def _run(run_id: str, answer: str, *, priced: Cost | None = None) -> Run[NoOutput]:
    """A completed run carrying one answer, as a recording replays it."""
    return Run[NoOutput](
        id=run_id,
        tenant=current_tenant().tenant,
        agent_name="support",
        agent_version="1.0.0",
        model="recorded",
        state=RunState.COMPLETED,
        messages=[Message(role="assistant", content=[TextPart(text=answer)])],
        usage=Usage(input_tokens=800, output_tokens=40, cost=priced),
        started_at=0.0,
        ended_at=0.9,
    )


async def _replay(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
    """Answer from a recording, the self-hosted case deliberately unpriced."""
    priced = None if case.id == "unpriced" else Cost(output=Decimal("0.05"), currency="USD")
    return _run(run_id, ANSWERS[case.id], priced=priced)


async def _measured() -> MetricReport:
    """Run the little suite and measure it, which is what a consumer's harness does."""
    outcome = await SuiteRunner(_replay).run(SUITE)
    return measure(SUITE, outcome, (ExactMatch(), CostPerCase()))


def _pair(
    tmp_path: Path, before: dict[str, float], after: dict[str, float]
) -> tuple[str, str, str]:
    """A baseline, a candidate and a policy on disk, as CI passes them."""
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    _baseline(before).write(baseline)
    _baseline(after, provenance=NOW).write(candidate)
    policy.write_text(_policy(tolerances=(CORRECTNESS,)).model_dump_json(), encoding="utf-8")
    return str(baseline), str(candidate), str(policy)
