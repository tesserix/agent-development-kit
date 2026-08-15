"""Measuring a guard against a corpus, and the gate that stops one weakening quietly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core.content_policy import ContentSeverity, Thresholds
from tesserix_adk.core.guards import GuardrailPipeline, GuardResult, GuardStage, GuardVerdict
from tesserix_adk.guardrails import ContentFilterGuard, Guard, InjectionGuard, PIIGuard
from tesserix_adk.testing import (
    CORPUS_VERSION,
    GUARD_CORPUS,
    GuardCase,
    GuardFamily,
    GuardMetrics,
    GuardrailConformance,
    GuardThresholds,
    RecordedGuard,
    assert_allows,
    assert_blocks,
    assert_fails_closed,
    assert_pipeline_order,
    assert_redacts,
    assert_synthetic,
    measure,
    sampled,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from tesserix_adk.core.protocols import Guardrail

pytestmark = pytest.mark.anyio

INJECTION_BAR = GuardThresholds(recall=0.71, false_positives=0.3, p95_seconds=0.05)
PII_BAR = GuardThresholds(recall=1.0, false_positives=0.0, p95_seconds=0.05)
POLICY_BAR = GuardThresholds(recall=0.75, false_positives=0.0, p95_seconds=0.05)


class Everything(Guard):
    """A guard with perfect recall and no use, which only the control set exposes."""

    name = "everything"

    async def check_input(self, content: str) -> GuardResult:
        del content
        return GuardResult.blocked(code="everything")

    async def check_output(self, content: str) -> GuardResult:
        return await self.check_input(content)


class Nothing(Guard):
    name = "nothing"


class Broken(Guard):
    name = "broken"

    async def check_input(self, content: str) -> GuardResult:
        raise RuntimeError(len(content))

    async def check_output(self, content: str) -> GuardResult:
        return await self.check_input(content)


class Permissive(Guard):
    """A guard that answers allow to everything, including what it could not read."""

    name = "permissive"


class TestTheCorpusItself:
    def test_every_payload_is_synthetic(self) -> None:
        assert_synthetic()

    def test_a_contributed_case_carrying_a_live_key_is_refused(self) -> None:
        case = GuardCase(
            name="a pasted deploy key",
            content="export key=AKIAIOSFODNN7EXAMPLE now",
            family=GuardFamily.PII,
        )

        with pytest.raises(AssertionError, match="live-looking"):
            assert_synthetic((case,))

    def test_a_case_carrying_a_real_address_is_refused(self) -> None:
        case = GuardCase(
            name="a real address",
            content="mail traveller@qantas.com about it",
            family=GuardFamily.PII,
        )

        with pytest.raises(AssertionError, match="real domain"):
            assert_synthetic((case,))

    def test_a_case_carrying_a_card_number_that_checksums_is_refused(self) -> None:
        case = GuardCase(
            name="a real-looking card",
            content="charge 4012 8888 8888 1881 for it",
            family=GuardFamily.PII,
        )

        with pytest.raises(AssertionError, match="Luhn"):
            assert_synthetic((case,))

    def test_it_carries_a_control_set_large_enough_to_price_false_positives(self) -> None:
        benign = [case for case in GUARD_CORPUS if case.family is GuardFamily.BENIGN]

        assert len(benign) >= len(GUARD_CORPUS) / 5

    def test_the_version_is_stated_so_two_recall_figures_can_be_compared(self) -> None:
        assert CORPUS_VERSION


class TestWhatACaseAccepts:
    def test_a_case_wanting_a_block_is_not_satisfied_by_a_redaction(self) -> None:
        case = GuardCase(name="x", content="y", family=GuardFamily.INJECTION)

        assert not case.satisfied_by(GuardVerdict.REDACT)
        assert case.satisfied_by(GuardVerdict.BLOCK)

    def test_a_case_wanting_a_redaction_is_satisfied_by_a_block(self) -> None:
        case = GuardCase(name="x", content="y", family=GuardFamily.PII, expect=GuardVerdict.REDACT)

        assert case.satisfied_by(GuardVerdict.BLOCK)
        assert not case.satisfied_by(GuardVerdict.ALLOW)


class TestMeasuringAGuard:
    async def test_a_guard_is_judged_only_on_the_families_it_claims(self) -> None:
        metrics = await measure(PIIGuard(tenant="acme"), families={GuardFamily.PII})

        assert metrics.recall == 1.0
        assert metrics.adversarial == len(
            [case for case in GUARD_CORPUS if case.family is GuardFamily.PII]
        )

    async def test_the_control_set_runs_whatever_the_guard_claims(self) -> None:
        metrics = await measure(Nothing(), families={GuardFamily.PII})

        assert metrics.control == len(
            [case for case in GUARD_CORPUS if case.family is GuardFamily.BENIGN]
        )

    async def test_a_guard_that_blocks_everything_has_perfect_recall_and_fails_anyway(
        self,
    ) -> None:
        metrics = await measure(Everything(), families={GuardFamily.INJECTION})

        assert metrics.recall == 1.0
        assert metrics.false_positive_rate == 1.0
        assert metrics.failures(GuardThresholds())

    async def test_a_guard_that_allows_everything_names_what_it_let_through(self) -> None:
        metrics = await measure(Nothing(), families={GuardFamily.INJECTION})

        assert metrics.recall == 0.0
        assert "a direct instruction to drop the operator's own" in metrics.missed

    async def test_a_guard_that_raises_is_counted_as_having_refused(self) -> None:
        metrics = await measure(Broken(), families={GuardFamily.INJECTION})

        assert metrics.recall == 1.0
        assert metrics.false_positive_rate == 1.0
        assert len(metrics.errored) == metrics.adversarial + metrics.control

    async def test_the_report_is_machine_readable_and_says_which_corpus(self) -> None:
        report = (await measure(Nothing(), families={GuardFamily.PII})).as_dict()

        assert report["corpus_version"] == CORPUS_VERSION
        assert report["guard"] == "nothing"
        assert isinstance(report["recall"], float)

    async def test_latency_is_reported_so_a_slow_guard_is_visible(self) -> None:
        metrics = await measure(Nothing(), families={GuardFamily.PII})

        assert metrics.p95_seconds >= 0.0


class TestTheGate:
    def test_a_run_that_measured_nothing_never_passes(self) -> None:
        empty = GuardMetrics(
            guard="nothing", corpus_version=CORPUS_VERSION, adversarial=0, control=0
        )

        assert empty.failures(GuardThresholds())

    def test_a_corpus_with_no_case_of_a_guards_family_reports_neither_as_a_score(self) -> None:
        """1.0 and 0.0 here mean "nothing was measured", which `failures` refuses to pass."""
        empty = GuardMetrics(
            guard="nothing", corpus_version=CORPUS_VERSION, adversarial=0, control=0
        )

        assert (empty.recall, empty.false_positive_rate, empty.p95_seconds) == (1.0, 0.0, 0.0)

    def test_a_drop_in_recall_names_the_cases_that_regressed(self) -> None:
        metrics = GuardMetrics(
            guard="injection",
            corpus_version=CORPUS_VERSION,
            adversarial=2,
            control=1,
            missed=("a forged chat turn borrowed from a model's template",),
        )

        reasons = metrics.failures(GuardThresholds(recall=1.0))

        assert any("missed: a forged chat turn" in reason for reason in reasons)

    def test_a_rise_in_false_positives_names_the_benign_cases_refused(self) -> None:
        metrics = GuardMetrics(
            guard="injection",
            corpus_version=CORPUS_VERSION,
            adversarial=1,
            control=2,
            refused=("an ordinary booking request",),
        )

        reasons = metrics.failures(GuardThresholds(recall=0.0))

        assert any("refused: an ordinary booking request" in reason for reason in reasons)

    def test_a_guard_slower_than_its_declared_ceiling_fails(self) -> None:
        metrics = GuardMetrics(
            guard="slow",
            corpus_version=CORPUS_VERSION,
            adversarial=1,
            control=1,
            durations=(2.0, 2.0),
        )

        assert any("p95" in reason for reason in metrics.failures(GuardThresholds(recall=0.0)))

    def test_a_permissive_guard_passes_against_its_own_declared_bar(self) -> None:
        """An internal agent's guard states its own threshold rather than failing a shared one."""
        metrics = GuardMetrics(
            guard="internal", corpus_version=CORPUS_VERSION, adversarial=4, control=4, missed=("a",)
        )

        assert not metrics.failures(GuardThresholds(recall=0.7, p95_seconds=1.0))


class TestSampling:
    def test_the_same_seed_chooses_the_same_cases(self) -> None:
        assert sampled(GUARD_CORPUS, 8, seed="abc123") == sampled(GUARD_CORPUS, 8, seed="abc123")

    def test_a_different_seed_may_choose_differently(self) -> None:
        first = sampled(GUARD_CORPUS, 8, seed="abc123")
        second = sampled(GUARD_CORPUS, 8, seed="def456")

        assert first != second

    def test_a_sample_at_or_above_the_corpus_is_the_whole_corpus(self) -> None:
        assert sampled(GUARD_CORPUS, len(GUARD_CORPUS) + 5, seed="x") == GUARD_CORPUS

    def test_a_sample_keeps_corpus_order_so_reports_read_alike(self) -> None:
        chosen = sampled(GUARD_CORPUS, 6, seed="x")

        assert list(chosen) == [case for case in GUARD_CORPUS if case in chosen]


class TestTheAssertionHelpers:
    async def test_a_block_is_asserted_and_the_code_handed_back(self) -> None:
        result = await assert_blocks(Everything(), "anything", stage=GuardStage.INPUT)

        assert result.code == "everything"

    async def test_an_allow_where_a_block_was_required_is_a_failure(self) -> None:
        with pytest.raises(AssertionError, match="block was required"):
            await assert_blocks(Nothing(), "anything", stage=GuardStage.OUTPUT)

    async def test_a_redaction_is_asserted_on_the_stage_it_happens(self) -> None:
        guard = PIIGuard(tenant="acme")

        result = await assert_redacts(guard, "mail me at a@example.com", stage=GuardStage.OUTPUT)

        assert "example.com" not in (result.content or "")

    async def test_a_block_where_a_redaction_was_required_is_a_failure(self) -> None:
        with pytest.raises(AssertionError, match="redaction was required"):
            await assert_redacts(Everything(), "anything", stage=GuardStage.INPUT)

    async def test_an_allow_is_asserted_for_the_control_set(self) -> None:
        await assert_allows(Nothing(), "find me a seat", stage=GuardStage.INPUT)

    async def test_a_refused_benign_payload_names_the_code_that_refused_it(self) -> None:
        with pytest.raises(AssertionError, match="everything"):
            await assert_allows(Everything(), "find me a seat", stage=GuardStage.INPUT)

    async def test_a_guard_that_raises_counts_as_failing_closed(self) -> None:
        await assert_fails_closed(Broken(), "anything", stage=GuardStage.INPUT)

    async def test_a_guard_that_blocks_counts_as_failing_closed(self) -> None:
        await assert_fails_closed(Everything(), "anything", stage=GuardStage.OUTPUT)

    async def test_a_guard_that_allows_what_it_could_not_read_does_not(self) -> None:
        with pytest.raises(AssertionError, match="could not evaluate"):
            await assert_fails_closed(Permissive(), "anything", stage=GuardStage.INPUT)

    def test_pipeline_order_is_asserted_by_name(self) -> None:
        pipeline = GuardrailPipeline((PIIGuard(tenant="acme"), InjectionGuard()))

        assert_pipeline_order(pipeline, ("pii", "injection"))

    def test_a_reordered_pipeline_is_a_failure_rather_than_a_detail(self) -> None:
        pipeline = GuardrailPipeline((InjectionGuard(), PIIGuard(tenant="acme")))

        with pytest.raises(AssertionError, match="guards run as"):
            assert_pipeline_order(pipeline, ("pii", "injection"))


class TestARemoteClassifierStaysOffline:
    async def test_recorded_verdicts_are_replayed_rather_than_fetched(self) -> None:
        guard = RecordedGuard("remote", {"hello": GuardResult.allow()})

        assert (await guard.check_input("hello")).verdict is GuardVerdict.ALLOW

    async def test_content_nobody_recorded_blocks_rather_than_guessing(self) -> None:
        guard = RecordedGuard("remote", {})

        result = await guard.check_output("never seen")

        assert result.verdict is GuardVerdict.BLOCK
        assert result.code == "not_recorded"

    async def test_a_recording_can_be_measured_like_any_other_guard(self) -> None:
        recorded = {
            case.content: GuardResult.blocked(code="recorded")
            for case in GUARD_CORPUS
            if case.family is GuardFamily.INJECTION
        }
        guard = RecordedGuard("remote", recorded, default=GuardResult.allow())

        metrics = await measure(guard, families={GuardFamily.INJECTION})

        assert metrics.recall == 1.0
        assert metrics.false_positive_rate == 0.0


class TestTheShippedGuardsAgainstTheirDeclaredBars:
    """The CI gate. Raising a bar is a change to this file; lowering one needs a reason."""

    async def test_the_injection_guard_holds_its_bar(self) -> None:
        metrics = await measure(InjectionGuard(), families={GuardFamily.INJECTION})

        assert not metrics.failures(INJECTION_BAR)

    async def test_the_pii_guard_holds_its_bar(self) -> None:
        metrics = await measure(PIIGuard(tenant="acme"), families={GuardFamily.PII})

        assert not metrics.failures(PII_BAR)

    async def test_the_content_filter_holds_its_bar_at_the_severity_it_is_given(self) -> None:
        guard = ContentFilterGuard(thresholds=Thresholds(default=ContentSeverity.MEDIUM))

        metrics = await measure(guard, families={GuardFamily.POLICY})

        assert not metrics.failures(POLICY_BAR)


class TestTheContractSuiteRunsAgainstAnyGuard(GuardrailConformance):
    """The suite a third-party guard inherits, run here against a shipped one."""

    def make_guard(self) -> Guardrail:
        return PIIGuard(tenant="acme")

    def families(self) -> Collection[GuardFamily]:
        return {GuardFamily.PII}

    def thresholds(self) -> GuardThresholds:
        return PII_BAR
