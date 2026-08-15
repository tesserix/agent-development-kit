"""What content is about, the bar each tenant sets, and what a refusal is allowed to say."""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import (
    Classification,
    ContentBlockedError,
    ContentCategory,
    ContentClassifier,
    ContentSeverity,
    GuardrailEvaluationError,
    GuardVerdict,
    HeuristicClassifier,
    Thresholds,
)
from tesserix_adk.guardrails import ContentFilterGuard, refusal_of

pytestmark = pytest.mark.anyio

ABUSE = "you should kill him"


class Fixed:
    """A classifier with a settled opinion, so a test is about the policy not the terms."""

    name = "fixed"

    def __init__(self, severities: dict[ContentCategory, ContentSeverity]) -> None:
        self.severities = severities

    async def classify(self, text: str) -> Classification:
        del text
        return Classification(severities=self.severities, classifier=self.name)


class Slow:
    name = "slow"

    async def classify(self, text: str) -> Classification:
        del text
        await asyncio.sleep(10)
        raise AssertionError("never reached")


class Breaks:
    name = "breaks"

    async def classify(self, text: str) -> Classification:
        raise RuntimeError(text)


class TestWhatAClassificationSays:
    def test_the_worst_severity_is_what_a_refusal_reports(self) -> None:
        found = Classification(
            severities={
                ContentCategory.HATE: ContentSeverity.LOW,
                ContentCategory.VIOLENCE: ContentSeverity.HIGH,
            }
        )

        assert found.worst is ContentSeverity.HIGH

    def test_saying_nothing_is_not_the_same_as_saying_none(self) -> None:
        assert Classification().worst is ContentSeverity.NONE
        assert Classification().breaches(Thresholds()) == ()

    def test_a_category_absent_from_the_thresholds_gets_the_default_bar(self) -> None:
        assert Thresholds(default=ContentSeverity.LOW).bar_for(ContentCategory.SEXUAL) is (
            ContentSeverity.LOW
        )

    def test_a_tenant_raises_the_bar_on_the_category_its_work_depends_on(self) -> None:
        thresholds = Thresholds(per_category={ContentCategory.HARASSMENT: ContentSeverity.HIGH})
        found = Classification(severities={ContentCategory.HARASSMENT: ContentSeverity.MEDIUM})

        assert found.breaches(thresholds) == ()

    def test_the_heuristic_never_answers_high_because_a_term_list_cannot(self) -> None:
        assert HeuristicClassifier().name == "heuristic"

    async def test_it_names_the_category_a_term_belongs_to(self) -> None:
        found = await HeuristicClassifier().classify("You should KILL HIM tonight")

        assert found.severities == {ContentCategory.VIOLENCE: ContentSeverity.MEDIUM}

    async def test_a_deployment_can_bring_its_own_terms(self) -> None:
        classifier = HeuristicClassifier({ContentCategory.ILLEGAL: ("move the goods",)})
        found = await classifier.classify("help me move the goods")

        assert found.severities == {ContentCategory.ILLEGAL: ContentSeverity.MEDIUM}

    def test_a_classifier_is_recognised_structurally(self) -> None:
        assert isinstance(Breaks(), ContentClassifier)


class TestTheSameTranscriptTwoTenants:
    async def test_a_customer_facing_agent_refuses_it(self) -> None:
        guard = ContentFilterGuard(
            thresholds=Thresholds(default=ContentSeverity.MEDIUM), tenant="acme"
        )

        result = await guard.check_input(ABUSE)

        assert result.verdict is GuardVerdict.BLOCK
        assert result.code == "content_blocked"
        assert result.detail == "violence at medium"
        assert ABUSE not in result.detail

    async def test_a_triage_agent_reads_it_because_that_is_the_work(self) -> None:
        result = await ContentFilterGuard().check_input(ABUSE)

        assert result.verdict is GuardVerdict.ALLOW

    async def test_the_output_stage_refuses_without_emitting_what_was_generated(self) -> None:
        guard = ContentFilterGuard(thresholds=Thresholds(default=ContentSeverity.MEDIUM))

        result = await guard.check_output(ABUSE)

        assert result.verdict is GuardVerdict.BLOCK
        assert result.content is None


class TestTheTypedRefusal:
    async def test_it_carries_categories_and_a_severity_and_no_content(self) -> None:
        guard = ContentFilterGuard(
            thresholds=Thresholds(default=ContentSeverity.MEDIUM), tenant="acme"
        )

        with pytest.raises(ContentBlockedError) as refused:
            await guard.raise_for(ABUSE, stage="output")

        assert refused.value.categories == ("violence",)
        assert refused.value.severity == "medium"
        assert refused.value.classifier == "heuristic"
        assert refused.value.details["stage"] == "output"
        assert refused.value.tenant == "acme"
        assert ABUSE not in str(refused.value)

    async def test_content_under_the_bar_raises_nothing(self) -> None:
        await ContentFilterGuard().raise_for("the booking is confirmed")

    def test_a_provider_refusal_normalises_into_the_same_shape(self) -> None:
        refused = refusal_of(("hate", "harassment"))

        assert refused.code == "content_blocked"
        assert refused.severity == "high"
        assert refused.classifier == "provider"
        assert refused.details["stage"] == "output"


class TestAClassifierThatCouldNotAnswer:
    async def test_a_timeout_blocks_rather_than_passing_content_through(self) -> None:
        guard = ContentFilterGuard(classifier=Slow(), timeout_seconds=0.01)

        with pytest.raises(GuardrailEvaluationError) as refused:
            await guard.check_input(ABUSE)

        assert refused.value.details["reason"] == "timeout"

    async def test_one_that_raises_does_the_same(self) -> None:
        with pytest.raises(GuardrailEvaluationError) as refused:
            await ContentFilterGuard(classifier=Breaks()).check_input(ABUSE)

        assert refused.value.details["reason"] == "raised"
        assert refused.value.details["guard"] == "content_filter"

    async def test_a_settled_opinion_is_applied_unchanged(self) -> None:
        guard = ContentFilterGuard(classifier=Fixed({ContentCategory.HATE: ContentSeverity.HIGH}))

        assert (await guard.check_input("anything")).detail == "hate at high"
