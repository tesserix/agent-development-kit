"""Measuring what compression saves and what it costs, in the same run."""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import ConfigurationError, EvalIncompleteError
from tesserix_adk.evals import (
    DEFAULT_FIXTURES,
    DEFAULT_FLOORS,
    Answer,
    CaseMeasurement,
    CompressionCase,
    CompressionFixtures,
    CompressionReport,
    Floor,
    FloorPolicy,
    measure_compression,
)
from tesserix_adk.memory import ContentKind, ContentRouter, ReversibleRouter
from tesserix_adk.runtime import MemoryClaimCheckStore

pytestmark = pytest.mark.anyio


class TestTheFixtureSet:
    """Deterministic, synthetic and spanning every kind the router routes."""

    def test_it_spans_the_routed_content_types(self) -> None:
        kinds = {case.kind for case in DEFAULT_FIXTURES.cases}
        assert kinds == {
            ContentKind.JSON,
            ContentKind.TABULAR,
            ContentKind.CODE,
            ContentKind.PROSE,
            ContentKind.UNKNOWN,
        }

    def test_every_case_declares_what_a_right_answer_contains(self) -> None:
        assert all(case.expected for case in DEFAULT_FIXTURES.cases)

    def test_the_same_fixtures_come_back_every_time(self) -> None:
        assert DEFAULT_FIXTURES.model_dump_json() == DEFAULT_FIXTURES.model_dump_json()

    def test_two_cases_sharing_an_id_are_refused(self) -> None:
        case = CompressionCase(
            id="one", kind=ContentKind.PROSE, content="a " * 400, question="?", expected="a"
        )
        with pytest.raises(ConfigurationError, match="one case per id"):
            CompressionFixtures(name="f", version="1", cases=(case, case))


def _router(threshold: int = 8) -> ReversibleRouter:
    """A reversible router over the default compressors, so handles exist to retrieve."""
    return ReversibleRouter(ContentRouter(threshold_tokens=threshold), MemoryClaimCheckStore())


class _Reader:
    """A solver that answers from what it was given, and retrieves when it must.

    It stands in for a model: it can only answer from the text in front of it, so a
    compressor that removed the answer is measured as having removed it.
    """

    def __init__(self, *, retrieves: bool = True) -> None:
        self.retrieves = retrieves
        self.seen: list[str] = []

    async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
        """Read the admitted content, expanding the handle where the answer is not in it."""
        self.seen.append(case.id)
        if case.expected in content:
            return Answer(text=case.expected)
        if handle and self.retrieves:
            return Answer(text=case.expected, expanded=(handle,))
        return Answer(text="I could not find it")


class TestMeasuringBothAtOnce:
    """Tokens before, tokens after, and whether the task still came out right."""

    async def test_it_reports_savings_and_accuracy_per_content_type(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        for kind in (ContentKind.JSON, ContentKind.TABULAR, ContentKind.CODE, ContentKind.PROSE):
            measured = report.kind(kind)
            assert measured.n > 0
            assert 0.0 <= measured.accuracy <= 1.0
            assert measured.savings >= 0.0

    async def test_a_faithful_compressor_passes(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        assert report.ok is True
        assert report.exit_code == 0

    async def test_every_case_says_what_it_cost(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        one = report.case(DEFAULT_FIXTURES.cases[0].id)
        assert one.tokens_before > one.tokens_after
        assert one.compressor
        assert one.outcome in {"kept", "lost", "recovered"}


class TestAnAggregateImprovementNeverMasksARegression:
    """The failure this gate exists for: a better ratio bought with accuracy."""

    async def test_one_content_type_below_its_floor_fails_the_build(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES,
            _router(),
            _Reader(retrieves=False),
            floors=DEFAULT_FLOORS,
        )
        assert report.ok is False
        assert report.exit_code == 1

    async def test_it_names_the_content_type_and_the_cases(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES,
            _router(),
            _Reader(retrieves=False),
            floors=DEFAULT_FLOORS,
        )
        failing = {measured.kind for measured in report.failing()}
        assert failing
        for measured in report.failing():
            assert measured.cases
            assert "floor" in measured.reason

    async def test_the_summary_says_which_type_rather_than_only_the_total(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES,
            _router(),
            _Reader(retrieves=False),
            floors=DEFAULT_FLOORS,
        )
        assert report.summary().startswith("compression")
        assert any(measured.kind.value in report.summary() for measured in report.failing())


class TestRetrievalIsAnOutcomeNotAnExcuse:
    """A case whose answer needs the elided detail must actually go and get it."""

    async def test_answering_without_retrieving_is_a_failure_even_when_right(self) -> None:
        class Lucky:
            """Answers correctly from memory, never expanding the handle."""

            async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
                del content, handle
                return Answer(text=case.expected)

        fixtures = CompressionFixtures(
            name="detail",
            version="1",
            cases=(_needing_detail(),),
        )
        report = await measure_compression(fixtures, _router(), Lucky(), floors=_floors())
        measured = report.case("elided")
        assert measured.correct is True
        assert measured.retrieved is False
        assert measured.outcome == "lost"
        assert report.ok is False

    async def test_retrieving_the_original_recovers_the_case(self) -> None:
        fixtures = CompressionFixtures(name="detail", version="1", cases=(_needing_detail(),))
        report = await measure_compression(fixtures, _router(), _Reader(), floors=_floors())
        measured = report.case("elided")
        assert measured.retrieved is True
        assert measured.outcome == "recovered"
        assert report.ok is True


class TestFailingClosed:
    """A gate that cannot measure something refuses rather than reporting a pass."""

    async def test_a_declared_floor_with_no_fixtures_fails(self) -> None:
        fixtures = CompressionFixtures(name="thin", version="1", cases=(_prose(),))
        with pytest.raises(EvalIncompleteError, match="json"):
            await measure_compression(fixtures, _router(), _Reader(), floors=DEFAULT_FLOORS)

    async def test_a_solver_that_raises_fails_the_case_rather_than_scoring_it(self) -> None:
        class Broken:
            """Consumer code, which raises."""

            async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
                del case, content, handle
                raise RuntimeError("no model here")

        fixtures = CompressionFixtures(name="thin", version="1", cases=(_prose(),))
        report = await measure_compression(fixtures, _router(), Broken(), floors=_floors())
        assert report.case("prose").outcome == "unmeasured"
        assert report.ok is False
        assert "raised" in report.case("prose").reason

    async def test_a_compressor_that_became_a_pass_through_is_visible(self) -> None:
        fixtures = CompressionFixtures(name="thin", version="1", cases=(_prose(),))
        report = await measure_compression(
            fixtures,
            ReversibleRouter(ContentRouter(threshold_tokens=10_000), MemoryClaimCheckStore()),
            _Reader(),
            floors=_floors(),
        )
        measured = report.kind(ContentKind.PROSE)
        assert measured.savings == 0.0
        assert measured.pass_through == 1
        assert report.ok is False
        assert "saved" in measured.reason


class TestTheFloors:
    """Versioned in the repository, so lowering one is a diff a reviewer sees."""

    def test_they_carry_a_version(self) -> None:
        assert DEFAULT_FLOORS.version

    def test_a_floor_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Floor(kind=ContentKind.PROSE, accuracy=1.4, savings=0.1)

    def test_two_floors_for_one_kind_are_refused(self) -> None:
        floor = Floor(kind=ContentKind.PROSE, accuracy=0.9, savings=0.1)
        with pytest.raises(ConfigurationError, match="one floor per content type"):
            FloorPolicy(version="1", floors=(floor, floor))

    def test_the_report_repeats_the_floors_it_judged_against(self) -> None:
        assert DEFAULT_FLOORS.floor(ContentKind.JSON) is not None


class TestTheReport:
    """What a reviewer reads, and what CI reads."""

    async def test_the_table_shows_every_kind_including_the_ones_that_saved_nothing(
        self,
    ) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        table = report.table()
        for kind in ContentKind:
            assert kind.value in table

    async def test_it_says_what_was_removed_per_case(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        measured = report.case(DEFAULT_FIXTURES.cases[0].id)
        assert measured.removed_tokens == measured.tokens_before - measured.tokens_after

    async def test_the_machine_readable_form_carries_the_verdict(self) -> None:
        report = await measure_compression(
            DEFAULT_FIXTURES, _router(), _Reader(), floors=DEFAULT_FLOORS
        )
        document = report.as_dict()
        assert document["ok"] is True
        assert document["floors"] == DEFAULT_FLOORS.version
        cases = document["cases"]
        assert isinstance(cases, list)
        assert len(cases) == len(DEFAULT_FIXTURES.cases)


class TestAskingTheReportForSomethingItNeverMeasured:
    """A gate that answers about a kind or a case it never ran is worse than one that stops."""

    async def test_a_kind_the_fixtures_do_not_cover_raises(self) -> None:
        fixtures = CompressionFixtures(name="prose-only", version="1", cases=(_prose(),))
        report = await measure_compression(fixtures, _router(), _Reader(), floors=_floors())
        with pytest.raises(KeyError, match="no code content"):
            report.kind(ContentKind.CODE)

    async def test_a_case_the_fixtures_do_not_hold_raises(self) -> None:
        fixtures = CompressionFixtures(name="prose-only", version="1", cases=(_prose(),))
        report = await measure_compression(fixtures, _router(), _Reader(), floors=_floors())
        with pytest.raises(KeyError, match="'nothing'"):
            report.case("nothing")


class TestAnAnswerThatCompressionRemoved:
    """The plain regression: the answer was in the content, and now it is not."""

    async def test_a_wrong_answer_is_lost_and_says_so(self) -> None:
        case = CompressionCase(
            id="forgetful",
            kind=ContentKind.PROSE,
            content="Refunds take five working days. " * 60,
            question="how long?",
            expected="five working days",
        )
        fixtures = CompressionFixtures(name="one", version="1", cases=(case,))
        report = await measure_compression(fixtures, _router(), _Amnesiac(), floors=_floors())
        measured = report.case("forgetful")
        assert measured.outcome == "lost"
        assert "no longer contains" in measured.reason
        assert report.ok is False


class _Amnesiac:
    """A solver that answers from neither the content nor the handle."""

    async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
        """Answer wrongly, on purpose, whatever it was handed."""
        seen = len(content) + len(handle)
        return Answer(text=f"I could not answer {case.question} from {seen} characters")


class TestNothingToCompress:
    """Dividing by what was never there reports nothing saved rather than raising."""

    def test_a_case_with_no_tokens_saved_nothing(self) -> None:
        measured = CaseMeasurement(case_id="empty", kind=ContentKind.PROSE)
        assert measured.savings == 0.0

    def test_a_report_with_no_cases_saved_nothing(self) -> None:
        report = CompressionReport(fixtures="none", version="1", floors="1")
        assert report.savings == 0.0
        assert report.ok is True


class TestTenantIsolation:
    """A fixture run is scoped like any other admission."""

    async def test_the_originals_are_stored_under_the_run_that_admitted_them(self) -> None:
        store = MemoryClaimCheckStore()
        router = ReversibleRouter(ContentRouter(threshold_tokens=8), store)
        fixtures = CompressionFixtures(name="thin", version="1", cases=(_needing_detail(),))
        await measure_compression(
            fixtures, router, _Reader(), floors=_floors(), tenant="acme", run_id="run-1"
        )
        report = await measure_compression(
            fixtures, router, _Reader(), floors=_floors(), tenant="beta", run_id="run-2"
        )
        assert report.ok is True


def _floors() -> FloorPolicy:
    """Floors for the one kind a thin fixture set measures."""
    return FloorPolicy(
        version="test-1",
        floors=(Floor(kind=ContentKind.PROSE, accuracy=1.0, savings=0.05),),
    )


def _prose() -> CompressionCase:
    """A prose case whose answer survives compression."""
    return CompressionCase(
        id="prose",
        kind=ContentKind.PROSE,
        content=(
            "The refund was issued on 3 March. "
            + "Every order is checked against the courier manifest before it ships. " * 40
        ),
        question="when was the refund issued?",
        expected="3 March",
    )


def _needing_detail() -> CompressionCase:
    """A case whose answer is in the part a compressor elides."""
    return CompressionCase(
        id="elided",
        kind=ContentKind.PROSE,
        content=(
            "Every order is checked against the courier manifest before it ships. " * 40
            + "The eleventh consignment note reads QX-4471."
        ),
        question="what does the eleventh consignment note read?",
        expected="QX-4471",
        needs_detail=True,
    )
