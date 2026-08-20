"""A dataset that survives an edit, and a suite that replays it the same way twice."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    Message,
    NoOutput,
    Run,
    RunState,
    TextPart,
    Usage,
)
from tesserix_adk.core.tenancy import current_tenant
from tesserix_adk.evals import (
    DATASET_FORMAT,
    CaseStatus,
    EvalCase,
    EvalSuite,
    SuiteRunner,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


def _case(case_id: str, **overrides: Any) -> EvalCase:
    fields: dict[str, Any] = {
        "id": case_id,
        "input": "how late is the 8.15 to Sydney",
        "tenant": "acme",
        "expected": "delayed by nine minutes",
    }
    return EvalCase(**(fields | overrides))


def _suite(*cases: EvalCase, name: str = "timetable", version: str = "2026-08-01") -> EvalSuite:
    return EvalSuite(name=name, version=version, cases=cases or (_case("c1"),))


def _run(case: EvalCase, run_id: str, *, state: RunState = RunState.COMPLETED) -> Run[NoOutput]:
    return Run[NoOutput](
        id=run_id,
        tenant=case.tenant,
        user=case.user,
        agent_name="timetable",
        agent_version="1.0.0",
        model="fake-1",
        state=state,
        messages=[Message(role="assistant", content=[TextPart(text="delayed by nine minutes")])],
        usage=Usage(input_tokens=120, output_tokens=8),
    )


def _answering(**overrides: Any) -> Any:
    async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
        return _run(case, run_id, **overrides)

    return execute


class TestTheCase:
    def test_a_case_without_an_id_is_refused(self) -> None:
        """An id is how a result is compared to yesterday's; a blank one compares nothing."""
        with pytest.raises(ConfigurationError, match="id"):
            _case("  ")

    def test_two_cases_sharing_an_id_are_refused_naming_it(self) -> None:
        with pytest.raises(ConfigurationError, match="c1"):
            _suite(_case("c1"), _case("c1", input="another"))

    def test_a_case_carries_the_context_it_runs_under(self) -> None:
        case = _case("c1", tenant="beta", user="ada", tags=("refunds",))
        assert case.tenant == "beta"
        assert case.user == "ada"
        assert case.tags == ("refunds",)

    def test_cases_can_be_selected_by_tag(self) -> None:
        suite = _suite(_case("c1", tags=("refunds",)), _case("c2", tags=("search",)))
        assert [case.id for case in suite.tagged("refunds")] == ["c1"]


class TestTheDatasetOnDisk:
    def test_a_suite_written_and_read_back_is_the_same_suite(self, tmp_path: Path) -> None:
        suite = _suite(_case("c1"), _case("c2", tags=("search",)))
        path = tmp_path / "timetable.jsonl"
        suite.to_jsonl(path)
        assert EvalSuite.from_jsonl(path) == suite

    def test_the_header_declares_the_format_the_name_and_the_version(self, tmp_path: Path) -> None:
        path = tmp_path / "timetable.jsonl"
        _suite().to_jsonl(path)
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header == {
            "format": DATASET_FORMAT,
            "name": "timetable",
            "version": "2026-08-01",
        }

    def test_a_format_from_the_future_is_refused_with_where_to_migrate(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "timetable.jsonl"
        _suite().to_jsonl(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = json.dumps({"format": DATASET_FORMAT + 1, "name": "t", "version": "1"})
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="migrat"):
            EvalSuite.from_jsonl(path)

    def test_a_dataset_with_no_header_is_refused_rather_than_guessed_at(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "timetable.jsonl"
        path.write_text(json.dumps(_case("c1").model_dump(mode="json")) + "\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="header"):
            EvalSuite.from_jsonl(path)

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "timetable.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="header"):
            EvalSuite.from_jsonl(path)

    def test_a_first_line_that_is_not_json_is_refused_as_a_header(self, tmp_path: Path) -> None:
        path = tmp_path / "timetable.jsonl"
        path.write_text("name,input,tenant\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="header"):
            EvalSuite.from_jsonl(path)

    def test_personal_data_is_redacted_on_the_way_to_disk(self, tmp_path: Path) -> None:
        """A dataset is committed, and a committed email outlives every run that used it."""
        path = tmp_path / "timetable.jsonl"
        _suite(_case("c1", input="refund ada@example.com for order 12")).to_jsonl(path)
        written = path.read_text(encoding="utf-8")
        assert "ada@example.com" not in written
        assert "order 12" in written

    def test_an_identity_field_that_cannot_be_masked_refuses_to_be_written(
        self, tmp_path: Path
    ) -> None:
        """Masking an id would break the comparison it exists for, so refuse instead."""
        path = tmp_path / "timetable.jsonl"
        with pytest.raises(ConfigurationError, match="id"):
            _suite(_case("sk-live-" + "a" * 32)).to_jsonl(path)
        assert not path.exists()

    def test_a_real_person_in_the_user_field_refuses_to_be_written(self, tmp_path: Path) -> None:
        path = tmp_path / "timetable.jsonl"
        with pytest.raises(ConfigurationError, match="user"):
            _suite(_case("c1", user="ada@example.com")).to_jsonl(path)
        assert not path.exists()


class TestReplay:
    async def test_the_same_suite_run_twice_produces_the_same_digest(self) -> None:
        suite = _suite(*(_case(f"c{index}") for index in range(50)))
        runner = SuiteRunner(_answering())
        first = await runner.run(suite)
        second = await SuiteRunner(_answering()).run(suite)
        assert first.digest() == second.digest()

    async def test_results_are_in_dataset_order_whatever_order_they_finished_in(self) -> None:
        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            await asyncio.sleep(0.01 if case.id == "c1" else 0.0)
            return _run(case, run_id)

        result = await SuiteRunner(execute).run(_suite(_case("c1"), _case("c2"), _case("c3")))
        assert [each.case_id for each in result.results] == ["c1", "c2", "c3"]

    async def test_the_run_id_is_derived_so_two_machines_agree_on_it(self) -> None:
        suite = _suite(_case("c1"))
        first = await SuiteRunner(_answering()).run(suite)
        second = await SuiteRunner(_answering()).run(suite)
        assert first.results[0].run_id == second.results[0].run_id

    async def test_a_different_seed_gives_a_different_run_id(self) -> None:
        suite = _suite(_case("c1"))
        first = await SuiteRunner(_answering(), seed="a").run(suite)
        second = await SuiteRunner(_answering(), seed="b").run(suite)
        assert first.results[0].run_id != second.results[0].run_id

    async def test_the_digest_ignores_how_long_the_cases_took(self) -> None:
        """Wall-clock time is not a result, and a digest that includes it never matches."""
        slow = _suite(_case("c1"))

        async def dawdling(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            await asyncio.sleep(0.02)
            return _run(case, run_id)

        assert (await SuiteRunner(dawdling).run(slow)).digest() == (
            await SuiteRunner(_answering()).run(slow)
        ).digest()


class TestWhenACaseCannotRun:
    async def test_a_missing_recording_errors_that_case_with_the_reason(self) -> None:
        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            if case.id == "c2":
                raise LookupError("no recording for 'c2'; re-record with adk eval --record")
            return _run(case, run_id)

        result = await SuiteRunner(execute).run(_suite(_case("c1"), _case("c2")))
        errored = result.errored()
        assert [each.case_id for each in errored] == ["c2"]
        assert "no recording" in errored[0].reason

    async def test_a_case_that_could_not_run_is_never_counted_as_a_pass(self) -> None:
        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:  # noqa: ARG001 — the executor's shape
            raise LookupError("stale cassette")

        result = await SuiteRunner(execute).run(_suite())
        assert not result.ok
        assert result.exit_code == 1
        assert result.results[0].status is CaseStatus.ERRORED

    async def test_one_case_failing_does_not_stop_the_others(self) -> None:
        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            if case.id == "c1":
                raise LookupError("stale cassette")
            return _run(case, run_id)

        result = await SuiteRunner(execute).run(_suite(_case("c1"), _case("c2")))
        assert result.results[1].status is CaseStatus.COMPLETED

    async def test_a_run_stopped_by_the_iteration_cap_is_incomplete_not_complete(self) -> None:
        result = await SuiteRunner(_answering(state=RunState.RUNNING)).run(_suite())
        assert result.results[0].status is CaseStatus.INCOMPLETE
        assert not result.ok

    async def test_the_cases_that_stopped_short_are_listed_together(self) -> None:
        result = await SuiteRunner(_answering(state=RunState.RUNNING)).run(_suite())
        assert [each.case_id for each in result.incomplete()] == ["c1"]

    async def test_a_failed_run_still_completed_and_carries_its_terminal_state(self) -> None:
        """A wrong answer is a result to measure, not an error in the harness."""
        result = await SuiteRunner(_answering(state=RunState.FAILED)).run(_suite())
        assert result.results[0].status is CaseStatus.COMPLETED
        assert result.results[0].run is not None
        assert result.results[0].run.state is RunState.FAILED
        assert result.ok


class TestTheContextACaseRunsUnder:
    async def test_the_case_declares_the_tenant_and_the_runner_binds_it(self) -> None:
        seen: list[str] = []

        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            seen.append(current_tenant().tenant)
            return _run(case, run_id)

        await SuiteRunner(execute).run(_suite(_case("c1", tenant="beta")))
        assert seen == ["beta"]

    async def test_the_user_is_bound_too_so_a_scoped_case_is_really_scoped(self) -> None:
        seen: list[str | None] = []

        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            seen.append(current_tenant().user)
            return _run(case, run_id)

        await SuiteRunner(execute).run(_suite(_case("c1", user="ada")))
        assert seen == ["ada"]

    async def test_a_run_answering_for_another_tenant_errors_that_case(self) -> None:
        """The runner never widens what a case declared, and it checks what came back."""

        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:  # noqa: ARG001 — the executor's shape
            return _run(_case("c1", tenant="somebody-else"), run_id)

        result = await SuiteRunner(execute).run(_suite())
        assert result.results[0].status is CaseStatus.ERRORED
        assert "somebody-else" in result.results[0].reason


class TestConcurrency:
    async def test_no_more_cases_run_at_once_than_were_allowed(self) -> None:
        inflight = 0
        peak = 0

        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            finally:
                inflight -= 1
            return _run(case, run_id)

        suite = _suite(*(_case(f"c{index}") for index in range(8)))
        await SuiteRunner(execute, concurrency=2).run(suite)
        assert peak == 2

    def test_a_concurrency_of_zero_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="concurrency"):
            SuiteRunner(_answering(), concurrency=0)


class TestArtefacts:
    async def test_each_case_gets_its_own_directory(self, tmp_path: Path) -> None:
        await SuiteRunner(_answering(), artefacts=tmp_path).run(_suite(_case("c1"), _case("c2")))
        assert (tmp_path / "timetable" / "c1" / "run.json").exists()
        assert (tmp_path / "timetable" / "c2" / "case.json").exists()

    async def test_the_artefact_holds_the_transcript_and_the_usage(self, tmp_path: Path) -> None:
        await SuiteRunner(_answering(), artefacts=tmp_path).run(_suite())
        written = json.loads((tmp_path / "timetable" / "c1" / "run.json").read_text("utf-8"))
        assert written["usage"]["input_tokens"] == 120
        assert written["messages"][0]["content"][0]["text"] == "delayed by nine minutes"

    async def test_a_case_is_marked_incomplete_before_it_starts(self, tmp_path: Path) -> None:
        """A suite killed part way leaves artefacts that say so rather than half a pass."""
        started = asyncio.Event()

        async def never_finishes(case: EvalCase, *, run_id: str) -> Run[NoOutput]:  # noqa: ARG001 — the executor's shape
            started.set()
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

        task = asyncio.create_task(SuiteRunner(never_finishes, artefacts=tmp_path).run(_suite()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        written = json.loads((tmp_path / "timetable" / "c1" / "result.json").read_text("utf-8"))
        assert written["status"] == CaseStatus.INCOMPLETE

    async def test_timings_are_kept_apart_from_the_result_they_would_destabilise(
        self, tmp_path: Path
    ) -> None:
        await SuiteRunner(_answering(), artefacts=tmp_path).run(_suite())
        case_dir = tmp_path / "timetable" / "c1"
        assert "seconds" not in json.loads((case_dir / "result.json").read_text("utf-8"))
        assert "seconds" in json.loads((case_dir / "timings.json").read_text("utf-8"))

    async def test_an_errored_case_still_records_why(self, tmp_path: Path) -> None:
        async def execute(case: EvalCase, *, run_id: str) -> Run[NoOutput]:  # noqa: ARG001 — the executor's shape
            raise LookupError("stale cassette")

        await SuiteRunner(execute, artefacts=tmp_path).run(_suite())
        written = json.loads((tmp_path / "timetable" / "c1" / "result.json").read_text("utf-8"))
        assert written["status"] == CaseStatus.ERRORED
        assert "stale cassette" in written["reason"]
