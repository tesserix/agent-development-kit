"""Catching the code that cannot replay, before a worker runs it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import replay_check

from tesserix_adk import workflows
from tesserix_adk.core import NonDeterminismError
from tesserix_adk.workflows import (
    RULES,
    WORKFLOW_MARKER,
    DeterministicIds,
    Patches,
    RecordedHistory,
    WorkflowClock,
    assert_replays,
    guard,
    guard_source,
    stable,
)

MARKED = f"{WORKFLOW_MARKER} = True\n"


def module(body: str, *, marked: bool = True) -> str:
    """A module, workflow-marked unless the test says otherwise."""
    return (MARKED if marked else "") + body


class TestRefusingWhatCannotReplay:
    """The build fails naming the file, the line and where the call belongs instead."""

    def test_a_provider_called_on_the_workflow_path_is_refused(self) -> None:
        found = guard_source(
            module("async def run(provider):\n    return await provider.complete(request)\n"),
            source="agent/workflow.py",
        )

        assert [finding.code for finding in found] == ["ADK-W001"]
        assert found[0].source == "agent/workflow.py"
        assert found[0].line == 3
        assert "model_call_activity" in found[0].remedy

    def test_an_id_from_randomness_is_refused(self) -> None:
        found = guard_source(module("import uuid\n\ndef key():\n    return uuid.uuid4().hex\n"))

        assert [finding.code for finding in found] == ["ADK-W002"]
        assert "DeterministicIds" in found[0].remedy

    def test_the_wall_clock_is_refused(self) -> None:
        found = guard_source(module("import time\n\ndef at():\n    return time.time()\n"))

        assert [finding.code for finding in found] == ["ADK-W003"]

    def test_datetime_now_is_refused_however_it_is_reached(self) -> None:
        found = guard_source(
            module("import datetime\n\ndef at():\n    return datetime.datetime.now()\n")
        )

        assert [finding.code for finding in found] == ["ADK-W003"]

    def test_network_io_is_refused(self) -> None:
        found = guard_source(module("import httpx\n\ndef fetch():\n    return httpx.get(url)\n"))

        assert [finding.code for finding in found] == ["ADK-W004"]

    def test_a_helper_two_frames_deep_is_still_caught(self) -> None:
        found = guard_source(
            module(
                "import uuid\n\n"
                "def _key():\n    return uuid.uuid4().hex\n\n"
                "def run():\n    return _key()\n"
            )
        )

        assert [finding.code for finding in found] == ["ADK-W002", "ADK-W005"]
        assert found[1].line == 8

    def test_every_finding_says_what_to_do_instead(self) -> None:
        assert all(rule.remedy for rule in RULES)

    def test_findings_come_back_in_line_order(self) -> None:
        found = guard_source(
            module("import time\nimport uuid\n\ndef run():\n    time.time()\n    uuid.uuid4()\n")
        )

        assert [finding.line for finding in found] == [6, 7]

    def test_a_call_with_no_readable_name_is_not_guessed_at(self) -> None:
        found = guard_source(module("def run(table):\n    return table['run']()\n"))

        assert found == ()

    def test_a_file_that_does_not_parse_is_not_a_file_that_passes(self) -> None:
        with pytest.raises(SyntaxError):
            guard_source("def broken(:\n")


class TestNotFiringOnCodeItDoesNotGovern:
    """A guard that cries wolf is a guard the consumer turns off."""

    def test_an_unmarked_module_is_left_alone(self) -> None:
        found = guard_source(
            module("import uuid\n\ndef key():\n    return uuid.uuid4()\n", marked=False)
        )

        assert found == ()

    def test_a_module_marked_false_is_left_alone(self) -> None:
        found = guard_source(f"{WORKFLOW_MARKER} = False\nimport uuid\n\nuuid.uuid4()\n")

        assert found == ()

    def test_an_activity_calling_a_provider_is_fine(self) -> None:
        found = guard_source(
            "async def model_call(request, *, provider):\n"
            "    return await provider.complete(request)\n"
        )

        assert found == ()

    def test_ordinary_workflow_arithmetic_is_not_refused(self) -> None:
        found = guard_source(
            module("def run(state):\n    return state.model_copy(update={'i': state.i + 1})\n")
        )

        assert found == ()


class TestScanningATree:
    """What the CI step runs over the repository."""

    def test_it_reads_every_marked_module_under_the_paths(self, tmp_path: Path) -> None:
        (tmp_path / "workflow.py").write_text(
            module("import uuid\n\ndef key():\n    return uuid.uuid4()\n"), encoding="utf-8"
        )
        (tmp_path / "activity.py").write_text("import uuid\n\nuuid.uuid4()\n", encoding="utf-8")

        report = guard([tmp_path])

        assert report.scanned == 1
        assert len(report.findings) == 1
        assert report.ok is False
        assert report.exit_code == 1

    def test_a_clean_tree_passes(self, tmp_path: Path) -> None:
        (tmp_path / "workflow.py").write_text(
            module("def run():\n    return 1\n"), encoding="utf-8"
        )

        report = guard([tmp_path])

        assert report.ok is True
        assert report.exit_code == 0
        assert "0 replay-safety problem" in report.summary()

    def test_a_single_file_can_be_named_directly(self, tmp_path: Path) -> None:
        path = tmp_path / "workflow.py"
        path.write_text(module("import time\n\ndef at():\n    return time.time()\n"), "utf-8")

        report = guard([path])

        assert report.scanned == 1

    def test_the_summary_names_each_finding(self, tmp_path: Path) -> None:
        (tmp_path / "workflow.py").write_text(
            module("import time\n\ndef at():\n    return time.time()\n"), encoding="utf-8"
        )

        summary = guard([tmp_path]).summary()

        assert "ADK-W003" in summary
        assert "WorkflowClock" in summary

    def test_the_kits_own_workflow_modules_are_clean(self) -> None:
        report = guard([Path(workflows.__file__).parent])

        assert report.ok is True


class TestReplayingARecordedHistory:
    """A divergence is never a pass."""

    def test_the_same_commands_replay(self) -> None:
        history = RecordedHistory(run_id="r1", commands=("model:0", "tool:0:c0", "model:1"))

        assert_replays(history, ["model:0", "tool:0:c0", "model:1"])

    def test_a_changed_command_names_the_diverging_index(self) -> None:
        history = RecordedHistory(run_id="r1", commands=("model:0", "tool:0:c0"))

        with pytest.raises(NonDeterminismError) as diverged:
            assert_replays(history, ["model:0", "tool:0:c1"])

        assert diverged.value.command == 1
        assert diverged.value.expected == "tool:0:c0"
        assert diverged.value.actual == "tool:0:c1"
        assert diverged.value.retryable is False

    def test_a_replay_that_stops_short_has_diverged_too(self) -> None:
        history = RecordedHistory(run_id="r1", commands=("model:0", "tool:0:c0"))

        with pytest.raises(NonDeterminismError) as diverged:
            assert_replays(history, ["model:0"])

        assert diverged.value.command == 1
        assert diverged.value.actual == ""

    def test_a_replay_that_asks_for_more_has_diverged_too(self) -> None:
        history = RecordedHistory(run_id="r1", commands=("model:0",))

        with pytest.raises(NonDeterminismError) as diverged:
            assert_replays(history, ["model:0", "tool:0:c0"])

        assert diverged.value.command == 1
        assert diverged.value.expected == ""

    def test_the_committed_fixtures_still_replay(self) -> None:
        assert replay_check.replayed() == []


class TestTheCiStep:
    """What `make replay-check` returns to the build."""

    def test_a_clean_repository_passes(self) -> None:
        assert replay_check.main() == 0

    def test_the_kits_own_source_holds_no_findings(self) -> None:
        assert replay_check.scanned() == []

    def test_a_history_that_no_longer_replays_names_the_fixture(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "moved.json").write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "responses": [{"content": "done"}],
                    "commands": ["model:0", "tool:0:c0"],
                }
            ),
            encoding="utf-8",
        )

        problems = replay_check.replayed(tmp_path)

        assert len(problems) == 1
        assert "moved.json" in problems[0]
        assert capsys.readouterr().err == ""

    def test_a_finding_fails_the_build_and_says_what_to_do(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(replay_check, "scanned", lambda: ["workflow.py:3: ADK-W002 ..."])

        code = replay_check.main()

        assert code == 1
        printed = capsys.readouterr().err
        assert "ADK-W002" in printed
        assert "activity" in printed


class TestTheWorkflowSafeReplacements:
    """What the code uses once the guard has refused the wall clock and uuid4."""

    def test_the_clock_reads_the_runs_own_time(self) -> None:
        clock = WorkflowClock(started_at=1_700_000_000.0)

        assert clock.advanced(30.0).advanced(15.0).now() == 1_700_000_045.0

    def test_the_clock_never_goes_backwards(self) -> None:
        with pytest.raises(ValueError, match="only moves forward"):
            WorkflowClock(started_at=1.0).advanced(-1.0)

    def test_ids_are_the_same_on_the_second_execution(self) -> None:
        first, generator = DeterministicIds(run_id="r1").next("payment")
        second, _ = generator.next("payment")
        replayed_first, replayed_generator = DeterministicIds(run_id="r1").next("payment")
        replayed_second, _ = replayed_generator.next("payment")

        assert (first, second) == (replayed_first, replayed_second)
        assert first != second

    def test_two_runs_never_share_an_id(self) -> None:
        one, _ = DeterministicIds(run_id="r1").next("payment")
        two, _ = DeterministicIds(run_id="r2").next("payment")

        assert one != two

    def test_a_registry_is_read_in_an_order_a_deploy_cannot_change(self) -> None:
        assert stable({"search": 2, "book": 1}) == (("book", 1), ("search", 2))
        assert stable({"book": 1, "search": 2}) == (("book", 1), ("search", 2))

    def test_a_history_recorded_before_a_patch_takes_the_old_path(self) -> None:
        assert Patches().applied("prompt-v2") is False
        assert Patches(known=("prompt-v2",)).applied("prompt-v2") is True

    def test_reaching_a_patch_the_first_time_records_it(self) -> None:
        patches = Patches().applying("prompt-v2")

        assert patches.applied("prompt-v2") is True
        assert patches.applying("prompt-v2").known == ("prompt-v2",)
