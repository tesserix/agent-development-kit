"""One eval command gives local and CI the same deterministic verdict."""

from __future__ import annotations

import io
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tesserix_adk.cli.eval_run import (
    CASE_ERROR,
    CONFIGURATION_ERROR,
    GATE_FAILED,
    EvalTarget,
    main,
)
from tesserix_adk.core import Message, Run, RunState, TextPart
from tesserix_adk.evals import EvalCase, EvalSuite

if TYPE_CHECKING:
    from pathlib import Path


def suite(path: Path, *cases: EvalCase) -> Path:
    """Write one versioned dataset."""
    EvalSuite(name="answers", version="1", cases=cases).to_jsonl(path)
    return path


def completed(case: EvalCase, *, run_id: str, answer: str = "wrong") -> Run[Any]:
    """Return a terminal run with one visible answer."""
    return Run(
        id=run_id,
        tenant=case.tenant,
        user=case.user,
        agent_name="evaluated",
        agent_version="1.0.0",
        model="fake",
        state=RunState.COMPLETED,
        messages=[Message(role="assistant", content=[TextPart(text=answer)])],
    )


async def test_two_gate_failures_are_in_junit_and_have_inspectable_artifacts(
    tmp_path: Path,
) -> None:
    dataset = suite(
        tmp_path / "suite.jsonl",
        EvalCase(id="case-a", input="a", tenant="acme", expected="right"),
        EvalCase(id="case-b", input="b", tenant="acme", expected="right"),
    )

    async def execute(case: EvalCase, *, run_id: str) -> Run[Any]:
        return completed(case, run_id=run_id)

    output = io.StringIO()
    artifacts = tmp_path / "artifacts"
    code = await main(
        [str(dataset), "--report", "junit", "--output", str(artifacts)],
        resolve=lambda _reference: EvalTarget(deterministic=execute),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == GATE_FAILED
    assert '<testsuite name="answers" tests="2" failures="2" errors="0"' in output.getvalue()
    assert "case-a" in output.getvalue()
    assert "answer_match=0" in output.getvalue()
    for case_id in ("case-a", "case-b"):
        artifact = artifacts / "answers" / f"{case_id}.jsonl"
        assert artifact.exists()
        assert '"type":"complete"' in artifact.read_text(encoding="utf-8")


async def test_case_errors_have_a_distinct_exit_code_and_json_report(tmp_path: Path) -> None:
    dataset = suite(
        tmp_path / "suite.jsonl",
        EvalCase(id="broken", input="a", tenant="acme"),
    )

    async def execute(case: EvalCase, *, run_id: str) -> Run[Any]:
        del case, run_id
        raise RuntimeError("cassette is missing")

    output = io.StringIO()
    code = await main(
        [str(dataset), "--report", "json", "--output", str(tmp_path / "artifacts")],
        resolve=lambda _reference: EvalTarget(deterministic=execute),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == CASE_ERROR
    report = json.loads(output.getvalue())
    assert report["cases"][0]["status"] == "errored"
    assert report["cases"][0]["reason"] == "cassette is missing"


async def test_live_spend_above_the_ceiling_refuses_before_every_provider_call(
    tmp_path: Path,
) -> None:
    dataset = suite(
        tmp_path / "suite.jsonl",
        EvalCase(id="live", input="a", tenant="acme"),
    )
    calls = 0

    async def live(case: EvalCase, *, run_id: str) -> Run[Any]:
        nonlocal calls
        calls += 1
        return completed(case, run_id=run_id)

    target = EvalTarget(
        deterministic=live,
        live=live,
        estimate_live=lambda _suite: Decimal("12.50"),
        live_ceiling=Decimal("5.00"),
    )
    output = io.StringIO()

    code = await main(
        [str(dataset), "--live", "--yes", "--output", str(tmp_path / "artifacts")],
        resolve=lambda _reference: target,
        out=output,
        stdin=io.StringIO(),
    )

    assert code == CONFIGURATION_ERROR
    assert calls == 0
    assert "estimate=USD 12.50" in output.getvalue()
    assert "ceiling=USD 5.00" in output.getvalue()


async def test_deterministic_is_default_and_parallelism_is_throttled_to_target_limit(
    tmp_path: Path,
) -> None:
    dataset = suite(
        tmp_path / "suite.jsonl",
        EvalCase(id="offline", input="a", tenant="acme"),
    )
    deterministic_calls = 0
    live_calls = 0

    async def deterministic(case: EvalCase, *, run_id: str) -> Run[Any]:
        nonlocal deterministic_calls
        deterministic_calls += 1
        return completed(case, run_id=run_id, answer="right")

    async def live(case: EvalCase, *, run_id: str) -> Run[Any]:
        nonlocal live_calls
        live_calls += 1
        return completed(case, run_id=run_id, answer="right")

    output = io.StringIO()
    code = await main(
        [str(dataset), "--parallel", "99", "--output", str(tmp_path / "artifacts")],
        resolve=lambda _reference: EvalTarget(
            deterministic=deterministic,
            live=live,
            max_parallel=2,
        ),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == 0
    assert deterministic_calls == 1
    assert live_calls == 0
    assert "parallel=2 (requested 99, throttled by target)" in output.getvalue()


async def test_filter_selects_tags_and_empty_selection_is_configuration_error(
    tmp_path: Path,
) -> None:
    dataset = suite(
        tmp_path / "suite.jsonl",
        EvalCase(id="smoke", input="a", tenant="acme", tags=("smoke",)),
        EvalCase(id="slow", input="b", tenant="acme", tags=("slow",)),
    )

    async def execute(case: EvalCase, *, run_id: str) -> Run[Any]:
        return completed(case, run_id=run_id, answer="right")

    output = io.StringIO()
    assert (
        await main(
            [
                str(dataset),
                "--filter",
                "absent",
                "--output",
                str(tmp_path / "artifacts"),
            ],
            resolve=lambda _reference: EvalTarget(deterministic=execute),
            out=output,
            stdin=io.StringIO(),
        )
        == CONFIGURATION_ERROR
    )
    assert "no cases" in output.getvalue()
