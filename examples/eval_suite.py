"""Writing a golden dataset, replaying it twice, and proving the two runs agree.

Run it with `uv run python examples/eval_suite.py`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from tesserix_adk.core import Message, NoOutput, Run, RunState, TextPart, Usage
from tesserix_adk.core.tenancy import current_tenant
from tesserix_adk.evals import CaseStatus, EvalCase, EvalSuite, SuiteRunner

ANSWERS = {
    "late-refund": "a refund is on its way",
    "wrong-size": "we have posted a replacement",
}


async def replay(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
    """Answer from a recording. A live executor would call the model here instead."""
    if case.id not in ANSWERS:
        raise LookupError(f"no recording for {case.id!r}; re-record before gating on it")
    return Run[NoOutput](
        id=run_id,
        tenant=current_tenant().tenant,
        agent_name="support",
        agent_version="1.0.0",
        model="recorded",
        state=RunState.COMPLETED,
        messages=[Message(role="assistant", content=[TextPart(text=ANSWERS[case.id])])],
        usage=Usage(input_tokens=90, output_tokens=12),
    )


async def main() -> None:
    """Round-trip a dataset through disk, then replay it twice and compare the digests."""
    suite = EvalSuite(
        name="refunds",
        version="2026-08-01",
        cases=(
            EvalCase(id="late-refund", input="my order never arrived", tenant="acme"),
            EvalCase(id="wrong-size", input="reach me on ada@example.com", tenant="acme"),
            EvalCase(id="not-recorded", input="where is my parcel", tenant="beta"),
        ),
    )

    with tempfile.TemporaryDirectory() as workspace:
        home = Path(workspace)
        dataset = home / "refunds.jsonl"
        suite.to_jsonl(dataset)
        redacted = "ada@example.com" not in dataset.read_text(encoding="utf-8")
        print(f"the email never reached disk: {redacted}")  # noqa: T201

        read_back = EvalSuite.from_jsonl(dataset)
        first = await SuiteRunner(replay, artefacts=home / "artefacts").run(read_back)
        second = await SuiteRunner(replay).run(read_back)

        print(f"two replays, one digest: {first.digest() == second.digest()}")  # noqa: T201
        for result in first.results:
            print(f"  {result.case_id}: {result.status} {result.reason}".rstrip())  # noqa: T201

        missing = first.errored()[0]
        print(f"the suite exits {first.exit_code} because {missing.case_id} never ran")  # noqa: T201
        answered = first.results[0].status is CaseStatus.COMPLETED
        print(f"the first case answered: {answered}")  # noqa: T201
        kept = sorted(each.name for each in (home / "artefacts" / "refunds").iterdir())
        print(f"evidence kept at {kept}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
