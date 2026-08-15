"""The CI step that refuses code which cannot replay, and proves the recorded runs still do.

Two failures reach production the same way: a workflow that calls a provider directly, and a
workflow whose logic changed under a run that was already in flight. The first is caught by
reading the source; the second by re-driving a committed history through the current code.

`make replay-check` runs both. The histories in `tests/histories/` are plain JSON and the
replay uses scripted activities, so this needs no worker, no broker and no network.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import ModelResponse, NonDeterminismError, ToolCall
from tesserix_adk.workflows import (
    ActivityContext,
    AgentWorkflow,
    ModelCallResult,
    RecordedHistory,
    ToolCallResult,
    WorkflowState,
    assert_replays,
    guard,
)

if TYPE_CHECKING:
    from tesserix_adk.workflows import ModelCallInput, ToolCallInput

__all__ = ["HISTORIES", "SRC", "histories", "replayed", "scanned"]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HISTORIES = ROOT / "tests" / "histories"


class Scripted:
    """Activities that answer from a recorded fixture, so a replay needs no worker."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.commands: list[str] = []

    async def model_call(self, request: ModelCallInput) -> ModelCallResult:
        """Answer with the response the fixture recorded for this iteration."""
        self.commands.append(request.step)
        iteration = int(request.step.split(":")[1])
        return ModelCallResult(
            response=self.responses[min(iteration, len(self.responses) - 1)],
            history=f"{request.history}+{request.step}",
        )

    async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
        """Record that the tool ran, which is all a history remembers about it."""
        self.commands.append(request.step)
        return ToolCallResult(call_id=request.call_id, content="", history=f"h+{request.step}")


def _response(recorded: dict[str, Any]) -> ModelResponse:
    """One model response, as the fixture writes it."""
    return ModelResponse(
        content=recorded.get("content", ""),
        tool_calls=tuple(
            ToolCall(id=call["id"], name=call["name"]) for call in recorded.get("tool_calls", ())
        ),
    )


def histories(directory: Path = HISTORIES) -> list[Path]:
    """Every committed history, in a fixed order."""
    return sorted(directory.glob("*.json"))


async def _commands(fixture: dict[str, Any]) -> list[str]:
    """The commands the current code issues for this fixture's run."""
    activities = Scripted([_response(recorded) for recorded in fixture["responses"]])
    workflow = AgentWorkflow(activities=activities, model=fixture.get("model", "test-model"))
    await workflow.run(
        WorkflowState(run_id=fixture["run_id"], history="h0"),
        context=ActivityContext(run_id=fixture["run_id"], tenant=fixture.get("tenant", "acme")),
    )
    return activities.commands


def replayed(directory: Path = HISTORIES) -> list[str]:
    """Re-drive every committed history and return one message per divergence."""
    problems = []
    for path in histories(directory):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        history = RecordedHistory(run_id=fixture["run_id"], commands=tuple(fixture["commands"]))
        try:
            assert_replays(history, asyncio.run(_commands(fixture)))
        except NonDeterminismError as diverged:
            problems.append(f"{path.name}: {diverged}")
    return problems


def scanned(root: Path = SRC) -> list[str]:
    """Every replay-safety finding under `root`, as a build log prints them."""
    report = guard([root])
    return [str(finding) for finding in report.findings]


def main() -> int:
    """Fail the build on unsafe workflow code, or on a history that no longer replays."""
    problems = scanned() + replayed()
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    if problems:
        sys.stderr.write(
            f"\n{len(problems)} replay-safety problem(s). Move the call into an activity, or "
            f"gate the changed logic behind a patch name so runs already in flight keep the "
            f"path they recorded.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
