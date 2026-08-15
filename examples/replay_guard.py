"""Workflow code that cannot replay, refused before a worker runs it.

Reads two modules — one that calls a provider, reads the clock and invents an id, and one
that has been moved onto the safe replacements — then replays a recorded history against a
run whose logic has since changed.

Run it with `python examples/replay_guard.py`.
"""

from __future__ import annotations

from tesserix_adk.core import NonDeterminismError
from tesserix_adk.workflows import (
    DeterministicIds,
    Patches,
    RecordedHistory,
    WorkflowClock,
    assert_replays,
    guard_source,
    stable,
)

UNSAFE = """
__adk_workflow__ = True

import time
import uuid


def _idempotency_key():
    return uuid.uuid4().hex


async def run(provider, request):
    response = await provider.complete(request)
    return {"at": time.time(), "key": _idempotency_key(), "answer": response.content}
"""

SAFE = """
__adk_workflow__ = True


async def run(activities, state, clock, ids):
    key, ids = ids.next("payment")
    result = await activities.model_call(state.request)
    return {"at": clock.now(), "key": key, "answer": result.response.content}
"""


def refusals() -> None:
    """What the build prints for a module that cannot replay."""
    for finding in guard_source(UNSAFE, source="agent/workflow.py"):
        print(finding)  # noqa: T201

    print("\nthe rewritten module:", guard_source(SAFE, source="agent/workflow.py") or "clean")  # noqa: T201


def replacements() -> None:
    """The same answers on the second execution as on the first."""
    clock = WorkflowClock(started_at=1_700_000_000.0).advanced(30.0)
    first, _ = DeterministicIds(run_id="run-7").next("payment")
    again, _ = DeterministicIds(run_id="run-7").next("payment")

    print(f"\nclock now={clock.now()} key={first} stable_on_replay={first == again}")  # noqa: T201
    print("tools in a fixed order:", [name for name, _ in stable({"search": 1, "book": 2})])  # noqa: T201


def divergence() -> None:
    """A logic change that reaches a run already in flight."""
    history = RecordedHistory(run_id="run-7", commands=("model:0", "tool:0:c0", "model:1"))

    patches = Patches(known=history.patches)
    steps = ["model:0", "tool:0:c0", "model:1"]
    if patches.applied("verify-before-charge"):
        steps.insert(1, "tool:0:verify")

    assert_replays(history, steps)
    print("\nthe run recorded before the patch still replays")  # noqa: T201

    try:
        assert_replays(history, [*steps[:1], "tool:0:verify", *steps[1:]])
    except NonDeterminismError as diverged:
        print(f"ungated, the same change diverges at command {diverged.command}: {diverged}")  # noqa: T201


if __name__ == "__main__":
    refusals()
    replacements()
    divergence()
