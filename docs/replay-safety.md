# Replay safety

A workflow function is re-executed from the start every time a run resumes. Every decision it
makes has to come out the same way it came out the first time, and the four things that break
that — a model call, a random id, the wall clock, network I/O — all work perfectly in
development. The divergence arrives on the first replay in production, on a run that is
already in flight, where it either wedges the run or quietly re-decides it.

The guard catches that before a worker sees the code, and the replay test catches the day the
logic changed underneath a history that was already recorded.

## Marking a module

A module opts in by declaring the marker at module level:

```python
__adk_workflow__ = True
```

Nothing else is scanned. Activity modules and ordinary in-process runtime code are allowed
everything the guard refuses — a guard that fires on code it does not govern is a guard
consumers turn off.

## The ruleset

| Code | Refused | Do this instead |
|---|---|---|
| `ADK-W001` | a model provider called on the workflow path | call it through `model_call_activity`, which records its result in history |
| `ADK-W002` | an id from `uuid4` or `random` | `DeterministicIds`, or take the id from an activity result |
| `ADK-W003` | `time.time`, `time.sleep`, `datetime.now` | `WorkflowClock`, whose instants come from the run's own state |
| `ADK-W004` | `requests`, `httpx`, `urllib`, `socket` | move it into an activity, where its result is recorded once |
| `ADK-W005` | a call to a helper that is itself unsafe | move the helper behind an activity |

`ADK-W005` is why a consumer's own wrapper does not get past the guard: the module's
functions are read first, and any function that does something unreplayable makes its callers
unsafe too. Two frames away is still the same replay.

Findings name the file, the line and the call, and every one carries its remedy:

```
src/agent/workflow.py:41: ADK-W001 a model provider is called on the workflow path (provider.complete)
    call it through model_call_activity, which records its result in history
```

## Running it

```bash
make replay-check
```

`guard(paths)` returns a `ReplayReport`; `report.exit_code` is what the CI step returns and
`report.summary()` is what it prints. `guard_source(text, source=...)` reads one module, which
is what a pre-commit hook or an editor integration wants.

## The workflow-safe replacements

```python
from tesserix_adk.workflows import DeterministicIds, Patches, WorkflowClock, stable

clock = WorkflowClock(started_at=started).advanced(30.0)   # never a wall clock
key, ids = DeterministicIds(run_id=run_id).next("payment")  # the same key on replay
for name, tool in stable(registry):                         # an order a deploy cannot change
    ...
```

`stable` is not cosmetic. A tool registry iterated in load order feeds the prompt in load
order, and that order changes when an import does — which re-decides a run that was already
in flight.

## Changing the logic under a running fleet

Agent logic legitimately changes. A run started before the change has to keep taking the old
path, or its replay diverges from the history it is being replayed against. `Patches` is that
decision:

```python
if patches.applied("prompt-v2"):
    ...  # runs started after the change
else:
    ...  # runs already in flight keep the path they recorded
```

## Replaying a recorded history

`RecordedHistory` is a run's command sequence, committed as a fixture. `assert_replays` drives
the current code against it and raises `NonDeterminismError` at the first index where they
differ, naming both sides. A replay that stops short, or asks for one more command than the
history holds, has diverged too: a replay is not allowed to be a prefix of the truth.

The kit's own fixtures live in `tests/histories/` as plain JSON and replay through scripted
activities, so `make replay-check` needs no worker, no broker and no network.

## Related

- [Durable runs](durable-runs.md) — the workflow and activity split these rules protect.
- [`examples/replay_guard.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/replay_guard.py) — a runnable walk through both halves.
