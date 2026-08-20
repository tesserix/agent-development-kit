# Golden datasets and deterministic replay

An eval number that moves on its own gates nothing. Run the same dataset twice against a
live model and the two answers differ, so a drop of four points is either a regression or a
Tuesday, and nobody can tell which. A suite is only useful when a rerun that changed nothing
produces the same result.

```python
suite = EvalSuite.from_jsonl(Path("evals/refunds.jsonl"))
result = await SuiteRunner(replay, concurrency=4, artefacts=Path("artefacts")).run(suite)
sys.exit(result.exit_code)
```

## The dataset is source code

It is reviewed, diffed and read again in a year, so the file says what it is:

```jsonl
{"format": 1, "name": "refunds", "version": "2026-08-01"}
{"id": "late-refund", "input": "my order never arrived", "tenant": "acme", "expected": "refund offered"}
```

The header carries the format number, the suite name and the dataset version. A reader that
meets a format it does not know refuses and names the migration, rather than dropping the
fields it does not recognise and scoring the remains. Results only compare within one
dataset version — bump it when a case changes meaning, and the comparison starts fresh
instead of silently comparing two different questions.

Every case declares its `tenant`. There is no default: a case that does not say whose data
it touches cannot be replayed safely, and a suite is exactly where a tenant leak would go
unnoticed.

## Redaction happens on the way to disk

`to_jsonl` scrubs each case through the same shape-based redaction the runtime uses, because
a customer email committed once outlives every run that used it.

Identity fields — `id`, `tenant`, `user` — are the exception: masking an id would break the
comparison the id exists for, so a credential-shaped one is refused and left for the author
to fix. Nothing is written until every case has rendered, so a refusal leaves no half-file
behind.

## What the runner fixes, and what it does not

Fixed here:

| | |
|---|---|
| Order | Results come back in dataset order, whatever order they finished in |
| Run ids | Derived from suite, version, case and seed — two machines agree without coordinating |
| Concurrency | Bounded, because an unbounded fan-out is rate-limited into flakiness |
| Digest | Covers the answers, never the timings |

Not fixed here: whether the model answers the same way twice. That is the executor's
business — a cassette-backed one replays, a live one does not. `evals` cannot import
`testing`, by design, so the test doubles stay out of the shipped judgement path and the
consumer injects what it wants:

```python
async def replay(case: EvalCase, *, run_id: str) -> Run[Answer]:
    async with cassette(f"evals/{case.id}"):
        return await runner.run(agent, case.input, tenant=case.tenant, run_id=run_id)
```

## A case that did not run is not a pass

Three outcomes, and only one of them is a measurement:

- `COMPLETED` — the agent answered. A *failed* run is still completed: a wrong answer is a
  result to measure, not a broken harness.
- `ERRORED` — the executor raised. A missing or stale cassette lands here, carrying the
  reason it gave.
- `INCOMPLETE` — the run came back non-terminal, or the suite was stopped before the case
  finished.

`SuiteResult.ok` is true only when every case completed, and `exit_code` follows it. A
harness that scored what it could and passed on the rest would report green on the day the
recordings went stale, which is the day it most needed to report red.

The runner also checks what came back: a run answering for a tenant the case did not declare
errors that case rather than being scored under the case's tenant.

## Artefacts

With `artefacts=`, each case gets `<artefacts>/<suite>/<case-id>/` holding `case.json`,
`run.json`, `result.json` and `timings.json`. Timings live in their own file on purpose —
`result.json` is what two runs are diffed on, and a wall-clock field in it would make every
diff noisy.

A case's `result.json` is written as `incomplete` *before* it starts, so a suite killed part
way leaves evidence saying so rather than a directory that reads like a pass.

A runnable version of all of the above is `examples/eval_suite.py`.
