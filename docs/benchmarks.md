# Benchmarks

A performance gate has two ways to die. One fails on a noisy runner and is switched off
within a fortnight; the other passes whatever a noisy runner produces and defends nothing.
This harness is built around that: it measures its own noise alongside the numbers, refuses
to draw a conclusion the noise could explain, and gates only on metrics that mean the same
thing on somebody else's machine.

```bash
make bench          # measure and compare against the committed baseline
make bench-quick    # fewer rounds and iterations, for a check before opening a change
make bench-record   # record a new baseline — a reviewed commit of its own
```

## Exit codes

| Code | Meaning | What CI does |
|------|---------|--------------|
| 0 | Everything held | Passes |
| 1 | A metric regressed past its threshold | Fails, naming scenario, metric and delta |
| 2 | The run was too noisy to say | Passes, with the reason in the job summary |
| 3 | The suite could not be loaded | Fails — a typo in `--suite` must not read as an empty run |

## What is measured

`benchmarks/suite.py` holds the paths a consuming product pays for on every call:
`single-turn`, `tool-turn`, `streaming`, `structured-output`, `embedding-batch` and
`run-fanout`. Every scenario runs against scripted providers and local fakes. Nothing here
needs a credential, and a scenario that reached for one would be measuring somebody's
network rather than the kit.

Per scenario: `latency_p50` / `p95` / `p99`, `throughput`, `peak_bytes`, `allocations`, and
`tokens` where the scenario reports the prompt it assembled.

## Which metrics gate, and why only those

The committed baseline records `tokens` and `peak_bytes`. Everything else is measured and
printed on every run with its value, and compares as `unrecorded` — visible, never a gate.

- **`tokens` has a threshold of zero.** Prompt assembly growing by one token is a change
  somebody made, never machine noise, and every consumer pays for it on every call.
- **`peak_bytes`** is the high-water mark of a round's working set. It moved by under
  0.6% across repeated runs here, which is a number worth defending.
- **Wall clock** — `latency_*` and `throughput` — is a property of the runner as much as of
  the code. A baseline recorded on a laptop fails on a shared CI runner having a bad
  afternoon, and a gate that cries wolf gets deleted.
- **`allocations`** counts blocks still live after a collection: real, but on a scenario
  that retains almost nothing it sits near a handful of blocks, where one block reads as a
  ten-percent regression. It is reported, not gated.

Recording more metrics is a decision, not a code change: `make bench-record` takes
`--only`, and dropping it records everything the run measured.

## Variance controls

- **Warm-up** iterations run before each round and are never timed. Tokens are bracketed
  around the measured iterations alone, so the warm-up's own consumption is not billed to
  the round it was warming up for.
- **Rounds** — the whole measured block repeats, and the *slowest* round is discarded when
  there are at least three, so one stall does not become the result.
- **Percentiles** are nearest-rank over the kept iterations. **Spread** is the relative
  standard deviation of the kept round means: the run's own noise, recorded beside the
  numbers.
- **Memory is measured apart from the timings**, because tracing every allocation costs
  more than most of what is being measured. Tracing starts *after* the warm-up and a
  collection runs before the count, so neither a cache built on the first pass nor garbage
  nobody has swept is billed to the scenario. The median across rounds is taken.
- **Inconclusive, not green.** Where the spread exceeds the noise ceiling *and* covers the
  delta, the verdict is `inconclusive` with what a conclusive run would need — a quieter
  runner, or more rounds. Exit code 2.
- **A floor beside the threshold.** A percentage of a very small number is not a
  measurement, so each metric also carries the absolute change below which no verdict is
  drawn (`DEFAULT_FLOORS`).
- **Size is part of the comparison.** `peak_bytes` scales with iterations per round, so a
  shortened run — `make bench-quick`, `--iterations` — reports it as inconclusive rather
  than judging it against a full-size baseline.

Baselines are keyed by `(scenario, interpreter)`. A run on 3.12 never borrows 3.13's
numbers; it reports `unrecorded` instead.

## Updating a baseline — deliberately

A harness that re-records what it just measured ratchets performance downwards and reports
green every time. So:

1. A check run **never** writes the baseline. Not on success, not on failure, and it does
   not create one that was absent. Two tests pin this.
2. `make bench-record` writes, and writing merges: re-recording on one interpreter leaves
   the other's numbers alone.
3. A baseline moves in **its own commit**, with the reason in the message — the work got
   genuinely more expensive and here is what bought it, or the machine changed. Reviewers
   should be able to read why the number is allowed to be worse without opening the diff of
   anything else.

## Reproducing a CI result locally

`make bench` is what CI runs. The numbers will differ from the runner's; the verdicts on
the gated metrics should not, because those are the metrics that travel. Where a local run
disagrees, run it again — a single inconclusive verdict is the harness saying the machine
was too busy, not that the code changed.

## Adding a scenario

Add a `Scenario` to `benchmarks/suite.py`, run `make bench` to see it reported as
`unrecorded`, then `make bench-record` in a commit of its own. A scenario must be
deterministic, must not touch the network, and should measure a path a consumer actually
pays for.
