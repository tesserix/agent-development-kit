# Measuring compression on both axes at once

A compression ratio on its own is not a result. Any ratio is reachable by deleting more, and
the way a compression change actually fails is that the headline number improves while the
cases nobody put in the suite start coming out wrong. `measure_compression` runs both
measurements over the same fixtures in the same pass, per content type, against floors
checked into the repository.

```python
report = await measure_compression(DEFAULT_FIXTURES, router, solver, floors=DEFAULT_FLOORS)
print(report.table())
sys.exit(report.exit_code)
```

## An aggregate improvement never masks a regression

The verdict is the conjunction of the content types, not a mean over all cases. JSON getting
better cannot pay for code getting worse, because each kind is judged against its own floor
and one failing kind fails the report. `report.failing()` names the kinds and the case ids
inside them, so the reviewer opens the four cases that broke rather than a percentage.

## Retrieval is an outcome, not an excuse

A fixture marked `needs_detail` has its answer in exactly the part a compressor elides. The
solver is expected to expand the handle and go and get it. Answering such a case *without*
retrieving is recorded as `lost` even when the answer is right, because an answer produced
from memory rather than from the content will be wrong on the next input. Where the handle
was expanded and the answer is right the outcome is `recovered` — the compression cost a
round trip, which is a real cost, and the report shows it separately from `kept`.

## Failing closed

- A floor naming a content type the fixture set does not cover raises `EvalIncompleteError`.
  A gate that reports a pass on a type it never measured is worse than no gate.
- A solver that raises leaves the case `unmeasured`, and an unmeasured case fails its kind.
  Consumer code is allowed to break; it is not allowed to be silently skipped.
- A savings floor above zero is what catches a compressor that quietly became a pass-through.
  Accuracy would be perfect, `pass_through` would be the case count, and the kind fails.

Each case is also answered from the *uncompressed* content, recorded as `correct_whole`, so
a fixture the solver could never answer is visible as such rather than blamed on compression.

## The fixtures

`DEFAULT_FIXTURES` spans every kind the router routes — JSON, tabular, code, prose and
unknown — with synthetic, deterministic content and no network. Fixtures carrying real
customer data turn a test suite into a data store, so none of them do. A project measuring
its own corpus builds its own `CompressionFixtures`; ids are stable because a failing report
names them.

`DEFAULT_FLOORS` is what this kit holds its own compressors to. Lowering one is a diff in a
pull request, not a configuration tweak.

## Known limitations

- Correctness is substring containment against `expected`. A project wanting a judge scores
  its own answers and builds the `Answer` from that.
- The budget every case is squeezed into is a fixed fraction of what the content costs
  whole. It measures compressors under pressure; it is not a model of a real context budget.
- Retrieval is measured as "the handle was expanded", not as how many tokens the expansion
  itself cost.

A runnable version of all of the above is `examples/compression_gate.py`.
