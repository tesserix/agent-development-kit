# Usage and cost

What a run consumed and what that came to are two different questions with two different
answers, and the kit keeps them apart. `Usage` is counted — by the vendor where it reports,
by the kit where it does not. `Cost` is decided, by a price list with a date on it. Folding
them into one number loses both: the cache saving disappears into a total, and a hundred
thousand fractions of a cent drift away from the invoice.

## `Usage` — what was consumed

```python
Usage(
    input_tokens=1000,      # everything sent, cache reads included
    cached_tokens=400,      # of that, what the vendor served from its own cache
    cache_write_tokens=200, # what it charged to put into that cache
    output_tokens=500,      # generated and shown
    reasoning_tokens=300,   # hidden reasoning, beside the answer rather than inside it
    image_units=2,          # images, tiles or audio seconds, priced per unit
    source=CountSource.PROVIDER,
)
```

Two containment rules, and they are not the same rule:

- `cached_tokens` is **part of** `input_tokens`, because those tokens were sent. Pricing
  bills `input_tokens - cached_tokens` at the fresh rate and the remainder at the cache rate.
- `reasoning_tokens` is **beside** `output_tokens`, not inside it. OpenAI reports reasoning
  within its completion total, so the adapter subtracts it; Gemini reports the two apart and
  the adapter leaves them apart. One workload therefore reads the same way whoever answered
  it, which is the point — a field that lands in `extras` under one vendor and in a column
  under another cannot be budgeted against.

`extras` holds usage a vendor reports that the kit does not model. Nothing in the kit reads
it; it is evidence, not input.

### Who counted

`source` is `PROVIDER`, `TOKENISER` or `HEURISTIC`, and `usage.estimated` is true for the
last two. An endpoint that reports nothing is counted with the model's own tokeniser and
marked `TOKENISER`; a prompt the kit never got a count for at all is characters over a
constant and marked `HEURISTIC`. Adding two usages keeps the weaker of the two sources: a
total is only as trustworthy as its worst part, and a budget enforced against a guess
presented as a count is not enforcement.

### Failed attempts are on the ledger

A vendor that read a prompt and then rate-limited still charged for reading it. Every
`ATTEMPT_FAILED` event carries a `HEURISTIC` usage for what that attempt burned, and the
run's total includes it. A run that never got an answer still says what it spent.

## `Cost` — what that came to

Every component is a `Decimal`. Vendors quote in millionths of a dollar per token and a run
makes enough calls for binary floating point to disagree with the invoice. Components stay
separate and unrounded; `quantised()` rounds for presentation, which is the only place to
round.

`confidence` says how much of the number is known:

| | |
|---|---|
| `COUNTED` | Vendor-reported usage at a price the list knows. |
| `ESTIMATED` | A known price applied to counts the kit worked out. |
| `UNKNOWN` | No price for this model on this day. |

`UNKNOWN` has zero components because there is nothing to put in them, not because the call
was free — `Cost.unknown()` is never a silent zero. Totalling keeps the weakest confidence,
and totalling two currencies raises rather than producing a number true in neither.

## Prices

A price list is not a constant, so a `PriceCard` carries the day it took effect, the request
shape it answers for, and the rate:

```toml
[[cards]]
ref = "anthropic:claude-sonnet-4-5"
effective_from = 2026-03-01
rate = { input_per_mtok = "2.40", output_per_mtok = "12.00", cache_read_per_mtok = "0.24" }
```

Point `ADK_PRICE_LIST` at the file, or pass the path to `price_list()`. Nothing is
discovered by convention: a deployment billing against a file nobody named is one where the
answer to "what is this costing" lives on somebody's laptop. An unreadable or wrong-shaped
file is a `ConfigurationError` rather than a quiet fall back to list price — that gets found
on the invoice.

The kit ships the catalogue snapshot's rates, effective from the snapshot day and not
before. A deployment costing older runs supplies its own dated cards; back-dating the
shipped ones would be a fiction about what was charged.

Selection is by date and request shape, never by position. The narrowest card wins — the
batch tier where a batch call asked for it, then the highest long-context threshold the
prompt clears — and among cards of one shape, the latest already in force. A price change is
a new card, never an edit: overwriting one rewrites what last week's runs cost. Two cards
for one shape on one day is a `ConfigurationError`, because a deployment cannot bill both.

`overridden_by()` lays a negotiated list over the shipped one, replacing every shipped card
for the models it names. A negotiated rate that only won until the vendor's next list price
landed would be no agreement at all.

## Reading a run

```python
run = await runner.run(agent, "…", tenant="acme")
print(run.usage.input_tokens, run.usage.output_tokens, run.usage.source)
print(run.usage.cost.total if run.usage.cost else "unpriced")
```

The run total is summed as the run goes, never recomputed by the caller. `cost_of()` prices
a usage directly where something outside a run needs the number:

```python
cost_of(usage, "anthropic:claude-sonnet-4-5", at=date(2026, 8, 7))
```

A model no card covers on that day warns `UnknownPricing` and returns `Cost.unknown()`. A
deployment that requires every call to be priced turns that warning into an error with a
filter.

## What this does not do

Ceilings are not enforced here — that is the budget policy's job, and this module only
supplies the number it enforces against. Chargeback attribution is not here either.

Exercised by [`examples/cost.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/cost.py) and `tests/test_cost.py`.
