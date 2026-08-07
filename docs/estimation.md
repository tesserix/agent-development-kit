# Estimation

What a run is expected to cost, worked out before it starts, from what runs of that agent
have actually done. It is the number you show a person, check against what is left of a
budget, or refuse on — not a number the kit invented to have something to display.

An estimate is never a guarantee. Everything below exists to keep it from being read as one.

## Getting one

```python
from datetime import date

from tesserix_adk.models.pricing import pricing_at
from tesserix_adk.runtime import InMemoryHistory, estimate_run

estimate = estimate_run(
    researcher,
    "what happened to the Antikythera mechanism?",
    provider=provider,
    pricing=pricing_at(date.today()),
    history=history,
)
estimate.point.total     # the typical case
estimate.low.total       # the tenth-percentile case
estimate.high.total      # the ninetieth
estimate.confidence      # measured | inferred | unknown
```

The provider is asked for one thing: `count_tokens` on the assembled prompt. It is never
asked to complete anything. An estimate that costs a paid round trip is a bill for asking
about a bill.

## The confidence ladder

| Value | What it means |
|---|---|
| `measured` | Built from finished runs of **this agent at this version**. |
| `inferred` | Built from other versions of this agent, or from the kit's defaults. |
| `unknown` | Nothing prices this model. Token counts are real; the money is not. |

`assumptions.runs_observed` says how many finished runs are behind it. Zero means the
shape came from the kit's defaults, and the estimate says so rather than looking confident.

## What it rests on

`estimate.assumptions` is the whole argument, in a form somebody can disagree with:

| Field | Where it comes from |
|---|---|
| `model` | `agent.model`, or its task class where routing decides later. |
| `prompt_tokens` | The provider's own count of the assembled first prompt. |
| `iterations`, `output_tokens`, `tool_calls` | The median of the recorded runs. |
| `iterations_high`, `tool_calls_high` | The ninetieth percentile — what a ceiling is set to. |
| `tool_result_tokens` | `TOOL_RESULT_TOKENS`, what one tool result adds to the next prompt. |
| `cached_fraction` | How much of a repeated prompt the recorded runs had served from cache. |
| `runs_observed` | How many finished runs the distribution came from. |

Only completed runs are recorded. A run that was refused or died partway says nothing
about what a run of that agent costs.

Prompt tokens are not `prompt_tokens × iterations`. Each turn carries the previous turns'
output and tool results forward, so the accumulated context is charged too — which is why a
five-turn run costs more than five times a one-turn run, and why an estimator that ignores
it reads low on exactly the runs worth refusing.

## Turning one into a ceiling

An estimate is not a limit until somebody says it is:

```python
limits = estimate.as_limits(headroom=Decimal("1.25"))
```

Headroom applies to the **money only**. What varies between an estimate and an invoice is
what tokens cost, not how many turns the agent was allowed, so the shape ceilings
(`max_iterations`, `max_tool_calls`, the token ceilings) come from the high case as they
are.

## Refusing before the first call

```python
from tesserix_adk.runtime import affordable, refuse_unaffordable

decision = affordable(estimate, remaining)     # a BudgetDecision, naming what did not fit
refuse_unaffordable(estimate, remaining)       # or raise BudgetExceededError
```

Both check the **high case**. A ceiling the typical run fits and a bad one does not is not
a ceiling. A limit reached mid-run has already spent whatever it took to get there; this
is the check that costs nothing.

## Showing it to a person

```python
record = approval_for(estimate, agent, run_id=run_id, tenant=tenant)
```

The approval reason carries the range and the confidence, not a single figure — a single
number shown to somebody reads as a promise nobody made. The assumptions travel as the
record's arguments, so the approval is auditable against what was actually assumed.

## Being wrong, measured

```python
c = calibrate(estimate, finished_run)
c.ratio          # actual over estimated, or None where nothing priced the run
c.within_range   # did the actual land between low and high?
```

Nothing clamps the ratio. A run that cost fifty times its estimate reads as fifty, because
the outliers are the ones the estimator has something to learn from. `within_range` is the
number worth tracking across a fleet: a range that contains the actual less than eight
times in ten is a range that is too narrow to refuse on.

## Multi-agent runs

An estimate is **parent-only** by default — `estimate.scope` says so. A supervisor that
will start three children is not estimated by estimating the supervisor:

```python
whole = supervisor_estimate.with_children(*child_estimates)
whole.scope         # with_children
whole.confidence    # the weakest of the parts
```

Confidence does not survive averaging: one child nobody has measured makes the total
`inferred`, and one child nobody prices makes it `unknown`.

## Prices

Estimation holds no opinion about where prices live — it takes a `Pricer`, a callable from
usage and model to `Cost`. `pricing_at(date, ...)` is the shipped adapter over the kit's
dated price list; a deployment with negotiated rates passes its own list, or its own
callable entirely. See [`docs/cost.md`](cost.md).

## Where the history comes from

`RunHistory` is a one-method protocol:

```python
def observed(self, agent_name: str, version: str) -> Observed | None: ...
```

`InMemoryHistory` implements it for tests and small deployments. A real one reads the runs
the deployment already stores. Without any history the kit's defaults are used and
`confidence` is `inferred` — never `measured`.

## Known limitations

- An unpriced model raises `EstimateUnavailableError` rather than returning a plausible
  figure. `allow_unknown=True` returns the token counts with the money marked unknown.
- The estimate prices the model the agent names. Where routing or fallback sends the run
  somewhere else, the actual is billed to whoever answered — see
  [`docs/cost-attribution.md`](cost-attribution.md).
- Repair turns, retries after a provider failure, and streamed runs that abort all land in
  the actual, not the estimate, beyond what the recorded runs already show.
