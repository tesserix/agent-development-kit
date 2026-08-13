# Escalation ladder

Multi-agent designs are adopted before they are earned. Every agent added to a design adds
a model call, a context handoff, a failure mode and a trace hop, and most tasks that get
split would have been answered better by one agent with good tools and a workflow that
does not involve a model at all.

This page is the bar. It does not forbid any shape — the kit ships all of them, and
[`delegation.md`](delegation.md), [`planning.md`](planning.md),
[`parallel.md`](parallel.md) and [`routing.md`](routing.md) document them. It requires
each step up to be **paid for by a measurement somebody took**, so that a design review has
something to reject a proposal with other than taste.

## The rungs

| Rung | Shape | Kit surface |
|---|---|---|
| 1 | One agent with tools | `Agent`, `@tool`, `AgentRunner` |
| 2 | One agent plus a deterministic workflow | `dispatch`, `Planner` over a fixed step list |
| 3 | A router with specialist agents | `Roster`, `Specialist`, `TaskClass` |
| 4 | Collaborating agents | `Supervisor.delegate`, `handoff`, `fan_out` |

Start at rung 1. Climb one rung at a time, and only after the bar below has been cleared on
the suite the agent was agreed against — `AgentDefinition.evaluation_suite`, see
[`agent-definition.md`](agent-definition.md).
Skipping a rung is not a shortcut: rung 3 measured against rung 1 cannot tell you whether
the win came from specialisation or from the workflow you also added on the way.

## The bar for each step

Each bar is a delta on a suite, not an opinion about a transcript. All three of quality,
cost and latency are stated, because a change that improves one by wrecking another has not
improved anything — it has moved the problem to whoever pays the bill.

### Rung 1 → 2 — add a deterministic workflow

**Bar:** suite pass rate improves by **≥ 5 percentage points absolute**, at **no more than
+10% cost per task** and **no more than +10% `latency_p95`**.

**Or, independently sufficient:** the workflow removes a *class* of failure — a step that
must always run, an ordering that must always hold, a rule with a legal or financial
consequence. A rule that must be true every time is code, and its bar is that it is now
impossible to violate rather than unlikely to be.

Most wins attributed to multi-agent designs are actually this rung. Fixing the order of
steps, forcing a retrieval before an answer, or validating an argument before a call costs
no extra model calls at all.

### Rung 2 → 3 — split into a router with specialists

**Bar:** the weakest slice of the suite improves by **≥ 10 percentage points absolute**, no
other slice drops by more than **2 points**, and the whole is **no more than +15%
`latency_p95`** and **+20% cost per task**.

"Slice" means a labelled subset of the suite — one intent, one document type, one language.
A router pays for itself by lifting the slice a generalist was worst at. If no slice is
distinctly worse than the mean, there is nothing for a router to route on, and the win you
measured was noise or prompt tuning that rung 2 could have had for free.

### Rung 3 → 4 — let agents collaborate

**Bar, quality route:** **≥ 10 percentage points absolute** on the suite over rung 3, at
**no more than +50% cost per task**.

**Bar, latency route:** **≥ 30% reduction in `latency_p95`** at **no quality regression**
(within the harness's noise verdict), for a task with genuinely independent branches.

Cost is expected to rise here and the bar says so explicitly. What is not negotiable is
that the rise is **attributed**: `Aggregate.spent` reports per branch, including branches
excluded from the answer, and a fan-out whose cost cannot be split per branch has not
cleared this bar regardless of its quality.

## Measuring it

Every figure above is available from surfaces the kit already ships. Nothing here needs a
new harness.

| Figure | Where it comes from |
|---|---|
| Suite pass rate, per slice | The gold set named by `tesserix_adk.core.AgentDefinition.evaluation_suite` |
| Cost per task | `tesserix_adk.core.Usage`, priced through `tesserix_adk.core.cost` |
| Cost per participant | `tesserix_adk.runtime.Aggregate.spent`, `tesserix_adk.core.budget.RunBudget` |
| `latency_p50` / `latency_p95` / `tokens` | `make bench` over `benchmarks/suite.py` |
| Was the delta real | The bench harness's own noise verdict — see [`benchmarks.md`](benchmarks.md) |

Three rules about the measurement itself, which matter more than the thresholds:

- **An inconclusive comparison is not a justification.** The harness reports `inconclusive`
  when its measured spread covers the delta ([`benchmarks.md`](benchmarks.md)). That is a
  verdict, not a near-miss, and it does not clear a bar.
- **A delta on a handful of cases is not a delta.** Fewer than 50 suite cases cannot
  distinguish 5 points from a coin. State the case count next to the number.
- **Compare adjacent rungs, on the same suite, on the same day.** A rung-4 design measured
  against a rung-1 baseline from three months ago measures the three months.

## Roles are not services

Router, planner, executor, reviewer and human approver are **roles a run passes through**,
not processes that stay running. One deployment moves through several of them, and none of
them implies a separate agent, a separate container or a separate trace root.

| Role | What plays it | Doc |
|---|---|---|
| Router | Capability match over a `Roster`, or a `TaskClass` an operator resolves | [`delegation.md`](delegation.md), [`routing.md`](routing.md) |
| Planner | `Planner`, which cannot dispatch | [`planning.md`](planning.md) |
| Executor | `AgentRunner` | [`run-loop.md`](run-loop.md) |
| Reviewer | The guardrail chain — deterministic, ordered, fail-closed | [`guardrails.md`](guardrails.md) |
| Human approver | The approval gate and the autonomy grant | [`tool-approval.md`](tool-approval.md), [`autonomy.md`](autonomy.md) |

Reading these as four permanently running agents is the most common way a design arrives at
rung 4 without ever deciding to. "We need a reviewer agent" is nearly always "we need a
guardrail", and a guardrail is cheaper, ordered, and cannot be argued with by the thing it
reviews.

## Reasons that need no measurement

Two reasons justify a split on their own, because what they buy is not quality and no suite
would show it.

- **A distinct tool grant.** If one part of the work must hold `refund` and the rest must
  not, a specialist holding it is least privilege, not architecture. One agent holding the
  union of every grant is the design this replaces.
- **A distinct trust boundary.** Work over untrusted content should run in an agent that
  holds nothing worth reaching, so a prompt injection lands somewhere with no authority.
  See [`trust-boundary.md`](trust-boundary.md) and [`tool-results.md`](tool-results.md).

A third is weaker but real: **distinct ownership** — a different owner, a different
`evaluation_suite` and a different release cadence make a shared agent a shared bottleneck.
Weigh it, do not assume it.

All three still cost a hop. They justify the split; they do not make it free, and the cost
and latency figures should still be recorded so the next review starts from a number.

## Deterministic rules stay in code

A rule with a known answer is never a model call. Eligibility thresholds, tax, retry
policy, field validation, ordering constraints and routing on an enum belong in code, where
they are testable, free, instant and identical on every run.

The failure this prevents is not cost. It is a business rule that holds 97% of the time and
whose 3% is unreproducible.

## A worked example

The same task — *"classify an inbound support email, look up the account, and draft a
reply"* — built at rung 2 and at rung 4, over a 120-case suite.

> The figures below were taken against scripted providers to make the comparison
> reproducible offline. They are here to show the shape of the decision and the evidence a
> review should expect, not as a benchmark of any model or vendor.

| | Rung 2 — one agent + workflow | Rung 4 — supervisor + 3 workers |
|---|---|---|
| Suite pass rate | 88.3% (106/120) | 90.8% (109/120) |
| Cost per task | 1.00× (baseline) | 2.4× |
| `latency_p95` | 1.00× (baseline) | 1.9× |
| Model calls per task | 2 | 7 |
| Failure modes | provider error, tool error | those, plus handoff context loss, partial aggregate, delegation ceiling |

**Verdict: rejected.** +2.5 points is inside what the suite can distinguish at 120 cases,
and it is bought with 2.4× the cost, 1.9× the latency and three new failure modes. The bar
for rung 4 is 10 points, and this is not close to it.

**What the same evidence did justify:** the per-slice breakdown showed one slice —
non-English mail — at 61%, twenty-seven points below the mean. That is a rung-3 case, and
routing that slice to a specialist lifted it to 84% while no other slice moved by more than
a point. Rung 3 cleared its bar; rung 4 never had one to clear.

## In review

A design proposing a rung answers these, in writing:

1. Which rung is it, and which rung is it being compared against?
2. Which suite, how many cases, and what is the per-slice breakdown?
3. Quality, cost and latency deltas — all three, with the harness's noise verdict.
4. If it clears no bar: which reason from *Reasons that need no measurement* applies?
5. Which deterministic rules did this design turn into model calls, and why?

An answer of "it seemed better in the transcripts we looked at" is the answer this page
exists to reject.

## Known limitations

- The thresholds are chosen, not derived. They are set where a delta is large enough to
  survive a 120-case suite and a change of provider. A team with a 2000-case suite can
  justify smaller ones; a team with 30 cases cannot justify any of them.
- The ladder measures a design against another design on one suite. It says nothing about
  which suite is the right one, and a suite that does not contain the cases users actually
  send will clear any bar you like.
- Cost per task is compared as a ratio, because absolute money depends on dated prices
  ([`cost.md`](cost.md)) that move underneath a stored baseline.
- Nothing here is enforced in code. It is a review instrument, and a review that does not
  ask for the numbers gets designs that do not have them.
