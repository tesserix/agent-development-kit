# Multi-agent tracing and cost attribution

A supervisor delegates to three workers, one of which calls a peer over A2A and one of which
drops an activity on a queue. Five processes, five traces, five cost figures — and the one
question anybody asks afterwards ("what did that run cost, and which agent spent it?") has no
answer, because no artefact covers the whole run.

This page is about the artefact that does. [`cost-attribution.md`](cost-attribution.md) covers
one run's spend read off its own events; this covers a run made of several.

## The context that crosses the gap

`Run` has no parent. `TraceContext` carries the link a run cannot: the root, the parent, the
depth, the pattern and — for a fan-out — which branch.

```python
root = TraceContext.root_of(supervisor_run)
leg = root.child(run_id="w0", agent="worker0", pattern=Pattern.FAN_OUT, branch="leg0")
```

`Pattern` names why a hop happened: `root`, `delegation`, `handoff`, `fan_out`, `peer`,
`activity`. It is on every span, so "the fan-out costs four times what the delegation does" is
a query rather than an archaeology exercise.

A child may run for a different tenant than its parent — a run acting on somebody's behalf
bills the tenant it ran as. `child(..., tenant=...)` sets that without widening anything above
it.

## Leaving the process

Over NATS, HTTP or A2A the context travels as one header, `adk-trace`:

```python
publish(subject, payload, headers=leg.carried())
...
here = TraceContext.restored(headers, run_id="a1", agent="activity")
```

Header values are percent-encoded, so an agent named `x;tenant=globex` cannot rewrite the
field next to it and move somebody's money.

**A missing or unreadable header is recorded, not raised.** `restored` gives back a new root
with `broken=True`, and `adk.trace_broken` goes on the span. Dropping the work because its
trace went missing loses the spend as well as the trace; a break that is marked is a break
somebody can find. Breaks carry forward to children, so a node three hops down is never read
as though it were attached.

A `pattern` this version does not know reads as `activity` rather than failing to parse. A new
shape in a newer sender must not sever the chain.

## The tree, and what it refuses

```python
assembled = tree([node_of(supervisor_run, root), *workers, peer])
```

`node_of` reads a participant that ran here — usage, cost, metered steps, its own clock's
timings. `peer_node` takes a participant that ran somewhere else, from whatever it reported
back; it carries no records, because a peer reports a total and inventing steps for it would
put a shape on the bill that nobody measured.

`tree` refuses anything it cannot read as one run, with `AttributionError.reason`:

| `reason` | What happened |
|---|---|
| `empty` | No participants, so no run |
| `no_root` | The first node names a parent |
| `two_roots` | A later node has none — two trees, not one run |
| `duplicate` | One run id twice; a participant totalled twice |
| `orphan` | A parent that is not in the tree |

Each of those would otherwise produce a tree that silently drops a participant, and a dropped
participant is dropped spend.

## Totals that say what is missing

```python
totals = assembled.totals
totals.cost            # the attributed money, in the root's currency
totals.unattributed    # who is outside that figure
totals.lower_bound     # True when anything is
```

A worker that crashed before reporting is **named, never counted as zero**. An unknown spend
folded in as zero is how a budget ceiling stops meaning anything: the figure looks complete and
is not. `render` says so in the summary line — `lower bound: w0 reported nothing`.

Money is never converted on the kit's authority. A peer billing in another currency reaches the
total only through a `Rate` somebody recorded — source, timestamp, multiplier, both currencies
— and the converted figure carries `CostConfidence.ESTIMATED` and is listed in
`totals.converted`. Without a covering rate the peer is held as unattributed, which is a smaller
lie than a number that is true in neither currency.

Two processes do not share a clock. A child whose end precedes its start is marked `skewed` and
its `latency_ms` is `0.0` — the disagreement is a fact about the clocks, recorded rather than
corrected into a negative duration.

Roll-ups over the local participants use the same attribution surface as a single run:
`assembled.totals_by("agent")`, `totals_by("tenant", "model")`, `assembled.by_step()`.

## Export

```python
record_tree(assembled, tracer=tracer, meter=meter, dimensions=Dimensions(...))
```

One `adk.participant` span per participant, and counters **per participant, whatever the
sampler did**. That is what stops a wide fan-out losing its cost to sampling: the money never
travels on a span in the first place. Counters carry `attributed=false` for a participant
outside the total, so a dashboard can show the gap rather than absorb it.

A participant with no tenant fails the export closed — `AttributionError(reason="no_tenant")`,
raised before the first span leaves. Spend exported with no owner is worse than spend not
exported: it fills a bill with money nobody can claim, and the owner is unrecoverable once the
run is gone.

Nothing here is wired into the run loop. Export reads a finished tree, so a collector outage
cannot reach into the run that produced the numbers.

## Reading one afterwards

```
$ python -m your_app.inspect run_1
planner (run_1)  root  0.200000 USD, 150 tokens, 812 ms
  worker0 (w0)  fan_out/leg0  0.100000 USD, 150 tokens, 400 ms
  peer (p1)  peer  0.050000 USD, 150 tokens, 3000 ms
total 0.350000 USD over 3 participants, 450 tokens
```

`tesserix_adk.cli.inspect.main(argv, lookup=...)` draws it. Where the trees are kept is the
deployment's business, so it supplies the lookup; the kit supplies the rendering and the exit
codes: `0` drawn, `1` nothing kept under that id — which is not the same as a run that spent
nothing — `2` a command line it could not read. There is no console-script entry point; a kit
that installs a global `adk` binary fights with whatever the consumer already ships.

## Known limitations

Assembly is in-process: the kit builds a tree from nodes you hand it and does not fetch
participants from anywhere. Peer figures are taken on trust — a peer that under-reports is
under-reported here, and the tree records what was claimed, not what was verified. `Rate` is
supplied, never looked up.

Runnable: [`examples/multi_agent_trace.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/multi_agent_trace.py).
