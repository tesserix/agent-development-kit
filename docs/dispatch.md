# Dispatch — declared dependencies, derived schedule

Two lookups feed one comparison. Three retrievals feed one summary. Written as nested
sequential and parallel steps, that shape makes branches wait for each other for no reason,
and the hand-written schedule goes stale the first time a step is added.

So the dependencies are declared and the schedule comes from them:

```python
graph = Dispatch(
    (
        DispatchNode("statutes", find_statutes),
        DispatchNode("precedent", find_precedent),
        DispatchNode("compare", compare, needs=("statutes", "precedent")),
    )
)
result = await graph.run()
result.value("compare")
```

`statutes` and `precedent` run together; `compare` starts when both have returned, and is
given exactly what they returned, keyed by their names. A node that reaches around the
graph for its input has a dependency nobody declared and nobody scheduled.

## What is refused at construction

| | |
|---|---|
| A cycle | `DependencyCycleError`, naming the nodes in it. |
| A dependency no node declares | `ConfigurationError`, naming the node and the name it waits on. |
| One name used twice | `ConfigurationError` — a dependent could mean either. |
| An empty graph, or `width=0` | `ConfigurationError`. |

All of it happens while the graph is built, not while it runs. A cycle discovered at
runtime is a set of tasks waiting on each other, which is indistinguishable from work that
is merely slow.

`order` shows the derived grouping — what the graph *permits* to run together. The run is
finer than that: a node starts as soon as its own dependencies are done, without waiting
for the rest of its group.

## When a node fails

A failure is contained, not fatal.

```python
result.nodes["parse"].outcome      # NodeOutcome.SKIPPED
result.nodes["report"].blocked_by  # ('fetch',) — the failure, not the skip in between
result.values                      # only what completed
```

Everything downstream of a failure is skipped rather than run with a missing input, because
a join over an absent branch produces an answer built on nothing. Everything that did not
depend on it still finishes: after a partial failure the useful question is which parts of
the answer exist.

`failures` carries the exception itself, so a caller can re-raise it or match on its type
instead of parsing a message. Asking a failed or skipped node for its value raises
`KeyError` rather than returning `None` — a missing value read as `None` is how a partial
answer is mistaken for a whole one.

Cancellation is not a failure. Cancelling the run withdraws the question, so
`asyncio.CancelledError` propagates instead of being recorded as a node that could not
answer.

## Width

`Dispatch(nodes, width=4)` caps how many nodes are in flight at once. Without it the graph
runs everything whose dependencies are met, which is the point of declaring them; with it,
a graph whose nodes each hold a connection cannot open more than four.
