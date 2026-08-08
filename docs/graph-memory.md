# Graph memory

Relationship-shaped memory — who travelled with whom, which supplier failed which
booking, what an entity used to be — over a temporal knowledge graph.

It is the one adapter where a write costs money. Extraction is a model call, so the
first thing a write does is ask whether this tenant may still spend.

```python
graph = GraphMemoryStore(
    open_graphiti(GraphSettings(backend="neo4j", uri=SecretStr(os.environ["GRAPH_URI"]),
                                model="extract-small")),
    extractor=EntityExtractor(provider, settings=settings, clock=clock, meter=meter),
    companion=durable,          # everything that is not a relationship
    budget=run.budget,
    clock=clock,
)
```

Install it with the `graphiti` extra. The engine is injected, so Graphiti over Neo4j or
FalkorDB is what the kit ships a wrapper for and not what the adapter depends on.

## What the graph answers, and what it does not

`relations(scope, as_of=…)` is the read the graph exists for. Everything else — working
memory, profiles, the episode records themselves, semantic recall over them — goes to
`companion`, the same composition `RoutedMemoryStore` uses. A graph is a poor key-value
store and forcing four kinds of memory into one shape helps nobody.

`log` and `index` do both: the record lands in the companion, and the relations it
carries land in the graph.

## Two ceilings

`BudgetPolicy` bounds one run. `ExtractionMeter` bounds a tenant across every run they
have, which is the number that turns up on the invoice.

```python
meter = ExtractionMeter(ceilings={"acme": Decimal("50.00")})
```

A tenant absent from the mapping may spend nothing. An unlisted tenant is a
configuration gap, and defaulting a gap to unlimited is how the gap gets discovered.

The check runs before the call, so an exhausted ceiling raises `BudgetExceededError`
naming the limit and the spend to date, with no model call made and nothing half-written
into the graph. Every extraction that does run is reported on `charges` with tokens,
latency and cost, priced by the provider where it priced the call and by `rate_per_1k`
where it did not.

## When the model returns nonsense

`ExtractionError`, nothing committed, the raw payload on the error. A subgraph half of
which the model invented reads exactly like one it derived from the text, and there is no
later signal that tells them apart. An edge naming a node the extraction did not declare
is the same refusal: that is the model inventing an entity.

The episode itself is wrapped as untrusted data on the way in. Retrieved text reaching an
extraction model is the same injection surface as retrieved text reaching any other model.

## One entity, one tenant

Node ids are tenant-qualified, so the same person named in two sessions is one node and
the same name in two tenants is two. That boundary is in the id rather than in a filter
applied afterwards.

## Time is an interval

An edge carries `valid_from` and `valid_to`. `valid_to` of `None` means it still holds,
which is the common case, so an `as_of` read after the edge began returns it rather than
skipping it for a missing value.

## Paying twice for one extraction

A backend that goes away after the extraction was paid for does not cost the extraction:
the subgraph goes to `pending` and `retry_pending()` commits it without calling the model
again.

```python
await graph.log(scope, record)      # MemoryUnavailableError; graph.pending holds the work
await graph.retry_pending()         # commits, no second model call
```

## Writes that do not block the turn

`enqueue` returns before the model is called; `flush` extracts and commits what is
waiting; `writing()` drains in the background for as long as the block runs.

```python
async with graph.writing():
    await graph.enqueue(scope, record)
```

The queue is bounded, and a full one raises `WriteQueueFullError` rather than dropping a
write. A dropped write is one nobody will ever look for; refusing is the caller's decision
to make — wait, shed, or write through with `log` and pay the latency.

## Erasure

`erase` drops the partition and the companion's rows, and the receipt names both
adapters. The extraction is registered as a derivation of the record it came from, so it
is reached by the same erasure rather than surviving it.

## Not here

**Relationship-aware ranking beyond the engine's own.** Re-ranking a graph result by hand
is a second retrieval system wearing the first one's results.

**Migration from relational memory into the graph.** Replaying a year of episodes through
an extraction model is a bill, not a migration, and it belongs to whoever is paying it.
