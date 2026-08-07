# Naming the job, not the model

A model id written at a call site is a cost decision compiled into a product. Retuning it
means a code change in every consumer that wrote it down, and there is no way to say that
a classification step wants the cheap model while a planning step wants the reasoning one.
An agent names a **task class**; where that resolves to is configuration an operator owns.

```python
Agent(name="triage", instructions="Classify the ticket.", task_class=CHEAP)
```

Nothing in that agent changes when the deployment moves `cheap` to another vendor.

## The vocabulary

`CHEAP`, `SMART` and `REASONING` ship because they are the three every consumer reinvents.
`TaskClass` is a `str` subclass rather than an enum because the set is open: a deployment
that meters `transcription` separately writes `TaskClass("transcription")` and routes it.

An agent also states what it needs of whatever answers:

```python
Agent(
    name="exhibit-reader",
    instructions="Describe the exhibit.",
    task_class=CHEAP,
    requires={Capability.VISION},
)
```

Requirements are checked **before** the choice, not after it. A router that picks first and
validates second has already recorded the wrong model against the run.

## The table

One TOML file, one version, rules in any order:

```toml
version = 1

[[rules]]
task_class = "cheap"

  [[rules.candidates]]
  provider = "openai"
  model = "gpt-4o-mini"
  capabilities = { tool_calling = true, streaming = true, context_window_tokens = 128000 }

  [[rules.candidates]]
  provider = "anthropic"
  model = "claude-haiku-4-5"
  capabilities = { tool_calling = true, vision = true, context_window_tokens = 200000 }

[[rules]]
task_class = "cheap"
tenant = "acme"
agent = "planner"

  [[rules.candidates]]
  provider = "anthropic"
  model = "claude-sonnet-4-5"
  capabilities = { tool_calling = true, vision = true, context_window_tokens = 200000 }
```

```python
router = TableRouter(routing_table())          # path argument, else ADK_ROUTING_TABLE
runner = AgentRunner(provider=default, providers={"openai": ..., "anthropic": ...}, router=router)
```

`routing_table()` reads the path it is given, then `ADK_ROUTING_TABLE`. Nothing is
discovered by convention: a deployment routing by a file nobody named is one where the
answer to "which models is this billing" lives on somebody's laptop.

### Precedence

The **narrowest matching rule wins** — `class + tenant + agent`, then one of the two, then
the bare class. Within a rule the candidates are tried **in written order**, because that
order is the operator's preference and reordering it silently is choosing on their behalf.
The first candidate meeting the requirements answers.

Two rules at the same scope are refused at construction. Otherwise which candidates answer
a class would depend on rule order rather than on anything an operator wrote.

### Entitlements

Where a deployment restricts who may be routed where, the router carries it:

```python
TableRouter(table, entitlements={"acme": frozenset({"openai:gpt-4o-mini"})})
```

A tenant with no entry is unrestricted. A tenant with an empty entry is entitled to
nothing — a thing an operator can mean, so it is not read as "unset".

## Nothing falls back

A class with no rule, a rule whose candidates cannot do the work, and a pin nothing knows
are all `NoEligibleModelError`. There is no downgrade path: a model that cannot do the job
is not a cheaper way to do it, and an answer produced by a model the run record does not
name is worse than no answer. The error carries `task_class`, `unsatisfied` and every
`rejected` candidate with the requirement it failed, so "why not that one" is in the
record rather than in somebody's head.

## Validated at boot

`RoutingTable` refuses, at construction, a table that would fail on a later request:

| Refused | Because |
|---|---|
| A version this kit does not read | The schema moved under the file |
| No rules at all | It would refuse every class on the first request |
| Two rules at one scope | Rule order, not the operator, would decide |
| A rule with no candidates | It routes a class to nothing |
| A candidate declaring no capabilities | Silence is not a claim; it satisfies no requirement |
| A candidate the model catalogue does not list | A retired id answers every request with an error |

The last check applies only to providers the catalogue covers. A self-hosted endpoint or a
proxy is left alone: absence of a card there is absence of a card, not evidence the model
is gone.

## Pinning

A pin reproduces a run against one model:

```python
router.resolve(CHEAP, pinned=ModelRef(provider="openai", model="gpt-4o-mini"))
```

It is still checked against the requirements and the tenant's entitlements. A pin is a
choice between known models, not a way past the checks. The table's record of a model wins
over the published card, since a deployment that narrowed one — a self-hosted endpoint
serving smaller weights — knows something the catalogue does not.

## On the run

A routed run records a `MODEL_ROUTED` event naming the chosen model and one line of
reasoning, and `run.model` is what actually answered:

```
model_routed  openai:gpt-4o-mini  cheap -> openai:gpt-4o-mini (rule cheap, 2 considered)
```

A run whose agent named `model` outright is not routed at all: it keeps the runner's
default provider and records no routing event, because there was no decision to explain
and an event saying otherwise is a fiction.

## Reloading

A table changes without a restart:

```python
runner.reload(TableRouter(routing_table()))
```

The next run uses the new table. A run already in flight keeps the model it resolved
before its first call — resolving twice would make one record describe two runs.

## Deliberately not here

- **No cost optimiser.** The kit does not price candidates and pick the cheapest; the
  order in the file is the preference.
- **No health-based failover.** A candidate that is rejected is rejected on capability or
  entitlement, never on how it behaved last minute. Routing a fault to another vendor is
  the gateway epic.
- **No per-request class override.** The agent names the class. A caller that wants a
  different model pins one, which is recorded as a pin.
- **`unsatisfied_by` reads declared capabilities only.** A vendor that supports something
  it never declared is treated as not supporting it.

See also [`docs/providers.md`](providers.md) for what a candidate declares, and
[`examples/routing.py`](../examples/routing.py) for a runnable walk-through.

## Where a model may be used

Routing chooses among models that can do the work; the trust boundary decides which of
them a run may be moved to when one is unavailable. See
[`trust-boundary.md`](trust-boundary.md).
