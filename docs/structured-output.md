# Structured output

An agent answers in a declared shape, and the runtime proves the answer has that shape
before the run can be called finished. The alternative — a `completed` run carrying prose
the caller parses with a regex — is the failure this page exists to remove.

## Every agent declares the shape of its answer

`Agent` accepts exactly one of `output_type` and `free_text`. Declaring neither is a
configuration error raised where the agent is built:

```python
Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5")
# ValidationError: declare exactly one of output_type or free_text
```

Free text is a decision somebody made, not a state reached by forgetting:

```python
structured = Agent(..., output_type=TripPlan)
prose = Agent(..., free_text=True)
```

There is no third case, and no path that falls back into free text when validation is
inconvenient. A run whose answer does not validate fails; it does not complete with the
raw string.

## What the model is told

The schema comes from `schema_for(output_type, dialect=STRICT_SUBSET)` — see
[schemas.md](schemas.md) — so descriptions come from the type's docstring and every object
is closed. Its `schema_hash` travels with it.

| | Provider declares `structured_output` | Provider does not |
|---|---|---|
| Where the schema goes | `ModelRequest.output_schema` | a system message in the prompt |
| Who enforces it | the provider | nobody, until the answer comes back |
| How the answer is validated | `output_type.model_validate` | identically |

A provider declares the capability on its `ModelCapabilities` record — see
[providers.md](providers.md).
Silence is not a claim: an undeclared capability is treated as absent, because assuming it
is present means discovering the schema was ignored from a run that already completed.

`ModelRequest` carries `output_schema` only where the provider enforces it, and
`output_schema_hash` either way, and the
contract is part of `Prompt.version` — a changed answer type is a changed prompt, so it
does not quietly reuse a cached prefix or replay a cassette recorded against the old shape.

## What is accepted back

One thing: a JSON object that the declared type validates. Recorded as
`output_validated`, and carried on the run as `run.output`.

An enclosing code fence is stripped, explicitly, and the strip is recorded as
`output_unwrapped`:

```python
unwrap_fenced('```json\n{"a": 1}\n```')   # ('{"a": 1}', True)
unwrap_fenced('Here you go:\n```json\n{"a": 1}\n```')   # unchanged, False
```

The whole answer must be the fence. Prose either side of it is not scraped for an object,
because a pattern that finds JSON inside a sentence also finds it inside a refusal, and
answering from the wrong half of a refusal is worse than failing.

## What fails, and how it is reported

| The model returned | Outcome |
|---|---|
| Prose | `failed`, `schema_violation` |
| JSON cut off mid-object | `failed` — never repaired by guessing the closing braces |
| An object missing a required field | `failed`, with that field named |
| An object of nulls | `failed` — a null is a value, and the required check still runs |
| A conforming object, fenced | `completed`, with the unwrapping recorded |
| A conforming object | `completed` |

`OutputContract.parse` raises `SchemaViolationError` carrying the raw output as `payload`,
every failing dotted path in `paths`, each path's problem in `problems`, the refusing type
in `model` and the schema identity in `details["schema_hash"]`. The run loop catches it,
records the paths and the hash on the event and ends the run `failed` — a failure is a
state, never an exception escaping the loop.

An agent that declares a repair budget gets a bounded chance to correct itself first; see
[repair.md](repair.md). Nothing is repaired unless it is asked for, because a further
attempt spends a caller's budget on a decision they did not make.

## Retrieved content inside a field stays data

A structured field can hold text the agent did not author — a search result, a document
excerpt. Where an agent is structured, content echoed back into the conversation for the
next turn is wrapped with `wrap_untrusted(..., source="model_output")`, so an instruction
that arrived inside a field stays a string in a field rather than becoming the next turn's
system prompt. The validated value on `run.output` is verbatim: it is data the caller
reads, never text the kit re-interprets.
