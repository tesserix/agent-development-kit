# The model policy

Data crosses an agent's boundaries as provider payloads, tool arguments, config blocks and
records read back from a store. Where those stay loose dicts, the first call site to touch
a field decides what it means, and a typo in a tool argument becomes an `AttributeError`
deep inside a tool body, several steps after the mistake. Every boundary in this kit is a
model instead, so the failure happens at the crossing and names the field.

A runnable version of everything here is [`examples/models.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/models.py),
which needs no network and no credentials and is executed in CI.

## `AdkModel`

Every model the kit puts on a boundary derives from `AdkModel`, so strictness is a property
of the kit rather than something each new model has to remember. Its config is three
decisions:

| Setting | Means | Because |
|---|---|---|
| `extra="forbid"` | An undeclared field is an error, not a passenger. | A misspelt setting that is silently ignored is a setting that never took effect. |
| `strict=True` | `"12"` is not `12`, and `"false"` is not `False`. | A coerced value hides a real integration defect until the day the string is `"twelve"`. |
| `frozen=True` | A validated record cannot be edited. | What validated is what the next layer reads; anything else means re-validating on every access or trusting that nobody did. |

`tests/test_models.py` asserts that every `BaseModel` in the package derives from it, so a
model added next week cannot quietly opt out.

## Extras are declared, never loose

Forbidding extras must not mean a provider cannot evolve. Where a provider reports fields
the kit does not model, they go into a declared map — `Usage.extras`, `Message.metadata` —
which is typed, round-trips, and is documented as something the kit itself never reads.
The same field loose on the model is refused:

```python
Usage(input_tokens=1, output_tokens=1, extras={"reasoning_tokens": 7})   # kept
validated(Usage, {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 7})  # refused
```

The difference matters at review time: an extras map says "the kit does not understand
this", where an accepted unknown field says "somebody will read this eventually", and
nobody ever does.

## A violation says where

`validated(model, payload)` normalises pydantic's `ValidationError` into
`SchemaViolationError`, which carries:

- `model` — the model that refused it, by name.
- `paths` — every failing field path, sorted, `content.0.binary.media_type` style. List
  indices and union members are kept, because "a content part is wrong" is not a location.
- `problems` — path to reason.
- `payload` — the raw input, for a debugger.

Every failing field is reported at once. One field per round trip is how a five-field
config takes five deploys to fix. A violation is not `retryable`: the same payload
validates the same way, so asking again spends more to be refused again.

## Unions resolve by discriminator

`ContentPart` is `Annotated[TextPart | BinaryPart, Field(discriminator="kind")]`. A part
with an unknown `kind` is refused naming `kind`, rather than being tried against each
member until one fits — first-match guessing turns a new part type into a wrong part type,
silently, in a system where the part is evidence.

## Sensitive fields

`Annotated[bytes, Sensitive("...")]` marks a field that must not reach telemetry.
`telemetry_dump(model)` drops those fields and masks any `SecretStr`, recursing into nested
models, sequences and maps; `model_dump_json` keeps them. The split is deliberate: a
checkpoint rehydrates a run, and a run rehydrated without its credentials or its exhibit
rehydrates broken, while telemetry is the copy that leaves the process and leaves under a
retention policy nobody reads. `BinaryPart.data` is the first field marked this way — a
scanned exhibit is evidence in one system and a retention problem in another.

`telemetry_dump` emits fields in declaration order, so two dumps of one model diff cleanly.

## Aliases

There are none, and `tests/test_models.py` fails if one appears. Two names for one field
is two spellings in every config file, every payload and every piece of documentation, and
a reader who has only ever seen one of them cannot search for the other.

## Strings from the environment

An environment variable is a string whatever it means, so the strictness that protects
every model inside the kit gives way exactly once, at that edge, explicitly:
`parsed_from_strings(annotation, value)` parses a string against its own field annotation
and raises `SchemaViolationError` if it is not that type after all. `resolve_config` calls
it for values from the `env` layer only; a value passed in code is already typed, and one
read from a TOML file already has the type TOML gave it.

Durations take either spelling — `TESSERIX_ADK_PROVIDER__REQUEST_TIMEOUT=45` and `PT45S`
mean the same 45 seconds — because a config file written by a person and one written by a
generator disagree about which is obvious.

## Field changes and semver

A model's fields are part of the public surface, so they follow
[the deprecation policy](versioning.md):

| Change | Is | Because |
|---|---|---|
| Adding an optional field with a default | Minor | Existing payloads still validate and existing constructions still compile. |
| Adding a required field | Major | Every existing construction and every stored payload becomes invalid. |
| Removing or renaming a field | Major, after a deprecation release | `extra="forbid"` means a payload still carrying the old name is refused, not ignored. |
| Widening a field's type | Minor | Everything that validated before still validates. |
| Narrowing a field's type | Major | Payloads that validated stop validating. |
| Adding a member to a discriminated union | Minor for producers, major for exhaustive consumers | The kit treats it as minor and says so in the changelog entry. |
| Marking a field `Sensitive` | Minor | It drops out of telemetry, which is not a contract; it still round-trips. |

Every one of these shows up in `docs/api-surface.txt`, so the decision is made in the pull
request that causes it rather than in the release that ships it.
