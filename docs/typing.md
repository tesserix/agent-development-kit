# Typing

`Agent[TripPlan]` runs to a `Run[TripPlan]`, and `run.output` is a `TripPlan`. The claim
this document makes is narrower than "the kit is typed": it is that a consumer's own
`mypy --strict` sees the answer type without a cast, and that the parameter keeps meaning
the same thing across releases.

```python
agent = Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5",
              output_type=TripPlan)
run = await runner.run(agent, "four nights in Kyoto", tenant="acme")
if run.output is not None:
    print(run.output.nights)     # int, not object, and no cast
```

## The parameter has a default

`OutputT` defaults to `NoOutput`, a model with no fields. Three consequences, all
deliberate:

- An agent that declares `free_text=True` needs no annotation. `Agent(...)` is
  `Agent[NoOutput]` and its run's output is `NoOutput | None`, which is `None` in practice
  and which a checker refuses to hand to anything expecting a real answer type.
- Every existing bare `Run` and `Agent` annotation still reads. Adding the parameter was
  not a breaking change to code that says nothing about types.
- The sentinel is a type rather than `None` or `Never`, so there is one type parameter with
  one bound rather than two overloaded shapes, and pydantic can build a schema for it.

Defaults on type parameters are PEP 696. The inline syntax is 3.13 and the supported floor
is 3.12, so `OutputT` comes from `typing_extensions` — see
[`security/admissions/typing-extensions.toml`](../security/admissions/typing-extensions.toml).
When the floor moves to 3.13 the import moves to `typing` and the dependency is dropped;
that is an implementation change, not an API one.

## Rehydration names the type

A checkpoint is JSON, and JSON does not say which type the answer was:

```python
Run[TripPlan].model_validate_json(payload)   # output is a TripPlan
Run.model_validate_json(payload)             # ValidationError
```

The unparameterised read is refused rather than quietly returning a run with the answer
dropped or left as a dict. A caller that has forgotten which type it stored has a bug that
is cheaper to find at the read than three layers later.

Which type a run declared is not on the wire on purpose: a payload that names its own
Python type is a payload that decides what to import.

## Where `Any` is used, and where it is not

Inside the run loop the internal helpers are annotated `Run[Any]` and `Agent[Any]`. The
answer type is not established until validation, so threading the parameter through forty
private signatures would state a guarantee the loop cannot make at that point. The public
signatures — `AgentRunner.run`, `run_sync`, `Run.with_output` — are parameterised, and they
are the ones a consumer's checker sees.

## What is promised, and how it would change

The generic signatures are part of the public API surface: they appear in
`docs/api-surface.txt`, so widening one shows up in a pull request's diff and follows
[`docs/versioning.md`](versioning.md). Specifically:

| Change | Treated as |
|---|---|
| Adding a type parameter with a default | Additive |
| Removing the `NoOutput` default | Breaking — every unannotated use becomes an error |
| Changing the `BaseModel` bound | Breaking |
| Narrowing a public return from `Run[OutputT]` to a concrete type | Breaking |
| Replacing `Any` with the parameter in a *private* signature | Not part of the surface |

A change in the breaking rows ships with a deprecation period under the same policy as any
other, with the old shape kept working for one minor release.

## What the tests check

[`tests/test_typed_results.py`](../tests/test_typed_results.py) asserts inference two ways.
`assert_type` states what the checker must infer and does nothing at runtime. A
`# type: ignore[code]` on a line of deliberate misuse states that the checker must keep
rejecting it — `warn_unused_ignores` fails the build if it stops. Both run under
`mypy --strict` in `make check`, so a widened signature fails in CI rather than in a
consumer's editor.

`py.typed` ships in the wheel; without it none of the above is visible downstream.
