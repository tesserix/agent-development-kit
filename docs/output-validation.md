# Output validation

What a caller is handed is either an instance of the declared type or a typed error. There
is no third outcome — no partially parsed object, and no missing field filled in with a
default so that it validates.

## Two checks, in order

| Check | What it decides | Where it lives |
|-------|-----------------|----------------|
| `SchemaGuard` | Is this the declared type at all? | `guardrails.output` over `runtime.structured.OutputContract` |
| `PolicyGuard` | Is this answer allowed here? | `guardrails.output` over `core.output_policy` |

A schema says a quote has a total. It does not say the total is inside the band this tenant
sells at. That is what a policy is for, and it runs in ordinary Python:

```python
BAND = Bounded("total", minimum=10, maximum=100)
guard = PolicyGuard((BAND, OneOf("currency", ("AUD", "NZD"))))
guard.raise_for(quote)
```

Rules are never enforced by asking the model. A rule the model checks is a suggestion — it
is the same thing that produced the answer, marking its own work.

## What a violation is allowed to say

`OutputValidationError` carries the violated rule identifiers, the failing paths and the
attempt count. It does not carry the value:

```
answer violated total_within_band
```

A refusal that quotes the out-of-band total has published the out-of-band total, which is
the failure the guard exists to prevent. The raw payload is kept on `.payload` for a
debugger and is never logged by the kit.

## The built-in rules

- `Bounded(path, minimum=…, maximum=…)` — decimal comparison, so a provider returning
  `"90.10"` and one returning `90.10` are judged the same way.
- `OneOf(path, allowed)` — NFC comparison, so a composed and a decomposed character are the
  same member.
- `RequiresCitation(path, citations)` — an assertion drawn from retrieved content that
  arrives with nothing behind it is rejected. Without this, a run that retrieved and a run
  that guessed produce the same-shaped answer.
- `Invariant(name, holds, detail)` — anything about the answer as a whole.

All of them take `required=False` by default, so a legitimately empty or null-valued answer
is expressible rather than tripping a required-field rule.

## Abstention is an answer

```python
SchemaGuard(contract, abstention=True)
```

Where "I do not know" cannot be expressed, a model that does not know invents something that
validates — the worst failure mode, because nothing downstream can detect it. An `Abstention`
carries a required `abstained: Literal[True]`, so it cannot be confused with an answer that
happens to have a reason field, and policies are not applied to one: it quoted nothing.

## Repair is opt-in and bounded

```python
answer = await validated(content, schema=schema, policy=policy, reask=ask_again, attempts=3)
```

A re-ask is a real model call against the run's budget, so:

- Without a `reask` there is no repair at all. That is the default; nobody asked for the
  spend.
- `attempts` counts every answer including the first, so the loop cannot become unbounded.
- The correction text is built only from what actually failed, and never supplies a value.
  An answer the kit dictated is the kit's answer with the model's name on it.
- When the cap is spent the run fails closed with the last violation and its attempt count.

## Streaming

Validation is on the terminal result. A caller consuming a stream is consuming provisional
output, and must not treat a chunk as an answer — the final validation may still reject the
whole thing. See [`docs/run-progress.md`](run-progress.md) for how provisional output is
marked.

## Known limitations

- A policy runs against the parsed result, so a rule about text the model wrote outside the
  declared type has nowhere to attach.
- `reach` walks attributes, so a rule cannot point inside a `dict`-typed field.

## Related

- [`docs/structured-output.md`](structured-output.md) — the schema half, and how a provider
  is asked
- [`docs/guardrails.md`](guardrails.md) — the chain these guards sit in
- [`examples/output_validation.py`](../examples/output_validation.py)
