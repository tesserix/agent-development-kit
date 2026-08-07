# Schemas

The schema a model is told is derived from the Python type the code parses. There is one
declaration, not two, so the pair cannot drift.

```python
from tesserix_adk.core import schema_for, schema_hash

schema = schema_for(Itinerary)
version = schema_hash(schema)
```

`schema_for` accepts a pydantic model, a dataclass, a `TypedDict`, or a callable with
annotated parameters. What comes back is normalised JSON Schema (Draft 2020-12): titles
dropped, keys ordered, `required` sorted — so two generations of one type are the same
bytes and a diff of two schemas shows only what changed.

## Descriptions come from the docstring

The summary line describes the type. The Google-style `Args:` block describes its fields,
and each entry lands on the matching property:

```python
class Waypoint(AdkModel):
    """A stop on the way.

    Args:
        city: Where the traveller stops.
        nights: How long they stay, in nights.
    """

    city: str
    nights: int = 1
```

`Field(description=...)` wins where both exist. A missing docstring, a missing `Args:`
block, an entry naming a field that no longer exists, or a line the parser cannot read
costs a description and nothing else — documentation is guidance, and a typo in it is not
a reason to refuse to run.

The `Args:` block is never used as the type's own description: pydantic would otherwise
emit the whole docstring, repeating field-by-field guidance the schema already carries.

A description has to have been written for this type. `schema_for(str)` is `{"type": "string"}`
and not `str`'s own docstring: a builtin's `__doc__` is written for a Python reader, and
telling the model that a string is `str(object='') -> str` spends tokens to confuse it.

## What can be described

A pydantic model, a dataclass, a `TypedDict`, a callable, a bare type, and a type spelled in
a way `isinstance` does not recognise — `list[str]`, `str | None`, `dict[str, int]` — which
is how a tool's parameters usually arrive.

`annotations_of` is the resolution `schema_for` uses on a callable, exported because a caller
inspecting a signature needs the same answer: `get_type_hints` with `include_extras=True`,
falling back to the `__call__` of a callable object, which `get_type_hints` refuses outright.

## Excluding a parameter

`schema_for(callable, exclude=("ctx",))` describes every parameter but the named ones. It is
for arguments the caller injects — a request context, a connection, a tenant — which are not
the model's to choose and often cannot be rendered as JSON Schema anyway. Excluding is
structural rather than cosmetic: the parameter is not described, so nothing the caller
injects can be overridden by a model that guessed the name. `@tool` uses it for
[`ToolContext`](tools.md#context-is-injected-never-chosen).

## Provider dialects

Providers disagree about what a schema may contain, so the dialect is a parameter rather
than something each caller patches afterwards.

| Dialect | What it emits | For |
|---|---|---|
| `JSON_SCHEMA` (default) | Draft 2020-12, `$defs` and `$ref` intact | Anything that accepts JSON Schema |
| `STRICT_SUBSET` | `additionalProperties: false` on every object | Structured-output modes that demand closed objects |
| `INLINE_REFS` | Every `$ref` substituted, no `$defs` | Providers that will not follow a reference |

`STRICT_SUBSET` closes objects and does nothing else. Some provider guides also widen
`required` to every field; the kit does not, because a schema that demands keys the type
treats as optional refuses payloads the code would have accepted.

A dialect is anything satisfying `SchemaDialect` — a `name`, a `forbidden` set of schema
keywords, and an `adapt`. Nothing about the three built-ins is privileged:

```python
@dataclass(frozen=True)
class NoUnions:
    name: str = "no-unions"
    forbidden: frozenset[str] = frozenset({"anyOf"})

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        return schema
```

`forbidden` is checked after `adapt`. A keyword the provider will not accept raises
`CapabilityError` naming the dialect, the keyword and where it appeared — the schema is
never rewritten into something looser that the provider would take and the code would not.

## The hash

`schema_hash` is a SHA-256 over the canonical form, prefixed with the algorithm that
produced it. Key order does not change it; anything about the shape does.

That is deliberate. A renamed field, a widened bound, a new union member and a different
dialect all produce a different hash, so a cassette recorded against the old shape misses
loudly instead of replaying an answer for a type that no longer exists. It is the same
property the run fingerprint relies on (`docs/determinism.md`).

## What fails, and when

Every failure below happens where the type is declared, not on the first call that sends
it. A schema that accepts more than the type does is the worst outcome available: the model
satisfies it, the code refuses the answer, and the run fails in production.

| Situation | Raised |
|---|---|
| A parameter with no annotation | `SchemaGenerationError` naming the parameter |
| `*args` or `**kwargs` on a tool callable | `SchemaGenerationError` naming it |
| `Any` in a required position, at any depth | `SchemaGenerationError` naming the field |
| A type pydantic cannot render as JSON Schema | `SchemaGenerationError` naming the field and the type |
| A schema past `max_bytes` | `SchemaGenerationError` with both sizes |
| A dialect that forbids a keyword the type needs | `CapabilityError` naming dialect and keyword |
| A recursive type under `INLINE_REFS` | `CapabilityError` naming the type |
| A type nested deeper than `InlineRefs(max_depth=...)` | `CapabilityError` naming the depth |

`Any` is refused rather than emitted as `{}`, which accepts everything. Where a field
genuinely holds arbitrary provider data, model it as a declared map — the same rule as
`Usage.extras` in [`docs/models.md`](models.md).

A schema over a provider's size limit is refused whole. Truncating it would describe a
different type, and the model would be told a shape the code does not accept.

Recursion is fine wherever references are allowed:

```python
schema_for(Node)                        # $defs/Node referring to itself
schema_for(Node, dialect=INLINE_REFS)   # CapabilityError: inlining a cycle does not terminate
```

## Field changes and versions

The semver rules for changing a field are in [`docs/models.md`](models.md#field-changes-and-semver).
The schema hash is how those changes become visible: any of them changes it, and a changed
hash invalidates cassettes and cached prompts rather than letting a stale one answer.
