# The event contract

An emitted event becomes another team's contract the moment somebody subscribes to it. A
field renamed in a kit minor breaks every dashboard reading it, and the break is invisible
from here — the publisher's tests all pass.

So the contract is registered, generated as JSON Schema, committed to
[`docs/event-schemas.json`](event-schemas.json), published as a release asset, and diffed in
CI. An incompatible change fails the build naming the event, the version and the field.

## What is stable

| Stable | Meaning |
|---|---|
| `event_id` | A ULID, unique for ever. Safe as a dedupe key and as a primary key. |
| `type` | The vocabulary of `EventType`. New members are added; none is removed within a major. |
| `schema_version` | Which version of that type's body the event speaks. |
| Envelope fields | `occurred_at`, `tenant`, `user`, `run_id`, `trace_id`, `span_id`, `correlation_id`, `causation_id`, `attributes`. |
| Attribute meanings | A name keeps its meaning for the life of a major. |

## What is not

* **Attribute ordering.** `attributes` is a mapping; nothing may depend on key order.
* **Transport subject internals.** `adk.events.<tenant>.<kind>` is the platform's to change.
* **The set of attributes present on any one event.** A new optional attribute may appear at
  any minor; a consumer reads what it knows and ignores the rest.
* **Anything under `tesserix_adk.experimental`.**

## The compatibility rules

Within a major:

* fields may be **added**, and only as optional;
* a field is never removed, and a name is never reused for a different meaning — a rename is
  both at once, which is why the check reports it as one;
* an enum may gain members, never lose them, so a consumer must treat an unrecognised member
  as unknown rather than as an error;
* unknown fields are tolerated on the way in. `read_envelope` drops envelope fields this
  version has never heard of and keeps unknown attributes, so a newer publisher on the same
  stream does not stop an older consumer.

## Changing an event

**A correct additive change.** `ToolCallCompleted` gains an optional `attempt`:

```python
class ToolCallCompletedV2(ToolCallCompleted):
    version: ClassVar[int] = 2
    attempt: int = 0

EVENT_SCHEMAS.register(ToolCallCompletedV2)
EVENT_SCHEMAS.register_upcaster(
    EventType.TOOL_CALL_COMPLETED,
    from_version=1,
    upcast=lambda attributes: attributes | {"attempt": "1"},
)
```

`make event-schemas` records it, the diff is additive, and a consumer written against
version one keeps working unchanged.

**An incorrect one.** Renaming `iterations` to `iterations_count` on the existing version:

```
incompatible event contract change:
  run_completed@1.iterations was removed or renamed; within a major a field name keeps its
  meaning, and a rename is both a removal and a reuse
```

The build stays red. The fix is a new version of the event type, not a smaller diff.

## Deprecating a version

1. Register the new version and its upcaster; publish the new version.
2. Emit both for **one minor**, so consumers can move at their own pace.
3. Record the deprecation in the changelog with the release the old version is removed in.
4. Remove the old model and its upcaster only at a **major**.

## Reading events

```python
from tesserix_adk.core.event_schema import EVENT_SCHEMAS, read_envelope

envelope = read_envelope(message.data)      # tolerant decode, upcast to the current version
payload = EVENT_SCHEMAS.payload_of(envelope)  # validated against the model for that version
```

Two publishers at different kit versions on one stream is the normal case, not an incident.
The older one's events are upcast on the way in; the newer one's are parked:

| Situation | What happens |
|---|---|
| Older version, upcaster registered | Upcast one step at a time to the current version. |
| Older version, a step missing | `UnsupportedEventVersionError`. Guessing across a missing step invents data. |
| Newer version than this kit reads | `UnsupportedEventVersionError`, so the consumer parks the message instead of crash-looping on every redelivery. |
| An event type this kit has never heard of | `UnknownEventTypeError`, parked the same way. |
| An unknown attribute or envelope field | Kept, or dropped, respectively. Never an error. |

Park means dead-letter it and keep consuming — see
[idempotent consumption](idempotent-consumption.md), whose poison path is the same one.

## Publishing

`Eventing` stamps the current version of the payload's type and refuses a type with no
registered schema:

```
UnknownEventTypeError: invented has no registered schema, so no consumer has been told what
it means; register a payload model before publishing it
```

That refusal is at the publish site, not at the consumer, because an event nobody has a
contract for is one nobody can act on.
