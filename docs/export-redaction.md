# Redaction in the export path

Attaching an attribute should never be a data-protection decision. `RedactingSpanProcessor`
rewrites every attribute, span event and exception message on the way out, so a value that
should not have been attached still cannot reach the trace store.

```python
from tesserix_adk.observability import RedactingSpanProcessor, RedactionPolicy

processor = RedactingSpanProcessor(RedactionPolicy())
exported = processor.process(span)
```

## Layered, because each layer catches what the last one missed

1. A `SecretStr` renders as `[redacted]`, whatever key it sits under.
2. A denylisted key — `authorization`, `api_key`, `password`, `cookie` and the rest of
   `DENIED_KEYS`, matched on the last dotted segment — is dropped whatever its value looks
   like. A key denylist alone only catches the callers who already knew to be careful.
3. A payload attribute travels only if the deployment allowlisted it.
4. Everything else is scanned for sensitive shapes, recursively where it is structured.

The kit's own `adk.` attributes are structural identity — a tenant id, a token count — and
pass through unscrubbed. Redacting them would leave spend attributed to nobody.

## Payload capture is off by default

`PAYLOAD_ATTRIBUTES` — prompt, completion, tool arguments and results, retrieved text,
memory content — are the most useful attributes to debug with and the most likely to hold a
passport number. They are dropped unless named in the policy:

```python
RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}))
```

An allowlisted payload still goes through redaction and the size cap. A dropped one leaves
`adk.prompt.ref = "sha256:…"`, so a developer can tell two runs apart, group by payload and
correlate with a value that never left the process. Set `content_references=False` where
even a digest is unwelcome.

## Structured values

A tool argument is JSON, and a top-level string match misses everything inside it. Values
that parse as an object or an array are walked: strings at the leaves are scrubbed, and a
denylisted key anywhere in the tree has its value replaced.

## Fail closed

If a detector raises or a value cannot be scanned, that attribute is dropped and
`stats.failures` is incremented. The span still exports. Dropping the whole span would lose
the causality that explains the failure; exporting it unredacted is the incident this
exists to prevent. `RedactionStats.failures` should sit at zero in steady state.

Size caps are applied before scanning, so a large payload costs a bounded scan rather than
an unbounded one, and a secret cannot survive by sitting past the cap.

## Scope

The processor covers what the kit exports. Third-party instrumentation emitting into the
same process is outside it and needs its own processor. Logs and metrics need the same
treatment or the leak simply moves channel — `tesserix_adk.core.scrub` is the same scan
applied to a single value.

Detector implementations and the PII taxonomy belong to the guardrails path; this surface
consumes a detector rather than defining one.

## Related

- [One set of attribute names](telemetry-convention.md) — what a conforming span carries.
- [Spans without wiring](auto-instrumentation.md) — what produces the spans.
