# PII redaction and content policy

Two guards that decide what an agent is allowed to hold and what it is allowed to say.
They are separate because they fail differently: an identifier is removed and the turn
continues, while content over a tenant's bar ends the turn.

## Why redaction cannot live at the agent boundary

A passport number in a prompt is a passport number in the provider's logs, in the
checkpoint, in the memory record and in the span attribute. Redaction applied only where
the agent hands off has already lost, because four other things emitted first.

So the detectors live in `tesserix_adk.core.pii` and three paths use the same ones:

| Path | What applies them |
|------|-------------------|
| Guardrail chain, before prompt assembly | `guardrails.PIIGuard` |
| Memory write path | `memory.erasure.PIIRedactor` |
| Telemetry export | `observability.redaction.Redactor(pii_tenant=…)` |

One set of detectors also means one shape of placeholder. Hand-rolled redaction gives the
same person `[REDACTED]` in one store and `xxxx1234` in another, and nothing can be joined.

## The placeholder keeps the type and the subject

```python
redact("mail ada@example.test", tenant="acme").text
# 'mail [email:9f2c…]'
```

The type stays because an agent reasoning about `[redacted]` cannot tell an email from a
card number and starts asking for the value again. The pseudonym stays because "the same
traveller as last week" is most of what a memory store is for, and it is answerable without
holding the value.

The pseudonym is salted per tenant by construction. One digest shared across tenants would
let a tenant confirm the presence of another tenant's subject by guessing it.

## Over-redaction is a real cost, so it is a setting

A booking reference shaped like a passport number, redacted, breaks the agent's actual job.
Two dials, both per tenant:

- **`threshold`** — every match carries a confidence. The unambiguous shapes (an email, a
  checksummed card, a bearer token) are `1.0`; the ambiguous ones (a passport, a phone
  number) are `0.6`. Raising the bar drops the ambiguous ones.
- **`allow`** — shapes this tenant has said are not identifiers, matched in full against
  the text a detector claimed.

A card is also checked against Luhn, because without it every sixteen-digit order reference
is a card number.

Where two detectors claim overlapping text, the earlier one wins: `DEFAULT_DETECTORS` puts
the checksummed shapes first, so a card is a card rather than a phone number.

## Content policy is a taxonomy, not the provider's mood

Without one, safety behaviour is whatever the model vendor happens to refuse, returned as an
opaque error whose shape differs per vendor. A run that changes provider changes its safety
behaviour and nobody wrote that down.

`ContentCategory` and `ContentSeverity` are declared, and `refusal_of` normalises a
provider's native refusal into the same `ContentBlockedError` the kit raises itself.

## What a passage is about, and what to do about it, are separate

A classifier answers the first, and its answer is the same for every tenant. `Thresholds`
answers the second, and it is not:

```python
customer_facing = ContentFilterGuard(thresholds=Thresholds(default=ContentSeverity.MEDIUM))
triage = ContentFilterGuard(
    thresholds=Thresholds(per_category={ContentCategory.HARASSMENT: ContentSeverity.HIGH})
)
```

An internal triage agent reading an abusive support transcript is doing the work it exists
for; blocking it stops that work. A customer-facing agent handling the same text is not.
The difference is configuration, not code.

`HeuristicClassifier` ships so the policy path can be wired before a deployment has a model
for it. It answers `MEDIUM` and never `HIGH`, because a term list cannot tell a slur from a
quotation of one. It is not a shipping classifier.

## Both fail closed

A detector that raises, or a classifier that times out or raises, produces
`GuardrailEvaluationError` rather than an allow. Content nobody could evaluate is not
content that was found clean.

The refusal itself carries categories, a severity and a classifier name — never any part of
what was said. An output-stage block that quoted the payload into its error message would
have emitted the payload.

## Related

- [`docs/guardrails.md`](guardrails.md) — the chain these two sit in
- [`docs/prompt-injection.md`](prompt-injection.md) — the other thing content is screened for
- [`docs/erasure.md`](erasure.md) — getting out what should not have been stored
- [`examples/pii_redaction.py`](../examples/pii_redaction.py)
