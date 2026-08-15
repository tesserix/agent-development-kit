# Keeping business logic out of prompts

Prompts accumulate pricing, refund thresholds, cancellation windows and eligibility
conditions written in English. Those are business rules with none of the properties a
business rule needs: untested, unversioned against the code that depends on them, and
advisory — the model may simply not apply one, and nothing fails when it does not. That is
how a refund rule becomes a suggestion.

The split is the kit's own: **agents reason, code transacts.**

| Belongs in the prompt | Belongs in code |
|---|---|
| Role and task framing | Monetary thresholds and shares of an amount |
| Tone, style, verbosity | Eligibility conditions and rule chains |
| The output contract | Time windows that govern an outcome |
| What to ask a tool for | Authorisation decisions |
| How to cite what it used | Irreversible actions |

## Running it

```python
from tesserix_adk.core import lint_directory

report = lint_directory(Path("prompts"))
print(report.summary())
raise SystemExit(report.exit_code)
```

`lint_prompt(text, source=...)` does the same for a prompt held inline rather than in a
directory, so a project migrating to a registry can lint what it has today.

## The rules

| Code | What it catches | Severity |
|---|---|---|
| `ADK-P000` | A suppression with no recorded justification | error |
| `ADK-P001` | A monetary threshold or a share of an amount | error |
| `ADK-P002` | A conditional rule chain | error |
| `ADK-P003` | A time window governing an outcome | error |
| `ADK-P004` | Language granting the model authority to decide | error |
| `ADK-P005` | An instruction to perform an irreversible action directly | error |
| `ADK-P006` | An embedded endpoint, URL or service name | warning |

Every rule carries a remedy, because a check that only says no is a check projects turn off.
Warnings are reported and do not fail the run.

## A worked refactor

Before, with the rule in the prompt:

> If the booking was made more than 24 hours ago, refund 50% of the fare.

Three findings: the share of an amount, the rule chain, the time window. After, with the
rule in a tool and the prompt asking it:

```python
@tool
def refund_quote(booking_id: str) -> RefundQuote:
    """Return what this booking is entitled to, and why."""
```

> Ask `refund_quote` what the booking is entitled to. Tell the customer the amount and the
> reason it gives, and do not offer anything it did not return.

The threshold now lives beside the code that charges, has a test, and refuses when it should.
`ADK-P005` pairs with the tools epic's approval gate: a prompt is not an enforcement point,
and a retry of one is a second refund.

## False positives, and suppression

Ordinary numbers pass. "Return at most 5 suggestions, in 2 sentences each" is output shaping,
not a business rule, and nothing flags it — the rules key on currency, shares, time units and
authorisation verbs rather than on digits.

Where a finding is genuinely wrong or genuinely accepted, suppress it with a reason:

```
Waive the fee for orders over $500.  # adk-lint: allow ADK-P001 — legacy tariff, ticket PLAT-91
```

The comment covers its own line and the line below, and only the code it names. A suppression
with no reason raises `ADK-P000` in its place, which is an error, so silencing the check
costs the same as fixing it unless somebody writes down why. Every report counts its
suppressions, so a project that suppressed everything reads as `12 suppressed` rather than as
green.

## Known limitations

* The patterns are English. A tenant-specific prompt written in another language will pass a
  rule this would catch in English.
* A rule split across two lines, two prompts or two versions is not detected: each line is
  read on its own.
* Prompt text is matched, not understood. The lint is a reviewer's floor, not a proof.
