# Threat model

What `tesserix-adk` actually guarantees, what each guarantee assumes, and what it does
not defend against. A consuming product that over-trusts a boundary here has a
vulnerability the kit cannot fix for it.

The kit is a library. It runs inside your process, with your credentials, against models
and stores you choose. It is not a sandbox, not a proxy and not a policy engine.

## Who this is written against

| Adversary | Reaches the kit through |
|---|---|
| Untrusted input to an agent | A user turn, a tool result, a document that reached retrieval |
| A hostile or compromised model provider | Completions, tool-call arguments, embeddings |
| A hostile or compromised tool | Tool results, MCP server responses |
| A curious or careless operator of the consuming product | Logs, traces, exported telemetry |
| The dependency supply chain | A package the kit or your product installs |

## What the kit guarantees

### Guarantee: guardrails hold at the boundary

Input and output guardrails run at the edge of an agent step, on every path that crosses
it, including tool results and retrieved documents on the way in and model output on the
way out. A guardrail that raises stops the step; there is no path that skips the check
because a caller passed a flag.

**Assumption:** every path into the agent goes through the kit's boundary. Content your
product injects into a prompt directly, outside the kit's own call, is not seen by any
guardrail and is not covered.

**Assumption:** the guardrail you configured actually detects the class of content you
care about. The kit guarantees the check runs, not that a particular classifier is
correct. A guardrail is a control, not a proof.

### Guarantee: redaction happens before export

Telemetry is redacted at the point of export, not at the point of logging. A span
attribute carrying prompt or completion content is redacted before it leaves the process,
so a misconfigured collector receives redacted data rather than raw content.

**Assumption:** the content is in a field the kit produced. Your own spans, your own log
statements and anything you attach to a span outside the kit's instrumentation are yours
to redact.

**Assumption:** redaction is pattern-based and therefore incomplete by nature. It reduces
exposure; it does not make a trace safe to publish.

### Guarantee: spend is bounded

A run carries a budget ceiling. When the ceiling is reached the run fails rather than
continuing, and the failure names the budget, not a generic error.

**Assumption:** the accounting is the provider's own token counts, which arrive with the
response. A provider that under-reports, or a request that fails after the tokens were
charged, is counted wrong. The ceiling bounds runaway loops; it is not billing.

### Guarantee: errors fail closed

Every failure path in the kit raises rather than degrades. A guardrail that errors is a
blocked step, not an allowed one. A store that is unreachable is an error, not an empty
result. A model response that does not match the declared type is an error, not a partial
object.

**Assumption:** your product does not catch and continue. Fail-closed inside the kit is
undone by a bare `except` in the caller.

### Guarantee: no network in the unit path

The shipped pytest plugin blocks socket access in unit suites, so a test that would have
reached a provider fails locally instead of leaking a request — and a credential — from
a developer machine or CI.

**Assumption:** it is enabled. It is opt-in per suite, and a suite that opts out has no
guard.

## What the kit does not defend against

Stated plainly, because each of these has been mistaken for a guarantee before:

- **Prompt injection, in general.** Guardrails reduce a known class of attack. Retrieved
  or tool-returned content that persuades a model to act is an open research problem, and
  no configuration of this kit closes it. Design the blast radius of a tool assuming the
  model can be talked into calling it.
- **A malicious tool or MCP server.** Tool results are data the kit passes along. It does
  not sandbox a tool, verify a server's identity beyond TLS, or constrain what a tool
  does with the credentials you gave it.
- **A malicious model provider.** A provider that returns crafted tool-call arguments is
  indistinguishable, to the kit, from one that does not.
- **Secrets in your own configuration.** The kit does not store, rotate or scope your
  provider keys. It reads them from your environment and hands them to the client.
- **Multi-tenant isolation.** Nothing here separates one tenant's memory, retrieval index
  or cache from another's. That boundary belongs to your product, and the kit will
  faithfully retrieve whatever you stored under the key you asked for.
- **Denial of service and cost exhaustion by an authorised caller.** The budget bounds a
  run, not a user, and there is no rate limiting.
- **The content of what a model produces.** No claim is made about accuracy, bias or
  safety of generated text beyond the guardrails you configured.
- **Anything after the process boundary.** Your logs, your collector, your database, your
  deployment.

## Reporting

A flaw in one of the guarantees above is in scope for
[the security policy](https://github.com/tesserix/agent-development-kit/blob/main/SECURITY.md). Something listed under *does not defend against*
is a design boundary rather than a vulnerability — if you think a boundary is drawn in
the wrong place, that is a discussion worth having in a public issue.
