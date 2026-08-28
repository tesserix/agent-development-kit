# Security policy

`tesserix-adk` is a library that other products build their own security posture on. A
flaw in a guarantee the kit makes is a flaw in every product that relied on it, so it is
fixed and communicated once, centrally, rather than product by product.

## Reporting a vulnerability

**Do not open a public issue.** Report privately through GitHub Security Advisories:

<https://github.com/tesserix/agent-development-kit/security/advisories/new>

That channel is private to the maintainer rota, gives you a place to attach a proof of
concept, and becomes the advisory when the fix ships. If you cannot use it, open a public
issue containing *only* a request for a private channel and no detail of the flaw.

Please include the version, the install profile (base or which extra), what you expected
the kit to prevent, and what happened instead.

You will be credited in the advisory unless you ask not to be. There is no bounty.

## What we commit to

<!-- generated: response-targets -->

| Severity | Acknowledged within | Fix released within |
| --- | --- | --- |
| critical | 1 day(s) | 7 day(s) |
| high | 2 day(s) | 14 day(s) |
| moderate | 5 day(s) | 30 day(s) |
| low | 5 day(s) | 90 day(s) |
<!-- end generated: response-targets -->

Days are calendar days from the report. The targets are keyed by severity and nothing
else: a flaw reachable only through an optional extra exposes every product that installs
that extra, so it is never deprioritised for living outside the base install.

These are not prose. Every report becomes a record in `security/advisories/`, and
`make disclosure` — which CI runs — fails if an acknowledgement missed its target, if a
supported minor never got a patched release, or if consumers were notified after
publication rather than with it.

## Supported versions

Fixes land on `main` and are backported to the previous minor, per the support window in
[`docs/versioning.md`](docs/versioning.md). Before 1.0 that means the current minor and
the one before it. Nothing older receives a fix; the answer for an older version is to
upgrade.

A security fix may break behaviour faster than the deprecation window allows. That
exception is documented, is used for nothing else, and still produces a deprecation
record and a changelog entry saying the window was not met and why.

## What happens after you report

1. **Acknowledge** — inside the target above, with a severity and a named maintainer.
2. **Fix privately** — in a private fork, with a regression test that fails on the
   released version. The issue stays private until the release.
3. **Patch every supported minor** — not just the newest one. A fix that lands only on
   the current minor leaves the other supported minor exposed, and the gate fails on it.
4. **Notify the embedding products** — on or before publication. Consuming teams learn
   from this process, not from noticing a version bump on the index.
5. **Publish** — advisory, patched releases, changelog entry and release notes together.
   The advisory names the affected versions, the fixed versions, the interim mitigation
   and the reporter.

### If it is public before there is a fix

The embargo is over and speed beats completeness:

1. Publish the advisory immediately with an **interim mitigation** — a setting, a version
   to pin, or a call to stop making. The record cannot be published without one; the gate
   requires a mitigation on any advisory disclosed with no fix.
2. Notify consuming teams directly the same day.
3. Ship the patch releases as they become ready, updating the advisory in place.

### If the report is about a product, not the kit

Flaws in a product that embeds the kit belong to that product's own security contact. We
will forward a misdirected report to the right team and tell you we have done so; we will
not disclose the product's details, and the clock above does not apply to it.

## Published advisories

<!-- generated: advisories -->

No advisories have been published.
<!-- end generated: advisories -->

## What the kit actually guarantees

Over-trust is its own vulnerability. The guarantees the kit makes, the assumptions each
one rests on, and — more importantly — what it does **not** defend against are set out in
[`docs/threat-model.md`](docs/threat-model.md). Read it before treating any of the kit's
boundaries as a security control in your own product.

## Known limitations

- **GitHub private vulnerability reporting depends on repository settings.** Confirm it is
  enabled as part of the public-launch checklist. If the reporting link returns a 404,
  open a public issue asking for a private channel and include no vulnerability detail;
  a maintainer will open the advisory.
- The gate checks the process against its own records. It cannot tell you a report was
  never recorded — a flaw handled quietly outside `security/advisories/` is invisible to
  it, which is why the rota, not the tool, owns the acknowledgement.
- Severity is assigned by the maintainer rota. Nothing here computes CVSS, and a reporter
  who disagrees with a severity should say so on the advisory; the target follows the
  severity, so the disagreement is worth having before the clock is set.

## Related

- [`docs/security.md`](docs/security.md) — advisory, secret and licence scanning gates.
- [`docs/verifying.md`](docs/verifying.md) — verifying a release's signature and provenance.
- [`docs/dependencies.md`](docs/dependencies.md) — dependency policy and the advisory fast lane.
