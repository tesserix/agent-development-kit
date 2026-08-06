# Security scanning

A vulnerability in a kit dependency is a vulnerability in every product that imports the
kit, at once. The same is true of a credential committed to a fixture: the next sdist
distributes it to every consumer's source tree. Both are scanned continuously rather than
at adoption.

Two gates run in `.github/workflows/security.yml`, on every pull request, on every push to
`main`, and daily at 03:17 UTC. The schedule is the point of the workflow as much as the
pull-request trigger is: an advisory published the day after a merge is not caught by a
merge trigger.

Neither job uses a repository secret, so both run on a pull request from a fork — which is
where an unreviewed dependency is most likely to arrive — instead of skipping.

## Advisories

`tools/audit.py` runs `pip-audit` against the installed environment, which CI installs
from `uv.lock` with `--frozen`. The audit is of the set that ships, not of a fresh
resolution that nobody will get.

Each finding is then given two things `pip-audit` does not supply:

- **A severity**, read from the OSV record for the advisory.
- **A blast radius**, computed from the lockfile by `tools/lockfile.py`: whether the
  package is reachable from the runtime dependencies, only through a named extra, or only
  from a development group.

```
make audit
```

### The severity policy

| Severity | Verdict |
|---|---|
| `critical`, `high` | Blocks the build |
| `unknown` (no rating in OSV) | Blocks the build |
| `medium`, `low` | Reported and tracked, does not block |
| Any severity, development-only package | Reported and tracked, does not block |

An unrated advisory blocks. A scan that could not decide is not a scan that found nothing,
and the cost of being wrong in that direction is one triage; the cost of the other is a
shipped vulnerability nobody looked at.

A finding reachable only through an optional extra still blocks, and the report says which
extra, because a consumer who installs that extra is exposed exactly as much as anyone.
Only a development-group dependency is exempt: it is not in any published artefact.

### An advisory with no fix

There is no fixed version to move to, so the choices are a mitigation or a suppression
with a short expiry — never an open-ended exclusion. Record the mitigation in the
suppression's `mitigation` field and set `expires` to the date the fix is expected, or 90
days out, whichever is sooner. When the fix lands, the suppression is deleted, not renewed.

## Secrets

Two passes, because they see different things:

- `tools/secret_scan.py` over every tracked file, for eight credential shapes. It also
  checks recorded provider traffic under a `cassettes/` or `recordings/` directory for
  email addresses and phone numbers — a cassette was recorded from a live exchange and can
  carry someone's details even when it carries no key. Those rules apply only there: a
  maintainer address in `CODEOWNERS` is the point of that file.
- `gitleaks`, pinned by version and run as a binary, over the full history. A credential
  removed in a later commit is still in the history and still live.

```
make secrets
```

Matched values are truncated to eight characters in every report. A scanner that prints
the credential has published it to the build log, where it is readable by anyone who can
see the run.

### If the scan fires

**Rotate the credential first.** It is compromised from the moment it was pushed, and
rewriting history does not un-publish it — it was in a build log, in a fork, in a clone,
and in whatever mirrors the repository. History rewriting is cleanup; rotation is
remediation. Do it in that order.

Then remove the value, and only if it is a deliberate fixture, declare it.

## Suppressions

Everything both scanners choose not to fail on is declared in `security/policy.toml`, in
one shape:

```toml
[[suppression]]
id = "GHSA-xxxx-xxxx-xxxx"          # advisory id, or "<rule>:<filename>" for a secret
kind = "advisory"                    # or "secret"
owner = "@sam123ben"                 # a person, not a team alias
reason = "Reachable only from the docs build; upstream fix tracked in #123."
expires = "2026-10-01"               # at most 90 days out
mitigation = "…"                     # optional, and required in practice when no fix exists
```

Rules, enforced by `tools/security_policy.py` and its tests:

- Every field except `mitigation` is required. A missing owner makes the suppression
  nobody's problem, which is how it survives for two years.
- `reason` is at least 30 characters. "false positive" is not a reason.
- `expires` is at most 90 days out. A suppression is a decision to accept a risk for a
  while, not a decision to stop looking.
- **An expired suppression fails the build**, with the same weight as the finding it was
  covering. This is the only mechanism that makes the expiry real.
- A `secret` suppression does not cover an advisory, or the reverse. The kinds are
  separate on purpose.

A redaction fixture has to contain something shaped like a real key in order to test
redaction. That is legitimate, and it is declared with `kind = "secret"` rather than
inferred — deciding automatically which keys are fake is exactly how the real one gets
through.

## Known limitations

- Severity comes from OSV. A package with an advisory OSV has not rated is treated as
  blocking, so the first sight of a new advisory can be a red build with no severity.
- `pip-audit` covers PyPI advisories only. A vulnerability in a system library that a
  wheel bundles is not visible to it.
- The tree scan matches shapes, not provider lookups. A credential in a format no rule
  describes passes; the history scan is the second net under that.
