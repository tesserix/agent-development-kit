# Dependencies

A library has two obligations that pull in opposite directions.

Its own builds must be reproducible: the same commit resolves to the same graph a year
later, or a green pipeline proves nothing about what was published. That is `uv.lock`,
which pins every direct and transitive package to an exact version and hash, and which
every job in CI installs with `--frozen`.

Its *published* requirements must not over-constrain the product that installs it. The
kit is a dependency, not an application; a consumer already depends on `pydantic` and
`httpx` for their own reasons. So the requirements that travel with the distribution
carry floors and, almost never, an upper bound.

## Exact for builds, loose for consumers

| | Where | What it says | Who it binds |
|---|---|---|---|
| Lockfile | `uv.lock` | Exact version + hash for every package, direct and transitive | This repository's builds, tests and releases |
| Published requirements | `pyproject.toml` `dependencies` and `optional-dependencies` | A floor, and only a recorded cap | Every product that installs the kit |

Nothing in `uv.lock` reaches a consumer. Installing `tesserix-adk` resolves against the
published requirements alone, which is why those are governed separately.

## Why speculative upper bounds are the anti-pattern

`pydantic>=2.9,<3` looks like caution. What it does is guarantee that on the day the
consumer's product moves to pydantic 3, the kit becomes a resolution error they cannot
fix — not by configuration, not by an override, only by waiting for a release or forking
the kit. The cap was written against a break nobody had seen, and it fires against an
upgrade somebody chose.

So the rule is: **cap only where an incompatibility is proven, and record it.**

`security/dependencies.toml` holds both halves of the policy:

- `[[floor]]` — the oldest version the kit claims to work with, and why that one.
  A floor is a testable claim, and the `lowest-direct` CI leg tests it.
- `[[cap]]` — an upper bound, the incompatibility that earned it, the trigger that
  removes it, and the owner who removes it. A cap with no trigger is a permanent cap
  that nobody decided to make permanent.

`tools/dependency_policy.py` enforces it in the `published requirements` CI job:

```bash
make deps
```

It fails on a cap nothing recorded, a cap that disagrees with its record, a floor the
policy does not justify, a requirement with no floor at all, and on a record for a
package nothing depends on any more — the stale record is a violation because it
outlives the dependency and misleads the next reader.

Development groups are outside the policy. A cap in the `dev` group constrains nobody;
it is never in a consumer's resolution.

## Proving both ends of every range

A floor nothing resolves against is a floor nobody has checked. Two CI legs cover the
range the requirements claim:

- `lowest-direct` resolves with `--resolution lowest-direct`, installing every declared
  floor and running the suite against it.
- `test-matrix` resolves normally across the full supported Python matrix.

The fast lane on a pull request runs the two ends of the Python range. A pull request
labelled `dependencies` runs the full matrix as well, because an update has to prove all
of it, not the ends.

## Cadence

`.github/dependabot.yml` checks the `uv` and `github-actions` ecosystems every Monday.
A pinned action ages exactly like a pinned package and nothing else moves it.

Routine minor and patch updates arrive **grouped into one pull request** per ecosystem.
Forty separate pull requests are not reviewed, they are approved.

Majors are deliberately outside every group. A provider or store SDK major gets its own
change, its own migration note in `changes/`, and its own CHANGELOG entry stating what
moved for a consumer.

## Security updates do not wait for Monday

Advisories are not on the cadence. A dependency with a known advisory is fast-tracked:

1. `make audit` — the advisory scan, and the blast radius it reports (which install
   profile reaches the package, and whether the kit's code path uses the affected API).
2. Bump, lock, run the suite, merge. No grouping, no waiting for the weekly batch.
3. If no fixed version exists, record a suppression in `security/advisories.toml` with
   an owner, a reason and an expiry — never an indefinite one.

See [`security.md`](security.md) for the scanning gates themselves.

## Changes that need more than a green suite

Some updates are not a version bump:

- **Behaviour-affecting dependencies.** A model client, tokeniser or reranker version
  can change output without failing a single unit test. Those go through the evaluation
  gate before merging, not the unit suite alone.
- **A new dependency.** Adding one is a supply-chain decision, reviewed against its own
  criteria — not folded into an update pull request.
- **An unmaintained package.** When a dependency stops receiving fixes, the decision is
  replacement or vendoring, recorded as a decision with an owner. Pinning it forever and
  suppressing its advisories is not a third option.

## Adding a dependency

A package added here is added to every product that installs the kit, and none of those
teams got to review it. So an addition is a decision with an owner, not a line in a
manifest.

### Prefer, in this order

1. **The standard library.** Slower to write, nothing to inherit.
2. **An existing dependency.** If `httpx` already ships, a second HTTP client is a
   liability with no gain.
3. **An optional extra.** Anything specific to one provider, store or protocol — every
   integration SDK is here, behind a protocol, so a consumer who wants none installs
   none. `mcp`, `temporalio`, `graphiti-core`, `redis` and `psycopg` in the base install
   are a hard failure of the gate, by name.
4. **Vendor it.** Forty lines copied in with the licence header beats a package every
   consumer inherits for a function they will never call. This is a recordable outcome:
   `profile = "vendored"`.
5. **A base requirement.** Last, and only when the kit's central promises depend on it.

### What the record answers

Each direct requirement has a file in `security/admissions/` answering, in prose a
reviewer can disagree with:

| Field | The question it answers |
| --- | --- |
| `need` | What breaks without it, in terms of what the kit promises. |
| `alternatives` | What was rejected, and why. A record naming none is refused. |
| `maintenance` | Release cadence, maintainer count, whether it is pre-1.0. |
| `licence` | The SPDX identifier, cross-checked by the licence gate. |
| `transitive` | How many packages it drags in — the number a one-line diff hides. |
| `native_build` | Whether a consumer without a wheel needs a compiler. |
| `security_history` | Advisory record, and how fast fixes landed. |
| `review_by` | When this approval stops being current. |

`review_by` is the field that keeps the set honest. A package that goes unmaintained
otherwise stays approved forever on the strength of a decision made when it was healthy.

### The resolved graph is committed

`security/inventory.toml` lists all 51 packages a consumer can end up with and the
profiles that reach each one. It is generated — `make admissions` — and CI fails when the
lock disagrees with it. That is the point: a routine version bump that quietly adds a new
transitive package fails until someone regenerates the file, and then the arrival is a
line in the diff rather than something nobody sees. Development-only packages are absent;
they are never in a consumer's resolution and carry the maintainer's own bar instead.

## Review

`pyproject.toml`, `uv.lock`, `security/` and `.github/dependabot.yml` have a named owner
in `.github/CODEOWNERS`. Unowned update pull requests accumulate until somebody merges
the pile without reading it, which is the failure the weekly cadence was supposed to
prevent.

## Known limitations

- **Dependabot cannot update `uv.lock` while `required-version` excludes its bundled uv.**
  `pyproject.toml` pins `required-version = ">=0.12,<0.13"`; the hosted updater currently
  runs uv 0.11.31 and reports `tool_version_not_supported` for every locked package, so
  the `uv` ecosystem opens manifest-only pull requests and its run is red. The pin stays:
  dropping it would let a different uv rewrite the lock and take reproducibility with it.
  Until the updater ships uv 0.12, locked-package updates are done on the weekly rota by
  hand — `uv lock --upgrade`, run the suite, open the change with the `dependencies`
  label. The `github-actions` ecosystem is unaffected.
- **A label Dependabot is told to apply must already exist on the repository.** A label
  named in `.github/dependabot.yml` that does not exist is dropped silently, and a
  dependency pull request without the `dependencies` label never starts the full matrix.
  `dependencies` and `actions` exist; adding a third means creating it first.
- **The committed inventory describes the committed lock, and nothing else.** The
  `lowest declared versions resolve and pass` job re-resolves the lock on purpose, so the
  two graphs differ by design and that job skips the inventory assertions. A floor that
  drags in a package the pinned resolution does not is therefore not caught there — the
  gate sees it the moment the lock moves to it.
- **The admission gate reads records, not judgement.** It fails on a package with no
  record and on a graph that moved; it cannot tell you a record's `maintenance` claim was
  true when it was written and is false now. That is what `review_by` is for, and the
  rota, not the tool, does the reading.
- **`graphiti-core` brings a telemetry SDK.** The `graphiti` extra resolves `posthog`
  transitively, along with `openai`, `numpy` and `requests` — fifteen packages for one
  requirement. Its record says so and its re-review is set six months out rather than
  twelve. A consumer who does not install that extra inherits none of it.
- The floor check compares the declared floor against the recorded one as a string. A
  floor written `>=2.9.0` where the policy records `2.9` is reported as a disagreement;
  the fix is to write them identically, which is the point.
- Only the first lower and upper bound of a requirement is examined. A requirement with
  two upper bounds, or an environment-marked cap, is not something the kit publishes and
  is not modelled.
- The policy governs requirements the kit publishes. It says nothing about a *transitive*
  cap — if a dependency of a dependency caps `pydantic`, the consumer inherits it and no
  gate here catches it. The SBOM shows the graph; resolving that conflict is manual.
