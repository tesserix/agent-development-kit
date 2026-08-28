# Releasing

An index artefact cannot be replaced. A version uploaded to PyPI is that version for as
long as the index exists — a mistake is yanked and superseded, never corrected. Every
check in the release path therefore happens *before* the build, and the whole path is
driven by one thing: the tag.

[`docs/versioning.md`](versioning.md) says what a version number promises. This says how
one is produced.

## The tag is the version

`hatch-vcs` derives the distribution version from the git tag, and
`tesserix_adk.__version__` reads it back out of the installed distribution metadata.
Nothing in the tree states a version number, so nothing in the tree can disagree with the
release it claims to be. The build job asserts the artefact filename against the tag
anyway, because the one failure mode this design cannot detect on its own is a build run
from a tree that is not at the tag.

Tags are `v<major>.<minor>.<patch>`, optionally with an `a`/`b`/`rc` suffix:
`v0.3.0`, `v0.3.0rc1`, `v1.0.0`. Anything else is refused by the guard.

## Cutting a release

1. Make sure `main` is green.
2. Run the local gates: `make check`.
3. Ask what the version has to be, and why:

   ```
   $ make release-plan
   0.4.2 -> 0.5.0

   breaking: 1
     changes/126.breaking.md is breaking
   surface: 12
     changes/100.added.md adds surface
     ... and 7 more

   run `make release VERSION=0.5.0` to cut it.
   ```

   The number is derived from what is pending — the change fragments and the public API
   snapshot diff — against the policy in [`versioning.md`](versioning.md), so the releaser
   confirms an answer instead of remembering which digit moves. Where the fragments and
   the snapshot disagree the snapshot wins: a fragment is a claim, the snapshot is
   evidence, and a consumer meets the evidence. A breaking fragment carrying no migration
   note stops the release here rather than at the review.

   `VERSION=` is accepted to override the derived number — say, to take a major
   deliberately — and the plan says so out loud when you do.

4. Read what the release will say: `make notes VERSION=0.5.0`.
5. Fold the notes into the changelog and clear the consumed fragments, then push through
   the normal review path:

   ```bash
   make release VERSION=0.5.0
   git commit -am "chore(release): notes for 0.5.0"
   ```

6. Tag the reviewed commit and push the tag:

   ```bash
   git tag -a v0.5.0 -m v0.5.0
   git push origin v0.5.0
   ```

`make release` stops before the commit and the tag and prints both: pushing the tag *is*
the publish, so it stays a decision somebody makes. The tag push is the only trigger.
Pushing to `main` never publishes.

## What the workflow does

`.github/workflows/release.yml`, in order:

| Job | What it establishes |
|-----|---------------------|
| `guard` | The tag matches the documented format, points at a commit on `main`, and names a version the index does not already hold. Nothing irreversible has happened yet. |
| `gates` | The full CI workflow — the same one pull requests run, called rather than copied. |
| `sbom` | Checks every licence in the graph against the policy, then builds `sbom.cdx.json` from the lock at this tag and diffs it against the previous release. See [`security.md`](security.md). |
| `notes` | Assembles the release body from the repository, appends the dependency diff, and fails if any change in the range has nothing describing it to a consumer. |
| `build` | `uv build`, `twine check --strict` on the metadata, an assertion that the artefact filename carries the tag's version, and keyless signing with a build provenance attestation over `dist/*`. See [`verifying.md`](verifying.md). |
| `publish` | Trusted publishing to PyPI via workflow identity, behind the `pypi` environment. Runs only where the repository variable `PUBLISH_TO_PYPI` is `true`. |
| `mirror` | The same artefacts, `sbom.cdx.json` and the attestation bundles attached to a GitHub Release, with the assembled notes as its body. A mirrored install has no attestation store to reach, so the bundles travel with the artefacts. |
| `divergence` | Fails the release if PyPI succeeded and the mirror did not. |
| `smoke` | Downloads the *published* wheel from PyPI, verifies its attestation against this repository and `release.yml` before installing anything, then installs it in a clean virtualenv once per extra and runs `examples/getting_started.py`. Skipped with `publish`. |
| `smoke-mirror` | The same, against the release assets a consumer resolves with `--find-links`. This is the channel that is always exercised, because it is the one that always ships. |

`publish` is off until the trusted publisher exists on PyPI, and `mirror` is deliberately
not downstream of it — a skipped job skips everything behind it, so a mirror that waited
on `publish` would produce no consumable release at all. Until the
[one-time setup](#one-time-setup) is done, the GitHub Release **is** the release.

There is no signing key either, and none of these jobs holds a credential that outlives
the run: signing is keyless against the workflow's own identity.

There is no upload token. Trusted publishing mints a credential for the single workflow
run and it expires with it, so there is nothing to leak, rotate or revoke — a property
`tests/test_release_workflow.py` asserts as an absence, since that is the only way to
assert it.

Pre-releases take exactly this path. A separate route for `rc` builds would be a release
path nobody has tested by the time it matters.

### One-time setup

- PyPI: a trusted publisher for `tesserix/agent-development-kit`, workflow
  `release.yml`, environment `pypi`. A second one for `alpha.yml`, same environment.
- GitHub: a `pypi` environment with the reviewers who are allowed to approve a publish.
- GitHub: repository variable `PUBLISH_TO_PYPI=true`, once the publisher exists. Until it
  is set, a tag builds, attests, mirrors and smoke-tests the release without uploading to
  the index.
- GitHub: repository variable `PUBLISH_ALPHAS=true`, once both publishers exist. Until it
  is set, `alpha.yml` builds and checks the alpha on every merge but does not upload it —
  a merge that fails on setup nobody can do from the pull request teaches the team to
  ignore a red `main`.
- GitHub, when a consumer repository exists: variable `DOWNSTREAM_REPO` and a read-scoped
  `DOWNSTREAM_TOKEN`, which turn on the integration job.

## Release notes

Notes are derived, not written at release time. Hand-written notes are always incomplete,
and the entries that get left out are the breaking ones — the consumer then meets the
change as a failing test instead of as a line in the notes.

Four sources feed them:

- **Change fragments** in `changes/`, one file per change, written in the pull request
  that makes it. Format and kinds: [`changes/README.md`](https://github.com/tesserix/agent-development-kit/blob/main/changes/README.md).
- **Conventional commit subjects** since the last tag. `feat` / `fix` / `refactor` /
  `perf` / `revert` become entries; `docs` / `chore` / `test` / `ci` / `build` / `style`
  are housekeeping and produce none. An unrecognised type is treated as an unreadable
  subject, not as housekeeping, so `core: tidy up` does not slip through the gate.
- **The public API snapshot diff** against the last tag, attached verbatim.
- **The live `@deprecate` records**, with the version each is promised to be removed in.

Two things block a release outright:

1. A change with **neither** a fragment **nor** a readable subject. Nothing would
   describe it to a consumer, so it cannot ship silently.
2. A **breaking** change with no migration note. A breaking entry without instructions is
   the failure this whole mechanism exists to prevent.

Both are checked on every pull request by the `release-notes` job, which also renders the
notes into the run summary so reviewers see the consumer-facing wording before the tag —
not after it. `make notes` does the same locally.

A change spanning several subpackages appears once, attributed to the surface named in
its fragment, rather than once per commit. Changes under `tesserix_adk.experimental` are
rendered in their own section, because the stability promises do not apply to them.
Reverts get a section too: silently reverting a feature is itself a breaking change for
anyone who adopted it.

A hotfix runs this same path. There is no emergency route that skips the documentation,
because an emergency is exactly when notes get skipped and exactly when consumers need
them.

## The internal mirror

GitHub Packages has **no Python registry** — it hosts npm, Maven, NuGet, RubyGems,
Docker and Actions, and nothing that speaks the PyPI simple API. The internal mirror is
therefore the GitHub Release assets, which consumers install from directly:

```bash
pip install --find-links https://github.com/tesserix/agent-development-kit/releases/expanded_assets/v0.3.0 \
            "tesserix-adk[mcp]==0.3.0"
```

or, in `pyproject.toml` for a uv-managed consumer:

```toml
[[tool.uv.index]]
name = "tesserix-adk-mirror"
url = "https://github.com/tesserix/agent-development-kit/releases/expanded_assets/v0.3.0"
format = "flat"
```

Until `PUBLISH_TO_PYPI` is set this is the only source for the artefacts, and afterwards
it is a second one that does not depend on PyPI being reachable. Either way `smoke-mirror`
installs from it on every release, so the path in this section is tested rather than
described. It is not a resolvable index: it serves one version per URL and does no
dependency resolution of its own, so transitive dependencies still come from whichever
index the consumer has configured. A true second index — Azure Artifacts or GCP
Artifact Registry, both of which do speak the simple API — is the follow-up if a
fully air-gapped install is required.

## When the two indexes disagree

A version on PyPI but not on the mirror is worse than no release: consumers pinning the
mirror get a resolution failure for a version that visibly exists. The `divergence` job
detects exactly this and fails the release. To reconcile:

1. Do **not** re-tag and do **not** rebuild. The artefacts that went to PyPI are the
   artefacts that must go to the mirror; a rebuild can differ.
2. Download the `dist` artefact from the failed run:

   ```bash
   gh run download <run-id> --repo tesserix/agent-development-kit --name dist --dir dist
   ```

3. Create the release with those exact files:

   ```bash
   gh release create v0.3.0 dist/* --repo tesserix/agent-development-kit --generate-notes --verify-tag
   ```

4. Re-run the `divergence` job to confirm it now passes.

If instead the mirror holds a version PyPI does not — the publish failed after the
mirror ran, which the job order makes unlikely — delete the release assets and re-run
the workflow from the tag. Only the public index is immutable.

## Yanking

A release that is broken but not dangerous is **yanked**, not deleted. Yanking leaves the
file installable for anyone who has pinned it exactly, and removes it from resolution for
everyone else:

```bash
# On PyPI: Manage project → Releases → Options → Yank, with a reason.
```

Then ship the fix as a new patch version. Never delete a version and re-upload different
content under the same number: anyone who resolved it in between has a lockfile hash
that will never match again.

Deletion is reserved for a release that must not be installed at all — leaked
credentials, or a dependency confusion vector. That case is an incident, not a release
task: follow the security policy, and expect the version number to stay burned.

## The alpha channel

Every untagged merge to `main` builds a pre-release, and publishes it when
`PUBLISH_ALPHAS=true`. The channel exists so the earliest consumer can build against the
kit while the API can still change — which is the only time their feedback can still be
acted on. What each subpackage promises is in
[`stability.md`](stability.md); this section is the mechanics.

`.github/workflows/alpha.yml` runs the same gates as a release, numbers the version with
`tools/alpha.py`, tags it locally so `hatch-vcs` derives it, and builds it. When
`PUBLISH_ALPHAS=true`, it publishes through the same trusted publisher. The tag is never
pushed: an alpha is not a release and does not
belong in the repository's tag list.

Numbering: the base is the next minor after the last stable release, and the alpha number
follows the highest alpha of that same base. Release candidates are a separate series, so
an `rc1` does not push the next alpha to `a2`.

```bash
make alpha             # the version the next merge would publish
make alpha-retention   # the pre-releases that should be yanked
```

### Promotion

alpha → rc → stable is the same pipeline, not a different one. Nothing is rebuilt from a
different tree; each step is a tag on the commit that has already been through the gates:

1. `main` publishes `0.2.0a7`, and the consumer builds against it.
2. When the surface is settled, tag `v0.2.0rc1`. `release.yml` runs, the guard applies,
   and it publishes as a release — still opt-in, because it is still a pre-release.
3. After the soak, tag `v0.2.0` on the same commit. The release guard refuses it if that
   version is already on the index, so an rc that has to change gets `rc2`, not a re-cut.

An alpha is never "promoted" by re-uploading its file under a stable version. The stable
build is made from the same commit, and its artefact is a different file with a different
hash — which is what a consumer comparing the two should see.

### Retention and yanking

Pre-releases accumulate: one per merge is hundreds a year, and an index page nobody can
read is one nobody checks. `tools/alpha.py --retention` reports what to retire:

- Any alpha of a version that has since shipped stable — nobody should be building
  against a pre-release of released work.
- All but the newest ten alphas of the version currently open.
- Never a stable release, and never a release candidate.

Yanking a broken alpha does not touch the stable channel: they are separate versions, and
a consumer on a stable specifier never resolved the alpha in the first place. A consumer
who pinned it exactly still gets it, which is the point of yanking rather than deleting.

The report is a list, not an action: **PyPI has no yank API a trusted publisher can call**,
so the yanks are done by hand from the project page. The `retention` job writes the list
into the run summary on every alpha.

## Known limitations

- Retention is a report; yanking is manual, because PyPI exposes no yank API to a trusted
  publisher (above). A token with upload rights could do it, and creating one to avoid a
  monthly click is a worse trade than the click.
- The downstream integration job is configured but inert until a consumer repository
  exists: it runs only when the `DOWNSTREAM_REPO` variable is set.
- The mirror is flat release assets, not a simple-API index (above).
- The smoke job installs from PyPI only. If PyPI is slow to make a new file available it
  retries for a few minutes; a longer outage fails the job after the release has already
  happened, which is a signal to check the index rather than to re-tag.
- Builds are not yet reproducible bit-for-bit, and artefacts are not yet signed. Signing
  and provenance attestation are the Security & Supply Chain epic's work, and attach to
  the `build` job when they land.
