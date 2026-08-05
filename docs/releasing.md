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
2. Run the local gates: `make check`. The `release-check` target compares the public API
   snapshot at the last tag against the working tree and tells you which release channel
   the change requires — take that answer rather than guessing. `notes-check` fails if
   any change in the range would ship undocumented.
3. Read what the release will say: `make notes VERSION=0.3.0`.
4. Fold the notes into the changelog and clear the consumed fragments, then push through
   the normal review path:

   ```bash
   uv run python -m tools.release_notes --version 0.3.0 --release
   git commit -am "chore(release): notes for 0.3.0"
   ```

5. Tag the reviewed commit and push the tag:

   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

The tag push is the only trigger. Pushing to `main` never publishes.

## What the workflow does

`.github/workflows/release.yml`, in order:

| Job | What it establishes |
|-----|---------------------|
| `guard` | The tag matches the documented format, points at a commit on `main`, and names a version the index does not already hold. Nothing irreversible has happened yet. |
| `gates` | The full CI workflow — the same one pull requests run, called rather than copied. |
| `notes` | Assembles the release body from the repository, and fails if any change in the range has nothing describing it to a consumer. |
| `build` | `uv build`, `twine check --strict` on the metadata, and an assertion that the artefact filename carries the tag's version. |
| `publish` | Trusted publishing to PyPI via workflow identity, behind the `pypi` environment. |
| `mirror` | The same artefacts attached to a GitHub Release, with the assembled notes as its body. |
| `divergence` | Fails the release if PyPI succeeded and the mirror did not. |
| `smoke` | Installs the *published* wheel from PyPI in a clean virtualenv, once per extra, and runs `examples/getting_started.py`. |

There is no upload token. Trusted publishing mints a credential for the single workflow
run and it expires with it, so there is nothing to leak, rotate or revoke — a property
`tests/test_release_workflow.py` asserts as an absence, since that is the only way to
assert it.

Pre-releases take exactly this path. A separate route for `rc` builds would be a release
path nobody has tested by the time it matters.

### One-time setup

- PyPI: a trusted publisher for `tesserix/agent-development-kit`, workflow
  `release.yml`, environment `pypi`.
- GitHub: a `pypi` environment with the reviewers who are allowed to approve a publish.

## Release notes

Notes are derived, not written at release time. Hand-written notes are always incomplete,
and the entries that get left out are the breaking ones — the consumer then meets the
change as a failing test instead of as a line in the notes.

Four sources feed them:

- **Change fragments** in `changes/`, one file per change, written in the pull request
  that makes it. Format and kinds: [`changes/README.md`](../changes/README.md).
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

This gives a second source for the artefacts that does not depend on PyPI being
reachable. It is not a resolvable index: it serves one version per URL and does no
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

## Known limitations

- The mirror is flat release assets, not a simple-API index (above).
- The smoke job installs from PyPI only. If PyPI is slow to make a new file available it
  retries for a few minutes; a longer outage fails the job after the release has already
  happened, which is a signal to check the index rather than to re-tag.
- Builds are not yet reproducible bit-for-bit, and artefacts are not yet signed. Signing
  and provenance attestation are the Security & Supply Chain epic's work, and attach to
  the `build` job when they land.
