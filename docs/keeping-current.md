# Keep agents current safely

“Use the latest ADK” should mean **open a tested update pull request**, not “resolve an
unbounded version in production.” Every deployed agent should be reproducible from its
committed dependency declaration and lockfile.

## Choose a release channel

| Channel | Intended use | Update rule |
|---|---|---|
| Tagged GitHub Release | Production default while PyPI setup is pending | Install an exact release asset and commit the resulting lockfile. |
| Stable PyPI release | Production after trusted publishing is enabled | Declare a reviewed compatibility window and let the lockfile select one exact version. |
| Alpha | Canary agents and downstream compatibility | Opt in to one exact pre-release; never use an unconstrained `--pre` install in a deployment. |
| `main` | Source review and contribution | Do not use as a production dependency. A branch can change without preserving the artifact you tested. |

The [Releases page](https://github.com/tesserix/agent-development-kit/releases) is the
source of truth for the newest stable tag. The GitHub Release always carries the wheel,
source archive, SBOM, and provenance bundles. A plain `uv add tesserix-adk` works only
after the PyPI trusted publisher has been enabled; do not silently fall back to `main` if
the package is not yet present on PyPI.

## Pin the artifact an agent actually runs

Until PyPI publication is enabled, install the wheel from a specific public GitHub
Release. This example names the latest stable release at the time this guide was reviewed;
replace both occurrences when a newer stable release is approved.

For a `uv`-managed application:

```bash
uv add "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.0/tesserix_adk-0.53.0-py3-none-any.whl"
git add pyproject.toml uv.lock
```

For a standard virtual environment managed with `pip`:

```bash
python -m pip install "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.0/tesserix_adk-0.53.0-py3-none-any.whl"
python -c "import tesserix_adk; print(tesserix_adk.__version__)"
```

Record that exact direct reference in the application's dependency declaration and commit
the environment's hash-pinned requirements. `pip` installs packages; it does not make an
unlocked environment reproducible by itself. Extras use normal PEP 508 syntax, for example
`tesserix-adk[a2a,mcp] @ https://.../tesserix_adk-0.53.0-py3-none-any.whl`.

After PyPI trusted publishing is enabled, use a pre-1.0 compatibility window that accepts
patch fixes without silently accepting the next potentially breaking minor release:

```bash
uv add "tesserix-adk~=0.53.0"
python -m pip install "tesserix-adk~=0.53.0"
```

In the `uv` case, commit `uv.lock` and install with `uv sync --frozen`. In the `pip` case,
commit the consuming project's resolved, hash-pinned requirements and install with
`python -m pip install --require-hashes -r requirements.txt`. The dependency declaration
states what may be considered; the lock records the exact version and hashes that were
tested.

## Automate the proposal, not the decision

Each consuming agent repository should configure `.github/dependabot.yml` or Renovate to
open dependency update pull requests on a fixed cadence. An update pull request should:

1. select the newest allowed stable ADK version;
2. run `uv lock --upgrade-package tesserix-adk`;
3. run the agent's unit, integration, evaluation, provider, MCP, and A2A compatibility
   suites;
4. treat `DeprecationWarning` as an error in CI;
5. show the ADK release notes and API-surface change to a code owner; and
6. merge only after the required checks and review pass.

For GitHub Release URLs, the updater must change the version in the URL as well as the
declared requirement. Do not configure an updater to replace the URL with a branch name.

Provider and protocol upgrades deserve their own evidence. Re-run capability checks for
the exact deployed model, the official A2A conformance cases for `a2a-sdk`, and MCP or
gateway contract tests whenever those locked packages move.

## Canary the next ADK against a real agent

The ADK's `Alpha` workflow builds an attested pre-release after every untagged merge to
`main`. Publication is deliberately disabled until the PyPI trusted publisher exists and
the repository variable `PUBLISH_ALPHAS=true` is set.

Configure `DOWNSTREAM_REPO` and a read-scoped `DOWNSTREAM_TOKEN` only for a designated
canary consumer. The workflow runs that consumer's suite against its stable baseline and
the exact published alpha, then rejects regressions before stable promotion. Alpha success
is evidence for promotion; it is not permission for production agents to float to the
newest pre-release.

## Roll forward and roll back

Prefer a fixed patch over holding production on an old version. When an ADK update causes
a regression:

1. stop promotion of the update;
2. `git revert` the dependency update pull request so the previous declaration and
   `uv.lock` return together;
3. deploy through the consumer's normal pipeline; and
4. report the failing version, provider or protocol combination, and a minimal
   reproduction upstream.

Do not edit only the version in a running environment. Restoring the reviewed lockfile is
the rollback because it restores the complete transitive graph and artifact hashes.

## Maintainer controls

This repository uses the same pattern for its own dependencies: weekly reviewed update
pull requests, a frozen lock, admission and licence records, advisory scanning, strict
types, the full test matrix, and release artifacts derived from a tag. Stable publication
is controlled by `PUBLISH_TO_PYPI`; alpha publication by `PUBLISH_ALPHAS`; downstream
compatibility by `DOWNSTREAM_REPO`. See [Releasing](releasing.md),
[Stability](stability.md), and [Release verification](verifying.md) for the exact gates.
