# Contributing

## Toolchain

`uv` is the only resolver. The version is pinned in `pyproject.toml`
(`[tool.uv] required-version`) and in CI, and the interpreter is pinned by
`.python-version`, so a local resolution and a CI resolution produce the same
dependency set. Do not use `pip`, `poetry` or `pip-tools` against this repository —
they will resolve a different graph and the resulting failure will be attributed to
whatever changed last rather than to the resolver.

```bash
make sync        # install the frozen dependency set
make check       # lint, typecheck, coverage — everything CI runs
```

Individual targets: `make lint`, `make format`, `make typecheck`, `make test`,
`make cov`. Run `make` with no arguments to list them.

## The static gates

Four gates run on every change, in `make check`, in CI, and — after `make hooks` —
on every commit:

| Gate | Command | What it protects |
|------|---------|------------------|
| Rules | `ruff check` | Correctness, security (`S`), async misuse (`ASYNC`), annotations (`ANN`), docstrings (`D`) |
| Formatting | `ruff format --check` | One canonical layout, so diffs show intent |
| Types | `mypy --strict` | Signature conformance at every protocol seam |
| Layering | `lint-imports` | The RFC 0001 dependency direction |

`.pre-commit-config.yaml` contains only `local` hooks that shell out through
`uv run`. A `rev`-pinned mirror would pin ruff and mypy a second time, independently
of `uv.lock`, and the first symptom of the drift is a branch that is green locally
and red in CI. `tests/test_gate_parity.py` asserts the two stay identical.

Two suppression rules are enabled deliberately. `PGH003` rejects a bare
`# type: ignore` — use `# type: ignore[code]` so the suppression stops applying when
the underlying error changes. `PGH004` rejects a bare `# noqa` for the same reason.
Relaxations in `[tool.ruff.lint.per-file-ignores]` each require an owner and a reason
on the line above, enforced by `tests/test_lint_policy.py`; that list is meant to
shrink.

The `typecheck` job also publishes an annotation-coverage table
(`mypy --any-exprs-report`) to the run summary. `Any` is sometimes the honest type,
but an `Any` count nobody measures only ever grows.

## The public API surface

Everything a consumer may import is declared: every public module has `__all__`, and
`docs/api-surface.txt` records each exported symbol with its signature, one sorted
line each. The `api-surface` job regenerates that collection and fails on any
difference, printing the added, removed and changed symbols.

```bash
make api-check      # what CI runs
make api-snapshot   # regenerate after a deliberate change
```

**A snapshot change is never incidental.** If the diff surprises you, the change is a
bug; if it does not, it needs a `CHANGELOG.md` entry and a stability decision in the
same pull request. Removing an accidentally-public helper still counts as breaking and
follows the deprecation policy.

Two further checks run over the same collection:

- **Leak check.** A public signature naming a vendor type (`redis`, `httpx`, `openai`,
  …) or a concrete `Fake*` outside `tesserix_adk.testing` fails the job naming the
  symbol. Returning `KeyValueStore` keeps every implementation substitutable; returning
  a Redis-backed store couples every consumer to Redis.
- **Re-export allowlist.** Any exported name defined outside `tesserix_adk` must be
  listed in `RE_EXPORT_ALLOWLIST`. Re-exporting a third-party type adopts that
  project's release cadence as our compatibility problem, so the list is short and
  reviewed. It is currently empty.

`tesserix_adk.experimental` is excluded from the snapshot: it carries no stability
promise, and pinning it would imply one. Promoting a symbol out of it is a changelog
entry plus a stability statement.

## Extras and the base footprint

**A provider or store SDK may only ever appear in an extra, never in `[project]
dependencies.`** The base install is `pydantic`, `httpx` and `opentelemetry-api` and
nothing else, because every package in it lands in every consumer's image whether they
use it or not. A kit that drags in temporalio, redis and psycopg to run a two-tool agent
is a kit teams copy snippets out of instead of installing.

| Extra | Installs | Import name |
|---|---|---|
| `mcp` | `mcp` | `mcp` |
| `temporal` | `temporalio` | `temporalio` |
| `graphiti` | `graphiti-core` | `graphiti_core` |
| `redis` | `redis` | `redis` |
| `postgres` | `psycopg[binary,pool]` | `psycopg` |
| `all` | the five above, as a self-reference | — |

`tests/test_extras.py` enforces the rule from both ends: the base requirement set is
exactly those three, the locked base graph stays under a transitive package ceiling, and
importing every module in the kit in a fresh interpreter must leave `sys.modules` free of
all five SDKs — asserted that way so it holds in the `all` leg too, where the wheels are
installed but must still go untouched.

Each extra also gets its own CI leg, so an accidental unconditional import of `redis`
fails in the `mcp` leg rather than after release.

Reach an optional dependency through `require_extra`, never a bare import:

```python
from tesserix_adk.core import require_extra

redis = require_extra("redis", "redis.asyncio")
```

A consumer who has not installed the extra gets `MissingExtraError` naming the extra and
the exact command (`uv add 'tesserix-adk[redis]'`), rather than an `ImportError` about a
transitive module they have never heard of. `MissingExtraError` is also an `ImportError`,
so existing `except ImportError` guards keep working. An SDK that *is* installed but
fails its own import raises through unchanged: that is the SDK's bug, and telling the
consumer to install something they already have would waste their afternoon.

Two rules the tests hold that are easy to reason past:

- `all` is a pure union (`tesserix-adk[graphiti,mcp,postgres,redis,temporal]`). Listing
  the packages directly lets `all` quietly become the only tested combination.
- Extras gate *integrations*. They may never gate anything the kit promises
  unconditionally — redaction and budget enforcement are not opt-in, so `core`,
  `runtime`, `guardrails` and `observability` contain no `require_extra` call at all.

Two extras that need incompatible versions of a shared transitive dependency is a
resolution the consumer must see and decide on. Do not pin the shared package in the base
requirements to make the conflict disappear; that pushes it onto everyone who installed
neither extra.

## Versioning and deprecations

Removing or reshaping anything in `docs/api-surface.txt` needs a `@deprecate` record two
minor releases ahead of the removal, and the removal itself waits for a release allowed to
break. Full policy, including what counts as breaking and the security exception, is in
[versioning.md](versioning.md); every live deprecation is in
[deprecations.md](deprecations.md), generated by `make deprecations`.

`make release-check` compares the surface and deprecations published at the last tag with
the ones about to ship and refuses a removal nobody was warned about, a removal earlier
than promised, or a version bump too small for what changed.

## Releasing

You do not set a version number anywhere: `hatch-vcs` derives it from the git tag, so
pushing `v0.3.0` to `main` is the whole release. The guard, the gates, the mirror and the
per-extra smoke install are described in [releasing.md](releasing.md), along with what to
do when the two indexes disagree — the one failure that needs a human.

If your change is consumer-visible, write a change fragment in `changes/` in the same
pull request — `changes/<issue>.<kind>.md`, described in
[`changes/README.md`](../changes/README.md). A conventional commit subject is enough on
its own for most changes; a fragment is required for anything breaking, because that is
where the migration note lives. The `release-notes` CI job fails on a change with
neither, and renders the notes into the run summary so you can read the wording a
consumer will read.

Your merge to `main` also publishes a pre-release, so a change that lands is installable
by a consumer within the hour. Nobody gets it by accident — a stable specifier never
resolves a pre-release — but if your change alters what a subpackage promises, say so in
[stability.md](stability.md) in the same pull request.

The example the smoke job runs is `examples/getting_started.py`. Every example in that
directory is executed by `tests/test_examples.py`, so one that stops working fails at
review rather than after the release.

## Security scanning

A separate workflow audits the lockfile for advisories and scans the tree and the history
for credentials, on every pull request and daily. Locally:

```bash
make audit      # advisories against the locked set — needs the network
make secrets    # credential shapes in the tree — offline
make licences   # dependency licences against the policy — offline
make deps       # published requirements against the dependency policy — offline
make disclosure # regenerate the tables in SECURITY.md after an advisory record
```

Adding a dependency means adding its licence obligations to every consuming product. If
`make licences` blocks, the answer is a decision recorded in `security/licences.toml` with
your name against it, or a different dependency — not a wider allow list. Note that it
only sees what is installed, so run it after `uv sync --all-extras`; CI syncs every extra.

`make check` runs neither: the audit needs the network, and the repository-wide credential
scan already runs as a test inside `make cov`, so a leak fails the suite before it fails
the workflow.

Anything either scan is asked not to fail on goes in `security/policy.toml` with an owner,
a reason and an expiry — and an expired entry fails the build. The severity policy, the
blast-radius rules and what to do when an advisory has no fix are in
[security.md](security.md). If a scan fires on a credential, rotate it first: rewriting
history is cleanup, not remediation.

## Configuration

One resolution, at startup, into one frozen `AdkConfig`. There is no reload and no second
place a setting can come from, because a setting that can change under a running agent
turns a reproduction into a guess.

| Layer | Written as | Wins over |
|---|---|---|
| code | `resolve_config({"budget.max_input_tokens": 5000})` | everything |
| env | `TESSERIX_ADK_BUDGET__MAX_INPUT_TOKENS=5000` | file, default |
| file | `adk.toml`, or `[tool.tesserix-adk]` in `pyproject.toml` | default |
| default | the field default on the model | — |

Environment keys are the dotted key upper-cased, prefixed `TESSERIX_ADK_`, with `__`
between levels — a single underscore is ambiguous the moment a field name contains one.
The file is discovered by walking upward from the working directory, so a nested service
in a monorepo picks up the repository's file without being told where it is.

Three rules the tests hold:

- **Secrets are environment-only.** A `SecretStr` field supplied by a config file is a
  hard error, not a warning: config files get committed. Secret values are masked in
  `repr`, `str`, both dump forms, provenance and error messages.
- **Every problem is reported at once.** `ConfigError` carries a `ConfigProblem` per
  failure, each naming the key, the layer that supplied it and the offending literal.
  Fixing one, restarting and discovering the next is how ten minutes becomes an afternoon.
- **Every resolved key is attributable.** `resolve_config(...).explain()` prints the
  winning layer per key and what it overrode. An operator who cannot see which layer won
  cannot debug a wrong value.

Adding a key is a minor release. Renaming or removing one goes through the deprecation
policy — the old key keeps working for the deprecation window and warns, because a
configuration key is as public as a function signature and breaks at start-up in
production rather than in CI.

## The test matrix

| Lane | When | What |
|------|------|------|
| `test-fast` | every push and PR | Supported minors × Linux and macOS |
| `test-matrix` | after merge, and on demand | Every supported minor |
| `test-extras` | every push and PR | Bare install, each extra alone, and the union |
| `test-advisory` | every push and PR, never blocking | 3.14 and 3.14t (free-threaded) |

Supported minors come from the trove classifiers in `pyproject.toml`, and
`tests/test_ci_matrix.py` fails when the matrix and the classifiers disagree in
either direction — an untested claim and an unclaimed leg are both defects. Adding
support for a minor is one classifier line; the tests then demand the CI leg.

Order is randomised by `pytest-randomly`, and every CI leg passes
`--randomly-seed=${{ github.run_id }}` so a failure caused by state leaking between
tests is reproducible from the log with the same flag.

`filterwarnings = ["error"]` makes every warning fatal. That is how a
`DeprecationWarning` from a dependency reaches us before it reaches a consumer — and
it caught a socket this suite was leaking on the day it was turned on.

The coverage floor lives only in `[tool.coverage.report] fail_under`. A workflow
that repeats the number can lower it without anyone reviewing the change.

## Network isolation

`tesserix_adk.testing.pytest_plugin` is enabled in `tests/conftest.py` and is
published for consumers to enable the same way:

```python
pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]
```

Outbound TCP and DNS raise `NetworkAccessInTestError` naming the host that was
attempted. Unix sockets stay open, because local IPC is not the hazard and blocking
it breaks unrelated tooling. A genuine integration test opts in explicitly:

```python
@pytest.mark.allow_network
def test_against_a_real_provider(): ...
```

## Quarantine

A flaky test is quarantined, never retried indefinitely:

```python
@pytest.mark.quarantine(owner="@sam123ben", expires="2026-09-01", reason="races on CI runners")
```

All three fields are required — a quarantine without an owner is a test nobody will
fix. Until the expiry the test skips with that reason printed; on the day after, it
errors the run. Renewal is possible and deliberate, which is the point.

## Dependency changes

`uv.lock` is committed and CI runs `uv lock --check`, which fails when
`pyproject.toml` was edited without regenerating the lockfile in the same commit.
That is deliberate: a lockfile that trails its manifest means every contributor
installs something slightly different from what was reviewed.

**Adding a dependency**

1. Add it to the correct place in `pyproject.toml`:
   - `[project] dependencies` — needed at runtime by consumers of the kit
   - `[dependency-groups] dev` — tooling, never shipped
   - `[dependency-groups] test` — needed only to run the suite
   - `[dependency-groups] docs` — needed only to build documentation
2. Declare a range, not a pin: `>=2.9,<3`. The lower bound is a claim that the
   version works and is verified by the `lowest-direct` CI job; the upper bound is
   the next major.
3. Run `make lock` and commit `uv.lock` in the same commit.

**Updating** — `uv lock --upgrade-package <name>` for one package, `uv lock --upgrade`
for all. Never hand-edit `uv.lock`; on a merge conflict, take either side and
regenerate.

**Removing** — delete the declaration, run `make lock`, and check nothing still
imports it.

## Why the lower bounds are tested

The `lowest-direct` job installs the minimum of every declared range and runs the
suite. Without it a floor like `pydantic>=2.0` is a guess: everyone develops against
the newest release, the code quietly starts using a 2.9 API, and the first person to
resolve an older version gets an `AttributeError` the maintainers cannot reproduce.
If that job fails, raise the floor — do not relax the test.

## Platform coverage

The lockfile resolves for every supported platform, so a macOS contributor and a
Linux CI runner agree. A dependency that ships no wheel for a supported platform
forces a source build on every consumer, which means they inherit a compiler
toolchain requirement; raise that in review before adding it.

## Consumers are not constrained by the lockfile

`uv.lock` pins this repository's development environment only. It is not published
and it does not constrain anyone who installs `tesserix-adk` — their resolver uses
the ranges in `[project] dependencies`. Keep those ranges honest and wide; keep the
lockfile exact.
