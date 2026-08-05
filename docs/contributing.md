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
