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
