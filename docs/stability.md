# Stability

What a consumer may rely on, per subpackage, and what the pre-release channel promises.

Stability is a per-subpackage statement, not a per-release one: the kit ships one version
number, but `core` and `experimental` are not the same promise, and a consumer choosing
between them needs to know which they are taking on.

## Levels

| Level | Promise |
|-------|---------|
| `stable` | Breaking changes only in a major release, after a deprecation cycle. In `docs/api-surface.txt`. |
| `beta` | Breaking changes in a minor release, announced in the release notes with a migration note. In the surface snapshot. |
| `alpha` | Shape is still being decided. Breaking changes in any release, with an entry in the notes. In the surface snapshot so the change is at least visible. |
| `experimental` | No promise at all. May change or disappear without a note. Excluded from the surface snapshot. |
| `internal` | Not a public surface. Import it and the next release will break you. |

## Matrix

| Subpackage | Level | Notes |
|---|---|---|
| `core` | `beta` | Protocols, errors, config, extras. The one surface everything else is built on; shape is settled, names are not all final. |
| `testing` | `beta` | Conformance suites, fakes, pytest plugin. Moves with `core`. |
| `models` | `alpha` | Provider abstraction still being validated against a second provider. |
| `runtime` | `alpha` | Execution loop and step semantics under active design. |
| `tools` | `alpha` | Tool declaration and dispatch. |
| `code_intelligence` | `alpha` | Source context contracts and backend mappings are new. |
| `memory` | `alpha` | Store protocol settled; the retrieval surface is not. |
| `rag` | `alpha` | Retrieval pipeline surface is provisional. |
| `guardrails` | `alpha` | Policy interfaces provisional. |
| `observability` | `alpha` | Tracing surface follows OpenTelemetry; the kit-side helpers may change. |
| `adapters` | `alpha` | One adapter per integration; each moves with its upstream SDK. |
| `mcp` | `alpha` | Tracks the MCP specification, which is itself moving. |
| `a2a` | `alpha` | Tracks the A2A specification, which is itself moving. |
| `workflows` | `alpha` | Durable-execution surface provisional. |
| `evals` | `alpha` | Evaluation harness surface provisional. |
| `cli` | `alpha` | Command names and flags may change; the commands themselves are covered by tests. |
| `experimental` | `experimental` | No promise. Promotion out of it requires a stability statement and a changelog entry in the same pull request. |

## The alpha channel

Every merge to `main` publishes a pre-release (`0.2.0a3`, `0.2.0a4`, …). It carries the
level stated above for each subpackage and *no additional promise*: an alpha build is not
a release, it is `main` made installable.

Getting one is opt-in, by PEP 440's own rule — a stable specifier never resolves a
pre-release, so there is no way to land on an alpha by accident:

```bash
# Never resolves an alpha
uv add tesserix-adk

# Newest pre-release, explicitly asked for
uv add --prerelease=allow tesserix-adk
pip install --pre tesserix-adk

# The one you tested against, for a reproducible build
uv add "tesserix-adk==0.2.0a3"
```

There is one exception, and it is PEP 440's, not the kit's: a specifier that *only*
pre-releases can satisfy resolves one anyway. `tesserix-adk==0.2.*` before `0.2.0` ships
gets `0.2.0rc1`, because the alternative is an unsatisfiable pin with no explanation. As
soon as `0.2.0` exists the fallback stops and the same specifier resolves the stable
release. If that is not what you want, pin a lower bound that has shipped.

`tools/alpha.py` numbers them: the base is the next minor after the last stable release,
and the alpha number follows the highest alpha of that same base. Release candidates are a
separate series — an `rc1` does not make the next alpha an `a2`.

## Promotion, retention and yanking

Promotion and retention are documented with the rest of the release path in
[`releasing.md`](releasing.md#the-alpha-channel): alpha → rc → stable goes through the
same pipeline and the same guard, a broken alpha is yanked without touching the stable
channel, and `make alpha-retention` names the pre-releases that should be retired.
