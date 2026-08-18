# RFC 0001 — Package layout and boundaries

**Status:** Accepted
**Applies to:** every module under `tesserix_adk`

Four products import this kit. An import path, once published, is public forever —
moving it later is a breaking change for all of them simultaneously. This RFC fixes
the names and the permitted dependency direction once, and encodes the direction in
`.importlinter` so it is a machine-checked property rather than a convention that
survives only as long as reviewers remember it.

Amending this RFC and changing the layout are the same change. A story that needs a
module which fits no remit below is blocked until the amendment lands with it.

## 1. Distribution and import root

| | |
|---|---|
| Distribution | `tesserix-adk` |
| Import root | `tesserix_adk` |
| Source layout | `src/tesserix_adk/` |
| Minimum Python | 3.12 |

`src/` layout is not stylistic. Under a flat layout the test suite imports the
working tree rather than the installed artefact, so a packaging mistake — a
subpackage missing from the wheel, a missing `py.typed` — passes CI and fails for
the first consumer who installs it.

## 2. Subpackages

Sixteen, plus `experimental`. Each owns one remit; a module that needs two of them
is two modules.

| Subpackage | Remit |
|---|---|
| `core` | Frozen primitives, protocols and error types. The vocabulary every other layer speaks. |
| `runtime` | Run loop, caps, cancellation, timeout, checkpointing, streaming. |
| `models` | Model adapters and the ModelBus. Provider-specific code, behind a core protocol. |
| `tools` | Tool registry, argument schemas, invocation and the ToolBus. |
| `code_intelligence` | Backend-neutral source workspace, query and automatic context contracts. |
| `mcp` | Model Context Protocol client and server integration. |
| `a2a` | Agent-to-agent interoperability. |
| `memory` | Run state, working context, episodic and semantic stores. |
| `rag` | Retrieval, embedding, chunking and reranking. |
| `workflows` | Durable orchestration and long-running composition. |
| `guardrails` | Inline enforcement: policy, schema, approval and budget checks. |
| `evals` | Evaluation harness, gold sets and quality gates. |
| `observability` | Sideband tracing, metrics and audit emission. |
| `cli` | Command-line entrypoints. |
| `testing` | Fakes and fixtures published for consumers to test against. |
| `adapters` | Concrete backing stores and transports. |
| `experimental` | Unstable surfaces, exempt from the deprecation policy but not from layering. |

## 3. Dependency direction

Imports point inwards. Higher layers may import lower ones; the reverse is a build
failure. Packages on the same rank may not import each other at all.

```
cli                                    leaf — nothing imports it
testing                                leaf — nothing imports it at runtime
adapters                               leaf — nothing imports it
evals
workflows
a2a · mcp · rag · memory · guardrails · observability · tools · models
code_intelligence
runtime                                imports core only
core                                   imports nothing else in the kit
```

Three consequences worth stating explicitly, because each is a decision rather than
a derivation.

**`runtime` imports `core` only — not `guardrails`, not `observability`, not code
intelligence.** These are crossed during a run, so the obvious layout has the runtime
import them. It must not. Generic contributor contracts live in `runtime`; code
intelligence implements one from the layer above, while other protocols live in `core`
and the runtime holds those. Implementations are injected at composition time by
`adapters` or `cli`. Otherwise the runtime cannot be tested without every integration,
the enforcement packages cannot be swapped, and `core` acquires a transitive dependency
on everything.

**The integration rank is mutually exclusive.** `tools` may not import `models`,
`memory` may not import `rag`. Where two genuinely need a shared type, that type
belongs in `core`. This is the only permitted resolution of a would-be cycle —
never a deferred or function-local import, which hides the cycle from the linter
while keeping it in the program.

**Vendor types never travel inward.** An interop shim may depend on a vendor SDK,
but no signature in `core` or `runtime` may mention a vendor type. A consumer who
does not install that extra must still be able to typecheck against the kit.

## 4. Naming

- Modules are named for what they provide (`retry.py`, `budget.py`), never for what
  they contain (`utils.py`, `helpers.py`, `common.py`, `types.py`, `models.py`).
- Protocols carry no `I` prefix and no `Protocol` suffix: `ModelClient`, not
  `IModelClient`. The `Protocol` base already says so and the checker already knows.
- Errors end in `Error` and inherit from a single `AdkError` in `core`, so a
  consumer can catch this kit's failures without catching `Exception`.
- No stuttering: `tools.Registry`, not `tools.ToolRegistry`.
- A leading underscore marks a private module. Absence of one is a public promise.
- `tesserix_adk.experimental.<name>` for anything that may move. Graduating a module
  out of `experimental` is an addition at the new path plus a deprecation at the old
  one, never a silent move.

## 5. Decision log

**Single distribution, not one per subpackage.** Fifteen distributions would need
fifteen release cadences and a version-compatibility matrix between them, and every
consumer would pin a different subset. The cost of the single distribution is that a
consumer installs modules it does not import — cheap, since the heavy dependencies
are extras and an unimported module is a few kilobytes. Revisit only if an extra's
dependency footprint becomes unavoidable for consumers who do not use it.

**Flat package, not a namespace package.** A namespace package would let a separate
repository contribute `tesserix_adk.something`, which sounds like extensibility and
is in practice a way for an import path to resolve differently depending on what
else is installed. Third-party extensions register through an entry point instead,
which is explicit and inspectable.

**`testing` ships inside the distribution.** The alternative — a separate
`tesserix-adk-testing` package — keeps test-only helpers out of production installs,
at the cost of a second version to keep in step with the first. Fakes must match the
protocols they fake exactly, and the cheapest way to guarantee that is to have them
typechecked in the same run against the same source. The layering contract already
prevents anything importing `testing` at runtime, which is the risk that motivated
the split.

**Layering enforced by `import-linter`, not by review.** A layering rule that lives
only in this document is enforced when the reviewer happens to notice. The contract
in `.importlinter` names the forbidden edge and the contract it violates, in CI, on
the commit that introduces it.

## 6. Enforcement

`.importlinter` defines four contracts, run by `make lint` and in CI:

| Contract | What it catches |
|---|---|
| `layers` | Any import against the direction above |
| `core-is-independent` | Anything in `core` reaching into another subpackage |
| `leaves-are-not-imported` | Runtime code importing `cli`, `testing` or `adapters` |
| `no-vendor-types-inward` | A vendor SDK reached from `core` or `runtime` |
