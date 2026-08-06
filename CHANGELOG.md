# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

### Added

- Cancellation and deadlines that stop in-flight work. `runtime` gains
  `CancellationToken` and `Deadline`; `core` gains `DeadlineConfig`, `Agent.deadlines` and
  `Agent.idempotent_tools`; `AgentRunner` takes `deadlines`, and `run`/`run_sync` take
  `cancellation` and `deadline`. A deadline is an instant rather than a duration, so it
  survives being passed down and narrows but never extends. Nothing is bounded by
  default — a ceiling the kit invented would kill good runs on the slow hardware this kit
  targets — and a zero ceiling is refused at construction. Each model call, guardrail
  check and tool call is raced against the token and the deadline using the injected
  clock; work that ignores the abort is given a grace window and then dropped, recording
  `work_orphaned` rather than blocking the run. A tool stopped after dispatch records
  `tool_indeterminate` — whether its effect landed cannot be known — unless it is declared
  in `Agent.idempotent_tools`, which is validated against the agent's allowlist.
  `tesserix_adk.testing` gains `StallingProvider` and a manual-advance `FakeClock`, so
  timeout tests are deterministic and never sleep. **Stability:** additive — every name is
  new, and the new `run`/`run_sync`/`AgentRunner` arguments are keyword-only with defaults
  that preserve existing behaviour. Documented in `docs/run-loop.md` and exercised by
  `examples/cancellation.py`.

- `AgentRunner` — the run loop, from prompt assembly to exactly one terminal state.
  `runtime` gains `assemble_prompt`, `Prompt`, `ToolDeclaration`, `wrap_untrusted`,
  `ModelRequest`, `ModelResponse` and `SystemClock`; `core` gains `RunEvent`,
  `RunEventKind`, `Run.events`, `Run.output`, `Run.record_event`, `Run.with_output` and
  `ToolFailurePolicy` with `Agent.on_tool_error`. Assembly is deterministic and ordered
  (instructions, memory, history, input), content the agent did not author is fenced as
  untrusted data, and `Prompt.version` digests the cacheable prefix onto
  `Run.prompt_version`. The loop dispatches tools against the agent's allowlist, records
  every step on `Run.events` with its `Usage`, and always returns the run — a provider
  error, guardrail refusal, schema violation, budget ceiling, iteration cap or empty
  response is a terminal state rather than an escaped exception, and a tool failure is
  wrapped in `ToolExecutionError` and either shown to the model or made terminal per
  `on_tool_error`, never replaced with an invented result. An agent declaring a guardrail,
  budget or tool registry the runner was not given is refused before the run starts.
  `tesserix_adk.testing` gains `ScriptedProvider`, `FakeToolRegistry` and `FakeGuardrail`,
  so the whole loop runs without a network. **Stability:** additive — every name is new
  and semver-governed from this release; `docs/api-surface.txt` only grows. Documented in
  `docs/run-loop.md` and exercised by `examples/run_loop.py`.

- The core primitives every other layer speaks — `Agent`, `Message`, `TextPart`,
  `BinaryPart`, `ToolCall`, `Usage`, `Run`, `RunState`, `TenantContext`, `RunContext`,
  `deduplicate`, `legal_transitions` — and a typed error hierarchy under `AdkError`
  (`CapabilityError`, `ProviderError`, `ProviderTimeoutError`, `SchemaViolationError`,
  `ToolExecutionError`, `GuardrailViolationError`, `BudgetExceededError`,
  `CancelledError`, `MaxIterationsError`), each error carrying the run and tenant it
  happened in. All are frozen, forbid unknown fields and round-trip through JSON, so a
  run checkpointed by one process rehydrates in another and no primitive can hold a
  client or a socket. `Usage` records an unknown cost as unknown rather than zero and
  refuses to total two currencies; `Run.transition_to` refuses a move the transition
  table does not declare legal, naming the legal set. **Stability:** additive — every
  name is new and semver-governed from this release; `docs/api-surface.txt` only grows.
  `tesserix_adk.testing.BudgetExceededError` now *is* the core error rather than a
  distinct class of the same name, which fixes an `except` clause that would have passed
  in tests and failed in production. Documented in `docs/primitives.md` and exercised by
  `examples/typed_primitives.py`; docstring examples are now executed in CI.

- An admission gate for third-party dependencies (`tools/admissions.py`,
  `security/admissions/`, `security/inventory.toml`), run on every pull request by the
  `dependency admissions` job. Each published requirement carries a decision record
  naming the need, the rejected alternative, maintenance, licence, transitive count and a
  `review_by` date after which the approval is a violation; all eight existing
  requirements have retrospective records. The resolved graph is committed, so a version
  bump that adds a transitive package fails until it is regenerated and the arrival shows
  in the diff. The preference order is documented in `docs/dependencies.md`, and an
  integration SDK in the base install fails the gate by name.

- A security policy and coordinated disclosure process (`SECURITY.md`,
  `security/disclosure.toml`, `security/advisories/`, `tools/disclosure.py`): a private
  reporting channel, per-severity acknowledgement and fix targets, and a process that
  patches every supported minor and publishes the advisory in the same move. The
  commitments are checked — `make disclosure` fails on a missed acknowledgement, a
  supported minor left unpatched, publication before acknowledgement, consumers notified
  after the fact, or a public disclosure carrying no interim mitigation — and the tables
  in `SECURITY.md` are generated from the same records. Targets are keyed by severity
  alone, so a flaw in an optional extra is never deprioritised for being one.
- `docs/threat-model.md`: the guarantees the kit makes, the assumption behind each, and
  an explicit list of what it does not defend against, so a consuming product does not
  mistake a design boundary for a security control.
- A dependency policy the published requirements are held to
  (`tools/dependency_policy.py`, `security/dependencies.toml`, `docs/dependencies.md`),
  checked on every pull request by the `published requirements` job. Builds stay exact
  through `uv.lock`; what the kit *publishes* now carries justified floors and, with one
  recorded exception, no upper bound. Every cap names the incompatibility that earned it,
  the trigger that removes it and its owner, and a record for a package nothing depends
  on any more fails the job. Weekly grouped updates via `.github/dependabot.yml` with
  majors outside every group; a pull request labelled `dependencies` runs the full matrix
  as well as the lowest-direct leg.
- Signed releases with build provenance (`docs/verifying.md`): every artefact, on the
  release and alpha channels alike, is signed keyless by workflow identity and carries a
  SLSA provenance attestation naming the repository, workflow, commit and build inputs.
  `sbom.cdx.json` is attested by the same run, uploads carry a PEP 740 attestation to
  PyPI, and the bundles are attached to the GitHub Release so a mirrored or air-gapped
  install can verify offline. Both the release smoke job and the downstream consumer job
  verify before installing and fail closed. There is no signing key to steal or rotate.
- A bill of materials published with every release (`tools/sbom.py`): a CycloneDX 1.6
  document built from `uv.lock` inside the release job, so it describes the graph that was
  published rather than a later re-resolution, attached to the GitHub Release as
  `sbom.cdx.json`. Each component carries its purl, licence, source hash, the install
  profile that reaches it (`base` or `extra:<name>`) and every wheel with its platform tag
  and hash; development-only packages are excluded. A diff against the previous release's
  document goes into the release notes.
- A licence policy across the whole graph (`tools/licences.py`, `security/licences.toml`),
  checked on every pull request and again in the release before anything irreversible. An
  undeclared or disallowed licence blocks, `A AND B` needs both halves allowed, `A OR B`
  needs a recorded decision naming an owner, and a licence off the allow list can be
  accepted for one named package and one named licence only.
- Dependency and secret scanning (`docs/security.md`, `.github/workflows/security.yml`):
  advisories audited against the frozen lock on every pull request and daily, rated from
  OSV with an unrated advisory blocking, and reported with the first fixed version and the
  blast radius derived from the lockfile — a finding reachable only through an extra still
  blocks and names the extra. Credentials are matched by shape across the tree and by a
  pinned `gitleaks` across the history, with recorded provider traffic also checked for
  personal identifiers and every matched value truncated in the report. Suppressions live
  in `security/policy.toml` and require an owner, a reason and an expiry of at most 90
  days; an expired suppression fails the build. Neither job uses a repository secret, so
  both run on a pull request from a fork.
- Pre-release alpha channel (`docs/stability.md`, `.github/workflows/alpha.yml`): every
  merge to `main` publishes a pre-release through the same gates and trusted publisher as
  a release, numbered by `tools/alpha.py` from the next minor after the last stable.
  Consumers opt in explicitly — a stable specifier never resolves a pre-release — and the
  per-subpackage stability matrix states what each one promises. Promotion is
  alpha → rc → stable through the one pipeline, retention is reported rather than
  guessed, and a downstream job runs the earliest consumer's suite against the last
  stable and the alpha, failing only on a regression the alpha introduced.
- Release notes assembled from the repository (`tools/release_notes.py`): change
  fragments in `changes/`, conventional commit subjects, the public API snapshot diff and
  the live deprecation records. A change with neither a fragment nor a readable subject
  blocks the release, as does a breaking change with no migration note. The
  `release-notes` CI job renders the notes on every pull request, and the release body is
  the assembled notes rather than generated commit titles.
- Release path (`docs/releasing.md`): the git tag is the sole source of the version
  (`hatch-vcs` derives it, `tesserix_adk.__version__` reads it back from the installed
  distribution), a guard refuses a tag off `main`, in the wrong format, or naming a
  version already on the index, and publishing goes to PyPI by trusted publishing with
  no upload token in existence. Artefacts are mirrored to GitHub Release assets, a
  divergence between the two fails the release, and a per-extra smoke job installs the
  published wheel from the index and runs `examples/getting_started.py`.
- Versioning and deprecation policy (`docs/versioning.md`), with `@deprecate` recording
  the alternative and removal version, `AdkDeprecationWarning` warned once per call site
  and attributed to the caller's frame, `TESSERIX_ADK_DEPRECATIONS_AS_ERRORS` for consumers
  preparing an upgrade, a generated `docs/deprecations.md`, and a release check that
  refuses a removal with no deprecation record or a version bump too small for it.
- `tesserix_adk.core.config`: one frozen `AdkConfig` resolved once at startup from code,
  `TESSERIX_ADK_*` environment variables, `adk.toml` or `[tool.tesserix-adk]`, in that
  precedence. `resolve_config` additionally returns per-key provenance and `explain()`;
  `ConfigError` reports every problem at once with the layer that supplied each. Secrets
  are environment-only and masked everywhere they could be rendered. Adding a key is a
  minor release; renaming or removing one goes through the deprecation policy.
- Optional extras per integration (`mcp`, `temporal`, `graphiti`, `redis`, `postgres`,
  and `all` as their union), with `tesserix_adk.core.require_extra` and
  `MissingExtraError` naming the extra and its install command. Base requirements are
  `pydantic`, `httpx` and `opentelemetry-api` only, held there by a transitive package
  ceiling and a per-extra CI leg.
- Public API surface snapshot (`docs/api-surface.txt`) with a CI gate, a leak check
  rejecting vendor and concrete implementation types in public signatures, and an
  explicit allowlist for third-party re-exports.
- `tesserix_adk.testing.pytest_plugin`: network isolation for unit suites and a
  quarantine marker requiring owner, expiry and reason.
- `tesserix_adk.core`: protocols, error hierarchy, `verify_conformance`, and shipped
  conformance suites and fakes in `tesserix_adk.testing`.

### Changed

- Published requirements no longer carry speculative upper bounds. `pydantic`, `httpx`,
  `opentelemetry-api`, `mcp`, `temporalio`, `redis` and `psycopg` declare a floor only,
  so a consuming product that moves to a new major is not blocked by a resolution error
  it cannot fix without forking the kit. The one remaining cap is `graphiti-core<1`,
  recorded with its incompatibility and removal trigger in `security/dependencies.toml`.

### Notes

- `tesserix_adk.experimental` carries no stability promise and is deliberately excluded
  from the snapshot. Promoting a symbol out of it requires an entry here and a
  stability statement in the same pull request.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
