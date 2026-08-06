# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

### Added

- Lifecycle hooks for guardrails, budgets and approval gates, so policy attaches to the
  loop instead of being remembered at a call site. `core` gains `HookPoint` (seven points,
  `before_prompt_assembly` through `on_terminal`), `HookAction`, `HookDecision`,
  `HookSubject`, the `Hook` and `ApprovalGate` protocols, `HookChain`, `resolve_hooks`,
  `ApprovalRecord` / `ApprovalDecision`, `Agent.approval_required_tools`,
  `DeadlineConfig.hook_seconds`, the `hook_rewrite` / `hook_refusal` / `approval_required`
  / `approval_granted` / `approval_denied` events, and the `HookRegistrationError`,
  `HookEvaluationError`, `HookRefusedError`, `ApprovalDeniedError` and
  `ApprovalExpiredError` types; `AgentRunner` takes `hooks`, `approvals` and
  `approval_ttl_seconds`. The failure this closes is an agent that is safe in one product
  and unsafe in the next because the check lived in application code and the next caller
  did not write it. A hook returns a decision, never a mutation, and is handed facts rather
  than handles — no run, no config, no chain — so widening a tenant scope, disabling
  another hook or raising a cap is not something it can be talked into. The most
  restrictive decision wins and ties go to the first declared, so a chain resolves the same
  way on every process. Hooks fail closed: one that raises or outruns `hook_seconds` stops
  the run, because a check that did not run is not a check that passed — except at
  `on_terminal`, where the run is already over and there is nothing left to fail closed to,
  so a failure is recorded instead. The chain is sealed when a runner takes it, so a hook
  cannot register a permissive one behind itself or drop the one that would have refused
  it. A rewrite is logged as digests rather than content, so a replay can prove the same
  prompt was assembled without the redacted text living on in the log that was meant to
  remove it. An approval record carries a digest of the arguments and never the arguments,
  because an approval queue outlives the run and is read by people who are not party to it;
  a decision is honoured only if it echoes the record's id and lands inside
  `approval_ttl_seconds`, and a gate that fails or never answers is not a grant.
  **Stability:** additive — every name is new, and the new `AgentRunner` arguments are
  keyword-only with defaults, so a runner declared without `hooks` behaves exactly as
  before. Documented in `docs/run-loop.md` and exercised by `examples/hooks.py`.

- Caps on the shape of a run, enforced in the loop. `core` gains `LoopConfig` and
  `Agent.loop`, `RunContext.depth` and `Run.depth`, the `loop_limit_exceeded` terminal
  state, the `fan_out_refused` / `repeat_detected` / `depth_exceeded` events, and a
  `LoopLimitError` hierarchy (`RecursionLimitError`, `FanOutLimitError`,
  `RepeatedCallError`, with `MaxIterationsError` re-parented under it); `AgentRunner` takes
  `loop`, and `run` / `run_sync` take `parent`. Unlike deadlines and retries, loop shape is
  bounded by default (depth 4, 8 tool calls per turn, 32 per run, 3 identical repeats): a
  wall-clock ceiling the kit invented would kill good runs on slow hardware, where a cap on
  shape only ever stops a run that has stopped making progress, and costs nothing when it
  does not bind. An agent's own `LoopConfig` narrows the runner's and never widens it —
  how far a chain of agents may recurse is a property of the deployment paying for it, not
  of the agent. A turn that would break a cap is refused entire before any dispatch, since
  trimming a fan-out to fit leaves half a plan executed; depth is checked before a prompt
  is assembled, so a too-deep run costs nothing; and repeats are counted by request
  signature, with `Agent.idempotent_tools` exempt so polling one status endpoint is not
  mistaken for a cycle. Which cap bound is in the error type and named in the `terminated`
  event, and none of them are retryable: a cap is a decision, not a fault. **Stability:**
  additive apart from the `BudgetConfig` move noted under Changed; the new `AgentRunner`,
  `run` and `run_sync` arguments are keyword-only with defaults. Documented in
  `docs/run-loop.md` and exercised by `examples/loops.py`.

- Retry with full jitter and an explicit retryability policy. `core` gains `RetryConfig`,
  `RETRYABLE_STATUS`, `AdkError.retryable`, `ProviderError.status` / `retry_after` and
  `Agent.retry`; `runtime` gains `RetryPlan`; `AgentRunner` takes `retry` and `jitter`;
  `RunEventKind` gains `attempt_failed`. Nothing is retried by default — a retry is a
  second charge on someone's account and a second write to someone's database.
  Retryability is a property of the error rather than of the call site: a timeout and a
  transient status are faults worth a second attempt, while a rejected request, a
  guardrail refusal, a budget ceiling and a schema violation are answers, and asking again
  spends more to be told the same thing. Delays are drawn from the full window
  (`[0, min(base × multiplier^(n-1), cap))`) so a fleet does not retry one blip in unison,
  from an injected `Random` a test seeds to assert the schedule without waiting it out. A
  provider's `Retry-After` is believed over the computed backoff but refused beyond
  `max_retry_after_seconds`, a backoff that would land past the deadline is not taken, and
  a tool is retried only where `Agent.idempotent_tools` declares it safe to repeat — never
  on the shape of its exception, which says nothing about whether the side effect landed.
  **Stability:** additive — every name is new, and the new `AgentRunner` arguments are
  keyword-only with defaults that preserve existing behaviour (`max_attempts=1` means no
  run retries anything it did not before). Documented in `docs/run-loop.md` and exercised
  by `examples/retry.py`.

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

- `BudgetConfig.max_tool_calls_per_run` moved to `LoopConfig`, where it is enforced. It
  was declared on the budget and never checked, and a count of tool calls is loop shape
  rather than spend: `BudgetConfig` now carries only what is denominated in tokens and
  money. Pre-0.1.0 removal from the public surface, so there is no deprecation shim; the
  environment variable becomes `TESSERIX_ADK_LOOP__MAX_TOOL_CALLS_PER_RUN`.

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
