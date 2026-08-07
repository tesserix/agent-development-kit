# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

### Breaking changes

- `Attribution` gains a required `definition` dimension, exported on every span as
  `adk.definition`. A bill broken down by `agent_version` cannot tell two runs of one version
  apart when the version was edited between them; the definition revision names the exact
  reviewed artifact that spent the money. Required rather than defaulted, because attribution
  is derived from the run and a silent default is how a dimension quietly stops being
  populated. **Stability:** breaking for callers constructing an `Attribution` by hand — pass
  `definition=`, or `UNKNOWN` where the run had no definition — and for anything asserting on
  `Attribution.unknowns`, which now names `definition` for runs started from a bare agent.
  Attribution derived through `spend_of` needs no change. Documented in
  `docs/agent-definition.md` and `docs/cost-attribution.md`.

- The prompt's cacheable prefix is now an invariant with a name. `assemble_prompt` assembles
  five documented layers — `PROMPT_LAYERS`: system, tools, pinned, retrieved, conversation —
  and `Prompt.layers` labels every assembled message with the one it came from, so a change
  that reorders the prefix fails a test naming the regression instead of quietly doubling a
  bill. `Prompt.fingerprint` digests the prefix *as bytes*, pinned context included: equal
  fingerprints across two turns mean the inference server reuses the prefix it already
  evaluated, which on CPU is the difference between usable and unusable, since prefill
  dominates and a prompt that costs a second on an H100 costs tens of seconds without one.
  `Prompt.prefix` is the messages that digest covers and `Prompt.prefix_tokens` is how large
  they are, counted by `approximate_tokens` — four characters to a token, fine for a log line
  and wrong for a context-window check — or by any `Tokenizer` passed as `tokenizer=`. The
  `PROMPT_ASSEMBLED` run event carries the fingerprint and the prefix size, so a cache-hit
  ratio is measurable from the audit trail. Tool declarations are sorted by name rather than
  kept in registry order, and two tools sharing a name are refused.
  `assemble_prompt(memory=...)` is replaced by `pinned=` and `retrieved=`.
  **Stability:** breaking for callers passing `memory=`, for anything depending on tool
  declaration order, and for a persisted `Prompt`, which gains three fields. Documented in
  `docs/run-loop.md`, exercised by `examples/prompt_prefix.py`.

- How deep and how wide a run may go are caps on spend, so they live with the money.
  `BudgetLimits` gains `max_delegation_depth`, `max_parallel_tool_calls` and
  `max_peer_invocations`; `LoopConfig` loses `max_depth`, `max_tool_calls_per_turn` and
  `max_tool_calls_per_run`, keeping `max_repeated_calls`, which is not spend but the shape of
  a run that has stopped making progress. A turn wider than `max_parallel_tool_calls` is
  refused entire before any tool is dispatched, naming the cap and what was asked for, and the
  run total is checked against the whole turn so a fan-out cannot step over a ceiling one call
  at a time. Depth and peer invocations are checked before a prompt is assembled and the
  refusal prints the call path, carried on the new `Run.path` and `RunContext.path`; peer
  invocations are counted on the shared ledger, so a tree of runs cannot each stay under a cap
  they broke together and a delegated agent cannot vote itself more rope than its parent had.
  Tool calls in a cleared turn still go out one at a time, because the check between them is
  what stops the second call of a turn the caller cancelled during the first.
  **Stability:** breaking for anything constructing `LoopConfig` with the removed fields.
  Documented in `docs/run-loop.md` and `docs/budget.md`, exercised by `examples/loops.py`.

- A ceiling said once and honoured everywhere, and no way to build a runtime without one.
  `BudgetLimits` replaces `BudgetConfig` on `Agent.budget` and `AdkConfig.budget` and states
  money as `Decimal` with a currency alongside tokens, model calls, tool calls, iterations
  and wall-clock seconds; every field is optional to write and none is optional in effect,
  since `filled()` replaces what was left unsaid with `BudgetLimits.conservative()`. Limits
  attach to a run, agent, tenant or tenant window and `most_restrictive` resolves them per
  dimension with the winning scope recorded, refusing two scopes of one kind and two
  currencies rather than converting. `BudgetPolicy.record` now takes a `Usage` and typed
  counts rather than an integer, and the protocol gains `resolved`, `limits`, `child` and
  `check`; `BudgetExceededError` names the limit breached, its scope, the ceiling, the
  consumed amount and the remaining one, and the new `BudgetUnavailableError` is what a run
  fails closed on when the ledger holding a shared ceiling cannot be reached. A runner given
  no policy resolves one per run and records it on `run.budget`, so an agent declaring a
  budget with no policy is no longer a `ConfigurationError`; removing the ceiling takes
  `UnlimitedBudget(reason=...)`, which will not be built without a stated reason.
  **Stability:** breaking for anything constructing `BudgetConfig`, setting
  `budget.max_tokens_per_run`, or implementing `BudgetPolicy`. Documented in
  `docs/budget.md`, exercised by `examples/budget.py`.

- Usage and cost are two records, not one float. `Usage` is what a call consumed and `Cost`
  is what that came to; `Usage.cost` is now a `Cost` rather than a `float`, `Usage.currency`
  is gone because money without a currency is a number, and `Usage.estimated` is derived from
  a new `CountSource` saying who counted — the provider, the model's own tokeniser, or
  characters over a constant. `Usage` gains `cache_write_tokens`, `reasoning_tokens` and
  `image_units` as fields, and every vendor adapter normalises into them rather than leaving
  its own key in `extras`: OpenAI reports reasoning inside its completion total and the
  adapter splits it out, so one workload reads the same way whoever answered it. `Cost` is
  `Decimal` throughout with input, output, cache-read, cache-write, reasoning and image kept
  separate and unrounded until `quantised()`; totals keep the weaker confidence and refuse to
  add two currencies. **Stability:** breaking for anything reading `usage.cost` as a number
  or `usage.currency`, and for anything reading `extras["reasoning_tokens"]`,
  `extras["thoughts_tokens"]` or `extras["cache_creation_input_tokens"]`. Documented in
  `docs/cost.md`, exercised by `examples/cost.py`.

- One typed provider protocol, with what a model can do declared as data. `ModelProvider`
  now states `complete`, `stream`, `count_tokens` and `capabilities` over the kit's own
  request and response types, which moved into `core` so the protocol could be typed over
  them without inverting the layering; `tesserix_adk.models` re-exports the surface.
  `ModelCapabilities` carries `structured_output`, `tool_calling`, `parallel_tool_calls`,
  `vision`, `streaming`, `context_window_tokens` and `max_output_tokens`, every one
  defaulting to off or unknown — silence is not a claim. The record is read before anything
  is sent: a tool registry wired to a model that does not call tools fails at construction,
  an agent naming tools fails before its first request, and an image part or an over-long
  prompt fails the same way, the second with `ContextWindowExceededError` carrying the count
  and the limit. A payload that is not a response at all raises `ModelResponseError` with
  the raw payload and the provider's request id, distinct from a well-formed answer in the
  wrong shape, which stays repairable. `ModelRef` and `ModelSpec` address a model from
  configuration with the provider part of its identity; `declaring` and `with_capabilities`
  narrow a record without a subclass. `ModelProviderConformance` is the suite a third-party
  provider inherits. **Stability:** breaking for provider implementors, additive for
  callers — a provider must add `capabilities` and `count_tokens`, and
  `ScriptedProvider(structured=True)` becomes
  `ScriptedProvider(capabilities=CAPABLE.declaring(structured_output=True))`. No call site
  that only uses a runner changes. Documented in `docs/providers.md`, exercised by
  `examples/providers.py`.

### Added

- A run can now be put on the wire without each product reinventing the bridge.
  `tesserix_adk.adapters` gains `RunBroker`, which drives one run once however many
  transports read it, `sse_events` for correctly framed server-sent events with `SSE_HEADERS`
  and heartbeats that survive an intermediary, and `WebSocketBridge` for the same payloads
  plus a control channel carrying cancellation and approval decisions. A reconnecting client
  presents its last sequence id and receives what it missed or an explicit `StreamGap` —
  silently closing the gap is how a UI ends up showing a run that never happened. The
  boundary authorises the tenant before framing anything and gives one refusal for an unknown
  run id and another tenant's, since which ids exist is itself tenant information. Every
  event is re-scrubbed on its way out and an oversized one becomes a `PayloadElided`
  reference rather than truncated into invalid JSON. Framework-neutral: the websocket bridge
  asks for `send_text`, `receive_text` and `close`, so no web framework enters the core path.
  **Stability:** additive; nothing existing changes. `RunStream` gains `run_id`, readable
  before the run has produced anything, and abandoning a stream nobody ever read now cancels
  it rather than leaving a run to start later with no reader. Documented in
  `docs/transports.md`, exercised by `examples/transports.py`.

- A stream can now be consumed three ways, and what it holds mid-flight cannot be mistaken
  for a result. `RunStream` is awaitable and an async context manager: iterate then await
  for progress plus the authoritative record, await alone for the answer with no progress,
  or iterate and leave once you have seen enough. Awaiting the same stream from two places
  drives the run once and hands both the same `Run`. `stream.provisional` is a
  `Provisional[OutputT]`, which the type checker refuses everywhere an `OutputT` is required
  — half a JSON object is shaped exactly like a whole one, so the distinction cannot be left
  to a naming convention; `snapshot()` gives a plain mapping, and `None` while the object is
  still half-arrived. Only the run's own `output` is schema-validated. Leaving the context
  manager cancels a run nobody is reading any more, through the same cancellation path a
  caller's own token uses, and awaiting an abandoned stream raises `StreamInterruptedError`
  carrying what arrived rather than promoting it to a result. **Stability:** additive —
  `RunStream` gains `__await__`, `__aenter__`/`__aexit__`, `aclose` and `provisional`, and
  existing iteration is unchanged. Documented in `docs/run-progress.md`, exercised by
  `examples/stream_consumption.py`.

- A run can now be watched while it happens. `AgentRunner.stream` returns a `RunStream` that
  drives the same run `run` drives — same loop, same guardrails, same record — and yields a
  discriminated `ProgressEvent` union rather than raw text chunks: `RunStarted`,
  `IterationStarted`, `AnswerDelta`, `StructuredDelta`, the tool-call lifecycle,
  `GuardrailDecision`, `ApprovalRequired`, `UsageUpdated` and the three terminal variants,
  with `stream.run` giving the finished record once the stream is drained. Exactly one
  terminal event is emitted and it is last, derived from the finished run, so a connection
  that drops mid-answer fails the run instead of presenting accumulated text as an answer.
  Every event carries its `run_id` and a gapless `sequence` from zero, and `SequenceCheck`
  counts what was lost and rejects a late or duplicate event rather than reordering it into
  place. Tool arguments are scrubbed inside the runtime before emission; the answer itself is
  not, because deltas that no longer reassemble to the answer are a corrupted answer.
  `decode_progress` skips a variant this version has never heard of, so adding one stays a
  minor change, while a known variant that will not parse raises. **Stability:** additive —
  `run` is unchanged and a consumer that only wants the answer keeps calling it. Documented
  in `docs/run-progress.md`, exercised by `examples/run_progress.py`.

- Fallback can no longer trade a data-handling guarantee for an availability one. A chain
  that promotes a hosted vendor because the self-hosted endpoint is down, or the standard
  tier because the sealed one is, has made a decision nobody approved. `TrustBoundary` states
  where a model sits — `tier`, `hosting`, `residency` — and `ModelSpec` carries it; a
  fallback is legal only between models that share one. Routing drops the rest from the chain
  and records them in `RoutingDecision.excluded_by_boundary` and in `rejected` with the axes
  that differ, and a spent chain whose only remaining alternatives are outside the boundary
  fails the run closed with `TrustBoundaryError`, before any request reaches the
  out-of-boundary provider. A boundary nobody declared constrains nothing, so existing
  deployments route unchanged; declaring one on a single model is enough to protect it,
  since an undeclared target is an unknown one and unknown is not equal. `RoutingDecision`
  also records what chose the model — `required`, `min_context_window_tokens`, `boundary` —
  all of it drawn from a closed vocabulary, so a rationale carries no prompt content and can
  be kept for a sealed matter. Documented in `docs/trust-boundary.md`, exercised by
  `examples/trust_boundary.py`.

- An agent is now expressible as the artifact that was reviewed. `AgentDefinition` carries
  the declaration plus the `Owner` who answers for it, the evaluation suite that checks it,
  the prompt entry it was written for, the memory policy it runs under and the schema it
  answers in — none of which can be diffed in review or named by a finished run while they
  live in the call sites that construct the agent.
  `AgentDefinition.declared(..., known_tools=...)` refuses an allowlist naming a tool nobody
  registered, at construction rather than at the first production execution that reaches for
  it, and `Owner` refuses a contact nobody can be paged at. `revision` is a content-derived
  digest, so editing the instructions, the allowlist, the owner or the answer schema produces
  a new revision rather than moving what an old run pointed at; `output_schema` is stored as
  data precisely because `output_type` is a Python class the digest and the store both lose.
  `AgentRunner.run` and `run_sync` accept a definition wherever they accept an agent and pin
  its revision onto `Run.definition_revision` and every span. A run from a bare agent records
  `None`, attributed as `unknown`. Documented in `docs/agent-definition.md`, exercised by
  `examples/agent_definition.py`.

- Prompt-cache hit ratio is a first-class number at every level, because prefix stability is
  unfalsifiable without one: every change to prompt assembly reads as an improvement if
  nothing counts what the server re-evaluated, and on CPU prefill dominates. `Usage` gains
  `fresh_input_tokens` — input with cache reads taken out, never negative, since vendors
  disagree about whether a read sits inside the input count — plus `cache_hit_ratio` and
  `measured`. `Totals` gains `cached_tokens`, `cache_write_tokens`, `hit_ratio` and
  `measured`, so `totals_by` aggregates the cache question along the same dimensions as the
  money; writes total apart from reads because they are priced apart. Nothing read is a
  ratio of zero rather than a division error, and `measured` separates "the cache missed"
  from "nobody sent anything", which a dashboard otherwise draws identically. `record_spend`
  emits `adk.input_tokens` and `adk.cached_tokens` under the existing dimensions — two
  counters rather than one ratio, because a ratio cannot be re-aggregated across series.
  **Stability:** additive. Documented in `docs/cost-attribution.md`, exercised by
  `examples/cache_hit_ratio.py`.

- `ContextWindow` decides what goes into the context and what leaves it when there is no
  room, so a retrieval loop holds the most useful tokens rather than the most recent ones.
  Admission is keyed: a `Segment` whose `key` is already held in any layer is refused, and
  `admit` returns `False` to say so — re-injecting a chunk the model already has is a
  rounding error on a GPU and seconds of prefill per turn on CPU. `key=None` is never
  deduplicated. `fit()` evicts until what is held fits `limit_tokens` and returns what left,
  in the order it left: conversation first, oldest first; then retrieval, lowest-scored
  first. The cacheable prefix — system, tools, pinned — is never evicted even where dropping
  it would free the most tokens fastest, because trimming it invalidates the prefix and every
  downstream turn pays the refill; a prefix that cannot fit alone raises
  `ContextWindowExceededError` instead. Eviction frees the key, so a chunk dropped for room
  is admissible again later. `texts(layer)` hands what survived to `assemble_prompt`.
  **Stability:** additive. Documented in `docs/context.md`, exercised by
  `examples/context_window.py`.

- Agents now run on a machine with no GPU. `LlamaCppProvider` puts `llama-server` behind the
  OpenAI-compatible client, so an agent written against vLLM runs on CPU unchanged — same
  runner, same structured output, same usage accounting. Its timeout is minutes rather than
  seconds, because a first token waits for the weights to load, and every request asks for
  llama.cpp's prompt cache, which the server does not turn on by itself and without which
  every turn re-evaluates the whole prefix. `LlamaCppTuning` describes how the server was
  started — `threads`, `batch_size`, `micro_batch_size`, `context_tokens`, `prompt_cache` —
  renders `server_arguments()` for the launch command, and refuses a batch smaller than its
  own micro-batch; a field left unset renders no flag rather than a guess. `GgufModel`
  answers what a quantized model will need before anything loads it, splitting weights, KV
  cache and buffers, with the per-token KV cost a field because grouped-query attention moves
  it by an order of magnitude between two models of the same size. Given `weights` and
  `available_bytes`, the provider raises `ModelTooLargeError` — a `ConfigurationError` — at
  construction, naming the shortfall and a lighter quantization that would have fitted,
  instead of being OOM-killed mid-run. `quantization_for` picks a format and is never heavier
  than `Q4_K_M`, the published trade-off point, because more bits past it buys little quality
  and costs the memory bandwidth that is the whole budget on CPU.
  **Stability:** additive. Documented in `docs/cpu-inference.md`, exercised by
  `examples/cpu_inference.py`.

- A per-tenant ceiling now holds across replicas. `SpendLedger` is the protocol —
  `reserve` / `settle` / `release` / `record_progress` / `read_window` / `reconcile` /
  `forget` — with `InMemoryLedger` for one process, `RedisLedger` and `PostgresLedger` for a
  shared one, and `CoalescingLedger` in front of any of them for deployments that cannot
  afford a round trip per model call. A reservation counts against the ceiling before it
  settles, so eight replicas reserving against the same empty window cannot all be told yes.
  `Window` is rolling or calendar; time inside a ledger is monotonic, so a clock corrected
  backwards cannot open a second allowance, and a run crossing a boundary keeps the hold it
  took from the old window. Every hold carries a lease: `reconcile` settles a lapsed one
  against whatever progress it admitted and releases one that admitted none, so a dead
  replica neither keeps a tenant's allowance nor is credited with spend that happened.
  Everything fails closed with `BudgetUnavailableError`; degraded mode is off, configured in
  advance rather than inferred from a failure, and recorded on every hold it waves through.
  `LedgerKey` carries identifiers and amounts only, rejects names containing the key
  separator, and `forget(tenant)` reduces a tenant to a non-identifying aggregate. Shared
  stores translate one operation into one Lua script or one CTE statement, because a ceiling
  check and the write it authorises cannot be two round trips.
  `SpendLedgerConformance` holds a store you write yourself to the same behaviour.

- A run can be costed before it starts. `estimate_run` returns a `CostEstimate` — a point, a
  tenth- and ninetieth-percentile case, the token counts behind them, the `Assumptions` they
  rest on and a `Confidence` saying whether those assumptions came from this agent version's
  own finished runs, another version's, or the kit's defaults. The provider is asked only to
  count tokens, never to complete anything. A model nothing prices raises
  `EstimateUnavailableError` rather than a plausible-looking figure, unless `allow_unknown`
  asks for the token counts with the money marked unknown. Prompt growth across turns and the
  prompt cache are both modelled rather than assumed away. `affordable` and
  `refuse_unaffordable` check the high case against what is left of a budget, before anything
  is spent; `as_limits(headroom=…)` is the explicit conversion into `BudgetLimits`, with
  headroom on the money and the shape ceilings taken from the high case. `approval_for` puts
  the range and its confidence to a person rather than one number, `calibrate` holds the
  estimate against what the run actually cost without clamping the outliers, and
  `with_children` totals a multi-agent run and takes the weakest confidence in it. `RunHistory`
  is the protocol over runs a deployment already stores (`InMemoryHistory` ships for tests),
  and `Pricer` — with `models.pricing.pricing_at` as the shipped adapter — keeps the runtime
  free of any opinion about where prices live.
- A finished run can say who spent what. `spend_of(run)` returns one `SpendRecord` per metered
  step carrying the tenant, user, agent, agent version, model, prompt version, task class and
  run id that were true when the money went out — derived from the run, never supplied by the
  caller. A run that fell back bills each step against the model that answered it; an attempt
  that failed after burning tokens is a record rather than a gap; a run acting on another
  tenant's request bills the tenant it ran as; and what the run could not say resolves to an
  explicit `unknown` named by `Attribution.unknowns`. `totals_by` groups by any fields of
  `Attribution`, keyed by a tuple in the order asked for, refusing an unknown dimension by name
  and refusing to sum two currencies; `Totals.estimated` separates spend that will appear on a
  vendor invoice from spend that was counted rather than metered. `record_spend` reads a
  finished run rather than hooking the loop, emits full-identity spans under one `adk.` prefix,
  and emits the `adk.cost`, `adk.tokens` and `adk.calls` counters whatever the trace did,
  because a cost total taken from sampled spans looks precise and is wrong; `Dimensions` keeps
  metric cardinality bounded with an allow-list, bucketing the rest under `other` without ever
  dropping the money. Consumer attributes are pattern-scrubbed and the dropped keys named on an
  `adk.redacted` event, while `adk.` attributes pass through as structural identity. `Run` gains
  `prompt_version`, `task_class` and `depth`; `tesserix_adk.testing` gains `FakeMeter` and
  `MetricPoint`. **Stability:** additive. Documented in `docs/cost-attribution.md`, exercised by
  `examples/cost_attribution.py`.

- The ceiling is enforced where the spend happens. The run loop reserves before a model call,
  settles against what came back, charges every tool call before dispatch and re-checks every
  dimension at the top of each iteration, so a ceiling reached on the fortieth turn ends the run
  in `BUDGET_EXHAUSTED` with `run.output` empty and the work that did happen still on the run.
  Nothing is truncated, dropped or downgraded to squeeze a call under a limit. A failed attempt
  is charged, so retries and fallback cannot spend past a ceiling; a cancelled call settles what
  it had sent and still ends `CANCELLED`; and a non-idempotent tool that ran on a run that did
  not complete gets a `COMPENSATION_REQUIRED` event rather than being re-dispatched while
  unwinding. `BudgetDecision.overshoot` says how far past a ceiling a call landed and
  `as_error` turns a decision into its typed refusal. `budgeted_stream` holds a stream to the
  same ceiling, charging each reported total as an increment and ending the stream with
  `BudgetExceededError` rather than letting it trail off. New `RunEventKind` members:
  `BUDGET_EXCEEDED`, `COMPENSATION_REQUIRED`, `FAN_OUT_REFUSED`. **Stability:** additive.
  Documented in `docs/budget.md`, exercised by `examples/budget_enforcement.py`.

- Prices are dated data rather than a constant. `PriceCard` carries the day a rate took
  effect, the request shape it answers for and a `Rate`; `PriceList.rate_for` picks the
  narrowest card the request clears — batch tier, then the highest long-context threshold —
  and among those the latest already in force. A price change is a new card and never an edit,
  because overwriting one rewrites what last week's runs cost, and two cards for one shape on
  one day is refused. `price_list()` reads a TOML file named by `ADK_PRICE_LIST` or by path and
  never by convention, `overridden_by` lays negotiated rates over the shipped ones, and a file
  that will not parse is a `ConfigurationError` rather than a quiet fall back to list price. A
  model no card covers warns `UnknownPricing` and reports `Cost.unknown()` — zero components at
  `UNKNOWN` confidence, never a silent free call. Tokens burned on failed and retried attempts
  are on the ledger too, carried on each `ATTEMPT_FAILED` event and summed into the run, so a
  run that never got an answer still says what it spent. **Stability:** additive. Documented in
  `docs/cost.md`, exercised by `examples/cost.py` and `tests/test_cost.py`.

- A run survives a vendor that will not answer it. `FallbackChain` is the eligible
  candidates of the routing rule that already matched, chosen one first, so the fallback
  order and the routing order cannot drift and every link has already passed the run's
  capability floor. The chain moves only after that vendor's own retries are spent and only
  for failures another vendor could answer differently — rate limits, overloads and timeouts
  — while a bad key, an invalid request, a filtered prompt, a capability mismatch, an
  exhausted budget and anything unmapped stay terminal. Falling back after a tool has run
  replays the recorded result rather than re-invoking it, and a tool not listed in
  `Agent.idempotent_tools` blocks the fallback with `FallbackUnsafeError` instead. Every
  attempt is on the run, the run names the model that actually answered, failed attempts
  still charge the budget, and `FallbackExhaustedError` names every candidate that refused
  rather than only the last. See `docs/fallback.md`.
- An agent can name the job instead of the model. `Agent.task_class` names the kind of
  work — `CHEAP`, `SMART`, `REASONING`, or any `TaskClass` a deployment invents — and
  `Agent.requires` names what it needs of whatever answers, so retuning a deployment is an
  edit to a file rather than a code change in every consumer that wrote a model id at a
  call site. `RoutingTable` holds the rules (`task_class`, optional `tenant` and `agent`,
  candidates in preference order), `routing_table()` reads it from TOML at a given path or
  at `ADK_ROUTING_TABLE`, and `TableRouter` resolves against it with optional per-tenant
  `entitlements`. The narrowest rule wins; within it the first candidate meeting the
  requirements answers. A table that would fail on a later request fails at construction
  instead — no rules, two rules at one scope, a candidate declaring no capabilities, a
  candidate the catalogue no longer lists. Nothing falls back: an unrouted class, a rule
  nothing eligible answers and an unknown pin all raise `NoEligibleModelError`, carrying
  the unsatisfied requirements and every rejected candidate with its reason. `AgentRunner`
  takes `router` and a `providers` map, resolves once before the first call, records a
  `MODEL_ROUTED` event, and `reload()` applies a new table to the next run. Routing is
  opt-in — an agent naming `model` outright keeps the runner's single provider unchanged.
  See `docs/routing.md` and `examples/routing.py`.
- One error taxonomy over every vendor, and the two things now done in front of a call
  rather than after it. `RateLimitError`, `AuthenticationError`, `ContentFilteredError`
  and `InvalidRequestError` join the existing provider errors, and each adapter classifies
  a failure into them from the vendor's own code and status: a rate limit is
  `rate_limit_error` at Anthropic, `rate_limit_exceeded` at OpenAI and `RESOURCE_EXHAUSTED`
  at Google, and `RetryPlan` now reads the type rather than the body. A code nobody has
  mapped becomes a plain `ProviderError` whose retryability follows its status. A spent
  quota is `RateLimitError(quota=True)` and is **not** retried — a rate clears by waiting,
  an allowance clears when somebody pays. Every failure carries `provider`, `model`,
  `request_id`, `status`, `retry_after` and the vendor's `details["code"]`; it no longer
  carries the vendor's free-text message, because a 400 body quotes the request that caused
  it and the request body is the prompt. `redact_vendor_messages=False` restores it for an
  operator already entitled to read those prompts. Connecting and generating get separate
  budgets, documented as `PhaseTimeouts` / `PHASE_DEFAULTS` and settable per provider with
  `timeout` and `connect_timeout`, with whichever wait ran out named as `details["phase"]`.
  `RateLimiter` meters requests and tokens across every provider sharing one key, so
  concurrent runs space their calls instead of each believing they have the whole
  allowance. Additive: every new error subclasses `ProviderError` and every new argument
  defaults to today's behaviour, except the redaction, which is the one intended change.
  See `docs/resilience.md`.
- `OpenAICompatibleProvider`, so an endpoint you run yourself — vLLM, Ollama, TGI — is
  routable, costed and capability-checked like any vendor, with presets `VLLM`, `OLLAMA`
  and `TGI` carrying each server's deviations from the format it claims to speak. Two
  arguments have no default: `base_url`, because there is no host to guess for a service
  only the operator has named, and `capabilities`, because the deployment's flags decide
  them and no endpoint reports them honestly. The provider is named for the server rather
  than for OpenAI, so a call to a box in the cluster is not recorded against the vendor's
  bill, and `api_key_variable` is optional — omit it and no `Authorization` header is sent,
  which is the in-cluster case. Four deviations are reconciled rather than passed on,
  because each is a wrong answer instead of an error: an error object under a 200 raises
  `ProviderError` rather than being assembled into a response; a stop reason the server
  omitted is read off the answer, since `unknown` on a turn that asked for a tool ends a
  run with the call never made; tool calls arriving without ids are given positional ones,
  so a result has something to match back to and a recorded run stays replayable; and usage
  nobody reported is estimated from the provider's own token count and marked
  `Usage.estimated`, because zero tokens reads as a free call and a call on a GPU somebody
  pays for is not free — the cost stays `None`, the kit not knowing what that GPU hour is
  worth. `ProviderUnavailableError` is new, raised for a connection that never landed and
  for 502/503/504 by every HTTP provider, always `retryable`, believing any `Retry-After`
  the endpoint sent in preference to a computed backoff, since retrying a model still
  loading as fast as the policy allows is how it never finishes loading. And nothing is
  emulated unless asked for: an agent with an `output_type` against an endpoint that has
  not declared `structured_output` now raises `CapabilityError` before the run starts,
  rather than prompting a small model for JSON and calling the result a schema. Providers
  say which they are through the new `DeclaresEmulation` protocol. **Stability:** additive
  — `Usage.estimated` defaults to `False`, `ProviderUnavailableError` subclasses
  `ProviderError`, and `DeclaresEmulation` is opt-in, so no existing provider or consumer
  changes behaviour. Documented in `docs/providers.md`, exercised by
  `examples/self_hosted_provider.py`.

- Providers for Anthropic, OpenAI and Gemini, behind the one protocol. `AnthropicProvider`,
  `OpenAIProvider` and `GeminiProvider` take a model id as the vendor spells it, read their
  capabilities and prices from the model catalogue, and resolve their key on every call
  rather than at construction, so a rotated secret is picked up without a restart; the same
  agent definition runs on all three with only the model reference changed. They speak HTTP
  directly rather than through vendor SDKs, so each adapter is one request shape and one
  response shape instead of a second dependency graph and a second translation. The
  differences stay inside them: system prompt placement, structured output through a forced
  tool or `response_format` or `responseSchema` and never through parsing prose, tool
  results merged into one turn or sent one turn each or matched back by name, the tool-call
  ids Gemini does not send and the adapter mints, and the `STOP` Gemini reports whether or
  not it asked for a tool. Streaming is one event model across the three, terminating in a
  `StreamEnd` that carries the settled response; a stream that ends early raises
  `StreamInterruptedError` with the partial text rather than returning a fragment as a whole
  answer. `HttpCassette`, `HttpExchange`, `HttpReplay` and `FakeSecrets` in
  `tesserix_adk.testing` record and serve traffic at the HTTP layer, so the whole matrix
  runs in CI with no network and no keys, and `replay.sent` asserts what the adapter put on
  the wire — which a provider-level recording cannot see. **Stability:** additive; nothing
  existing changes. Documented in `docs/providers.md`, exercised by
  `examples/vendor_providers.py`.

- A conformance gate for the typing guarantee. Every `# type: ignore` in the source and
  every `Any` in an exported signature is declared in the new `typing-policy.toml` with a
  reason, an owner and a review date; `make typing-gate` fails on an escape the policy does
  not list *and* on an entry the code no longer contains, since a record that outlives its
  code is how an inventory stops describing anything. An entry whose owner the policy does
  not recognise is flagged for reassignment rather than silently inherited, and one past its
  review date fails — an exception nobody revisits is a permanent one. An `Any` entry names
  its kind: `json`, `variadic`, or `provisional`, which must name the issue that removes it.
  Every exported callable is checked for an unannotated parameter or return. At the
  third-party boundary `disallow_any_unimported` is on, so a dependency that ships no stubs
  fails at the import rather than widening a public signature to `Any`, and the two settings
  that would readmit an SDK's `Any` wholesale are forbidden by test. The checker is pinned
  in both the dev group and the policy, asserted equal. The gate runs in CI, in `make check`
  and as a pre-commit hook. **Stability:** additive — a repository gate with no runtime
  effect and no change to the public surface. Documented in `docs/typing.md`.

- Bounded validation repair, with the failure itself fed back. `core` gains `RepairConfig`
  and `Agent` gains `repair`, undeclared by default: an answer that fails validation stays
  terminal unless a budget was asked for, because a further attempt is a further charge on
  someone's account. Where one is declared, the loop sends the violation back through the
  new `OutputContract.repair_prompt` — every failing dotted path with what was wrong with
  it, plus the schema, and nothing else. No value is supplied for a failing field, no
  default is filled, no field is dropped and nothing is cast: a prompt that says what the
  answer should be is coercion with extra steps. A repair attempt is an ordinary model
  call, so its tokens land on `run.usage`, it is recorded against the budget policy and it
  is bounded by the run deadline and the iteration cap — repair can never spend past a
  ceiling. Attempts are recorded as the new `repair_requested` event naming the type, the
  failing fields and which attempt of how many, so repair rate is measurable per agent and
  prompt version. An answer that comes back with the identical failure after being told
  what it was stops the run with the new `repair_abandoned` event and a configuration
  error, since a constraint nothing can satisfy is a defect in the declaration rather than
  a budget to spend proving it. Running out fails the run carrying the last violation,
  never a best-effort object. **Stability:** additive — `Agent.repair` defaults to `None`,
  which is what every existing agent already did; `RunEventKind` gains two members.
  Documented in `docs/repair.md` and exercised by `examples/repair.py`.

- Structured output by default. An `Agent` declares exactly one of `output_type` and the
  new `free_text`; declaring neither is refused where the agent is built, so an answer
  whose shape nobody declared is a configuration error rather than a string the caller
  parses by guessing. Where a type is declared, the runtime derives its schema through
  `schema_for` in the closed dialect and sends it as `ModelRequest.output_schema` beside
  the new `output_schema_hash`, both folded into `Prompt.version` — a changed answer type
  is a changed prompt, so it neither reuses a cached prefix nor replays a cassette recorded
  against the old shape. A provider exposing a truthy `supports_structured_output` enforces
  the schema itself; one that does not is given the schema in the prompt and its answer is
  validated identically, because an undeclared capability is treated as absent. Validation
  happens before the run can reach `completed`: an enclosing code fence is stripped
  explicitly and recorded as the new `output_unwrapped` event, prose around JSON is never
  scraped, and truncation mid-object is a violation rather than something to repair by
  guessing closing braces. `runtime` gains `OutputContract` and `unwrap_fenced`; a
  violation raises `SchemaViolationError` carrying the raw output, every failing dotted
  path, the refusing type and the schema hash, which the loop records before ending the run
  `failed`. Content echoed into the next turn of a structured run is wrapped as untrusted
  data, so an instruction that arrived inside a field cannot become the next turn's prompt.
  **Stability:** breaking — an agent that declared neither now needs `free_text=True`, and
  `assemble_prompt` gains an `output` keyword. Documented in `docs/structured-output.md`
  and exercised by `examples/structured_output.py`.

- JSON Schema derived from the Python type, so the shape the model is told and the shape
  the code parses cannot drift apart. `core` gains `schema_for`, which accepts a pydantic
  model, a dataclass, a `TypedDict` or an annotated callable and returns normalised Draft
  2020-12 — titles dropped, keys ordered, `required` sorted — with descriptions read from
  `Field(description=...)` or the Google-style `Args:` block of the docstring, so the
  guidance the model reads is written once, where the field is declared. A missing or
  malformed docstring costs descriptions and nothing else. `schema_hash` digests the
  result: key order does not change it and any change of shape does, so a renamed field
  misses a cassette recorded against the old one instead of replaying an answer for a type
  that no longer exists. Provider differences sit behind the `SchemaDialect` protocol —
  `JSON_SCHEMA`, `STRICT_SUBSET` and `INLINE_REFS` ship with the kit, and any value with a
  `name`, a `forbidden` keyword set and an `adapt` is a dialect. Nothing is downgraded
  silently: a forbidden keyword, or a recursive type under an inlining dialect, raises
  `CapabilityError` naming the dialect. The new `SchemaGenerationError` is raised where the
  type is declared for anything that cannot be described faithfully — an unannotated or
  variadic parameter, `Any` in a required position at any depth, a type pydantic cannot
  render, or a schema past `max_bytes`, refused whole rather than truncated into a
  different type. **Stability:** additive. Documented in `docs/schemas.md` and exercised by
  `examples/schemas.py`.

- Strict validation at every boundary. `core` gains `AdkModel` — frozen, `strict=True`,
  `extra="forbid"` — and every model in the kit derives from it, so a misspelt field is an
  error rather than a passenger, `"12"` is never quietly read as `12`, and a validated
  record cannot be edited by the layer that reads it. Alongside it: `validated`, which
  normalises pydantic's error into `SchemaViolationError` carrying the model, every failing
  path at once (`content.0.binary.media_type` — the list index and the union member are
  both part of the location), the reason per path and the raw payload; `Sensitive`, a
  marker carried in `Annotated[...]` metadata; `telemetry_dump`, which drops sensitive
  fields and masks `SecretStr` while `model_dump_json` keeps both, because a run rehydrated
  without its credentials rehydrates broken; and `parsed_from_strings`, the one deliberate
  exception to strictness, for the environment, where every value is a string whatever it
  means. `BinaryPart.data` is now marked `Sensitive` — a scanned exhibit is evidence in one
  system and a retention problem in another. Extras stay possible where they are declared
  (`Usage.extras`, `Message.metadata`) and are refused where they are loose, so forbidding
  them never blocks a provider from evolving. Aliases stay forbidden outright, with a test
  that fails if one appears. **Stability:** additive for anything constructing models
  correctly; a payload that relied on coercion or on an ignored unknown field now raises
  `SchemaViolationError`, which is an existing defect surfacing rather than a new one.
  Documented in `docs/models.md` and exercised by `examples/models.py`.

- Determinism and offline replay against a recorded provider. `core` gains the `IdFactory`
  protocol and `AgentRunner` takes `ids`, so the last ambient source in the loop joins the
  clock and the jitter as something a test injects — a `uuid4` in the loop is a field no
  assertion can name. `runtime` gains `RunFingerprint`, `fingerprint_of` and
  `canonical_digest`: a canonical summary of the assembled prompt, the tool schemas the
  model was told about, the model, the output schema and the hook chain, normalised so that
  dict order is not a difference while list order still is. `testing` gains `Cassette`,
  `Interaction`, `RecordedError`, `RecordingProvider`, `ReplayingProvider`, `SequentialIds`,
  `assert_same_run`, `redacted` and the `CassetteMissError` / `CassetteVersionError` types.
  A replay serves the exchange that was recorded or fails naming the field that diverged;
  there is no live provider behind it to fall through to, because quietly reusing the
  nearest response is a green test asserting nothing. Recorded failures replay with their
  retries, so the recovery path is exercised rather than assumed. A cassette keeps digests
  of the request and never its content, redacts credential-shaped keys and values before
  anything is written, and is refused when it was recorded against a different provider,
  version or format — replaying across an SDK upgrade on trust proves nothing about the
  code now shipping. `assert_same_run` normalises timings but not sequence, which is also
  how a hook that reads the wall clock is caught. **Stability:** additive — every name is
  new and `ids` is keyword-only with a default that preserves random ids in production.
  Documented in `docs/determinism.md` and exercised by `examples/determinism.py`.

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

### Fixed

- A model step whose provider reported no usage is recorded as `Cost.unknown()` rather than
  a counted zero. A call at a price nobody knows is not a call that was free, and recording
  it as free is a false statement that totals up into a bill; the group it lands in now
  reads `UNKNOWN` confidence instead of quietly understating. A tool step has no price
  rather than an unknown one and stays `Cost.nothing()`.
  **Stability:** `Totals.cost.confidence` changes for groups containing an unpriced model
  step. The amount is unchanged; the total stops claiming to be counted.

### Changed

- `RoutingDecision.explain()` names the capability and context floors the work asked for and
  how many candidates the trust boundary excluded. **Stability:** breaking for a test
  asserting the exact line; the fields it reads are additive.

- Typed run results. `Agent[TripPlan]` runs to a `Run[TripPlan]`, and `run.output` is a
  `TripPlan` rather than a dict. `Agent` and `Run` take one type parameter bound to
  `BaseModel`, `AgentRunner.run` and `run_sync` carry it through, and `Run.with_output`
  takes the instance. The parameter defaults to the new `NoOutput`, a model with no
  fields, so an agent declaring `free_text=True` needs no annotation and every existing
  bare `Agent` or `Run` annotation still reads unchanged. A run built by the loop is
  parameterised at runtime as well, because a bare `Run` serialises a typed answer away to
  `{}` and the checkpoint of a typed run would otherwise lose it; rehydration names the
  type, and an unparameterised `Run.model_validate_json` is refused rather than dropping
  the answer, since nothing on the wire says which type it was. **Stability:** breaking —
  a caller reading `run.output["nights"]` reads `run.output.nights` instead. The
  serialised form of a run is unchanged, and the parameter's default keeps every
  unannotated use valid. Documented in `docs/typing.md` and exercised by
  `examples/typed_results.py`.

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
