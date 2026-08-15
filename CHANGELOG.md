# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

## 0.15.0

### Added

- **tesserix_adk.memory.prefix**: `tesserix_adk.memory` now draws the line compression may not cross. Compression and prefix
caching pull in opposite directions and the conflict is silent: recompressing the whole
context each turn produces slightly different bytes each time, the provider's cached prefix
is invalidated, and the next call pays a full prefill. On CPU inference that trade is
catastrophic — the compression saves perhaps a third of the tokens and the cache miss costs
tens of seconds of prefill on all of them.

`FrozenPrefix` holds one context's boundary. `advance` freezes what a completed turn sent,
monotonically and idempotently, so two concurrent turns cannot move it twice. `live` and
`compressible` answer what may be touched: content that arrived since the boundary and is
not already marked by `compressed`, because content is compressed once on arrival and then
becomes frozen history itself.

`verify` is what makes the rule enforceable. It fails with `PrefixDriftError` naming the
layer that moved and where, so a rewritten system prompt or a reordered retrieval block
stops a build rather than reaching production as a doubling of prefill latency nobody
attributes to anything for a week.

Giving the prefix up is a recorded event, not a quiet one: `reset` takes the reason —
an eviction, a redeployed instruction block — and counts it, because each reset is a full
prefill somebody paid for. `PrefixBoundary.attributes` reports the position and the reset
count as span or metric attributes, worth watching beside the cache hit ratio they exist to
protect. Positions and counts only; no prompt content reaches a metric.

The boundary is per context and holds nothing global, so a position never crosses a tenant.

### Public API surface

- Added: `tesserix_adk.core.PrefixDriftError`
- Added: `tesserix_adk.core.errors.PrefixDriftError`
- Added: `tesserix_adk.memory.COMPRESSED`
- Added: `tesserix_adk.memory.FrozenPrefix`
- Added: `tesserix_adk.memory.POSITION_METRIC`
- Added: `tesserix_adk.memory.PrefixBoundary`
- Added: `tesserix_adk.memory.RESETS_METRIC`
- Added: `tesserix_adk.memory.SECTION`
- Added: `tesserix_adk.memory.compressed`
- Added: `tesserix_adk.memory.prefix.COMPRESSED`
- Added: `tesserix_adk.memory.prefix.FrozenPrefix`
- Added: `tesserix_adk.memory.prefix.POSITION_METRIC`
- Added: `tesserix_adk.memory.prefix.PrefixBoundary`
- Added: `tesserix_adk.memory.prefix.RESETS_METRIC`
- Added: `tesserix_adk.memory.prefix.SECTION`
- Added: `tesserix_adk.memory.prefix.compressed`

## 0.14.0

### Added

- **tesserix_adk.memory.reversible**: `tesserix_adk.memory` now makes compression reversible. Every lossy scheme is a bet that the
model will not need the part that was removed, and the bet is occasionally catastrophically
wrong: the elided log line was the one naming the failing host, and the agent then answers
confidently from a hole in its context without knowing the hole is there.

`ReversibleRouter` wraps a `ContentRouter`, retains the original for the run that admitted
it, and appends a note to the compressed form naming the size of what was removed and the
tool that reads it back. `expand_content_tool` is that tool, resolving a handle within the
caller's `ToolContext`. An irreversible quality risk becomes a recoverable extra turn, which
is what makes an aggressive ratio defensible — and it makes the accuracy question testable,
because whether the model retrieves when it should is a behaviour, while whether a summary
kept the right sentence is not.

The handle is the claim check the kit already issues for oversized tool results, so its
scope rules come with it: the tenant and the run are hashed into the handle as well as
checked at the lookup. A handle cannot be transplanted between runs, derived for another
tenant, or read past its retention window, and every one of those failures is the same
`ClaimUnavailableError` — telling a model which condition failed tells it what handles other
runs hold, and it would do nothing different either way.

Retrieval is itself subject to admission: an original larger than the budget left comes back
as the leading window that fits, marked `truncated`, rather than putting back into the
prompt exactly what compression took out. Every expansion is audited with the handle, the
requester and the resulting size; refusals are audited too, and the content never reaches
the record. `Compressed` gains `handle`, and `ContentRouter` exposes the `count` it sizes
with so a caller can agree with it.

### Public API surface

- Added: `tesserix_adk.memory.EXPAND_TOOL`
- Added: `tesserix_adk.memory.Expansion`
- Added: `tesserix_adk.memory.ReversibleRouter`
- Added: `tesserix_adk.memory.reversible.EXPAND_TOOL`
- Added: `tesserix_adk.memory.reversible.Expansion`
- Added: `tesserix_adk.memory.reversible.ReversibleRouter`
- Added: `tesserix_adk.tools.ContentExpander`
- Added: `tesserix_adk.tools.Expanded`
- Added: `tesserix_adk.tools.expand_content_tool`
- Added: `tesserix_adk.tools.expansion.ContentExpander`
- Added: `tesserix_adk.tools.expansion.Expanded`
- Added: `tesserix_adk.tools.expansion.expand_content_tool`

## 0.13.0

### Added

- **tesserix_adk.memory.compression**: `tesserix_adk.memory` now compresses admitted content by what it is rather than by how long
it is. Tool output is the largest and least dense thing in an agent prompt, and the prompt
is rebuilt every turn, so a five-hundred-row result is paid for on every turn that follows
it. Most of what it costs is repeated keys, alignment whitespace and identical rows — none
of which the model was reading.

`ContentRouter.admit` classifies the content and dispatches it to a `Compressor` that
understands it. `StructuredCompressor` factors a homogeneous JSON array into a field list
and value rows, hoists fields identical in every row into `shared`, and folds consecutive
identical rows with a repeat count: a re-encoding rather than a summary, so every distinct
field name and every distinct value is still there to write the next query from.
`TabularCompressor` takes the terminal padding out of aligned columns. `CodeCompressor`
works from the syntax tree — a regex over source finds `def` inside a string literal — and
keeps signatures, annotations and every `raise` while eliding the bodies nobody reads.
`ProseCompressor` drops repeated sentences and trims at a sentence boundary. `PassThrough`
is the terminal fallback.

The fallback is the point of the design. Content that cannot be classified with confidence
is admitted untouched with the reason recorded, because a wrong compressor silently
destroying content is worse than paying full price for it: anything that half-looks like a
type it does not parse as is `UNKNOWN`. A compressor that raises falls back rather than
propagating, a compressor that expands its input is discarded and the original admitted, and
content below `threshold_tokens` never reaches the router at all.

Everything here is pure and offline: no network call, no model call, no state, and the same
bytes in produce the same bytes out — a compressor that varied its output would move the
prefix the provider is caching on. Classification parses content and never executes it or
follows a reference inside it, and `Compressed.untrusted` carries the trust of what came in
unchanged. Compression is not sanitisation.

### Public API surface

- Added: `tesserix_adk.memory.CodeCompressor`
- Added: `tesserix_adk.memory.Compressed`
- Added: `tesserix_adk.memory.Compressor`
- Added: `tesserix_adk.memory.ContentKind`
- Added: `tesserix_adk.memory.ContentRouter`
- Added: `tesserix_adk.memory.DEFAULT_COMPRESSORS`
- Added: `tesserix_adk.memory.PassThrough`
- Added: `tesserix_adk.memory.ProseCompressor`
- Added: `tesserix_adk.memory.SizeCounter`
- Added: `tesserix_adk.memory.StructuredCompressor`
- Added: `tesserix_adk.memory.TabularCompressor`
- Added: `tesserix_adk.memory.classify`
- Added: `tesserix_adk.memory.compression.CodeCompressor`
- Added: `tesserix_adk.memory.compression.Compressed`
- Added: `tesserix_adk.memory.compression.Compressor`
- Added: `tesserix_adk.memory.compression.ContentKind`
- Added: `tesserix_adk.memory.compression.ContentRouter`
- Added: `tesserix_adk.memory.compression.PassThrough`
- Added: `tesserix_adk.memory.compression.ProseCompressor`
- Added: `tesserix_adk.memory.compression.SizeCounter`
- Added: `tesserix_adk.memory.compression.StructuredCompressor`
- Added: `tesserix_adk.memory.compression.TabularCompressor`
- Added: `tesserix_adk.memory.compression.classify`
- Added: `tesserix_adk.memory.compression.estimate_tokens`
- Added: `tesserix_adk.memory.estimate_tokens`

## 0.12.0

### Added

- **tesserix_adk.memory.admission**: `tesserix_adk.memory` now decides what is allowed to become a durable fact, and records
where everything durable came from. Prompt injection affects one turn; the same instruction
written into long-term memory re-influences every future run for that user, and arrives on
later turns wearing the costume of a trusted internal fact. The kit already redacted on
write and erased on request, but nothing decided whether a thing should have been stored at
all.

`WriteGate.admit` judges a candidate against an `AdmissionPolicy` before the write. It never
returns the record it was given: what comes back carries its `Provenance` — origin, run,
turn, source, citations, and the policy that let it in — and whatever confidence the policy
is willing to believe it at. A refusal raises `MemoryAdmissionError` after the audit event
is written, so the refusal survives a caller that swallows the exception. The event names
the source and digests the value; an audit trail that quotes the poisoned fact has copied
the poisoned fact somewhere else.

The defaults are the ones worth arguing with. Retrieved content does not persist, because
it is data about a corpus rather than a fact about a user, and that distance is exactly the
one an injection has to travel. Tool output persists only with a citation. Nothing a model
inferred is believed as far as something a person said — an inference is capped at
`inferred_ceiling` and never silently promoted to an assertion, since a store that records
"the customer approves all refunds" at confidence 1.0 whether the model concluded it or the
customer said it has destroyed the signal that could tell them apart later.
Instruction-shaped content is refused whatever its origin.

`WriteGate.recall` is the boundary the write gate cannot cover. It re-judges stored records
against the policy in force *now* and re-runs the guardrail on the content. A fact persisted
before any policy existed carries no provenance, so it is `UNPROVEN`, so it does not come
back; one admitted under a policy since tightened is re-judged the same way. A record does
not become trustworthy by surviving.

`MemoryRecord` gains an optional `provenance`. `MemoryGuard` is the narrow part of a guard a
recall needs, so any `guardrails.Guard` satisfies it without memory importing that package.

### Public API surface

- Added: `tesserix_adk.core.MemoryAdmissionError`
- Added: `tesserix_adk.memory.Admission`
- Added: `tesserix_adk.memory.AdmissionPolicy`
- Added: `tesserix_adk.memory.INSTRUCTION_SIGNATURES`
- Added: `tesserix_adk.memory.MemoryGuard`
- Added: `tesserix_adk.memory.Origin`
- Added: `tesserix_adk.memory.Provenance`
- Added: `tesserix_adk.memory.Recall`
- Added: `tesserix_adk.memory.Verdict`
- Added: `tesserix_adk.memory.WriteGate`
- Added: `tesserix_adk.memory.admission.Admission`
- Added: `tesserix_adk.memory.admission.AdmissionPolicy`
- Added: `tesserix_adk.memory.admission.INSTRUCTION_SIGNATURES`
- Added: `tesserix_adk.memory.admission.MemoryGuard`
- Added: `tesserix_adk.memory.admission.Recall`
- Added: `tesserix_adk.memory.admission.Verdict`
- Added: `tesserix_adk.memory.admission.WriteGate`
- Added: `tesserix_adk.memory.admission.instruction_shaped`
- Added: `tesserix_adk.memory.instruction_shaped`
- Added: `tesserix_adk.memory.provenance.Origin`
- Added: `tesserix_adk.memory.provenance.Provenance`

## 0.11.0

### Added

- **tesserix_adk.tools.intake**: `tesserix_adk.tools` now reads scanned documents and recordings into text an agent can cite.
Intake is the same job in every product that has it, and rebuilding it per product means
each rebuild leaves out provenance: a page of OCR without a page number cannot be traced
back to, and a transcript without timestamps cannot be checked.

PaddleOCR and whisper.cpp are the CPU paths worth using and neither is installed here, for
the reason `models.onnx` gives about `onnxruntime`. The kit takes `OcrBackend` and
`TranscriptionBackend` protocols instead.

`ocr_document` yields an `OcrPage` at a time, so a hundred-page contract costs one page of
memory and a caller that wanted page three can stop after it. Each `Region` carries the box
it was read from as fractions of the page — a reference that survives a re-render at another
resolution — with the backend's confidence and a `RegionKind` from the layout pass, because
an answer built from a footer and one built from a clause are not equally trustworthy.
`OcrPage.scripts` names every writing system on the page, so a mixed-script document
announces itself. `transcribe_audio` is the same shape for audio, with `Segment.reference()`
giving the span it was said in and the speaker label where the backend supplied one.

The new `MediaIntakeError` says which check failed: `unsupported` (refused before the file
is opened), `missing`, `empty` or `corrupt`. `empty` covers a backend that produced nothing
at all, which is the case that matters — returning `""` for a document that could not be
read is indistinguishable from a blank page, and it ends with an agent answering
confidently from nothing. A backend raising `MediaIntakeError` itself is passed through
untouched.

`ocr_tool` and `transcribe_tool` are the tools a model calls. A model names a file; what
that is allowed to mean is decided by `root`, and a name resolving outside it is a
`ToolRefusal`. Output is windowed at `max_chars`, since a tool returning a whole contract
would put the document back into the context one call later, and every page and span is
prefixed with its reference so what the model quotes can be cited.
- **tesserix_adk.observability.latency**: `tesserix_adk.observability` now records the latency numbers that decide whether CPU
inference is usable. "Kit overhead under twenty milliseconds" is easy to state and says
almost nothing against a model call of five to thirty seconds; time to first token, the
sustained rate after it, and the share of the prompt that did not have to be prefilled are
what a user actually feels.

`RunTimer` times one run — `first_token()` is safe to call on every event, only the first
counts — and `LatencyReport` carries what it cost. The sustained rate excludes the wait for
the first token, because a slow prefill averaged into the decode rate flatters a run that
felt slow. Cold and warm, streamed and blocking, are dimensions on every metric and
separate scenarios in the benchmark suite: averaged together the cold number looks fine and
the warm one looks bad.

`CacheHits` tracks the prompt cache beside the latency rather than in another dashboard,
since prefill is where CPU latency goes. A provider that reports no cached-token count
leaves the ratio `None`. It is not zero and it is not one — both are answers, and an
unknown that reads as an answer is how a cache that quietly stopped working goes unnoticed
for a quarter.

Every number is emitted through `Meter` and not only attached to the span, for the reason
`observability.metrics` gives: traces are sampled, and a percentile computed from whatever
the sampler kept is precise-looking and wrong. Span attributes stay durations and counts,
never content.

The benchmark suite gains `first-token-cold`, `first-token-warm` and `sustained-stream`,
and the harness gains the three metrics with thresholds and noise floors. A shared runner
cannot measure CPU inference reproducibly, so model time is modelled from a declared
profile for the documented target CPU and the uncached token count is measured — that is
the part a change moves. A change that puts something volatile near the front of the prompt
now fails the gate naming both the first-token latency and the hit ratio.

### Public API surface

- Added: `tesserix_adk.core.MediaIntakeError`
- Added: `tesserix_adk.core.errors.MediaIntakeError`
- Added: `tesserix_adk.observability.CACHE_HIT_RATIO`
- Added: `tesserix_adk.observability.CacheHits`
- Added: `tesserix_adk.observability.LATENCY_SECONDS`
- Added: `tesserix_adk.observability.LatencyReport`
- Added: `tesserix_adk.observability.RunTimer`
- Added: `tesserix_adk.observability.TIME_TO_FIRST_TOKEN`
- Added: `tesserix_adk.observability.TOKENS_PER_SECOND`
- Added: `tesserix_adk.observability.latency.CACHE_HIT_RATIO`
- Added: `tesserix_adk.observability.latency.CacheHits`
- Added: `tesserix_adk.observability.latency.LATENCY_SECONDS`
- Added: `tesserix_adk.observability.latency.LatencyReport`
- Added: `tesserix_adk.observability.latency.RunTimer`
- Added: `tesserix_adk.observability.latency.TIME_TO_FIRST_TOKEN`
- Added: `tesserix_adk.observability.latency.TOKENS_PER_SECOND`
- Added: `tesserix_adk.tools.DEFAULT_INTAKE_CHARS`
- Added: `tesserix_adk.tools.OcrBackend`
- Added: `tesserix_adk.tools.OcrPage`
- Added: `tesserix_adk.tools.Region`
- Added: `tesserix_adk.tools.RegionKind`
- Added: `tesserix_adk.tools.SUPPORTED_AUDIO`
- Added: `tesserix_adk.tools.SUPPORTED_DOCUMENTS`
- Added: `tesserix_adk.tools.Segment`
- Added: `tesserix_adk.tools.TranscriptionBackend`
- Added: `tesserix_adk.tools.intake.DEFAULT_INTAKE_CHARS`
- Added: `tesserix_adk.tools.intake.OcrBackend`
- Added: `tesserix_adk.tools.intake.OcrPage`
- Added: `tesserix_adk.tools.intake.Region`
- Added: `tesserix_adk.tools.intake.RegionKind`
- Added: `tesserix_adk.tools.intake.SUPPORTED_AUDIO`
- Added: `tesserix_adk.tools.intake.SUPPORTED_DOCUMENTS`
- Added: `tesserix_adk.tools.intake.Segment`
- Added: `tesserix_adk.tools.intake.TranscriptionBackend`
- Added: `tesserix_adk.tools.intake.ocr_document`
- Added: `tesserix_adk.tools.intake.ocr_tool`
- Added: `tesserix_adk.tools.intake.transcribe_audio`
- Added: `tesserix_adk.tools.intake.transcribe_tool`
- Added: `tesserix_adk.tools.ocr_document`
- Added: `tesserix_adk.tools.ocr_tool`
- Added: `tesserix_adk.tools.transcribe_audio`
- Added: `tesserix_adk.tools.transcribe_tool`

## 0.10.0

### Added

- **tesserix_adk.models.onnx**: `tesserix_adk.models` now embeds and reranks on the CPU the operator already has, through a
quantized ONNX session. On a machine without a GPU a retrieval turn spends more of its
latency in the small models than in the large one, and this is what makes that interactive.

`onnxruntime` is deliberately not a dependency and not an extra: it is a native wheel of a
few hundred megabytes, and everyone installing `tesserix-adk[all]` would inherit it for a
feature they may not use. The kit defines two narrow protocols instead — `Tokenizing` and
`OnnxSession` — so an operator either supplies a session or calls `load_session`, which
imports the runtime by name and raises `ConfigurationError` naming the install command when
it is absent.

`ONNX_MODELS` is the documented set, each with a `Throughput` giving a measured rate
together with the batch and thread count it was measured at, because a rate without those
cannot be reproduced or held to. `onnx_model` looks one up by name and lists the set when
the name is unknown. One of them is multilingual.

`verify_artefact` checks the model file before anything loads it and raises the new
`ModelArtifactError` with `reason` set to `missing`, `empty` or `digest` — a half-downloaded
model should fail at startup for an operator, not at the first query for a user.

`OnnxEmbeddings` satisfies `EmbeddingProvider`. It batches, runs the synchronous session off
the event loop, and caches vectors under a key covering the model's name, version and
dimension, so a version bump cannot read the previous vectors. The cache is bounded and
evicts oldest-first. After every call `metrics` reports what was embedded, what was cached,
and whether the measured rate met the model's declared budget.

`OnnxCrossEncoder` scores query/passage pairs and is the `CrossEncoder` that
`rag.CrossEncoderReranker` takes. A GPU changes `device` and nothing else about either.

### Public API surface

- Added: `tesserix_adk.core.ModelArtifactError`
- Added: `tesserix_adk.core.errors.ModelArtifactError`
- Added: `tesserix_adk.models.Device`
- Added: `tesserix_adk.models.Encoded`
- Added: `tesserix_adk.models.ONNX_MODELS`
- Added: `tesserix_adk.models.OnnxCrossEncoder`
- Added: `tesserix_adk.models.OnnxEmbeddings`
- Added: `tesserix_adk.models.OnnxMetrics`
- Added: `tesserix_adk.models.OnnxModel`
- Added: `tesserix_adk.models.OnnxSession`
- Added: `tesserix_adk.models.Throughput`
- Added: `tesserix_adk.models.Tokenizing`
- Added: `tesserix_adk.models.load_session`
- Added: `tesserix_adk.models.onnx.Device`
- Added: `tesserix_adk.models.onnx.Encoded`
- Added: `tesserix_adk.models.onnx.ONNX_MODELS`
- Added: `tesserix_adk.models.onnx.OnnxCrossEncoder`
- Added: `tesserix_adk.models.onnx.OnnxEmbeddings`
- Added: `tesserix_adk.models.onnx.OnnxMetrics`
- Added: `tesserix_adk.models.onnx.OnnxModel`
- Added: `tesserix_adk.models.onnx.OnnxSession`
- Added: `tesserix_adk.models.onnx.Throughput`
- Added: `tesserix_adk.models.onnx.Tokenizing`
- Added: `tesserix_adk.models.onnx.load_session`
- Added: `tesserix_adk.models.onnx.onnx_model`
- Added: `tesserix_adk.models.onnx.verify_artefact`
- Added: `tesserix_adk.models.onnx_model`
- Added: `tesserix_adk.models.verify_artefact`

## 0.9.0

### Added

- **tesserix_adk.runtime.compaction**: `tesserix_adk.runtime` now compacts a long conversation without losing where its claims
came from. `compact_conversation` folds the oldest turns into one summary once the
conversation passes `threshold_tokens`, keeping the newest `keep_recent` verbatim, and does
nothing below the threshold so it can be called every turn.

Citation ids travel on the message itself — `cited` writes them under `adk.citations` and
`citations_of` reads them back — because a history is what gets persisted, replayed and
handed to a provider, and a parallel list of sources is what does not survive that. The
`Summariser` is handed the turns and returns the replacement message, so it can carry across
whatever else its messages hold; what is not left to it is the provenance. Every id carried
by a folded turn must be on the message replacing them, or the new `ProvenanceLostError`
names the ids that would have been dropped and the history is returned untouched. Prose is
what a summary may lose.

Compaction touches only the conversation layer, so `assemble_prompt` produces the same
prefix fingerprint before and after. The summary is marked `adk.compacted` and is not
summarised again, which makes a second pass a no-op and folds it in with new turns on a
later one. `Compaction.event` is a `CompactionEvent` giving the run, the turns folded, the
ids carried and the tokens before and after, with `attributes()` for the span — counts
only, never the conversation.

### Public API surface

- Added: `tesserix_adk.core.ProvenanceLostError`
- Added: `tesserix_adk.core.errors.ProvenanceLostError`
- Added: `tesserix_adk.runtime.CITATIONS`
- Added: `tesserix_adk.runtime.COMPACTED`
- Added: `tesserix_adk.runtime.Compaction`
- Added: `tesserix_adk.runtime.CompactionEvent`
- Added: `tesserix_adk.runtime.Summariser`
- Added: `tesserix_adk.runtime.citations_of`
- Added: `tesserix_adk.runtime.cited`
- Added: `tesserix_adk.runtime.compact_conversation`
- Added: `tesserix_adk.runtime.compaction.CITATIONS`
- Added: `tesserix_adk.runtime.compaction.COMPACTED`
- Added: `tesserix_adk.runtime.compaction.Compaction`
- Added: `tesserix_adk.runtime.compaction.CompactionEvent`
- Added: `tesserix_adk.runtime.compaction.Summariser`
- Added: `tesserix_adk.runtime.compaction.citations_of`
- Added: `tesserix_adk.runtime.compaction.cited`
- Added: `tesserix_adk.runtime.compaction.compact_conversation`

## 0.8.0

### Added

- **tesserix_adk.rag.quarantine**: Retrieved content now leaves `tesserix_adk.rag` as data. `quarantine` turns a
`RetrievalResult` into `UntrustedText` values, which are deliberately not `str`: every
instruction-position parameter in the kit is typed `str`, so putting a passage there does
not type-check, and `str()` of one raises `TrustBoundaryError` rather than producing prose
the model will read as its own. `Quarantined.for_layer` hands out the fenced blocks for the
retrieved prompt layer and refuses every other section by name, and the fence delimiter is
escaped inside each block, so a passage containing `</untrusted-data>` cannot close it
early.

`screen` normalises a passage the way the model reads it — NFKC, zero-width characters
stripped, Cyrillic homoglyphs folded, base64 runs decoded — and returns `InjectionSignal`
values naming what it recognised: `OVERRIDE`, `TOOL_SHAPED`, `FENCE`, `ENCODED`,
`SYSTEM_ECHO` for the agent's own instructions quoted back at it, `METADATA` for an
instruction in a field nobody reads as prose, and `SPLIT` for one assembled across two
adjacent chunks. Nothing is dropped: a flagged passage still reaches the prompt, fenced,
because pattern matching loses to paraphrase and a silently shortened corpus answers the
question wrongly while looking like it worked. `Quarantined.attributes` gives counts and
kinds under `adk.retrieval.injection_*`, never document text.

`tesserix_adk.testing.POISONED_CORPUS` is a corpus somebody has written to, covering each
of those shapes, for testing a pipeline against corpus poisoning without writing the
payloads by hand.

### Public API surface

- Added: `tesserix_adk.rag.InjectionSignal`
- Added: `tesserix_adk.rag.Quarantined`
- Added: `tesserix_adk.rag.SignalKind`
- Added: `tesserix_adk.rag.UntrustedText`
- Added: `tesserix_adk.rag.quarantine`
- Added: `tesserix_adk.rag.quarantine.InjectionSignal`
- Added: `tesserix_adk.rag.quarantine.Quarantined`
- Added: `tesserix_adk.rag.quarantine.SignalKind`
- Added: `tesserix_adk.rag.quarantine.UntrustedText`
- Added: `tesserix_adk.rag.quarantine.quarantine`
- Added: `tesserix_adk.rag.quarantine.screen`
- Added: `tesserix_adk.rag.screen`
- Added: `tesserix_adk.testing.POISONED_CORPUS`
- Added: `tesserix_adk.testing.retrieval.POISONED_CORPUS`

## 0.7.0

### Added

- **tesserix_adk.rag.citation**: `tesserix_adk.rag` now produces citations that resolve. `cite` turns a `RetrievalResult`
into `Citation` values pinning the document, the version that was retrieved, the chunk and
its character `Span`, along with the tenant, the score, the branches that found it and a
`SourceLocator`. A chunk whose metadata carries no version or no span raises
`ConfigurationError` rather than yielding a citation that would resolve against whatever
the document says now.

`CitedAnswer` is claims and citations rather than prose containing footnote-shaped strings:
each `Claim` names the `citation_ids` it rests on, several citations may support one claim,
and one may support several. `check_grounding` runs before an answer is returned and fails
closed — `UncitedClaimError` for a claim resting on nothing, `UngroundedCitationError` for
a citation this run did not retrieve or one whose version has moved, and
`TenantCrossingError` for a citation into another tenant. Nothing is stripped to make an
answer look sourced.

`excerpt` resolves a citation back to the exact characters of the document version it was
made from, and refuses a document that has since been updated. `CitationResolver` and
`ResolvedCitation` cover a live corpus, where an erased chunk resolves to a tombstone.

`answer.provenance()` and the new `MemoryRecord.citations` field carry the sources of a
fact derived from retrieved content into memory, and `citation_attributes` gives span
attributes — counts, document ids and versions, never document text.

### Public API surface

- Added: `tesserix_adk.core.UncitedClaimError`
- Added: `tesserix_adk.core.UngroundedCitationError`
- Added: `tesserix_adk.core.errors.UncitedClaimError`
- Added: `tesserix_adk.core.errors.UngroundedCitationError`
- Added: `tesserix_adk.rag.Citation`
- Added: `tesserix_adk.rag.CitationResolver`
- Added: `tesserix_adk.rag.CitedAnswer`
- Added: `tesserix_adk.rag.Claim`
- Added: `tesserix_adk.rag.ResolvedCitation`
- Added: `tesserix_adk.rag.SourceLocator`
- Added: `tesserix_adk.rag.Span`
- Added: `tesserix_adk.rag.check_grounding`
- Added: `tesserix_adk.rag.citation.Citation`
- Added: `tesserix_adk.rag.citation.CitationResolver`
- Added: `tesserix_adk.rag.citation.CitedAnswer`
- Added: `tesserix_adk.rag.citation.Claim`
- Added: `tesserix_adk.rag.citation.ResolvedCitation`
- Added: `tesserix_adk.rag.citation.SourceLocator`
- Added: `tesserix_adk.rag.citation.Span`
- Added: `tesserix_adk.rag.citation.check_grounding`
- Added: `tesserix_adk.rag.citation.citation_attributes`
- Added: `tesserix_adk.rag.citation.cite`
- Added: `tesserix_adk.rag.citation.excerpt`
- Added: `tesserix_adk.rag.citation_attributes`
- Added: `tesserix_adk.rag.cite`
- Added: `tesserix_adk.rag.excerpt`

## 0.6.0

### Added

- **tesserix_adk.rag.reranking**: `tesserix_adk.rag` now reranks behind a `Reranker` protocol. `RerankingRetriever` is a
`Retriever` wrapping a `Retriever`: it asks the inner one for `candidates`, has the
reranker score them, and returns the best `top_n`. Both counts are checked at construction,
so the fan-out cannot be unbounded, and a reranker declared unavailable raises
`CapabilityError` there rather than degrading on every call.

Every hit keeps its fusion `score` beside its new `rerank_score`, and ties break on chunk
id so a replayed retrieval gives the same ranking. A candidate the reranker did not score
keeps its fused position behind the ones that were scored, and a score for a chunk nobody
retrieved is ignored.

The reranker call is recorded against the `BudgetPolicy` as one model call. An exhausted
budget, a timeout or a failing reranker returns the fused order with `reranked=False` and
traces `adk.rerank.degraded` with the reason, rather than failing the retrieval.

Three rerankers ship: `NoReranking` for measuring the stage's own overhead,
`CrossEncoderReranker` over the one-method `CrossEncoder` protocol, and `ModelReranker`,
which sends passages as JSON data under an instruction that says they are never
instructions to follow, and reads no scores at all from a reply of the wrong shape.

### Public API surface

- Added: `tesserix_adk.rag.CrossEncoder`
- Added: `tesserix_adk.rag.CrossEncoderReranker`
- Added: `tesserix_adk.rag.DEGRADED`
- Added: `tesserix_adk.rag.ModelReranker`
- Added: `tesserix_adk.rag.NoReranking`
- Added: `tesserix_adk.rag.RerankScore`
- Added: `tesserix_adk.rag.Reranker`
- Added: `tesserix_adk.rag.Reranking`
- Added: `tesserix_adk.rag.RerankingRetriever`
- Added: `tesserix_adk.rag.reranking.CrossEncoder`
- Added: `tesserix_adk.rag.reranking.CrossEncoderReranker`
- Added: `tesserix_adk.rag.reranking.DEGRADED`
- Added: `tesserix_adk.rag.reranking.ModelReranker`
- Added: `tesserix_adk.rag.reranking.NoReranking`
- Added: `tesserix_adk.rag.reranking.RerankScore`
- Added: `tesserix_adk.rag.reranking.Reranker`
- Added: `tesserix_adk.rag.reranking.Reranking`
- Added: `tesserix_adk.rag.reranking.RerankingRetriever`

## 0.5.0

### Added

- **tesserix_adk.rag.retrieval**: `tesserix_adk.rag` now retrieves behind a `Retriever` protocol. `IndexRetriever` runs one
branch over a `SearchIndex`, and `HybridRetriever` runs several and fuses them by rank —
`ReciprocalRankFusion` by default, `WeightedSum` for a corpus that has been measured. Every
hit carries a `BranchScore` per branch that found it, so `hit.found_by(Branch.KEYWORD)`
answers whether a passage was an exact match or the vector's opinion.

The tenant predicate is set from the tenant context into `IndexQuery.tenant` and applied
inside the store's own query; a caller filter named `tenant` is refused with
`SchemaViolationError` rather than merged or dropped. Other filters are pushed down
alongside it, so nothing is filtered after the fetch.

A branch that fails or times out leaves the result `partial` and names the branches that
did answer; `require=` turns that into a `RetrievalDegradedError` naming what is missing,
and a retrieval where no branch answered always raises.

`tesserix_adk.adapters` adds `PgvectorIndex`, running both branches over one chunk table,
with `EXPECTED_CHUNK_SCHEMA` for the migration repository to own. `tesserix_adk.testing`
adds `FakeIndex`, `Indexed` and a `SearchIndexConformance` suite whose corpus holds a
second tenant's identical passage, so a store that filters after the fetch fails the suite.

### Public API surface

- Added: `tesserix_adk.adapters.CHUNK_SCHEMA_VERSION`
- Added: `tesserix_adk.adapters.ChunkTables`
- Added: `tesserix_adk.adapters.DEFAULT_CHUNK_TABLES`
- Added: `tesserix_adk.adapters.EXPECTED_CHUNK_SCHEMA`
- Added: `tesserix_adk.adapters.PgvectorIndex`
- Added: `tesserix_adk.adapters.PgvectorSettings`
- Added: `tesserix_adk.adapters.pgvector.CHUNK_SCHEMA_VERSION`
- Added: `tesserix_adk.adapters.pgvector.ChunkTables`
- Added: `tesserix_adk.adapters.pgvector.DEFAULT_CHUNK_TABLES`
- Added: `tesserix_adk.adapters.pgvector.EXPECTED_CHUNK_SCHEMA`
- Added: `tesserix_adk.adapters.pgvector.PgvectorIndex`
- Added: `tesserix_adk.adapters.pgvector.PgvectorSettings`
- Added: `tesserix_adk.core.RetrievalDegradedError`
- Added: `tesserix_adk.core.errors.RetrievalDegradedError`
- Added: `tesserix_adk.rag.Branch`
- Added: `tesserix_adk.rag.BranchScore`
- Added: `tesserix_adk.rag.Fusion`
- Added: `tesserix_adk.rag.Hit`
- Added: `tesserix_adk.rag.HybridRetriever`
- Added: `tesserix_adk.rag.IndexQuery`
- Added: `tesserix_adk.rag.IndexRetriever`
- Added: `tesserix_adk.rag.ReciprocalRankFusion`
- Added: `tesserix_adk.rag.RetrievalResult`
- Added: `tesserix_adk.rag.RetrievalScope`
- Added: `tesserix_adk.rag.Retriever`
- Added: `tesserix_adk.rag.SearchIndex`
- Added: `tesserix_adk.rag.WeightedSum`
- Added: `tesserix_adk.rag.retrieval.Branch`
- Added: `tesserix_adk.rag.retrieval.BranchScore`
- Added: `tesserix_adk.rag.retrieval.Fusion`
- Added: `tesserix_adk.rag.retrieval.Hit`
- Added: `tesserix_adk.rag.retrieval.HybridRetriever`
- Added: `tesserix_adk.rag.retrieval.IndexQuery`
- Added: `tesserix_adk.rag.retrieval.IndexRetriever`
- Added: `tesserix_adk.rag.retrieval.ReciprocalRankFusion`
- Added: `tesserix_adk.rag.retrieval.RetrievalResult`
- Added: `tesserix_adk.rag.retrieval.RetrievalScope`
- Added: `tesserix_adk.rag.retrieval.Retriever`
- Added: `tesserix_adk.rag.retrieval.SearchIndex`
- Added: `tesserix_adk.rag.retrieval.WeightedSum`
- Added: `tesserix_adk.testing.FakeIndex`
- Added: `tesserix_adk.testing.Indexed`
- Added: `tesserix_adk.testing.SearchIndexConformance`
- Added: `tesserix_adk.testing.conformance.CONFORMANCE_CORPUS`
- Added: `tesserix_adk.testing.conformance.SearchIndexConformance`
- Added: `tesserix_adk.testing.retrieval.FakeIndex`
- Added: `tesserix_adk.testing.retrieval.Indexed`

## 0.4.0

### Added

- **tesserix_adk.rag.embedding**: `tesserix_adk.rag` embeds behind an `Embedder` protocol. `BatchedEmbedder` wraps a
`VectorSource` — one provider call, one batch — and adds batching to the model's declared
limit, a cap on calls in flight, jittered retries through the kit's `RetryConfig`, and a
content-addressed cache keyed on model name, version, width and the normalised text, so an
unchanged corpus re-indexes for free. Cache entries are tenant-isolated by default, with
`shared=True` for a public corpus. A batch that cannot be embedded within the retry budget
raises `EmbeddingUnavailableError` naming the batch and the cursor to resume from, with
everything that did land already cached; nothing is ever substituted for a missing vector.
`MemoryEmbeddingCache` is a bounded in-process backend and `tesserix_adk.testing`
publishes `FakeEmbedder`, whose vectors are the same in every run.

### Public API surface

- Added: `tesserix_adk.core.EmbeddingUnavailableError`
- Added: `tesserix_adk.core.errors.EmbeddingUnavailableError`
- Added: `tesserix_adk.rag.BatchVectors`
- Added: `tesserix_adk.rag.BatchedEmbedder`
- Added: `tesserix_adk.rag.EmbeddedBatch`
- Added: `tesserix_adk.rag.Embedder`
- Added: `tesserix_adk.rag.EmbeddingCache`
- Added: `tesserix_adk.rag.EmbeddingModel`
- Added: `tesserix_adk.rag.EmbeddingStats`
- Added: `tesserix_adk.rag.MemoryEmbeddingCache`
- Added: `tesserix_adk.rag.Vector`
- Added: `tesserix_adk.rag.VectorSource`
- Added: `tesserix_adk.rag.embedding.BatchVectors`
- Added: `tesserix_adk.rag.embedding.BatchedEmbedder`
- Added: `tesserix_adk.rag.embedding.EmbeddedBatch`
- Added: `tesserix_adk.rag.embedding.Embedder`
- Added: `tesserix_adk.rag.embedding.EmbeddingCache`
- Added: `tesserix_adk.rag.embedding.EmbeddingModel`
- Added: `tesserix_adk.rag.embedding.EmbeddingStats`
- Added: `tesserix_adk.rag.embedding.MemoryEmbeddingCache`
- Added: `tesserix_adk.rag.embedding.Vector`
- Added: `tesserix_adk.rag.embedding.VectorSource`
- Added: `tesserix_adk.rag.embedding.embedding_key`
- Added: `tesserix_adk.rag.embedding.normalised`
- Added: `tesserix_adk.rag.embedding_key`
- Added: `tesserix_adk.rag.normalised`
- Added: `tesserix_adk.testing.FakeEmbedder`
- Added: `tesserix_adk.testing.embedding.FakeEmbedder`

## 0.3.0

### Added

- **tesserix_adk.rag.chunking**: `tesserix_adk.rag` chunks documents behind a `Chunker` protocol: fixed-token windows with
overlap, recursive structural splitting, sentence windows and code-aware splitting, chosen
per collection through a `ChunkerRegistry`. Chunks are sized by the tokeniser of the model
that will read them, carry the exact character span they came from, and take
content-addressed ids. Text that will not divide under the limit raises `ChunkingError`
naming the document and offset rather than emitting an over-long chunk.

### Public API surface

- Added: `tesserix_adk.core.ChunkingError`
- Added: `tesserix_adk.core.errors.ChunkingError`
- Added: `tesserix_adk.rag.Chunk`
- Added: `tesserix_adk.rag.Chunker`
- Added: `tesserix_adk.rag.ChunkerFactory`
- Added: `tesserix_adk.rag.ChunkerRegistry`
- Added: `tesserix_adk.rag.ChunkingSpec`
- Added: `tesserix_adk.rag.CodeAware`
- Added: `tesserix_adk.rag.Document`
- Added: `tesserix_adk.rag.FixedTokens`
- Added: `tesserix_adk.rag.Overflow`
- Added: `tesserix_adk.rag.SentenceWindow`
- Added: `tesserix_adk.rag.Structural`
- Added: `tesserix_adk.rag.TokenCount`
- Added: `tesserix_adk.rag.chunk_id`
- Added: `tesserix_adk.rag.chunking.Chunk`
- Added: `tesserix_adk.rag.chunking.Chunker`
- Added: `tesserix_adk.rag.chunking.ChunkerFactory`
- Added: `tesserix_adk.rag.chunking.ChunkerRegistry`
- Added: `tesserix_adk.rag.chunking.ChunkingSpec`
- Added: `tesserix_adk.rag.chunking.CodeAware`
- Added: `tesserix_adk.rag.chunking.Document`
- Added: `tesserix_adk.rag.chunking.FixedTokens`
- Added: `tesserix_adk.rag.chunking.Overflow`
- Added: `tesserix_adk.rag.chunking.SentenceWindow`
- Added: `tesserix_adk.rag.chunking.Structural`
- Added: `tesserix_adk.rag.chunking.TokenCount`
- Added: `tesserix_adk.rag.chunking.chunk_id`
- Added: `tesserix_adk.rag.chunking.tokens_via`
- Added: `tesserix_adk.rag.tokens_via`

## 0.2.0

### Added

- **tesserix_adk.core.tenant_config**: What a tenant is permitted is now data rather than a branch in agent code. A
`TenantConfigProvider` answers for one tenant at a time — `FileTenantConfig` reads a TOML
file per tenant, `CachingTenantConfig` puts a bounded, per-tenant-invalidated cache in
front of any provider — and `resolve_tenant_policy` resolves the bound tenant's
entitlement once, at the boundary that already knows whose work this is.
`tenant_policy(policy)` binds it; `current_policy()` is what everything below reads.

`TenantLimits` states the model allowlist, the task-class routing table, the tool
allowlist, the data region, the memory retention window and the budget ceiling.
`check_model`, `check_tool` and `check_region` refuse before the call rather than
reporting after it — a ceiling catches spend once it has happened, an allowlist stops it.
An absent setting states nothing and leaves the decision to whatever else is in force; an
empty one is a stated refusal.

Budget is the exception to "the tenant layer is highest": a tenant ceiling arrives as one
more `ScopedLimits` under `BudgetScope.TENANT` and `most_restrictive` decides, so a tenant
configuration can narrow a run's ceiling and never widen it. `AgentRunner` folds the bound
policy's ceiling in where it was given no explicit `BudgetPolicy`, so the same agent code
halts at the cheap plan's ceiling for one tenant and continues for another.

A tenant-scoped ceiling with no `TenantLedger` wired now holds the run it is on, where
before it was checked against a ledger total that nothing updated and so bound nothing.

Every path that could end in a permissive default ends in a refusal: an unknown tenant is
`UnknownTenantError` rather than the global defaults, an unreadable store is
`TenantUnconfiguredError` — a distinct type, because one is a request to reject and the
other is an outage to page on — and a cache entry past its window is not served even when
the store behind it is down. Secrets in tenant configuration are named by `SecretRef` and
resolved from the `SecretProvider` in force; a literal where a reference belongs does not
validate.

Documented in `docs/tenant-config.md`.

### Public API surface

- Added: `tesserix_adk.core.CachingTenantConfig`
- Added: `tesserix_adk.core.FileTenantConfig`
- Added: `tesserix_adk.core.SecretRef`
- Added: `tesserix_adk.core.TenantConfig`
- Added: `tesserix_adk.core.TenantConfigProvider`
- Added: `tesserix_adk.core.TenantLimitError`
- Added: `tesserix_adk.core.TenantLimits`
- Added: `tesserix_adk.core.TenantPolicy`
- Added: `tesserix_adk.core.TenantUnconfiguredError`
- Added: `tesserix_adk.core.UnknownTenantError`
- Added: `tesserix_adk.core.current_policy`
- Added: `tesserix_adk.core.errors.TenantLimitError`
- Added: `tesserix_adk.core.errors.TenantUnconfiguredError`
- Added: `tesserix_adk.core.errors.UnknownTenantError`
- Added: `tesserix_adk.core.policy_here`
- Added: `tesserix_adk.core.resolve_tenant_policy`
- Added: `tesserix_adk.core.tenant_config.CachingTenantConfig`
- Added: `tesserix_adk.core.tenant_config.FileTenantConfig`
- Added: `tesserix_adk.core.tenant_config.SecretRef`
- Added: `tesserix_adk.core.tenant_config.TenantConfig`
- Added: `tesserix_adk.core.tenant_config.TenantConfigProvider`
- Added: `tesserix_adk.core.tenant_config.TenantLimits`
- Added: `tesserix_adk.core.tenant_config.TenantPolicy`
- Added: `tesserix_adk.core.tenant_config.current_policy`
- Added: `tesserix_adk.core.tenant_config.policy_here`
- Added: `tesserix_adk.core.tenant_config.resolve_tenant_policy`
- Added: `tesserix_adk.core.tenant_config.tenant_policy`
- Added: `tesserix_adk.core.tenant_policy`

## 0.1.1

### Fixed

- **tesserix_adk.testing**: Importing a fake from `tesserix_adk.testing` no longer requires `pytest`. The package
imported its conformance suites and pytest plugin eagerly, and both import `pytest` at
module scope, so on a wheel install `import tesserix_adk.testing` raised
`ModuleNotFoundError: No module named 'pytest'` — which reached every consumer who only
wanted a fake, and the getting-started example with them.

Both modules are now loaded on first use behind a module-level `__getattr__`. Every name
is still importable exactly as before, still exported from `__all__`, and still carries
its real type for a type checker; a name the package does not have still fails as an
`AttributeError` rather than being swallowed.

**Stability:** no API change. `from tesserix_adk.testing import TracerConformance` and
subclassing the suites work unchanged; running them still needs `pytest`, as it always did.

## 0.1.0

### Breaking changes

- **tesserix_adk.core.ModelProvider**: One typed provider protocol, with what a model can do declared as data. `ModelProvider`
was a set of `Any` signatures a provider could satisfy by accident; it now states
`complete`, `stream`, `count_tokens` and `capabilities` over the kit's own request and
response types, which moved into `core` so the protocol could be typed over them without
inverting the layering. `tesserix_adk.models` re-exports the whole surface, which is where
a provider author looks for it.

`ModelCapabilities` carries `structured_output`, `tool_calling`, `parallel_tool_calls`,
`vision`, `streaming`, `context_window_tokens` and `max_output_tokens`, every one
defaulting to off or unknown: silence is not a claim, and a capability nobody declared is
one the kit will not assume. `declared`, `supports` and `require` read it, and
`CapabilityError` names the capability, the provider and the model rather than saying
"unsupported" and leaving the caller to work out which of the three to change.

The record is checked before anything is sent. A tool registry wired to a model that does
not call tools fails at construction; an agent naming tools fails before its first
request; an image part on a text-only model and a prompt past the declared window fail the
same way, the second with `ContextWindowExceededError` carrying the count and the limit —
a vendor handed an over-long prompt truncates it and answers anyway, so the first sign of
the problem is an answer that ignores the beginning of the case. A payload that is not a
response at all raises `ModelResponseError` carrying the raw payload and the provider's
request id, distinct from a well-formed answer in the wrong shape, which stays a
`SchemaViolationError` and goes to the repair flow.

`ModelRef` and `ModelSpec` make a model addressable from configuration, with the provider
part of its identity: a vendor API and an OpenAI-compatible proxy serve the same model ids
and are not the same model. `ModelCapabilities.declaring` and `ModelSpec.with_capabilities`
narrow a record without a subclass, because a self-hosted endpoint serves the weights it
was given rather than the ones on the model card. `ModelProviderConformance` is the suite
a third-party provider inherits.

**Stability:** breaking for provider implementors, additive for callers. Two members are
now required of anything passed as a provider, and `verify_conformance` reports their
absence at construction. No call site that only *uses* a runner changes. Documented in
`docs/providers.md`, exercised by `examples/providers.py`.

  *Migration:* A provider gains two members. Add a `capabilities` property returning a `ModelCapabilities` and a `count_tokens(messages)` returning the provider's own count; `tesserix_adk.testing.estimate_tokens` is a character-count stand-in where no tokeniser ships. `ScriptedProvider(structured=True)` becomes `ScriptedProvider(capabilities=CAPABLE.declaring(structured_output=True))`.
- **tesserix_adk.core, tesserix_adk.runtime**: How deep and how wide a run may go are caps on spend, so they now live with the money.
`BudgetLimits` gains `max_delegation_depth`, `max_parallel_tool_calls` and
`max_peer_invocations`; `LoopConfig` loses `max_depth`, `max_tool_calls_per_turn` and
`max_tool_calls_per_run` and keeps `max_repeated_calls`, which is not spend but the shape of
a run that has stopped making progress. Two policies is two places to raise a cap, and the
one nobody read is the one that lets a run away.

A turn wider than `max_parallel_tool_calls` is refused entire, before any tool is dispatched:
ten calls of two hundred executed and presented as the answer is worse than none, and the
partial side effects are real either way. The refusal names the cap and what was asked for,
and lands as a `FAN_OUT_REFUSED` event. The run total is checked against the whole turn in the
same place, so a fan-out cannot step over a ceiling one call at a time.

Delegation depth and peer invocations are checked before a prompt is assembled — a cycle costs
nothing to stop that way — and the refusal prints the call path, because `A→B→A` is the shape
of the bug and naming it is how somebody finds it. `Run.path` and `RunContext.path` carry that
path, root first; a depth alone says a run went too far, not where it went round. Peer
invocations are counted on the shared ledger rather than per hop, so a tree of runs each
staying under a cap they broke together is no longer possible, and a delegated agent asking
for a wider cap than its parent's gets its parent's.

Tool calls in a cleared turn still go out one at a time. Dispatching them concurrently would
save wall-clock and lose the check between calls, and that check is what stops the second call
of a turn the caller cancelled during the first.

**Stability:** breaking for anything constructing `LoopConfig` with the removed fields.
Documented in `docs/run-loop.md` and `docs/budget.md`, exercised by `examples/loops.py` and
`tests/test_fan_out.py`.

  *Migration:* State the caps on `BudgetLimits` instead — `LoopConfig(max_depth=…)` becomes `BudgetLimits(max_delegation_depth=…)`, `max_tool_calls_per_turn` becomes `max_parallel_tool_calls`, and `max_tool_calls_per_run` becomes `max_tool_calls`. `LoopConfig` keeps `max_repeated_calls` only.
- **tesserix_adk.core.ApprovalDenial**: A denied approval no longer fails the run. It reaches the agent as a `ToolRefusal` — code
`approval_denied`, or `approval_expired` for a decision that arrived outside its window — which
the agent can answer by proposing something the human will accept. Killing the run instead throws
away everything it has done because a human said no to one call, and "no" to a call is an answer,
not a crash. The `APPROVAL_DENIED` event and the guarantee that nothing dispatches are unchanged.

**Stability:** breaking for anyone asserting on `RunState.FAILED` after a denial, or relying on
the run stopping. `ApprovalDenial.FAIL_RUN` restores the previous behaviour exactly. A gate that
cannot be reached still fails the run: an unanswered request is not a denial, and treating an
outage as a refusal the agent may talk around is how the gate stops being one.

  *Migration:* Pass `AgentRunner(approval_denial=ApprovalDenial.FAIL_RUN)` to keep a denied or expired approval failing the run, and update assertions expecting `RunState.FAILED` after a denial to `RunState.COMPLETED` plus the `TOOL_REFUSED` event carrying `approval_denied` or `approval_expired`.
- **tesserix_adk.core.KeyValueStore, tesserix_adk.testing.FakeKeyValueStore, tesserix_adk.testing.KeyValueStoreConformance**: `tesserix_adk.core.MemoryStore` is now `KeyValueStore`, with `FakeMemoryStore` and
`MemoryStoreConformance` renamed to match. It was a three-method `get`/`put`/`delete`
protocol over untyped values with no scope in any signature — a key-value store, not a
memory system, and the name was the only thing suggesting otherwise.

**Stability:** breaking, and a rename only — the methods, semantics and conformance cases
are untouched, and nothing in the kit consumed it. The names it vacates are taken by the
memory protocol added alongside, which is what a consumer reaching for `MemoryStore`
was looking for.

  *Migration:* Rename `MemoryStore` to `KeyValueStore`, `FakeMemoryStore` to `FakeKeyValueStore`, and `MemoryStoreConformance` to `KeyValueStoreConformance`. The three methods are unchanged. `MemoryStore` and `MemoryStoreConformance` now name the memory protocol in `tesserix_adk.memory` and its suite in `tesserix_adk.testing`.
- **tesserix_adk.memory.ErasureReceipt, tesserix_adk.core.PartialErasureError**: `MemoryStore.erase` now returns an `ErasureReceipt` rather than a count. A number cannot
say which kinds went, which indices were spoken to, whether the erasure finished, or
when — and a right-to-erasure request answered with `5` is not answered.

**Stability:** breaking for anyone comparing the return of `erase` to a number;
`.records` is the same integer. The kit's own call sites and examples are updated.
Documented in `docs/erasure.md`.

  *Migration:* `MemoryStore.erase` returns an `ErasureReceipt` instead of an `int`. Replace `count = await store.erase(scope)` with `count = (await store.erase(scope)).records`. An adapter implementing the protocol must return a receipt and gains two members, `derived` and `derivations`.
- **tesserix_adk.core.Guardrail, tesserix_adk.core.GuardrailViolationError**: The `Guardrail` protocol splits `check(subject: Any) -> Any` into `check_input` and
`check_output`, both taking `str` and returning `GuardResult`. One method could not say
which end of the run it covered, so a pipeline had to call a guard to find out, and an
`Any` verdict left "allowed" and "returned something unreadable" indistinguishable — which
is exactly the case that must fail closed. Requiring both means what a guard covers is
readable from its type.

`GuardrailViolationError` now inherits `GuardrailError` rather than `AdkError` directly and
takes keyword-only `code`, `stage`, `guard` and `detail`. Code catching `AdkError` is
unaffected.

  *Migration:* Replace `async def check(self, subject)` with `async def check_input(self, content: str) -> GuardResult` and `check_output` — or subclass `tesserix_adk.guardrails.Guard`, which allows on both stages and lets a one-sided check override one method. Return `GuardResult.allow()` / `.redacted(content, code=…)` / `.blocked(code=…)` instead of raising `GuardrailViolationError` yourself; the pipeline raises it, with `code`, `stage`, `guard` and `detail` now keyword-only.
- **tesserix_adk.runtime.prompt**: The prompt's cacheable prefix is now an invariant with a name. `assemble_prompt` assembles
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
run records both: the `PROMPT_ASSEMBLED` event now carries the fingerprint and the prefix
size, so a cache-hit ratio is measurable from the audit trail.

Tool declarations are **sorted by name** rather than kept in registry order, so a registry
that iterates a dict differently between processes cannot cost a refill; two tools sharing
a name are now refused, because sorting hides the duplicate and the model cannot tell which
it is calling. `assemble_prompt(memory=...)` is replaced by `pinned=` and `retrieved=`:
context that holds for the conversation belongs in the prefix, context fetched for this
turn must not be, or the cache is invalidated every turn.

**Stability:** breaking for callers passing `memory=` — pass `retrieved=` — and for anything
depending on tool declarations arriving in the order they were given. `Prompt` gains three
fields, so a persisted one from an earlier version no longer validates. Documented in
`docs/run-loop.md`, exercised by `examples/prompt_prefix.py` and `tests/test_prompt.py`.

  *Migration:* Pass `assemble_prompt(retrieved=…)` where you passed `memory=…`, and move context that holds for the whole conversation to `pinned=…` so it joins the cacheable prefix. Sort your own expectations of `Prompt.tools` by name, or read them off `prompt.tools`; give two tools of the same name different names. Reassemble any persisted `Prompt` rather than validating an old one, since `layers`, `fingerprint` and `prefix_tokens` are new required fields.
- **tesserix_adk.observability.attribution**: `Attribution` gains a required `definition` dimension, exported as `adk.definition`. A bill
broken down by agent version cannot tell two runs of one version apart when the version was
edited between them; the definition revision names the exact reviewed artifact that spent
the money. It is required rather than defaulted because attribution is derived from the run
and a silent default is how a dimension quietly stops being populated.

  *Migration:* Pass `definition=` when constructing an `Attribution` by hand — the revision of the `AgentDefinition` the run declared, or `UNKNOWN` where it ran from a bare agent. Attribution derived from a run through `spend_of` needs no change. A dashboard that asserts on `Attribution.unknowns` will now see `"definition"` for runs started from a bare agent, which is the honest reading rather than a regression.

### Added

- **tools.release_notes**: Release notes are assembled from change fragments, conventional commit subjects, the
public API snapshot and the live deprecation records. A change with neither a fragment
nor a readable subject blocks the release, and a breaking change without a migration
note blocks it too.
- Pre-release alpha channel: every merge to `main` publishes a pre-release through the same
gates and the same trusted publisher as a release. Getting one is opt-in — a stable
specifier never resolves a pre-release — and `docs/stability.md` states what each
subpackage promises. Retention is reported by `tools/alpha.py`, and a downstream
integration job compares a consumer's suite against the last stable and the alpha so only
a regression the alpha introduced fails the build.
- Advisory and secret scanning (`.github/workflows/security.yml`), on every pull request and
daily. Advisories are audited against the frozen lock, rated from OSV — an unrated
advisory blocks — and reported with the first fixed version and the blast radius computed
from the lockfile. Secrets are matched by shape across the tree and by `gitleaks` across
the history, with recorded provider traffic additionally checked for personal identifiers.
Suppressions need an owner, a reason and an expiry of at most 90 days, and an expired one
fails the build. Neither job uses a secret, so both run on a fork's pull request.
- A CycloneDX 1.6 bill of materials, `sbom.cdx.json`, attached to every release. It is built
from `uv.lock` inside the release job, so it describes the graph that was published rather
than a later re-resolution, and each component carries its purl, licence, source hash, the
install profile that reaches it, and every wheel with its platform tag and hash.
Development-only packages are excluded. The release notes carry a diff against the previous
release's document, so dependency growth is visible.

A licence policy (`security/licences.toml`) gating the whole graph, on every pull request
and again before anything in the release is irreversible. An undeclared licence blocks, a
choice of licences needs a recorded decision with an owner, and a licence off the allow
list can be accepted for one named package only. See [`docs/security.md`](docs/security.md).
- Signed release artefacts with build provenance. Every artefact is signed keyless by the
workflow's own identity — there is no signing key in existence — and carries an attestation
naming the repository, workflow, commit and build inputs. The bill of materials is attested
by the same run, uploads carry a PEP 740 attestation to PyPI, and the bundles are attached
to the GitHub Release so a mirrored or air-gapped install can verify without reaching
GitHub. Alphas follow the same path, so no channel is unattested.

[`docs/verifying.md`](docs/verifying.md) has the exact commands a consumer runs, what a
correct output looks like, and what to do when the signing identity changes.
- A dependency policy the published requirements are held to
(`tools/dependency_policy.py`, `security/dependencies.toml`), checked on every pull
request. The kit's own builds stay exact through `uv.lock`; what it *publishes* now
carries floors and, with one recorded exception, no upper bound at all — a speculative
cap turns an upgrade the consuming product chose into a resolution error they cannot fix
without forking the kit. Every floor is justified and proved by the `lowest-direct` leg,
every cap names the incompatibility that earned it, the trigger that removes it and the
owner who removes it, and a record for a package nothing depends on any more fails the
job too. Updates arrive weekly and grouped, with majors deliberately outside every group
and advisories fast-tracked past the cadence; a pull request labelled `dependencies`
runs the full matrix as well as the lowest-direct leg, so both ends of every declared
range are proved before it merges. See `docs/dependencies.md`. Dependabot's hosted
updater cannot touch `uv.lock` while `required-version` excludes its bundled uv, so
locked-package updates stay manual on the weekly rota until it catches up; the pin stays,
because dropping it trades reproducibility for a convenience.
- A published security policy with a private reporting channel and a coordinated
disclosure process (`SECURITY.md`, `security/disclosure.toml`, `security/advisories/`,
`tools/disclosure.py`). Reports go to GitHub Security Advisories, are acknowledged inside
a stated per-severity target, are fixed privately, and land as patched releases for every
supported minor together with the advisory — never one before the other. The commitments
are not prose: each report is a record, and `make disclosure` (run daily by the Security
workflow, and by `make check`) fails on an acknowledgement that missed its target, a
supported minor that never got a patch, an advisory published before it was acknowledged,
consumers notified after publication rather than with it, or a flaw disclosed with no fix
and no interim mitigation. Response targets are keyed by severity and nothing else, so a
flaw reachable only through an optional extra is never deprioritised for it. The tables in
`SECURITY.md` are generated from the same records, so the page a reporter relies on cannot
drift from the gate. `docs/threat-model.md` states the guarantees the kit actually
makes — boundary guardrails, redaction before export, bounded spend, fail-closed errors,
no network in the unit path — the assumption behind each, and, at length, what the kit
does **not** defend against.
- An admission gate for third-party dependencies (`tools/admissions.py`,
`security/admissions/`, `security/inventory.toml`), run on every pull request by the
`dependency admissions` job and by `make check`. Every requirement the kit publishes now
carries a decision record naming what breaks without it, the alternative that was
rejected, its maintenance and licence position, how many packages it drags in, whether it
needs a compiler, and the date the approval stops being current — so a dependency that
goes unmaintained cannot stay approved on the strength of a decision made when it was
healthy. All eight existing requirements have retrospective records. The resolved graph
itself is committed: `security/inventory.toml` lists the 51 packages a consumer can end
up with and the profiles that reach each, and the gate fails when the lock disagrees, so
a version bump that quietly adds a transitive package becomes a line in the diff instead
of something nobody sees. The preference order — standard library, then an existing
dependency, then an optional extra, then vendoring, and a base requirement last — is
documented in `docs/dependencies.md` and partly enforced: an integration SDK in the base
install fails the gate by name. Development-only packages are excluded throughout; they
are never in a consumer's resolution.
- The core primitives every other layer speaks: `Agent`, `Message` with `TextPart` and
`BinaryPart` content, `ToolCall`, `Usage`, `Run` with an explicit `RunState` transition
table, `TenantContext` / `RunContext`, and a typed error hierarchy under `AdkError` —
`CapabilityError`, `ProviderError`, `ProviderTimeoutError`, `SchemaViolationError`,
`ToolExecutionError`, `GuardrailViolationError`, `BudgetExceededError`, `CancelledError`
and `MaxIterationsError`, each carrying the run and tenant it happened in. Everything is
frozen, forbids unknown fields and round-trips through JSON, so a run checkpointed by one
process rehydrates in another and no primitive can hold a client or a socket. `Usage`
treats an unknown cost as unknown rather than zero and refuses to total two currencies;
`Run.transition_to` returns a new run and refuses a move the table does not declare
legal, naming the legal set. `tesserix_adk.testing` now raises the kit's own
`BudgetExceededError` rather than a lookalike of the same name. Documented in
`docs/primitives.md`, exercised by `examples/typed_primitives.py`, and docstring examples
are executed in CI.
- `AgentRunner`, the run loop from prompt assembly to exactly one terminal state. Prompt
assembly (`assemble_prompt`, `Prompt`, `ToolDeclaration`, `wrap_untrusted`) is
deterministic and ordered — instructions, memory, history, input — with everything the
agent did not author fenced as untrusted data, and a `Prompt.version` digest of the
cacheable prefix landing on `Run.prompt_version`. The loop dispatches tools against the
agent's allowlist, feeds results back as data, records every step on `Run.events` with its
`Usage`, and always returns the run: a provider error, guardrail refusal, schema
violation, budget ceiling, iteration cap or empty response is a terminal state, not an
escaped exception. Tool failures are wrapped in `ToolExecutionError` and either surfaced
to the model or made terminal per the agent's new `on_tool_error` policy; no result is
ever invented. An agent declaring a guardrail, budget or registry the runner was not given
is refused before the run starts. `tesserix_adk.testing` gains `ScriptedProvider`,
`FakeToolRegistry` and `FakeGuardrail` so the whole loop runs with no network. Documented
in `docs/run-loop.md`, exercised by `examples/run_loop.py`.
- Cancellation and deadlines that stop in-flight work. `runtime` gains `CancellationToken`
and `Deadline`; `core` gains `DeadlineConfig`, `Agent.deadlines` and
`Agent.idempotent_tools`; `AgentRunner` takes `deadlines`, and `run`/`run_sync` take
`cancellation` and `deadline`. A deadline is an instant rather than a duration, so it
survives being passed down and narrows but never extends. Nothing is bounded by default —
a ceiling the kit invented would kill good runs on slow hardware — and a zero ceiling is
refused. Each model call, guardrail check and tool call is raced against the token and the
deadline using the injected clock; uncooperative work is cancelled, given a grace window,
then dropped with `work_orphaned` recorded rather than waited for. A tool stopped after
dispatch records `tool_indeterminate`, because whether its effect landed cannot be known,
unless it is declared in `Agent.idempotent_tools`. `tesserix_adk.testing` gains
`StallingProvider` and a manual-advance `FakeClock`, so timeout tests are deterministic.
Documented in `docs/run-loop.md`, exercised by `examples/cancellation.py`.
- Retry with full jitter and an explicit retryability policy. `core` gains `RetryConfig`,
`RETRYABLE_STATUS`, `AdkError.retryable`, `ProviderError.status`/`retry_after` and
`Agent.retry`; `runtime` gains `RetryPlan`; `AgentRunner` takes `retry` and `jitter`, and
`RunEventKind` gains `attempt_failed`. Nothing is retried by default — a retry is a second
charge on someone's account. Retryability is a property of the error: timeouts and
transient statuses are faults worth a second attempt, while a rejected request, a guardrail
refusal, a budget ceiling and a schema violation are answers. Delays are drawn from the
full window so a fleet does not retry a blip in unison, from an injected `Random` a test
can seed. A `Retry-After` is believed over the computed backoff but refused beyond
`max_retry_after_seconds`, a backoff that would overrun the deadline is not taken, and
tools are retried only where `Agent.idempotent_tools` declares them safe to repeat.
Documented in `docs/run-loop.md`, exercised by `examples/retry.py`.
- Caps on the shape of a run, enforced in the loop. `core` gains `LoopConfig`, `Agent.loop`,
`RunContext.depth`, `Run.depth`, the `loop_limit_exceeded` state, the `fan_out_refused` /
`repeat_detected` / `depth_exceeded` events and the `LoopLimitError` hierarchy
(`RecursionLimitError`, `FanOutLimitError`, `RepeatedCallError`, and `MaxIterationsError`
re-parented under it); `AgentRunner` takes `loop`, and `run`/`run_sync` take `parent`.
Unlike deadlines and retries, loop shape is bounded by default: a cap only ever stops a run
that has stopped making progress. An agent's own `LoopConfig` narrows the runner's and
never widens it. A turn that would break a cap is refused entire before any dispatch, since
half a fan-out is a set of side effects nobody chose, and depth is checked before a prompt
is assembled so a too-deep run costs nothing. `BudgetConfig.max_tool_calls_per_run` moves
to `LoopConfig`, where it is enforced. Documented in `docs/run-loop.md`, exercised by
`examples/loops.py`.
- Lifecycle hooks, so policy attaches to the loop instead of being remembered at a call site.
`core` gains `HookPoint` (seven points, from `before_prompt_assembly` to `on_terminal`),
`HookAction`, `HookDecision`, `HookSubject`, the `Hook` and `ApprovalGate` protocols,
`HookChain`, `resolve_hooks`, `ApprovalRecord` / `ApprovalDecision`,
`Agent.approval_required_tools`, `DeadlineConfig.hook_seconds`, the `hook_rewrite` /
`hook_refusal` / `approval_required` / `approval_granted` / `approval_denied` events and
the `HookRegistrationError`, `HookEvaluationError`, `HookRefusedError`,
`ApprovalDeniedError` and `ApprovalExpiredError` types; `AgentRunner` takes `hooks`,
`approvals` and `approval_ttl_seconds`. A hook returns a decision, never a mutation, and is
handed facts rather than handles, so it cannot widen a tenant scope, disable another hook
or raise a cap. The most restrictive decision wins and ties go to the first declared, so a
chain resolves the same way on every process. Hooks fail closed — one that raises or
outruns `hook_seconds` stops the run, because a check that did not run is not a check that
passed — except at `on_terminal`, where the run is already over and a failure is recorded
instead. The chain is sealed when a runner takes it, so a hook cannot register a permissive
one behind itself. Rewrites are logged as digests rather than content, so a replay can
prove the same prompt was assembled without the redacted text living on in the log. An
approval record carries a digest of the arguments and never the arguments, and a decision
is honoured only if it echoes the record's id and lands inside `approval_ttl_seconds`.
Documented in `docs/run-loop.md`, exercised by `examples/hooks.py`.
- Determinism and offline replay. `core` gains the `IdFactory` protocol; `AgentRunner` takes
`ids`, so the last ambient source in the loop joins the clock and the jitter as something a
test injects. `runtime` gains `RunFingerprint`, `fingerprint_of` and `canonical_digest` — a
canonical summary of the prompt, tool schemas, model, output schema and hook chain, which
names the field that moved rather than reporting a bare mismatch. `testing` gains
`Cassette`, `Interaction`, `RecordedError`, `RecordingProvider`, `ReplayingProvider`,
`SequentialIds`, `assert_same_run`, `redacted` and the `CassetteMissError` /
`CassetteVersionError` types. A replay serves what was recorded or fails saying which field
diverged; there is no live provider behind it to fall through to, because reusing the
nearest response is a green test asserting nothing. Recorded failures replay with their
retries, so the recovery path is exercised rather than assumed. A cassette keeps digests of
the request and never its content, redacts credential-shaped keys and values, and is
refused when it was recorded against another provider, version or format. Documented in
`docs/determinism.md`, exercised by `examples/determinism.py`.
- Strict validation at every boundary. `core` gains `AdkModel` — frozen, `strict=True`,
`extra="forbid"` — and every model in the kit now derives from it, so a string is never
quietly read as a number and an undeclared field is an error rather than a passenger.
`validated`, `parsed_from_strings`, `telemetry_dump` and the `Sensitive` marker come with
it. A rejection raises `SchemaViolationError` carrying the model, every failing path at
once (`content.0.binary.media_type`, so a list index and a union member are both in the
location), the reason per path and the raw payload. Extras stay possible where they are
declared — `Usage.extras`, `Message.metadata` — and are refused where they are loose, so
forbidding them never blocks a provider from evolving. `Sensitive` fields drop out of
`telemetry_dump` and `SecretStr` is masked, while `model_dump_json` keeps both: a run
rehydrated without its credentials rehydrates broken. `BinaryPart.data` is marked this way.
Environment values are parsed once at the one edge that has no types, so
`TESSERIX_ADK_PROVIDER__REQUEST_TIMEOUT` still takes `45` or `PT45S` under strict
validation. Documented in `docs/models.md`, exercised by `examples/models.py`.
- JSON Schema derived from the Python type. `core` gains `schema_for`, which turns a pydantic
model, a dataclass, a `TypedDict` or an annotated callable into normalised Draft 2020-12 —
titles dropped, keys ordered — with field descriptions read from `Field(description=...)`
or, failing that, from the Google-style `Args:` block of the docstring, so guidance is
written once where the field is declared. A missing or malformed docstring costs
descriptions, never the schema. `schema_hash` digests the result: key order does not change
it and any change of shape does, so a renamed field misses a cassette recorded against the
old shape rather than replaying it. Provider differences sit behind the `SchemaDialect`
protocol — `JSON_SCHEMA` (default), `STRICT_SUBSET` (every object closed) and `INLINE_REFS`
(no `$ref` at all) ship with the kit, and any value with a `name`, a `forbidden` keyword set
and an `adapt` is a dialect. Nothing is silently downgraded: a keyword the dialect forbids,
or a recursive type under an inlining dialect, raises `CapabilityError` naming the dialect.
Anything that cannot be described faithfully raises the new `SchemaGenerationError` where
the type is declared — an unannotated or variadic parameter, `Any` in a required position at
any depth, a type pydantic cannot render, or a schema past `max_bytes`, which is refused
whole rather than truncated into a different type. Documented in `docs/schemas.md`,
exercised by `examples/schemas.py`.
- Structured output by default. An `Agent` now declares exactly one of `output_type` and the
new `free_text`; declaring neither is refused where the agent is built, so an answer whose
shape nobody declared is a configuration error rather than a string the caller parses by
guessing. Where a type is declared, `runtime` derives the schema through `schema_for` in the
closed dialect, sends it as `ModelRequest.output_schema` alongside the new
`output_schema_hash`, and folds both into `Prompt.version` — a changed answer type is a
changed prompt. A provider that exposes a truthy `supports_structured_output` enforces the
schema itself; one that does not is given the schema in the prompt instead and its answer is
validated identically, because an undeclared capability is treated as absent. The answer is
validated before the run can reach `completed`: an enclosing code fence is stripped
explicitly and recorded as the new `output_unwrapped` event, prose around JSON is never
scraped, and truncation mid-object is a violation rather than something to repair by
guessing closing braces. `OutputContract` and `unwrap_fenced` are exported from `runtime`;
a violation raises `SchemaViolationError` carrying the raw output, every failing dotted
path, the refusing type and the schema hash, which the loop records before ending the run
`failed`. Content echoed into the next turn of a structured run is wrapped as untrusted
data, so an instruction that arrived inside a field cannot become the next turn's prompt.
Documented in `docs/structured-output.md`, exercised by `examples/structured_output.py`.

**Stability:** breaking. An `Agent` that declared neither `output_type` nor `free_text` was
previously accepted and answered in prose; it must now say so with `free_text=True`.
`assemble_prompt` gains an `output` keyword.
- Bounded validation repair, with the failure itself fed back. `core` gains `RepairConfig`
and `Agent` gains `repair`, undeclared by default: an answer that fails validation is still
terminal unless a budget was asked for, because a further attempt is a further charge on
someone's account. Where one is declared, the run loop sends the violation back through
`OutputContract.repair_prompt` — every failing dotted path with what was wrong with it,
plus the schema, and nothing else. No value is supplied for a failing field, no default is
filled, no field is dropped and nothing is cast: a prompt that says what the answer should
be is coercion with extra steps. A repair attempt is an ordinary model call, so its tokens
land on `run.usage`, it is recorded against the budget policy and it is bounded by the run
deadline and the iteration cap — repair can never spend past a ceiling. Attempts are
recorded as the new `repair_requested` event naming the type, the failing fields and which
attempt of how many, so repair rate is measurable per agent and prompt version. An answer
that comes back with the identical failure after being told what it was stops the run at
that point with the new `repair_abandoned` event and a configuration error: a constraint
that cannot be satisfied as instructed is a defect in the declaration, not a budget to
spend proving it. Running out of attempts fails the run carrying the last violation, never
a best-effort object. Documented in `docs/repair.md`, exercised by `examples/repair.py`.

**Stability:** additive. `Agent.repair` defaults to `None`, which is the behaviour every
existing agent already had. `RunEventKind` gains two members, which a consumer matching
exhaustively over it will see.
- A conformance gate for the typing guarantee. `mypy --strict` proves the code the checker
can see; it cannot prove that an exported symbol is annotated at all, that an `Any` in a
public signature was a decision, or that a `# type: ignore` was reviewed by anyone. Those
three are how a library's typing claim erodes, one plausible exception per release, so each
one now lives in `typing-policy.toml` with a reason, an owner and a review date, and
`make typing-gate` fails without it. The gate fails in both directions: an escape the
policy does not list, and an entry the code no longer contains — a record that outlives its
code is how an inventory stops describing anything. An entry whose owner the policy does
not recognise is flagged for reassignment rather than inherited by whoever touched the file
next, and an entry past its review date fails, because an exception nobody revisits is a
permanent one. An `Any` entry names its kind, and only three are accepted: `json` for a
document that is heterogeneous by definition, `variadic` for a sink that forwards what it
is handed, and `provisional` for a placeholder, which must name the story that removes it.
Entries are keyed by where a symbol is defined rather than where it is exported, so one
decision is reviewed once. At the third-party boundary `disallow_any_unimported` is on: a
dependency that ships no stubs, or drops them in an upgrade, fails at the import rather
than widening a public signature, and the two settings that would readmit an SDK's `Any`
wholesale are forbidden by test. The checker itself is pinned in both the dev group and the
policy, asserted equal, so a mypy that tightens a rule does so on a day someone chose. The
gate runs in CI, in `make check` and as a pre-commit hook, from the same command in each.

**Stability:** additive — a repository gate with no runtime or public API surface. It
changes what CI accepts, not what the kit exports. Documented in `docs/typing.md`.
- **tesserix_adk.runtime.progress, tesserix_adk.runtime.AgentRunner.stream**: A run can now be watched while it happens, as typed events rather than as text chunks.
`AgentRunner.stream` returns a `RunStream` that drives the same run `run` drives — same
loop, same guardrails, same record — and yields a discriminated `ProgressEvent` union:
`RunStarted`, `IterationStarted`, `AnswerDelta`, `StructuredDelta`, the tool-call
lifecycle, `GuardrailDecision`, `ApprovalRequired`, `UsageUpdated` and the three terminal
variants. `stream.run` is the finished record once the stream is drained.

Three properties hold whatever the run does. Exactly one terminal event is emitted and it
is last, derived from the finished run rather than from inside the loop, so a stream that
drops mid-answer fails the run instead of reading as a short one. Every event carries its
`run_id` and a gapless `sequence` from zero, so a consumer can tell a slow stream from a
lossy one — `SequenceCheck` counts what was missed and rejects a late or duplicate event
rather than reordering it into place. And tool arguments are scrubbed in the runtime,
before emission, because a transport that redacts has already handed the value to whatever
it logs to.

`decode_progress` returns `None` for a variant this version has never heard of, so adding
one stays a minor release; a known variant that will not parse raises instead. Where the
provider cannot stream, or nobody is watching, the answer arrives as a single delta and the
sequence is otherwise identical. Documented in `docs/run-progress.md`, exercised by
`examples/run_progress.py`.
- **tesserix_adk.runtime.Provisional, tesserix_adk.runtime.RunStream**: `RunStream` is now awaitable, an async context manager, and carries what has arrived so far
as a `Provisional`. Three consumption patterns fall out of one object: iterate then await
for progress plus the authoritative record, await alone for the answer with no progress,
and iterate then leave once you have seen enough. Awaiting the same stream from two places
drives the run once and hands both the same `Run`.

`Provisional[OutputT]` is what a consumer holds mid-stream, and the type checker refuses it
everywhere an `OutputT` is required — half a JSON object is shaped exactly like a whole one,
so the distinction cannot be left to a naming convention. `snapshot()` returns a plain
mapping, and `None` while the object is still half-arrived, because filling in the missing
half would be inventing content the model never sent. Only the run's own `output` is
schema-validated, and only once the run reached a terminal event.

Leaving the context manager cancels a run nobody is reading any more, through the same
cancellation path a caller's own token uses; `stream.run` is then the cancelled record. A
consumer that stops reading has stopped paying attention, not stopped paying. An exception
in the loop body takes the same exit and still propagates as the consumer's own. Awaiting an
abandoned stream raises `StreamInterruptedError` carrying what arrived, rather than
promoting accumulated partial content to a result. Documented in `docs/run-progress.md`,
exercised by `examples/stream_consumption.py`.
- **tesserix_adk.adapters.RunBroker, tesserix_adk.adapters.sse_events, tesserix_adk.adapters.WebSocketBridge, tesserix_adk.runtime.RunStream**: A run reaches a browser over SSE or a websocket without each product writing the bridge
again. `RunBroker` owns the run rather than the connection, because a reconnect is a second
reader of a run already in flight; the run starts when the first transport attaches and is
driven once however many attach after it. `sse_events` frames each event with its own kind
and its sequence as the SSE id, so a browser dispatches by event type and sends the id back
as `Last-Event-ID`. `WebSocketBridge` sends identical payloads and reads a control channel
for cancellation and approval decisions.

A reconnecting client presents its last sequence and receives the events it missed, or a
`StreamGap` naming how many are gone and where the stream resumes, because silently closing
the gap is how a UI ends up showing a run that never happened. A peer that vanishes without
a close frame cancels the run: a run nobody watches still calls providers and still bills.
An unknown control message is ignored rather than fatal, so a newer client cannot take an
older server down.

The boundary fails closed. `subscribe`, `cancel`, `run` and `serve` authorise the tenant
before anything is framed, and an unknown run id gives the same refusal as another tenant's,
since which ids exist is itself tenant information. Every outgoing event is re-scrubbed —
it may have arrived from a queue this code cannot see — and one above the payload limit
becomes a `PayloadElided` reference rather than truncated into invalid JSON.

`RunStream` gains `run_id`, which the broker keys on before the run has produced anything to
read it from, and abandoning a stream nobody ever read now cancels it rather than leaving a
run to start later with no reader. Documented in `docs/transports.md`, exercised by
`examples/transports.py`.
- **tesserix_adk.runtime.Backpressure, tesserix_adk.runtime.Pressure, tesserix_adk.runtime.AgentRunner.stream, tesserix_adk.runtime.RunStream.pressure**: A run stream's buffer is bounded. `stream(..., backpressure=Backpressure(...))` sets how many
events and how many bytes may wait unread and how long a reader may hold them; the defaults
bound a stream nobody configured, which is the case the unbounded buffer used to lose the
process to.

Pressure is answered by merging text, never by waiting and never by dropping. Above either
mark an arriving `AnswerDelta` or `StructuredDelta` is concatenated onto the one already
waiting, and `coalesced` on the event says how many were folded in, so the answer a consumer
renders is still the whole answer and a client measuring stream shape is told rather than
left to infer it. Lifecycle, tool, approval, usage and terminal events are never merged: a
run missing one of those is a run nobody can account for. An event larger than the entire
byte budget is admitted anyway and counted in `oversize`, because dropping it loses a tool
call and growing for it is the unbounded case again.

The run never blocks on its buffer. A queue the run waits on deadlocks exactly the run it
protects — the one whose own tool result feeds the stalled consumer — and makes one slow
client a slow answer for everyone.

A reader that stops reading stops the run. The stall clock runs from the last read and only
while events are waiting, so a quiet run is not a stalled one; past `stall_seconds` the run
is cancelled through the same path a caller's own token uses, and a dead client that never
disconnected stops costing provider spend. Await-only attaches no reader, so events are
numbered and discarded rather than buffered.

`RunStream.pressure` reports `buffered`, `peak`, `coalesced`, `oversize` and `stalled` during
the run rather than only after it, and `Backpressure.shared(total_bytes=…, streams=…)` turns
a process-wide allowance into a per-run budget. Documented in `docs/run-progress.md`,
exercised by `examples/backpressure.py`.
- **tesserix_adk.runtime.ToolCallIndeterminate, tesserix_adk.runtime.RunCancelled, tesserix_adk.adapters.RunBroker.cancel**: Stopping a stream now stops the work, and both sides agree on what happened. `run_cancelled`
carries the reason, the usage the run had accrued when it stopped and `last_sequence`, the
sequence of the last event before it: a run whose spend is knowable only on completion is
unattributable exactly when it did not complete, and a client that cannot tell where the
stream ended cannot tell a stop from a dropped connection.

A stop racing a natural completion gives one outcome. The terminal event is derived from the
state the run's own loop reached, so a stop arriving after that does not rewrite it and a
client never sees both `run_completed` and `run_cancelled`. Ordering is enforced rather than
trusted: an event posted after the run ended is dropped instead of delivered behind the
terminal one. Teardown is idempotent, and the first reason is the one that stands, so a
retrying client sending stop twice gets one cancelled run and one explanation.

A tool caught in flight is reported, not glossed. `ToolCallIndeterminate` says a call was
stopped after dispatch and that whether its effect landed cannot be known — claiming it was
rolled back when nobody rolled it back is the worse answer. A tool the agent named in
`idempotent_tools` is reported failed and safe to retry instead. The stream says exactly what
the run record already said, so there are not two accounts of one call.

`RunBroker.cancel` drives a run nobody ever attached to as far as a cancelled record, because
attribution cannot depend on a client still being there to be told; a token cancelled that
early reaches no provider. It still authorises the tenant first, and an unknown run id is
refused exactly as another tenant's is. Documented in `docs/run-progress.md`, exercised by
`examples/stream_cancellation.py`.
- **tesserix_adk.runtime.Ambient, tesserix_adk.runtime.LoopMonitor, tesserix_adk.runtime.WorkerPool, tesserix_adk.runtime.Workers, tesserix_adk.runtime.carrying, tesserix_adk.runtime.current_ambient, tesserix_adk.runtime.drive, tesserix_adk.runtime.AgentRunner.stream_sync, tesserix_adk.core.RunningLoopError, tesserix_adk.core.EventLoopStalledError, tesserix_adk.core.WorkersBusyError**: Both crossings between the async core and synchronous code are now named, and neither is
improvised at the call site. Going in, `run_sync` and the new `stream_sync` drive the same
run `run` and `stream` drive — one implementation, not a second one that drifts — and
refuse from inside a running loop with `RunningLoopError`, which names the async call to
use instead. Nesting a second loop runs two schedulers over one set of tasks and blocking
on the first deadlocks it against the work it is waiting for; a deadlock says nothing
about which line caused it. The refusal is also a `RuntimeError`, so existing guards
against 'this event loop is already running' keep working, and it is raised before any
coroutine is created, so a refused call leaves nothing un-awaited. `run_sync` now carries
`budget` as `run` does.

Going out, a blocking body has somewhere to go and nowhere to hide. `WorkerPool` runs it
on a bounded set of threads: a body that waits longer than `queue_seconds` is refused with
`WorkersBusyError`, because growing the pool trades a bounded wait for unbounded threads
and fails later, harder, and on someone else's request. A body nobody declared is caught
instead — every tool call is watched by a `LoopMonitor` that measures the loop's own lag,
and work that stopped the loop fails with `EventLoopStalledError` naming the tool. Lag
rather than duration, so a tool that legitimately awaits for a minute is left alone, and a
body that failed on its own terms is reported on its own terms.

Identity crosses with the work. A tool body receives the arguments the model chose, so the
tenant, the run and the caller's switch arrive as an `Ambient` bound for the call rather
than as arguments a caller can forget. Each `pool.call` copies the calling context, so
nothing a body sets crosses back or reaches the next body on that thread and two runs
sharing a worker cannot read each other's tenant. `current_ambient()` returns `None`
outside a run rather than an invented default. A thread cannot be interrupted, so a
cancelled body is told rather than killed: `Ambient.raise_if_cancelled` is how a long body
cooperates, and an abandoned one is reported indeterminate rather than claimed undone.

Documented in `docs/async-and-sync.md`, including the notebook pattern and why
`nest_asyncio` is not supported, and exercised by `examples/sync_surface.py`.
- **tesserix_adk.core.ConcurrencyConfig, tesserix_adk.core.ToolTimedOutError, tesserix_adk.core.Agent.concurrency, tesserix_adk.core.ToolDeclaration.parallel_safe**: A turn that asked for four independent lookups now costs roughly one lookup rather than
four. The calls in a single model response are dispatched together, and what each returns
is merged back in the order the model asked for, so a batch that resolved in whatever
order the network allowed still reads, and replays, as one deterministic transcript.

Firing them all at once is also how one agent turn becomes a rate-limit breach at a
partner, so the batch runs inside declared lanes. `ConcurrencyConfig` names three of them:
`max_concurrent_tools` bounds one turn, `per_tool` bounds one tool across the runner, and
`per_tenant` bounds one tenant across its runs. Per-tool and per-tenant lanes bound a
downstream shared with every other run, so they are the runner's to declare; an agent may
narrow the runner's widths with `Agent.concurrency` and never widen them. Every call takes
the lanes in the same order, so two calls cannot each hold what the other waits for, and a
sub-agent spends the lane its own caller is already standing in rather than queueing
behind itself.

`per_tool_seconds` gives a slow tool its own ceiling instead of the batch's: it fails with
`ToolTimedOutError` against its own call id while its siblings are kept. That is the shape
of every failure here — a tool that raises is reported against the call that caused it, no
placeholder is fabricated for it, and whether the batch survives is the agent's existing
`on_tool_error` declaration rather than a new one. A cancelled batch records what it did
distinctly from what it never started: a call still queued behind a lane is reported as
never dispatched, and one already in flight is reported indeterminate unless the agent
declared the tool idempotent.

A tool whose effect depends on the order it is called in declares
`ToolDeclaration(parallel_safe=False)` and is run alone, with everything the model asked
for before it already resolved. The runtime cannot infer that from a signature, so it is
declared. A blocking tool body remains the registry's to offload; `WorkerPool` is the
supported route and keeps the batch's siblings moving.

Documented in `docs/tool-concurrency.md` and exercised by `examples/tool_concurrency.py`.
- **tesserix_adk.models.ClientPool, tesserix_adk.models.PoolConfig, tesserix_adk.models.PoolMetrics, tesserix_adk.models.ClientKey, tesserix_adk.core.PoolExhaustedError**: Provider connections now outlive the run that opened them. `ClientPool` owns them and
decides what may be shared with what: same provider, endpoint, credential and transport
settings means the same warm client, and anything else means a different one. Providers
take it with `pool=`; a provider given none still owns its client, so nothing about the
single-provider case changes.

The key is the safety argument. It carries a truncated digest of the credential rather
than the credential, because a key is compared, logged and used as a metric label, and a
secret that reaches any of those has leaked — so two tenants against one endpoint can
never be handed each other's connection. The credential is resolved per request, so a
rotation lands without a restart: the next request opens a pool on the new key and the old
client is *retired* rather than closed, staying alive until the requests already on it
have finished.

Waiting for a connection is bounded by `acquire_seconds` and reported as
`PoolExhaustedError`, which is retryable — the endpoint is fine and the process is
over-subscribed. An unbounded wait is how a downstream slowdown becomes a run that queues
past its own deadline. A pool inherited across a `fork` is discarded rather than used,
since its descriptors belong to the parent's event loop, and `PoolMetrics` reports opened,
reused, retired, inherited, exhausted and currently held as one snapshot.

Documented in `docs/connection-pooling.md` and exercised by
`examples/connection_pooling.py`.
- **tesserix_adk.models.BatchingEmbedder, tesserix_adk.models.BatchConfig, tesserix_adk.models.EmbeddingProvider, tesserix_adk.models.EmbeddingLimits, tesserix_adk.models.EmbeddingMetrics, tesserix_adk.models.Vector**: Embedding one text at a time is how indexing a document becomes a few hundred sequential
round trips. `BatchingEmbedder` sits in front of an `EmbeddingProvider` and coalesces
concurrent single-text calls into provider batches; the caller still asks for one text and
still gets one vector back, and writes no batching loop to get there.

Identity is the guarantee that makes coalescing safe. A waiting caller carries the digest
of its own text and is answered by that digest rather than by a position in a list, the
provider's answer is checked for count and width before anyone is given anything, and a
short or wrong-width response is a `ModelResponseError` rather than a padded vector. The
kit never substitutes a zero vector or a neighbour's. Duplicate texts in one batch are
sent once and both callers answered; a cancelled caller drops out without disturbing its
siblings; a failed batch is bisected until the failure is isolated to the one text that
caused it, and everybody else in that batch still gets what they asked for.

Batches are keyed by model, tenant and dimensionality, so two tenants are never in one
batch even for the same model. A batch goes out when it is full, when the next item would
put it past a byte ceiling, when its window expires, or when the embedder closes — a batch
still filling is flushed on the deadline rather than held for a full one. Ceilings come
from `provider.limits(model)`, so a vendor that raises its batch size raises yours;
`BatchConfig` may narrow them and never widen them. `interactive=True` skips the window
entirely, since a query embedding queued behind a bulk flush is latency a person sees.
`EmbeddingMetrics` reports requests, batches, deduplicated, bypassed, isolated and how
each batch was triggered.

Documented in `docs/embedding-batching.md` and exercised by
`examples/embedding_batching.py`.
- **tesserix_adk.models.CachingProvider, tesserix_adk.models.CachePolicy, tesserix_adk.models.CacheKey, tesserix_adk.models.CacheEntry, tesserix_adk.models.CacheOutcome, tesserix_adk.models.CacheMetrics, tesserix_adk.models.CacheStatus, tesserix_adk.models.CacheStore, tesserix_adk.models.Cacheability, tesserix_adk.models.MemoryCacheStore, tesserix_adk.models.MemorySemanticIndex, tesserix_adk.models.SemanticConfig, tesserix_adk.models.SemanticIndex, tesserix_adk.models.not_cacheable, tesserix_adk.adapters.RedisCacheStore, tesserix_adk.adapters.DEFAULT_NAMESPACE**: Every product eventually puts a cache in front of its model calls, and it goes wrong the
same two ways: the key is the user's text, so editing a tool schema serves an answer shaped
for the old one, and the tenant is not in the key, so one customer is served another's
answer. `CachingProvider` is a `ModelProvider` wrapping another, so caching is a change to
where the provider is built and nothing else.

The key is the whole correctness argument: tenant, model, assembled prompt, tool schema
hash, output schema hash, declared parameters, prompt version and model version. A change
in any of them is a miss. The tenant is structural as well as keyed — a `CachingProvider`
is built for one tenant and has no way to ask the store for another's entry.

`CachePolicy` refuses rather than storing what must not be stored: a declared `temperature`
above zero or `n` above one, and anything inside `not_cacheable(...)` — a personalised
memory read, a side-effecting tool's result, an approval-gated answer. Serving a stored
random draw as a fact is fabricating determinism, not caching.

A cold key under concurrent load is one call, with the rest counted as `coalesced`; a
failed call is neither cached nor left wedging the key. A store that cannot be reached is a
slow run rather than a broken one — lookups and writes degrade to a live call and report
`STORE_UNAVAILABLE` — except `forget`, which raises, because erasure that failed silently
is worse than erasure that failed loudly.

The semantic tier is off unless configured, serves only at or above its threshold, only
within the tenant, and only where the entry was indexed by the same embedding model.
`RedisCacheStore` keys `<namespace>:<tenant>:<prompt version>:<model version>:<digest>`, so
a purge is one pattern rather than a scan of every value, and the model's reasoning is
dropped before writing.

Documented in `docs/response-caching.md` and exercised by `examples/response_caching.py`.
- **tesserix_adk.testing.benchmarks.Baseline, tesserix_adk.testing.benchmarks.BenchmarkReport, tesserix_adk.testing.benchmarks.Comparison, tesserix_adk.testing.benchmarks.DEFAULT_FLOORS, tesserix_adk.testing.benchmarks.DEFAULT_LIMITS, tesserix_adk.testing.benchmarks.Measurement, tesserix_adk.testing.benchmarks.Metric, tesserix_adk.testing.benchmarks.Scenario, tesserix_adk.testing.benchmarks.Thresholds, tesserix_adk.testing.benchmarks.Verdict, tesserix_adk.testing.benchmarks.compare, tesserix_adk.testing.benchmarks.load_baseline, tesserix_adk.testing.benchmarks.measure, tesserix_adk.testing.benchmarks.run_suite, tesserix_adk.testing.benchmarks.write_baseline**: A benchmark harness that fails CI on a real regression and says so plainly, without failing
on a runner having a bad afternoon. `measure` runs a `Scenario` over warm-up, rounds and
iterations, drops the slowest round where there are three to spare, and records the run's
own spread beside the numbers. `compare` judges a `Measurement` against a committed
`Baseline` per metric and per interpreter, and `python -m tools.benchmark` — `make bench` —
exits 0 held, 1 regressed, 2 too noisy to say, 3 suite unloadable.

The noise is the design. Where the spread exceeds the ceiling *and* covers the delta, the
verdict is `INCONCLUSIVE` with what a conclusive run would need rather than a coin-toss
pass or fail; CI reports that and merges. Each metric also carries an absolute floor
beneath which no verdict is drawn, because a percentage of a very small number is not a
measurement — two blocks becoming three is not a fifty-percent regression.

The committed baseline gates `tokens` and `peak_bytes`: the metrics that mean the same
thing on another machine. Token overhead has a threshold of zero, since prompt assembly
growing by one token is a change somebody made and every consumer pays for it on every
call. Wall clock and live block counts are measured and printed every run but gate nothing,
which is stated where a maintainer will read it rather than left as a surprise.

Memory is measured apart from the timings — tracing every allocation costs more than most
of what is being measured — with tracing started after the warm-up, a collection before the
count, and the median taken across rounds. `peak_bytes` scales with iterations per round,
so a shortened local run reports it inconclusive instead of judging it against a full-size
baseline.

A check run never writes the baseline: not on success, not on failure, and it does not
create one that was absent. Recording is `make bench-record`, in a commit of its own, and
merges rather than replacing, so re-recording one interpreter leaves the other's numbers
alone.

Six scenarios ship in `benchmarks/suite.py` — single turn, tool turn, streaming, structured
output, embedding batch, run fan-out — over scripted providers and local fakes, so nothing
reaches the network. Documented in `docs/benchmarks.md` and exercised by
`examples/benchmarks.py`.
- **tesserix_adk.models.providers**: Providers for Anthropic, OpenAI and Gemini, behind the one protocol. `AnthropicProvider`,
`OpenAIProvider` and `GeminiProvider` each take a model id as the vendor spells it, read
their capabilities and prices from the model catalogue, and resolve their key on every call
rather than at construction, so a rotated secret is picked up without a restart. The same
agent definition runs on all three with only the model reference changed.

They speak HTTP directly rather than through vendor SDKs. `httpx` is already a dependency,
so each adapter is one request shape and one response shape instead of a second dependency
graph and a second translation — and the traffic is recordable at the HTTP layer, which is
where half of an adapter's behaviour lives. `HttpCassette`, `HttpExchange` and `HttpReplay`
in `tesserix_adk.testing` record and serve exchanges through an `httpx` transport;
`replay.sent` is what the adapter actually put on the wire, which a provider-level
recording cannot see, because by then the translation has already happened. The whole
matrix runs in CI with no network and no keys, and `FakeSecrets` keeps a test out of the
environment.

The differences stay inside the adapters. System prompts go to `system`, a system turn or
`systemInstruction`; structured output goes through a forced tool, `response_format` or
`responseSchema`, never through parsing JSON out of prose; tool results are merged into one
user turn, sent one turn each, or matched back by tool name. Gemini sends no tool-call ids,
so the adapter mints them, and reports `STOP` whether or not it asked for a tool, so the
stop reason is read off the parts — believing it would return a finished turn for a model
that had asked for a tool, and the caller would never run it. `strict` is claimed for an
OpenAI schema only when the schema meets the vendor's strict subset, and a Gemini schema is
pruned of the keywords the vendor rejects.

Streaming is one event model across the three: text, reasoning and tool-call deltas, usage,
and a terminal `StreamEnd` carrying the settled response. A stream that ends before the
model said why it stopped raises `StreamInterruptedError` with the partial text and the
frame count rather than returning the fragment as a whole answer.

**Stability:** additive. Nothing existing changes; `tesserix_adk.models.providers` and the
HTTP cassette are new surface. The vendor endpoints are not versioned by this project, so a
wire-format change upstream is a fix rather than a break here. Documented in
`docs/providers.md`, exercised by `examples/vendor_providers.py`, and proved against the
shared suite in `tests/test_provider_conformance.py` plus a cross-provider run in
`tests/test_provider_matrix.py`.
- **tesserix_adk.models.providers**: `OpenAICompatibleProvider`, so an endpoint you run yourself — vLLM, Ollama, TGI — is
routable, costed and capability-checked like any vendor. Presets `VLLM`, `OLLAMA` and
`TGI` carry each server's deviations from the format it claims to speak. Two arguments
have no default: `base_url`, because there is no host to guess for a service only the
operator has named, and `capabilities`, because the deployment's flags decide them and no
endpoint reports them honestly. The provider is named for the server rather than for
OpenAI, so a call to a box in the cluster is not recorded against the vendor's bill, and
`api_key_variable` is optional — omit it and no `Authorization` header is sent, which is
the in-cluster case.

Four deviations are reconciled rather than passed on, because each is a wrong answer
instead of an error. An error object under a 200, which several compatible servers send
and mean, raises `ProviderError` rather than being assembled into a response. A stop
reason the server omitted is read off the answer, since `unknown` on a turn that asked for
a tool ends a run with the call never made. Tool calls that arrive without ids are given
positional ones, because a result has to match back to something. And usage nobody
reported is estimated from the provider's own token count and marked `Usage.estimated`:
zero tokens reads as a free call, and a call on a GPU somebody pays for is not free. The
cost stays `None`, because the kit does not know what that GPU hour is worth.

`ProviderUnavailableError` is new, raised for a connection that never landed and for
502/503/504 by every HTTP provider. It is a `ProviderError`, so existing handling catches
it unchanged, and always `retryable`; any `Retry-After` the endpoint sent is believed in
preference to a computed backoff, since retrying a model that is still loading as fast as
the policy allows is how it never finishes loading.

Nothing is emulated unless asked for. Where a model cannot enforce a schema, the kit used
to ask for JSON in the prompt and validate the reply; against a small self-hosted model
that is a schema enforced by nobody, so an agent with an `output_type` against an endpoint
that has not declared `structured_output` now raises `CapabilityError` before the run
starts. Providers say which they are through the new `DeclaresEmulation` protocol; one
that says nothing is emulated exactly as before, and `emulates=True` restores it here.

**Stability:** additive. `Usage.estimated` defaults to `False`, `ProviderUnavailableError`
subclasses `ProviderError`, and `DeclaresEmulation` is opt-in, so no existing provider or
consumer changes behaviour. Documented in `docs/providers.md`, exercised by
`examples/self_hosted_provider.py` and `tests/test_provider_compatible.py`.
- **tesserix_adk.core, tesserix_adk.models.providers, tesserix_adk.runtime**: One error taxonomy over every vendor. `RateLimitError`, `AuthenticationError`,
`ContentFilteredError` and `InvalidRequestError` join `ProviderTimeoutError`,
`ProviderUnavailableError` and `ContextWindowExceededError`, and every HTTP adapter now
classifies a failure into them from the vendor's own code and status. A rate limit is
`rate_limit_error` at Anthropic, `rate_limit_exceeded` at OpenAI and `RESOURCE_EXHAUSTED`
at Google; a consumer that branches on those strings has written three error handlers and
will write a fourth for the next endpoint. `RetryPlan` asks the error's `retryable` and
nothing else, so the retry decision follows from the type rather than from a string match
on a body. The default is explicit rather than absent: a code nobody has mapped becomes a
plain `ProviderError` whose retryability follows its status, because guessing that an
unknown failure is transient is how a broken deployment becomes a burst of identical
calls.

A spent quota is distinguished from a rate limit. `insufficient_quota` and a hard billing
limit arrive as `RateLimitError(quota=True)`, which is **not** retryable — a rate clears by
waiting and an allowance clears when somebody pays, so retrying the second is the same
call every time. Every failure carries `provider`, `model`, `request_id`, `status`,
`retry_after` and the vendor's own `details["code"]`, which is what a support ticket is
answered against.

The vendor's free-text message no longer reaches the error. A 400 body quotes the request
that caused it and the request body is the prompt, so a raw body copied into an exception
is prompt content in every log line the exception reaches. The status, the code and the
request id are enough to read the failure without it; an operator entitled to read those
prompts can set `redact_vendor_messages=False` per provider.

Connecting and generating are given separate budgets. `PhaseTimeouts` and `PHASE_DEFAULTS`
document them — 10s connect, 60s read, 30s write, 10s pool — `timeout` moves the read
budget, `connect_timeout` moves the connect budget, and a provider's own are on
`provider.timeouts`. One number for both means either a dead host is waited on for a
minute or a long answer is cut off. Whichever wait ran out is named on the error as
`details["phase"]`, since httpx reports a dead host and a slow model as the same
exception.

`RateLimiter` shapes calls before they are sent. A key's allowance belongs to the key, not
to the process holding it: twenty concurrent runs sharing one key each get a twentieth of
the limit, find that out as 429s, and retry into the same wall together. One limiter
passed to every provider on that key meters both requests and tokens, refills
continuously rather than on the minute, and spends in arrival order; `burst` controls how
much of a minute may go out at once. A caller cancelled while waiting spends nothing, a
limit that is not above zero is refused at construction, and a request larger than the
whole token allowance is refused rather than waited on.

**Stability:** additive. Every new error subclasses `ProviderError`, so a handler catching
that catches all of them unchanged, and the new provider arguments all default to today's
behaviour except the redaction of vendor messages, which is on by default and is the one
intended change: `details["body"]` and `details["message"]` are no longer populated unless
asked for. Documented in `docs/resilience.md`, exercised by `examples/resilience.py`,
`tests/test_provider_failures.py` and `tests/test_rate_limiter.py`.
- **tesserix_adk.core, tesserix_adk.models.routing, tesserix_adk.runtime**: An agent can name the job instead of the model. `Agent.task_class` says what kind of work
a step is — `CHEAP`, `SMART`, `REASONING`, or any `TaskClass` a deployment invents — and
`Agent.requires` says what it needs of whatever answers. Where a class resolves to is
configuration an operator owns, so retuning a deployment is an edit to a file rather than
a code change in every consumer that wrote a model id at a call site.

`RoutingTable` is that file: rules of `task_class` plus optional `tenant` and `agent`, each
offering candidates in preference order. The narrowest matching rule wins and, within it,
the first candidate meeting the requirements answers — that written order is the operator's
preference and reordering it silently is choosing on their behalf. `routing_table()` reads
TOML from a path or from `ADK_ROUTING_TABLE`; nothing is discovered by convention, because
a deployment routing by a file nobody named is one where the answer to "which models is
this billing" lives on somebody's laptop. `TableRouter` resolves against it and takes
optional per-tenant `entitlements`, where a tenant with no entry is unrestricted and a
tenant with an empty entry is entitled to nothing.

A table that would fail on a later request fails at construction instead: an unreadable
version, no rules, two rules at one scope, a rule with no candidates, a candidate declaring
no capabilities, or a candidate the model catalogue no longer lists. That last check is
skipped for providers the catalogue says nothing about, since absence of a card for a
self-hosted endpoint is not evidence the model is gone.

Nothing falls back. A class with no rule, a rule whose candidates cannot do the work, and a
pin nothing knows all raise `NoEligibleModelError`, which carries `task_class`, the
`unsatisfied` requirements and every `rejected` candidate with the reason it was passed
over. A model that cannot do the job is not a cheaper way to do it, and an answer produced
by a model the run record does not name is worse than no answer. Requirements are checked
before the choice rather than after it: a router that picks first and validates second has
already recorded the wrong model against the run.

`AgentRunner` takes a `router` and a `providers` map of vendor name to provider, resolves
once before the first call, and records a `MODEL_ROUTED` event naming the chosen model and
the rule that chose it. `runner.reload(router)` swaps the table for the next run; a run
already in flight keeps the model it resolved, since resolving twice would make one record
describe two runs.

**Stability:** additive. Routing is opt-in — an agent naming `model` outright keeps the
runner's single provider and records no routing event, so every existing runner behaves
exactly as before. Documented in `docs/routing.md`, exercised by `examples/routing.py`,
`tests/test_routing.py` and `tests/test_routed_runs.py`.
- **tesserix_adk.core, tesserix_adk.runtime**: A rate-limited vendor no longer ends a run another vendor could finish. `FallbackChain` is
the eligible candidates of the routing rule that already matched, chosen one first — there
is no separate fallback configuration, because an order invented apart from the routing
order is a second opinion on the same question and the two drift. Every link has already
passed the run's capability floor, so falling down the chain cannot quietly lose structured
output or tool calling, and a pinned model has a chain of one.

The chain moves only once that vendor's own `RetryConfig` is spent, and only for failures
another vendor could answer differently: `fallback_eligible()` admits rate limits (including
a spent quota, which waiting will never clear), overloads and timeouts. A bad key, an invalid
request, a filtered prompt, a capability mismatch, an exhausted budget and anything unmapped
are terminal — opening a second bill on a failure nobody has classified is how one broken
deployment becomes two. A stream that already emitted is never restarted transparently; the
partial text stays on `StreamInterruptedError` and restarting is the caller's explicit choice.

Falling back replays recorded tool results rather than re-invoking tools, which is sound only
where being invoked once was the whole story. A run that has already called a tool not listed
in `Agent.idempotent_tools` fails closed with `FallbackUnsafeError` naming the tool, and the
tool is not called again. When every candidate refuses, `FallbackExhaustedError` carries all
of them with their reasons rather than only the last.

Nothing is silent and nothing is evaded. Each attempt records `ATTEMPT_FAILED` with its model
and error class, each move records `MODEL_FELL_BACK`, and the run's `model` is the model that
actually answered. Failed attempts still reserve against the budget, cancellation is checked
between candidates, a candidate whose window cannot hold the assembled prompt or whose vendor
this runner was never given is skipped with the reason recorded, and a candidate already tried
is never returned to.

**Stability:** additive. A run with one candidate, a pin, or no router behaves exactly as
before, failing with the same error it always did. Documented in `docs/fallback.md`,
exercised by `examples/fallback.py` and `tests/test_fallback.py`.
- **tesserix_adk.core, tesserix_adk.models**: One honest number for what a run spent, whichever vendor answered it. `Usage` is what was
consumed and `Cost` is what that came to, kept apart because a price list decides the second
and only the vendor knows the first. `Usage` now carries `cache_write_tokens`,
`reasoning_tokens` and `image_units` as fields rather than vendor-specific `extras`, and
every adapter normalises into them: Anthropic's cache creation, OpenAI's reasoning (which it
reports inside the completion total, so the adapter splits it out) and Gemini's thoughts all
land in the same column. `cached_tokens` remains part of `input_tokens` because those tokens
were sent; `reasoning_tokens` sits beside `output_tokens` rather than inside it.

`Cost` is `Decimal` throughout with an ISO 4217 currency, and its input, output, cache-read,
cache-write, reasoning and image components stay separate and unrounded until `quantised()`
— a cache saving folded into a total is a saving nobody can question, and floating point
over a run's worth of millionths of a dollar disagrees with the invoice. Totals keep the
weaker of two confidences and refuse to add two currencies. `CountSource` records who
counted: `PROVIDER`, the model's own tokeniser, or characters over a constant, with
`weaker_source` making one guessed step enough to mark the whole total a guess.

Prices are dated data, not constants. A `PriceCard` carries `effective_from`, the request
shape it answers for, and a `Rate`; `PriceList.rate_for` picks the narrowest card the request
clears and, among those, the latest already in force. A price change is a new card and never
an edit, because overwriting one rewrites what last week's runs cost, and two cards for one
shape on one day is refused. `price_list()` reads a TOML file named by `ADK_PRICE_LIST` or by
path — never by convention — and `overridden_by` lays negotiated rates over the shipped ones
for the models they name. An unreadable or wrong-shaped file raises `ConfigurationError`
rather than falling quietly back to list price.

A model no card covers warns `UnknownPricing` and reports `Cost.unknown()`: zero components
at `UNKNOWN` confidence, because there is nothing to put in them and not because the call was
free. Tokens burned on failed and retried attempts are on the ledger too — a vendor that read
a prompt and then rate-limited still charged for reading it — carried on each `ATTEMPT_FAILED`
event and summed into the run, so a run that never got an answer still says what it spent.

**Stability:** breaking. `Usage.cost` is a `Cost` rather than a `float`, `Usage.currency` is
gone (it lives on `Cost`), `Usage.estimated` is now derived from `source`, and the reasoning
and cache-creation counts that vendors previously left in `extras` have named fields.
Documented in `docs/cost.md`, exercised by `examples/cost.py` and `tests/test_cost.py`.
- **tesserix_adk.core, tesserix_adk.runtime, tesserix_adk.testing**: A ceiling said once and honoured everywhere, and no way to build a runtime without one.
`BudgetLimits` is the vocabulary — money as `Decimal` with a currency, input and output
tokens, model calls, tool calls, iterations and wall-clock seconds — and every field is
optional to write while none is optional in effect: `filled()` replaces what was left unsaid
with `BudgetLimits.conservative()`, because forgetting is not a way to opt out. Saying there
is no ceiling takes `BudgetLimits.unbounded()`, a sentence a reviewer can see. A ceiling of
zero is refused as a field somebody meant to disable rather than a limit somebody meant to
set.

Limits attach to a run, an agent, a tenant or a tenant window, and `most_restrictive`
resolves them per dimension with the winning scope recorded: a run asking for 5.00 under a
tenant capped at 1.00 gets 1.00, and `resolved.sources["max_cost"]` says who said so — a
ceiling nobody can attribute is a ceiling nobody can raise. Two scopes of one kind, and two
currencies capping money, are `ConfigurationError` at resolution time rather than a
conversion at a rate nobody agreed to.

`RunBudget` enforces it before the spend: `reserve` holds an estimate so a call cannot start
that the ceiling could not cover, `record` replaces the hold with what was consumed, and
`check` answers without raising. `BudgetExceededError` now names the limit breached, its
scope, the ceiling, the consumed amount and the remaining one. `BudgetDecision.priced`
reports whether every call so far had a price, because a money ceiling checked against calls
nobody could price is not enforcement. `child()` hands a sub-agent the parent's remaining
allowance on the parent's ledger — a fresh allowance is a way to spend one ceiling twice.

Ceilings wider than one run live behind the `TenantLedger` protocol, whose `consume` adds
and returns the new total in one call so two concurrent runs cannot both see the whole
allowance as free. The window is pinned when the run starts, so a run beginning at 10:59 is
not a way to spend two hours of one hour's budget. A ledger that cannot be reached raises
`BudgetUnavailableError` and fails the run closed, distinct from exceeding a budget because
nobody knows whether this run would have; proceeding without it is
`LedgerFailure.PROCEED`, an explicit choice recorded on the run.

A runner given no policy is not a runner without a ceiling: each run gets a `RunBudget`
resolved from the agent's limits and the conservative defaults, and `run.budget` carries the
resolution and its sources. Removing the ceiling takes `UnlimitedBudget(reason=...)`, which
will not be built without a stated reason and puts that reason on every run it governed.

**Stability:** breaking. `BudgetConfig` is replaced by `BudgetLimits` on `Agent.budget` and
`AdkConfig.budget` (`budget.max_tokens_per_run` becomes `budget.max_input_tokens`, and
`max_cost_usd_per_run` becomes `max_cost` with a `currency`); `BudgetPolicy.record` takes a
`Usage` and typed counts rather than an integer, and the protocol gains `resolved`,
`limits`, `child` and `check`; an agent declaring a budget with no policy on the runner is
no longer a `ConfigurationError`, because the default policy applies. Documented in
`docs/budget.md`, exercised by `examples/budget.py` and `tests/test_budget.py`.
- **tesserix_adk.core, tesserix_adk.runtime**: The ceiling is now enforced where the spend happens rather than at the boundaries. The run
loop reserves before a model call, settles against what came back, charges every tool call
before dispatch and re-checks every dimension at the top of each iteration — so a ceiling can
be reached on the fortieth turn of a loop nobody expected to loop, and reaching it ends the
run in `BUDGET_EXHAUSTED` with `run.output` empty. The work that did happen stays on the run
as events; none of it is dressed up as a finished answer.

Nothing is squeezed under the ceiling. The prompt is not truncated, tools are not dropped and
the model is not downgraded to make a call fit — a degraded answer presented as a real one is
worse than no answer, and cheaper only in money. A ceiling that cannot cover the next call
means the call is not made.

Resilience is charged like anything else. A failed attempt burns the kit's own estimate of
what the vendor read and that estimate goes on the run and against the ceiling, so retries and
fallback cannot be used to spend past a limit. A cancelled call settles what it had already
sent and still ends the run in `CANCELLED`: cancellation and a breach racing each other have
one deterministic outcome, and the ledger stays reconcilable either way.

A tool with a side effect that ran before the run ended is named, not undone. Each
non-idempotent tool that executed on a run that did not complete gets a
`COMPENSATION_REQUIRED` event saying what ran and why it is now outstanding; the runtime never
re-dispatches while unwinding, because that is how one side effect becomes two. Tools listed
in `Agent.idempotent_tools` are left alone.

`BudgetDecision.overshoot` reports how far past a ceiling a call landed when its actual cost
exceeded the estimate that reserved for it, and the terminating event carries it. Rounding it
away is how a ceiling comes to mean something other than what it says. `BudgetDecision.as_error`
turns a decision into the typed refusal it amounts to, leaving the raise to the caller.

`budgeted_stream` holds a stream to the same ceiling: each running total the vendor reports is
charged as an increment, so a stream repeating a total is billed once, and passing the ceiling
mid-stream raises `BudgetExceededError` rather than letting the stream end quietly — a consumer
that sees a stream simply stop reads it as a finished answer. The source stream is closed on
the way out, so an aborted stream does not leave the vendor sending tokens charged to nobody.

New `RunEventKind` members: `BUDGET_EXCEEDED`, `COMPENSATION_REQUIRED`, `FAN_OUT_REFUSED`.

**Stability:** additive. Documented in `docs/budget.md`, exercised by
`examples/budget_enforcement.py` and `tests/test_budget_enforcement.py`.
- **tesserix_adk.observability, tesserix_adk.core, tesserix_adk.testing**: A finished run can now say who spent what. `spend_of(run)` returns one `SpendRecord` per
metered step, each carrying the tenant, user, agent, agent version, model, prompt version,
task class and run id that were true at the moment the money went out. Nothing is supplied by
the caller — a consumer that has to remember to tag its own spend forgets on one path and
mis-tags it on another, and the numbers are then wrong in a way nobody can see.

Spend that would otherwise be lost is kept. A run that fell back bills each step against the
model that actually answered it, not both against the first. A failed attempt that burned
tokens is a record rather than a gap, because the vendor invoices for what it read whether or
not anything came back. A run acting on another tenant's request bills the tenant it ran as,
never the requester. What the run could not say resolves to an explicit `unknown` and is named
by `Attribution.unknowns` rather than left blank.

`totals_by(records, "tenant", "agent")` groups by any fields of `Attribution`, keyed by a
tuple in the order asked for so a caller never branches on how many dimensions it requested.
A name that is not a dimension is refused rather than collapsed into one bucket, and a group
spanning two currencies raises instead of summing to a number true in neither.
`Totals.estimated` marks a group whose rows were counted rather than metered, which is what
separates spend that will appear on a vendor invoice from spend that will not.

`record_spend(run, tracer=…, meter=…)` exports it. Nothing is wired into the run loop: emission
reads a finished run, so a collector outage cannot reach into the run that produced the
numbers. Spans carry the full attribute set under one `adk.` prefix. Counters — `adk.cost`,
`adk.tokens`, `adk.calls` — are emitted whatever the trace did, because a cost total taken from
sampled spans looks precise and is wrong. Their dimensions are an allow-list stated by
`Dimensions`; a tenant outside it still has its money counted, under `other`, while the span
keeps the full identity for an investigation.

Attributes a consumer attaches are pattern-scrubbed on the way out — addresses, vendor keys,
bearer tokens, JWTs, long opaque hex, plus whatever `Redactor(extra_patterns=…)` a deployment
adds — and the dropped keys are named on an `adk.redacted` event so a missing value reads as a
decision. The kit's own `adk.` attributes pass through: they are structural identity, and
redacting them leaves spend attributed to nobody.

`Run` gains `prompt_version`, `task_class` and `depth`; the run loop sets `task_class` from the
routing decision. `tesserix_adk.testing` gains `FakeMeter` and `MetricPoint`.

**Stability:** additive. Documented in `docs/cost-attribution.md`, exercised by
`examples/cost_attribution.py` and `tests/test_cost_attribution.py`.
- **tesserix_adk.runtime, tesserix_adk.models.pricing, tesserix_adk.core**: A run can now be costed before it starts. `estimate_run(agent, user_input, provider=…,
pricing=…, history=…)` returns a `CostEstimate` carrying a point, a tenth- and
ninetieth-percentile case, the token counts behind them, the assumptions they rest on and a
confidence. The provider is asked only to count tokens and never to complete anything: an
estimate that needs a paid round trip is a bill for asking about a bill.

The number says how much of itself is measurement. `Confidence.MEASURED` is built from
finished runs of that agent at that version, `INFERRED` from another version or the kit's
defaults, `UNKNOWN` where nothing prices the model at all. `Assumptions.runs_observed`
carries how many runs are behind it, so an estimate with nothing behind it cannot be
mistaken for one with a fleet behind it. Only completed runs become samples — a run that was
refused or died partway says nothing about what a run of that agent costs.

The kit does not invent a figure. A model no price list covers raises
`EstimateUnavailableError` rather than returning something plausible; `allow_unknown=True`
returns the token counts with the money marked unknown. Prompt growth is modelled rather than
multiplied: each turn carries the previous turns' output and tool results forward, so the
accumulated context is charged, and the prompt cache is netted off at the fraction the
recorded runs actually saw.

`affordable(estimate, limits)` and `refuse_unaffordable(...)` check the **high** case against
what is left of a budget, because a ceiling the typical run fits and a bad one does not is no
ceiling. `estimate.as_limits(headroom=…)` converts an estimate into a `BudgetLimits`
explicitly — nothing does it implicitly, which is how a run gets killed for being averagely
expensive — and headroom applies to the money only, since what varies between an estimate and
an invoice is what tokens cost, not how many turns the agent was allowed.

`approval_for(estimate, agent, …)` puts it to a person as an `ApprovalRecord` whose reason
carries the range and the confidence rather than one number, with the assumptions travelling
as the record's arguments. `calibrate(estimate, run)` holds it against what the run actually
cost; nothing clamps the ratio, because the outliers are what the estimator has to learn
from. A supervisor's estimate is parent-only and says so via `Scope`; `with_children(...)`
totals the parts and takes the weakest confidence among them.

`RunHistory` is the one-method protocol a deployment implements over the runs it already
stores, with `InMemoryHistory` shipped for tests and small deployments. `Pricer` is the
callable estimation is built against, so nothing in the runtime holds an opinion about where
prices live; `tesserix_adk.models.pricing.pricing_at(date)` is the shipped adapter over the
dated price list.

**Stability:** additive. Documented in `docs/estimation.md`, exercised by
`examples/cost_estimate.py` and `tests/test_estimation.py`.
- **tesserix_adk.core.ledger, tesserix_adk.adapters.ledger, tesserix_adk.testing**: A per-tenant ceiling now holds across replicas. `SpendLedger` is the protocol —
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

**Stability:** additive. Documented in `docs/ledger.md`, exercised by `examples/ledger.py`,
`tests/test_ledger.py` and `tests/test_ledger_stores.py`.
- **tesserix_adk.tools.tool, tesserix_adk.tools.Tool, tesserix_adk.tools.ToolContext, tesserix_adk.core.ToolDefinitionError, tesserix_adk.core.schema.annotations_of**: A tool is now one typed function. `@tool` derives the model-facing schema from the signature
and the docstring, so the declaration the model reads and the code that runs are the same
declaration. The two-declaration alternative drifts silently — the model keeps sending the
argument renamed six months ago — and it surfaces as a call the code refuses in production.

Everything a model could be told wrongly is refused at decoration, which is import time: an
unannotated parameter, `*args`, `**kwargs`, `Any` at any depth, a type with no JSON Schema
form, an annotation that does not resolve, a generator, a self-referencing model under the
default inlining dialect, and a name another live tool already answers to. That last claim
is held for the tool's lifetime rather than the process's, so a reloaded module or a
per-test fixture takes back the name it held, and `Tool.release()` gives one up explicitly.

Decorating makes every tool awaitable. A synchronous body leaves the event loop — a thread
by default, a bounded `WorkerPool` when one is passed to `invoke` — so a blocking tool does
not stall every other run sharing the loop. `Tool.__call__` keeps the function's own typed
signature; `Tool.invoke` takes the mapping a provider chose.

`ToolContext` carries the run, tenant, user, trace and cancellation token that a model must
never be able to choose. A parameter annotated with it is excluded from the schema and
filled by the caller, and `invoke` overwrites an argument of that name if the payload
contains one. Without a default it is required, so a call outside a run is refused rather
than run against a guessed tenant.

`schema_for` grew what the decorator needed and any caller can use: a keyword-only
`exclude=` for parameters the caller injects, support for targets like `list[str]` and
`str | None`, and `annotations_of` for resolving a callable object's hints through its
`__call__`. A builtin's own docstring is no longer emitted as a description.

Documented in `docs/tools.md` and exercised by `examples/tools.py`.

One harness change came with it: `PytestUnraisableExceptionWarning` is no longer an error.
pytest-asyncio installs a fresh event loop after every async test and never closes it, so
which test is running when the collector finalises one is chance — as an error it fails
whichever test happened to allocate at the wrong moment. Every other warning still is one.
- **tesserix_adk.tools.ToolArgumentValidator, tesserix_adk.tools.ArgumentPolicy, tesserix_adk.tools.STRICT, tesserix_adk.tools.LENIENT, tesserix_adk.core.ToolArgumentValidationError.feedback**: Tool arguments are model output, and model output is now checked before any tool body runs.
`Tool.invoke` reads the payload into the tool's own signature first: a field the tool does
not declare is refused rather than dropped, a missing one is never filled in, and each
argument reaches the function as its declared type — a nested model arrives as that model,
not as the dictionary a provider sent. The alternative is a hallucinated field name reaching
business code and failing somewhere the traceback says nothing about the call, or not
failing, and being interpolated into a query.

Payload handling is the same whichever provider answered. Arguments sent as JSON text, or
inside a redundant `{"arguments": ...}` envelope, are normalised before they are read;
JSON that does not parse, JSON that is not an object, a repeated key, and a payload over
`max_bytes` are each the same typed refusal, and the ceiling is held before parsing rather
than after.

`ToolArgumentValidationError` names every field that failed with what was wrong, and never
what it held: a rejected argument may be a password or someone's address, and quoting it
back copies it into the next request and the provider's logs. `error.payload` keeps what
arrived for a debugger, and `error.feedback()` is what may be said to the model.

Inside a run that refusal is correctable rather than fatal. The tool did not run, so the
call is not retried against it — the same payload is the same refusal — and the feedback
goes back as the tool's result on the agent's declared repair budget, recorded as
`REPAIR_REQUESTED` and counted with every other repair the run has made. An agent that
declared no repair fails on the first rejection, and one that spends the budget still
calling the tool wrongly fails closed, naming the tool and the unsatisfied fields.

Strictness is a decision made once: `@tool(arguments=LENIENT)` takes the documented JSON
coercions, `ArgumentPolicy(max_bytes=...)` tightens the ceiling, and the default is strict
so a tool's contract does not depend on which vendor sent `2` and which sent `"2"`. Leniency
relaxes the coercions and nothing else. `ToolArgumentValidator` is the same check standing
alone, for a registry holding tools it did not build with `@tool`.

Documented in `docs/tools.md` and exercised by `examples/tool_arguments.py`.
- **tesserix_adk.tools.ToolRegistry, tesserix_adk.tools.AgentToolView, tesserix_adk.tools.ToolCallSpan, tesserix_adk.core.ToolNotFoundError, tesserix_adk.core.ToolNotPermittedError**: What an agent may call is now declared configuration held in one place. `ToolRegistry`
holds the tools a process has; `registry.view(allow=..., agent=...)` hands one agent the
subset it may call. Two agents in one service no longer share a callable set because they
share a process: a tool registered for one is not reachable by the other, and one agent's
refusal is invisible to the other. The allowlist is resolved when the view is made, so a
misspelled tool name fails there — naming what is registered — rather than at the first
call in production, and the view is frozen, so an agent's reach cannot widen mid-run.

A refused call is refused before dispatch. An allowlist checked after the call has already
had its side effect by the time the decision is recorded. `ToolNotPermittedError` names the
agent and what it may call, so the model can be told; `ToolNotFoundError` is deliberately a
different type, because a name nobody registered is a deployment mistake and a name this
agent may not call is a permission decision, and the fix is different. Neither is retried,
even for a tool the agent declared idempotent — asking again gets the same answer and only
spends the budget. An empty allowlist raises `ConfigurationError` at construction rather
than leaving an agent to discover it has no tools on its first turn.

A ceiling travels with the tool that needs it: `@tool(timeout=5.0)` is the author of the
network call saying what a healthy one costs, and a registry may override it per deployment.
When it elapses the underlying task is cancelled rather than orphaned and `ToolTimedOutError`
reaches the run loop, which never receives an invented result for the call. A body that
swallows cancellation is bounded by a hard abandonment path: waiting stops, the span records
`abandoned`, and the late result is discarded instead of being injected into a run that has
moved on.

`ConcurrencyConfig` bounds the registry as a whole and each tool individually — how much the
process may do at once, and how much one downstream may be asked to take. `@tool(parallel_safe=False)`
holds an order-dependent tool to one call at a time whatever the config says. Duplicate
registration of one name from two sources is refused with both origins named.

Every invocation emits a `ToolCallSpan` — tool, agent, permission decision, outcome class,
duration, and whether a timed-out body was abandoned. It carries neither the arguments nor
the result: a tool's payload is model output and its result is business data, and neither
belongs in telemetry leaving the process. An observer that raises is ignored, because a
broken exporter should not fail a tool call.

Stability is documented rather than implied: additive tool metadata is non-breaking,
allowlist semantics are versioned, and any change to default-deny is a major version.

Documented in `docs/tools.md` and exercised by `examples/tool_registry.py`.
- **tesserix_adk.runtime.ToolResultBoundary, tesserix_adk.runtime.ToolResult, tesserix_adk.runtime.ResultPolicy, tesserix_adk.runtime.ResultFinding, tesserix_adk.runtime.ReturningTool, tesserix_adk.core.ToolResultError, tesserix_adk.testing.INJECTION_FIXTURES, tesserix_adk.testing.InjectionFixture**: Everything a tool returns now crosses a boundary before it reaches the model.
`ToolResultBoundary` validates the value against the tool's declared return type, walks it
for injection heuristics, neutralises structural forgery, applies size and depth ceilings,
and hands back a `ToolResult` rendered as an explicitly untrusted-data envelope carrying the
tool, source, tenant and trust label. The run loop uses it by default: a defence that must be
opted into is a defence most runs will not have.

Structural forgery and instruction-shaped prose get different answers, because they are
different problems. Chat-template turn markers, escapes out of the envelope, null bytes and
bidi reordering characters are removed outright — no legitimate result emits them. Text that
merely reads like an instruction is flagged and delivered, because a support macro and a
refund policy discuss ignoring instructions in the same words an injection does, and
`ResultPolicy.on_suspicion` lets the consumer choose `annotate`, `truncate` or `fail`, per
tool. Scanning walks the whole structure rather than the top level, so an instruction in the
fourth search hit's body, in an image's alt text, inside an HTML comment or inside a base64
field is found where it sits; decoded content is scanned but never rendered back.

A tool's declared return type is now enforced rather than documented. A value that does not
match raises `ToolResultError` naming the tool and the violation and never quoting the value —
a rejected result may be someone's address, and quoting it copies it into the logs the
refusal was meant to keep it out of. Nothing is summarised or repaired into something
plausible: an invented result is indistinguishable from a real one once it is in the
conversation. A tool that annotates nothing has promised nothing and is taken at its word.

A result cannot authorise the next call. Once anything in a run has been flagged, a call to
an approval-required tool is refused before the approval gate is even asked, with the refusal
naming the flagged result rather than quoting it. This is run-scoped rather than
next-turn-scoped deliberately: "the very next call" is a window an attacker waits out.
Flagging records a `tool_result_flagged` event carrying the heuristic name and the path and
nothing else — content suspicious enough to flag is exactly the content that must not be
copied verbatim into telemetry or memory.

`tesserix_adk.testing.INJECTION_FIXTURES` publishes the payloads a boundary must survive —
direct overrides, envelope escapes, forged ChatML and Llama turns, an instruction in the
fourth search hit, image alt text, base64, HTML comments, bidi reordering, null bytes — each
stating whether it should be neutralised or flagged. No network, no model, nothing that only
passes against this implementation, so a consumer replacing the heuristics or the renderer
can show the replacement is no weaker.

`AgentToolView` gains `resolve`, so a view can say what a permitted name is without widening
what an agent may call.

Documented in `docs/tool-results.md` and exercised by `examples/tool_results.py`.
- **tesserix_adk.core.ToolError, tesserix_adk.core.ToolFailure, tesserix_adk.core.ToolRefusal, tesserix_adk.tools.ToolError, tesserix_adk.tools.ToolFailure, tesserix_adk.tools.ToolRefusal, tesserix_adk.tools.ToolErrorMap, tesserix_adk.tools.ToolErrorRule, tesserix_adk.tools.transient, tesserix_adk.tools.permanent, tesserix_adk.tools.refusal**: A tool that failed and a tool that declined are now different answers. `ToolFailure` carries a
stable `code`, a `retryable` flag and an optional `retry_after`; `ToolRefusal` says the tool
worked and said no. Before this, every tool problem arrived as a generic exception and the run
loop could not separate "the supplier was briefly unavailable" from "this booking is not
cancellable" — so it retried the refusal until the iteration cap fired, spending the budget to
be told the same thing and, worse, re-attempting an action the downstream had already declined.

A code is required at construction. A failure nobody can name is a failure nobody can write a
policy about, and an optional field would be omitted exactly where the run needed it.
`transient` defaults to `False`: an author who has not thought about whether repeating the call
repeats a side effect has not established that it does not.

A refusal reaches the model once, as data, through the same untrusted-result envelope as any
other tool output, carrying its reason code — so a reason string authored to read like an
instruction cannot become one. It is never retried and it does not fail the run.

`ToolErrorMap` translates the exceptions a tool's libraries actually raise, declaratively rather
than through hand-written `except` blocks — an author writing those eventually writes a bare
one, and a bare one classifies a bug as a retryable failure. The most specific rule on the
raised type's MRO wins; where no type matches, an exception carrying `status_code` or `status`
is read against `statuses`. Unmapped exceptions become a permanent `unmapped_failure` rather
than being optimistically retried; cancellation and anything else outside `Exception` is
re-raised rather than classified; an error already in the taxonomy is passed through, because
whoever raised it knew more than the map does. Messages are scrubbed as they are translated, so
a credential in an upstream exception string never reaches a span, a run event or memory.

The run loop honours `retryable` and `retry_after` for typed errors, and keeps the existing
idempotency gate for untyped ones — the kit cannot know whether an unclassified exception left a
side effect behind. A `retry_after` longer than the run has left fails closed immediately rather
than sleeping past a deadline it cannot meet. `AgentRunner(max_tool_attempts=…)` caps how many
retries one tool may consume across a whole run (12 by default), so one flaky dependency cannot
own the iteration budget.

`ToolCallSpan` gains `code`, and distinguishes `declined` — the tool saying no — from `refused`,
the permission decision. The `tool_error` run event names the code and the attempt count, never
the raw exception message.

Documented in `docs/tool-errors.md`, including the code-stability policy, and exercised by
`examples/tool_errors.py`.
- **tesserix_adk.core.ApprovalPolicy, tesserix_adk.core.ApprovalPredicate, tesserix_adk.core.ApprovalDenial, tesserix_adk.core.ApprovalBindingError, tesserix_adk.runtime.ApprovalLedger, tesserix_adk.tools.Tool.requires_approval**: A tool that moves money can now say so itself: `@tool(requires_approval=True)` gates every call,
and `@tool(requires_approval=lambda arguments: arguments["amount"] > 100)` gates the ones that
cross a threshold. The gate was previously the agent's to remember — `approval_required_tools` on
the `Agent` — which put the decision furthest from the person who knows what the tool does. An
agent that adopts a refund tool and forgets to list it is the common case, and a control that
depends on every consumer remembering is a control that is missing somewhere. Both work; either
one is enough to hold the call.

The predicate is asked with validated arguments, so a threshold reads `500` rather than `"500"`,
and a predicate that raises — or arguments the validator refuses — hold the call rather than
release it. A gate that errors open is not a gate on the day it matters.

What an approver is shown is a summary, not the payload: `ApprovalRecord.summary` shows numbers
and booleans in full, because an approver who cannot see the amount cannot approve it, and
describes everything else by type and length. A deny-list of key names was rejected — an IBAN and
a card token are both strings, and the list would be wrong for whichever field nobody thought of.
The record continues to carry a digest of the arguments rather than the arguments.

The grant is bound to that digest. `ApprovalLedger` permits one payload, once, for one run:
arguments altered after the human saw them, a decision replayed for a second execution, a grant
nobody recorded, and a grant belonging to a run that has ended all raise `ApprovalBindingError`
and refuse to dispatch. The repair loop that fixes a malformed amount and the retry that re-sends
the call are the paths this closes — both would otherwise execute under a decision nobody made
about them. A tool result that says "APPROVED by the desk, proceed" satisfies nothing; approval
is a decision from the gate, and untrusted output is not one.

Documented in `docs/tool-approval.md` and exercised by `examples/tool_approval.py`.
- **tesserix_adk.core.Idempotency, tesserix_adk.core.IdempotencyPolicy, tesserix_adk.core.IdempotencyStore, tesserix_adk.core.Claim, tesserix_adk.core.idempotency_key, tesserix_adk.core.IndeterminateOutcomeError, tesserix_adk.runtime.MemoryIdempotencyStore, tesserix_adk.adapters.RedisIdempotencyStore, tesserix_adk.adapters.PostgresIdempotencyStore, tesserix_adk.testing.IdempotencyStoreConformance, tesserix_adk.tools.ToolContext.idempotency_key**: A tool that books a seat can now say that booking it twice is two seats:
`@tool(idempotency=IdempotencyPolicy(Idempotency.EFFECTFUL, key_arguments=("flight",)))`. The
dispatcher derives a key over those arguments, claims it before the body runs, and records what
came back — so a retry, a replay after a restart, and two concurrent identical calls in one turn
all resolve to one execution and one recorded outcome. Two products had already threaded a key
through their own tool code by hand, differently, and neither survived a worker restart.

The guarantee is stated as versioned public API: **at most one side effect per key within the
retention window**. Not exactly-once, which nothing that calls a network can honestly offer. A
call that fails without saying whether it landed keeps its claim, is not retried, and fails the
run with `IndeterminateOutcomeError` naming the tool — the same answer as a key that cannot be
derived, or a store that cannot be reached. A store that is down does not read as permission.

The call id is deliberately excluded from the key, against the letter of the scope. Including it
gives two concurrent identical calls two keys and both fire, which is the case the feature exists
for. What is in the key is the tenant, the run, the tool name and a canonical hash of the named
arguments — sorted, NFC-normalised, floats formatted so `2` and `2.0` agree. Arguments are hashed
and never persisted, records are tenant-scoped, and `forget(tenant=...)` erases them.

`MemoryIdempotencyStore` covers one replica and tests; `RedisIdempotencyStore` and
`PostgresIdempotencyStore` survive a restart and claim in a single server-side operation, because
a read followed by a write is a window the other replica fits through. `IdempotencyStoreConformance`
is the suite a third implementation has to pass. A tool that declares no policy, or a runner
configured with no store, behaves exactly as it did before. Documented in
`docs/tool-idempotency.md`.
- **tesserix_adk.memory.MemoryStore, tesserix_adk.memory.MemoryScope, tesserix_adk.memory.MemoryKind, tesserix_adk.memory.MemoryRecord, tesserix_adk.memory.MemoryQuery, tesserix_adk.memory.MemoryHit, tesserix_adk.memory.MemoryCapabilities, tesserix_adk.memory.MemoryNeeds, tesserix_adk.memory.require_memory, tesserix_adk.testing.InMemoryMemoryStore, tesserix_adk.testing.MemoryStoreConformance, tesserix_adk.core.MemoryScopeError, tesserix_adk.core.MemoryCorruptionError, tesserix_adk.core.MemoryLimitError, tesserix_adk.core.EmbeddingDimensionError**: Memory is now a substitutable dependency rather than something each product invents. One
`MemoryStore` protocol covers the four kinds of remembering an agent does — working
(`write` / `read` / `append` / `expire`), profile (`upsert` / `profile`), episodic (`log` /
`episodes`) and semantic (`index` / `search`) — with `erase` across all of them. They share
one record type because they share one lifecycle; they do not share operations, because
what working memory does and what semantic memory does collapse into a shared get/put only
by lying about one of them.

`MemoryScope` is in every signature and there is no unscoped overload, so a call site
cannot forget one. `tenant_id` is required with no default and no shared sentinel, and a
record written under a scope that is not its own raises `MemoryScopeError` rather than
being filed under whichever of the two the adapter read first.

Capabilities are declared by the adapter and checked when it is bound: `require_memory(store,
MemoryNeeds(semantic=True))` raises `CapabilityError` naming every missing capability and the
adapter that lacks them. An adapter with no vector index otherwise answers every semantic
recall with an empty list, without an error, and nobody notices for a month. The same
operations still refuse at run time for a consumer who skipped the check — zero rows erased
and cannot erase are the same number and opposite facts.

What goes wrong is typed: `MemoryCorruptionError` carries the record id and the raw payload
and is never swallowed, because a recall that drops what it could not read assembles a
prompt nobody can explain; `MemoryLimitError` refuses an oversized value at the write rather
than truncating it into a profile that is subtly wrong forever; `EmbeddingDimensionError`
refuses a vector of the wrong width rather than ranking on the overlap.

`InMemoryMemoryStore` is the network-free implementation, and `MemoryStoreConformance` is
the suite every adapter must pass — ordered concurrent appends, erasure that is all-or-
nothing against a concurrent read, scope isolation, and capability-gated cases that skip
themselves against a store that declares it cannot do the thing. The protocol's stability
contract is documented in `docs/memory.md`: public API under semver, additive-only within a
minor, one minor of notice and a working shim before any removal.
- **tesserix_adk.memory.ContextAssembler, tesserix_adk.memory.ContextPlan, tesserix_adk.memory.SectionPlan, tesserix_adk.memory.AssembledContext, tesserix_adk.memory.SectionOutcome, tesserix_adk.memory.CompactionStrategy, tesserix_adk.memory.CompactionOutcome, tesserix_adk.memory.ContextEntry, tesserix_adk.memory.DropOldest, tesserix_adk.memory.PinAndFold, tesserix_adk.memory.SummariseSpan, tesserix_adk.memory.TokenCounter, tesserix_adk.memory.pinned, tesserix_adk.core.ContextBudgetError**: The prompt is now built from a declared plan under a budget the provider supplied, rather
than concatenated until the provider truncates it. Truncation is by position, so what a
long conversation loses is the constraint stated in turn three and not the small talk in
turn ninety, and nothing records that it happened.

`ContextPlan` names the sections, their order, and the share of the budget each may take.
Shares are of the whole budget rather than of what earlier sections left, so adding a
section cannot silently shrink the ones after it, and shares adding up to more than the
budget are refused when the plan is built. Token counting goes through
`ModelProvider.count_tokens`, and the budget falls back to the provider's declared
`context_window_tokens`; both are read on every assembly, so a model swapped mid-session
for a smaller one is budgeted for rather than overflowed once.

`pinned(message)` marks a message non-evictable and a section can be pinned whole. Pinned
content is allocated its room first and no strategy may drop it; where it exceeds the
budget on its own, assembly raises `ContextBudgetError` rather than deciding which of the
caller's constraints was optional.

Compaction is a protocol with three built-ins: `DropOldest` and `PinAndFold` cost nothing,
and `SummariseSpan` replaces the oldest unpinned span with a model-written summary, written
back to episodic memory with provenance to the turns it replaced. A section naming a
strategy the assembler does not have is refused at construction.

Everything fails closed. A summarisation call that fails, a summariser that returns nothing
usable, or a strategy that hands back more than it was allowed all raise
`ContextBudgetError`; the kit never emits an over-budget prompt and never presents a
fabricated summary as the record of a conversation. Cancellation stays cancellation.
Compaction sees only the messages it was given, so a span carrying redaction placeholders
is summarised from the placeholders.

`AssembledContext` reports what was kept, evicted and summarised per section, and
`span_attributes()` exports the counts without any message content. Assembly is
deterministic, so a recorded run replays identically in an eval.

**Stability:** public API under semver. Additive only within a minor; a new built-in
strategy or a new field on the report may appear, but no operation is removed or given a
new required parameter without one minor of notice and a working shim. Documented in
`docs/context-assembly.md`.
- **tesserix_adk.memory.Belief, tesserix_adk.memory.Contradiction, tesserix_adk.memory.ContradictionPolicy, tesserix_adk.memory.SupersedeMatching, tesserix_adk.memory.Supersession, tesserix_adk.memory.Resolution, tesserix_adk.memory.DecayPolicy, tesserix_adk.memory.HalfLife, tesserix_adk.memory.ConfidenceFloor, tesserix_adk.core.MemoryConflictError, tesserix_adk.core.MemoryContradictionError**: A profile fact that changes no longer destroys the one it replaced. `supersede` writes a
new version and closes the old one with `valid_to` and `superseded_by`, so an agent can
say what it believed in March as well as what it believes now, and why it changed its
mind. `history` returns the whole trail per key or per scope, which is what support and
dispute resolution read.

`MemoryRecord` gained `recorded_at`, `superseded_by`, `version`, `subject` and
`predicate`. `recorded_at` is separate from `valid_from` because a fact backdated to March
was still acted on from August, and an audit that cannot tell those apart cannot explain a
decision made in between. A `valid_from` in the future is on the trail immediately and is
not recalled until the time comes. `profile` and `belief` take `as_of`, and exactly one
record is live per instant however deep the chain runs.

`ContradictionPolicy` decides what an incoming record does to the live one.
`SupersedeMatching`, the default, replaces only an exact restatement — same subject, same
predicate aspects. Anything else branches: both records stay live, `belief` returns a
`Contradiction` carrying them, and `profile` raises `MemoryContradictionError` rather than
returning whichever record happened to sort first. A branch ends when a writer names the
records it settles via `resolves=`, so the decision is recorded rather than inferred.

Concurrent writers pass `expected_version`; the loser gets `MemoryConflictError` naming
both the version it read and the version that is live. The store never leaves two live
records for one write and never applies one silently over the other.

`DecayPolicy` weighs a record instead of expiring it. `HalfLife` decays by age with a
floor under which nothing is recalled, `ConfidenceFloor` by the record's own confidence.
Decay changes ranking and recall eligibility and never deletes: what a policy stops
surfacing, `history` still returns, and `Belief.decayed` and `Belief.weight` make a policy
aggressive enough to silence a whole scope visible rather than indistinguishable from a
scope nobody ever wrote to. Erasure remains the only thing that removes anything.

`MemoryCapabilities.supports_supersession` gates the new operations, and
`MemoryStoreConformance` holds every adapter that declares it to the same behaviour.

**Stability:** public API under semver, additive within a minor. `MemoryStore` gained
`supersede`, `belief` and `history`, and `profile` gained an optional `as_of` — an
existing implementation keeps compiling and keeps its old behaviour until it declares
`supports_supersession`. Documented in `docs/beliefs.md`.
- **tesserix_adk.memory.Derivation, tesserix_adk.memory.DerivedIndex, tesserix_adk.memory.MemoryRedactor, tesserix_adk.memory.PatternRedactor, tesserix_adk.memory.DEFAULT_REDACTOR**: Erasure now reaches what was derived from a record, and redaction keeps most of it from
being stored in the first place.

Every write path — `write`, `append`, `upsert`, `supersede`, `log`, `index` — masks its
value before storing it, using the same shape detector the progress stream and the
telemetry exporter already use, and names the masked paths on `MemoryRecord.redacted`.
Paths rather than a count, so a reader can tell a masked field from an empty one; nested,
because the token is never at the top level. `PatternRedactor(extra_patterns=...)` adds
what a deployment knows about, and `redactor=None` turns it off, which has to be asked for
rather than arrived at. Card numbers joined `SENSITIVE_SHAPES`, so they are masked
everywhere the kit redacts, not only in memory.

An embedding, a summary or a cache entry still says what the record said, so each one
registers a `Derivation` naming what it came from and which `DerivedIndex` holds it.
`erase` walks that registry rather than assuming deleting rows was enough. An artefact two
scopes derived is never purged for one of them.

Erasure is two-phase. Records are tombstoned first and stop being readable at once —
`read`, `profile`, `belief`, `episodes`, `search` and `history` all skip them — and
artefacts are purged second. An index that cannot be reached raises `PartialErasureError`
naming it, with an incomplete receipt attached; the records stay out of reach, and running
`erase` again resumes without re-counting what already went. `dry_run=True` reports
accurate counts and touches nothing, and is never `complete`, because it kept no promise
to anybody. Each real erasure publishes one `adk.memory.erased` event carrying counts and
adapter names only — an audit trail that quoted what was erased would have undone it.

**Stability:** public API under semver. `MemoryStore` gained `derived` and `derivations`
alongside the breaking `erase` signature. Documented in `docs/erasure.md`.
- **tesserix_adk.adapters.MemoryStoreSettings, tesserix_adk.adapters.RedisMemoryStore, tesserix_adk.adapters.PostgresMemoryStore, tesserix_adk.adapters.PgvectorMemoryStore, tesserix_adk.adapters.RoutedMemoryStore, tesserix_adk.adapters.MemoryPage, tesserix_adk.core.MemoryUnavailableError**: Memory now has real stores behind it: Redis for the working set, PostgreSQL for profiles
and episodes, pgvector for semantic recall, and `RoutedMemoryStore` composing the three
into the one `MemoryStore` a consumer binds.

Each store is partial on purpose and says so through its `capabilities`, so a plan that
needs ranking cannot be bound to a key-value store and find out a month later. Working
memory keys carry the whole scope, expiry is the server's job, appends are a single
server-side script rather than a read-modify-write, and an evicted key reads as absent —
which is what it is — while the position an append returns still lets a caller see that a
sequence was lost. Profiles and episodes are bitemporal: nothing is overwritten, versions
are closed by the `UPDATE`'s own predicate so two writers cannot both win, and an episode
insert is idempotent on its id so a retry after a failover books it once. A wide time
window is read in keyset pages via `page`, because an `OFFSET` re-reads every row before
it to skip them. Semantic recall ranks in the database with the scope filter pushed into
the predicate — a filter applied after the fetch has already read the rows it was meant to
exclude — and `verify()` catches a collection narrower than the embedder at startup rather
than on the first bad ranking.

Credentials come from `MemoryStoreSettings` and nowhere else. The DSN is a `SecretStr`, a
blank one is refused, and so is one carrying a shipped default password: an adapter that
connects out of the box connects out of the box for everybody. Driver failures are
retried with bounded, jittered backoff and surface as `MemoryUnavailableError` once the
budget is spent, so the run fails closed rather than continuing with an empty memory. An
exhausted connection pool is reported immediately as `PoolExhaustedError` instead of
retried, since retrying into a full pool is how a fan-out spike becomes an outage.

Schema DDL is deliberately absent — tables and indexes belong to the platform's migrations,
so importing a library can never be the thing that alters a production table. The
conformance suite runs against real Redis and PostgreSQL in `tests/integration`, opted into
with `-m integration`; the default lane reaches no network.

**Stability:** additive. Documented in `docs/memory-adapters.md`.
- **tesserix_adk.adapters.GraphMemoryStore, tesserix_adk.adapters.GraphSettings, tesserix_adk.adapters.GraphEngine, tesserix_adk.adapters.GraphitiEngine, tesserix_adk.adapters.GraphitiClient, tesserix_adk.adapters.open_graphiti, tesserix_adk.adapters.EntityExtractor, tesserix_adk.adapters.ExtractedSubgraph, tesserix_adk.adapters.ExtractedNode, tesserix_adk.adapters.ExtractedEdge, tesserix_adk.adapters.ExtractionMeter, tesserix_adk.adapters.ExtractionCharge, tesserix_adk.adapters.BACKENDS, tesserix_adk.adapters.EXTRACTION_INSTRUCTION, tesserix_adk.core.ExtractionError, tesserix_adk.core.WriteQueueFullError**: `GraphMemoryStore` adds relationship memory — who travelled with whom, which supplier
failed which booking, what an entity used to be — behind the same `MemoryStore` protocol
as every other adapter. It answers `relations(scope, as_of=…)` and delegates working
memory, profiles, episodes and semantic recall to a companion store, because a graph is a
poor key-value store and forcing four kinds of memory into one shape helps nobody.

It is the one adapter where a write costs money: relations are extracted by a model call.
So the first thing a write does is ask whether this tenant may still spend. `BudgetPolicy`
bounds the run, `ExtractionMeter` bounds the tenant across every run they have, and an
exhausted ceiling raises `BudgetExceededError` naming the ceiling and the spend to date
*before* the provider is called — no model call, no partial subgraph. Extraction goes
through the kit's `ModelProvider`, so the extraction model is a routing decision and its
tokens, latency and cost are attributed like any other call.

Output that violates the entity schema raises `ExtractionError` carrying the raw payload,
and the graph transaction rolls back: the kit never commits guessed entities or half a
subgraph. If the backend is unreachable *after* extraction was paid for, the extracted
subgraph is held for retry so the spend is not thrown away. Retrieved text is fed to the
extractor as data and never as instruction. Entities deduplicate within a tenant and never
across tenants — the scope is part of the key, not a filter applied afterwards.

Interactive runs need not wait for extraction: the store takes writes onto a bounded
queue, and a saturated queue raises `WriteQueueFullError` rather than dropping a write
quietly. Edges map onto the protocol's `valid_from` / `valid_to` / `as_of` semantics, an
`as_of` read against a still-open interval resolves, and erasure removes the derived
embeddings alongside the nodes and edges.

The engine is injected. `open_graphiti(GraphSettings(...))` ships a wrapper for Graphiti
(Apache-2.0) over Neo4j or FalkorDB, selected by config, and the adapter itself depends on
neither. Install with the `graphiti` extra.

**Stability:** additive. Documented in `docs/graph-memory.md`.
- **tesserix_adk.core.StateStore, tesserix_adk.core.StateKey, tesserix_adk.core.SessionRecord, tesserix_adk.core.RunRecord, tesserix_adk.core.StateDelta, tesserix_adk.core.StateQuery, tesserix_adk.core.StatePage, tesserix_adk.core.StateConflictError, tesserix_adk.core.StateNotFoundError, tesserix_adk.core.StatePersistenceError, tesserix_adk.core.StateInUseError, tesserix_adk.runtime.MemoryStateStore, tesserix_adk.testing.StateStoreConformance**: A run that exists only in one process ends when that process does, so state has to go
somewhere shared — and the moment it does, two workers holding the same run both write it
back and the second silently wins. The first worker's iteration, spend and cursor are gone
with nothing recording that they happened, which looks exactly like work that was never
done. `StateStore` is the shape state has here, and it makes that outcome unrepresentable
rather than unlikely.

Every write states the version it read. A store commits at that version plus one or raises
`StateConflictError` carrying both numbers, so the loser re-reads and decides again instead
of retrying against a stale copy. Version zero is a create, which is how two workers racing
to start the same run resolve to one winner rather than to one silent overwrite.

Amounts that accumulate go through `patch_run`, which takes a `StateDelta` of additions and
no version at all: additions commute, so ten workers each recording what they spent produce
the sum, where ten workers each writing a total produce whichever arrived last. Nothing in
a delta may be negative — that is how one worker unspends another's tokens.

`RunRecord` holds enough to resume and no more: the message cursor, the tool calls that were
asked for and never came back, spend, iteration count and the approval it is held on. Tool
arguments are scrubbed on the way in, because persisted state is queryable and a token that
reached an argument is one an operator can later grep for. `StateKey` carries the tenant, so
a cross-tenant read needs a deliberate key rather than a forgotten argument.

Listings are ordered by the store's own insertion counter and never by a clock: two workers'
clocks disagree, and a listing that pages by timestamp skips whatever was written during the
disagreement. `updated_before` filters rather than orders, which is how abandoned work is
found. Deleting a session that still has live runs is refused by name, since a live run
whose session has gone is work nothing will ever reap.

`StateStoreConformance` carries the guarantees an adapter has to keep, and `MemoryStateStore`
exists so they can be exercised without a database — not so a deployment can skip having one.

**Stability:** additive. Documented in `docs/state.md`, exercised by `examples/state.py`.
- **tesserix_adk.core.Checkpoint, tesserix_adk.core.CheckpointBoundary, tesserix_adk.core.CheckpointPolicy, tesserix_adk.core.CheckpointStore, tesserix_adk.core.PendingCall, tesserix_adk.core.ResumePlan, tesserix_adk.core.Resumption, tesserix_adk.core.ToolDisposition, tesserix_adk.core.CHECKPOINT_FORMAT, tesserix_adk.core.CheckpointTooLargeError, tesserix_adk.core.CheckpointFormatError, tesserix_adk.core.IndeterminateToolCallError, tesserix_adk.core.ResumeConflictError, tesserix_adk.runtime.Checkpointer, tesserix_adk.runtime.MemoryCheckpointStore, tesserix_adk.runtime.plan_resume, tesserix_adk.runtime.claim_resume, tesserix_adk.runtime.refuse_if_undecidable, tesserix_adk.runtime.AgentRunner.resume, tesserix_adk.testing.CheckpointStoreConformance**: A run that dies at iteration nine restarts at zero. Every model call is paid for twice, and
every tool call it had already made is made again — which for a read is waste and for a
booking is a second seat. Checkpointing writes the run's frontier as it goes, and
`AgentRunner.resume` carries it on from there rather than from the beginning.

Checkpoints are taken only at boundaries where what has happened and what has not is
unambiguous: a model answer received, a tool result recorded, an approval about to be waited
on. Anywhere else the frontier is mid-flight, and a resume from it either repeats or skips
work with nothing able to tell which. `CheckpointPolicy` chooses among the three and caps the
payload; a run whose conversation outgrows the cap stops being checkpointed rather than being
silently truncated, because half a frontier resumes into a conversation that never happened.
A checkpoint that could not be written never fails the live run — the runner carries on
uncheckpointed and reports what it lost.

Resuming asks, per outstanding call, whether that call ran. The answer comes from the
idempotency store rather than from a guess: a recorded outcome means it finished and is
replayed into the conversation rather than re-executed, a key nobody holds means it never
started, and a key held in flight by a process that is gone means nobody can say. That last
case raises `IndeterminateToolCallError`, which is deliberately not retryable — a retry is
exactly the duplicate booking checkpointing exists to prevent, and resolving it needs the
tool's own status endpoint or a person. A dispatched call with no idempotency key is
indeterminate too: the absence of a key is the absence of a guarantee.

Two workers resuming one run would spend one budget twice and dispatch each outstanding call
twice, so a resume takes an at-most-once claim on the run and the second worker is refused
with `ResumeConflictError`. The claim expires, so one crashed worker does not strand the run
forever. A definition whose revision is not the one the checkpoint pinned is refused as well:
resuming into a changed agent is a different run wearing the first one's identity.

Spend is not copied into the checkpoint. The ledger already survives the process, and a
second copy of a number that only ever goes up is a second copy to disagree with.

`CheckpointStoreConformance` carries the guarantees an adapter has to keep — one frontier per
run, last write wins, no cross-tenant read — and `MemoryCheckpointStore` exists so they can
be exercised without a database, not so a deployment can survive a restart without one.

**Stability:** additive. Documented in `docs/checkpointing.md`, exercised by
`examples/checkpointing.py`.
- **tesserix_adk.core.WorkItem, tesserix_adk.core.WorkPriority, tesserix_adk.core.WorkQueue, tesserix_adk.core.WorkState, tesserix_adk.core.QueuePolicy, tesserix_adk.core.QueueStats, tesserix_adk.core.LeaseLostError, tesserix_adk.core.QueueUnavailableError, tesserix_adk.core.WorkItemNotFoundError, tesserix_adk.runtime.MemoryWorkQueue, tesserix_adk.testing.WorkQueueConformance**: A run dispatched to a background worker has no owner once the pod is rolled: nothing times
it out, nothing retries it, and nothing can say afterwards whether it finished. `WorkQueue`
gives that work a lease instead. A worker claims an item rather than taking it, a living
worker renews the claim, and a dead one's lapses — at which point the reaper puts the item
back with its attempt counted, for whichever worker is still alive to carry on.

Delivery is at-least-once and the docs say so rather than implying otherwise: a worker that
is merely slow has its item redelivered while it is still working, so handlers are idempotent
or they are wrong. Lease expiry is evaluated by the store rather than by a worker, because
two workers' clocks disagree and a queue that trusted theirs would free work that is being
done and keep work that is not.

Attempts are bounded. A failure comes back after a capped exponential backoff, a failure the
caller says cannot succeed skips the remaining attempts, and an item that runs out goes to
the dead letter carrying the failure from every attempt — so a handler that crashes on the
same item forever stops rather than looping, and stops somewhere an operator can see it
rather than being dropped. Renewals are bounded too: a worker that heartbeats indefinitely
holds work nothing else can pick up, and a stuck run is indistinguishable from a busy one to
everything except that bound.

A worker restarting under its own name gives back what it held immediately rather than
waiting out a lease nobody is renewing, so a rolled deployment orphans nothing; the attempt
still counts, because the work was tried. Tenants are served in rotation and priority orders
a tenant's own work and nothing else — a priority that crossed tenants would be one every
tenant sets to urgent, and they would be right to. A dedupe key collapses a second enqueue
of the same live job, and re-enqueueing an id that is still in the queue returns what is
stored rather than resetting the attempts the cap depends on.

An enqueue that could not reach the store raises `QueueUnavailableError` rather than
returning: work dropped silently is work nobody is waiting for and nothing will reap, because
nothing ever recorded that it existed. A worker acting on a claim it no longer holds is
refused with `LeaseLostError`, which is not retryable — the item belongs to somebody else now.

`WorkQueueConformance` carries the guarantees an adapter has to keep, and `MemoryWorkQueue`
exists so they can be exercised without Redis, not so a deployment can survive a rolled pod
without one.

**Stability:** additive. Documented in `docs/work-queue.md`, exercised by
`examples/work_queue.py`.
- **tesserix_adk.adapters.RedisStateStore, tesserix_adk.adapters.RedisWorkQueue, tesserix_adk.adapters.RedisStoreSettings, tesserix_adk.adapters.InspectableRedis, tesserix_adk.adapters.COUNTERS, tesserix_adk.adapters.DEFAULT_STATE_NAMESPACE, tesserix_adk.adapters.DEFAULT_QUEUE_NAMESPACE**: `MemoryStateStore` and `MemoryWorkQueue` hold a run and its work in a dict, which is exactly
as durable as the process. `RedisStateStore` and `RedisWorkQueue` put both on a server that
outlives it, and pass the existing `StateStoreConformance` and `WorkQueueConformance` suites
unchanged — the point being that a consumer swaps one for the other and the loop above them
cannot tell.

A run is stored as a JSON blob beside a small hash. The hash holds the version and the five
accumulating numbers, so `patch_run` is `HINCRBY` and two workers patching the same run at
once both land rather than one silently overwriting the other's tokens. Because a patch bumps
the version, a `put_run` holding a version the hash has moved past is refused with
`StateConflictError` carrying both numbers — a failover to a stale replica cannot resurrect
an old record on top of a live one.

Keys are length-prefixed per segment, so a tenant called `a:b` holding `c` and a tenant called
`a` holding `b:c` cannot collide into one key. The tenant is the cluster hash tag, which keeps
a run, its counters and its session index in one slot and the scripts atomic. The queue tags on
its whole namespace instead, because a claim reads several tenants' ready sets in one script in
order to take turns between them; the consequence is that one queue lives in one slot, and
`docs/state-adapters.md` says so rather than leaving it to be discovered under cluster.

Leases are evaluated against an injected `Clock` rather than the server's `TIME`, which is what
lets the conformance suite move time and what obliges a deployment to run its workers off NTP.
A claim is two writes with no shared transaction: a crash between them leaves an item that is
leased and still says it is queued, which the next `reap` corrects — every settle is fenced on
the lease score it saw, so a reaper that lost a race to a live heartbeat writes nothing at all.
Retries, backoff and the dead letter moved into `QueuePolicy`, shared with the in-process
queue, because two stores that disagree about when an item is poisonous are two deployments
with different retry semantics and one set of tests.

Both refuse a server that cannot hold state. `preflight()` reads the configuration at startup
and raises `ConfigurationError` for an eviction policy or for persistence turned off entirely:
an instance configured as a cache drops a run mid-flight and then answers the next read as
though it never existed, weeks later, under load. A queue whose work can be recreated elsewhere
opts out with `durable=False`. Records over `max_value_bytes` are refused before anything is
sent, naming the store that would take them, and that refusal is not retried — an oversized
item is not a bad connection. A failover is waited out with jittered backoff; an exhausted pool
is reported at once as `PoolExhaustedError` rather than retried into.

**Stability:** additive. Documented in `docs/state-adapters.md`, exercised by
`examples/state_adapters.py`, and verified against a real server by `tests/integration`.
- **tesserix_adk.adapters.PostgresStateStore, tesserix_adk.adapters.PostgresWorkQueue, tesserix_adk.adapters.PostgresStoreSettings, tesserix_adk.adapters.StateTables, tesserix_adk.adapters.SqlSession, tesserix_adk.adapters.SqlTransactor, tesserix_adk.adapters.EXPECTED_SCHEMA, tesserix_adk.adapters.SCHEMA_VERSION**: `PostgresStateStore` and `PostgresWorkQueue` back the `StateStore` and `WorkQueue` protocols
with the database a deployment already runs, and pass both conformance suites unchanged. The
reason to prefer them over the Redis pair is `bound(session)`: a state change and the work it
queues go into a transaction the caller opened, so they commit together or neither does, and
the outbox class of bug stops existing rather than being handled. A bound adapter retries
nothing — a failed statement has poisoned the caller's transaction, and retrying inside it
would only fail again.

A claim is one statement using `FOR UPDATE ... SKIP LOCKED`: a worker that arrives while a
row is locked steps over it rather than waiting behind it or taking it twice. Every predicate
is repeated on the locked relation, which is not redundancy — under `READ COMMITTED`
PostgreSQL re-checks the predicate against the updated row when a concurrent claim commits
underneath, and a predicate it cannot re-check is a row two workers both take. Tenants are
served in rotation through `adk_queue_turns`, so a tenant with a backlog cannot starve the
tenant beside it. Both properties are asserted against a real server, not a fake: ten workers
claiming at once take ten different items.

The five accumulating numbers live in columns rather than in the blob, so `patch_run` adds in
SQL and two patches arriving together both land. Paging is by a `bigserial` nothing rewrites.
Every settle is fenced on the lease the caller last saw, so a reaper that lost a race to a
live heartbeat writes nothing, and rows the reaper cannot lock are skipped rather than waited
on — a lease held inside a caller's long transaction is work still in progress.

The schema stays the deployment's. `EXPECTED_SCHEMA` publishes the shape the adapters read so
a migration can own it; the kit never applies it and never alters a table. `verify()` reads
`adk_schema` at startup and refuses a version this release was not written for, because a
column that moved is a write into the wrong shape, and refuses a connection with no
`statement_timeout`, because a statement that can run forever holds a pooled connection until
the process dies. Table names come from `StateTables`, which refuses anything that is not a
plain identifier: a name is interpolated, not bound, because a placeholder cannot name a
relation. A serialization failure or deadlock is retried with jittered backoff and then
surfaced as `contended`; a missing table is `ConfigurationError` and is not retried at all.

**Stability:** additive. Documented in `docs/state-adapters.md`, exercised by
`examples/sql_state_adapters.py`, and verified against a real PostgreSQL by
`tests/integration`.
- surface: tesserix_adk.core.DelegationError, tesserix_adk.core.RunBudget.sliced, tesserix_adk.core.UnlimitedBudget.sliced, tesserix_adk.runtime.Supervisor, tesserix_adk.runtime.Roster, tesserix_adk.runtime.Specialist, tesserix_adk.runtime.DelegationResult
---

One agent handing work to another is now a helper in the kit rather than something each
product rebuilds around its own `runner.run` call. Every hand-rolled version so far passed
the whole transcript into a sub-agent that held its own tool allowlist and spent whatever
the run had left, which is a privilege escalation and an unbounded bill wearing the same
shape as a feature.

A `Roster` of `Specialist`s declares who may be handed work and what each can do.
`Supervisor.delegate(task, needs=...)` routes to the narrowest worker covering the need, and
a roster with nobody matching is a `DelegationError(reason="no_worker")` rather than the
supervisor quietly doing the work itself with its own wider access.

A worker holds the intersection of its own tools and what its caller holds under its
delegation scope; sharing none is refused before a model is called. Its answer comes back as
`DelegationResult.data` inside the `<untrusted-data>` envelope and through the supervisor's
guardrail chain, so peer output cannot become the supervisor's instructions.

`RunBudget.sliced(limits)` is the new primitive underneath: a tighter ceiling for one
delegated run that is still deducted from the caller's ledger, so delegating cannot create
allowance. Spend is attributed under the worker's agent name whether the run finished, was
cancelled or exhausted its slice. A worker that runs out ends in `BUDGET_EXHAUSTED` and
comes back as a refusal the supervisor can reason about, unless the call declared
`fatal=True`. `supervisor.cancel()` cancels workers in flight and still records what they
spent, and `delegate(..., writes=key)` refuses a second worker reaching for a key another
holds instead of letting whichever finished last become the answer.

**Stability:** additive. `sliced` is new on the concrete budgets and the `BudgetPolicy`
protocol is unchanged, so external implementations keep working; a policy that cannot be
sliced raises `ConfigurationError` rather than silently widening a worker's ceiling.
Documented in `docs/delegation.md`, exercised by `examples/supervisor.py`.
- surface: tesserix_adk.core.HandoffContractError, tesserix_adk.runtime.HandoffDesk, tesserix_adk.runtime.Handoff, tesserix_adk.runtime.HandoffContract, tesserix_adk.runtime.HandoffQueue, tesserix_adk.runtime.HandoffResult, tesserix_adk.runtime.Receiver, tesserix_adk.runtime.narrowed_to
---

A conversation can now change hands under a declared contract instead of by forwarding the
transcript. Every staged flow that hand-rolled this — triage answers, then passes everything
it has to a specialist — paid for it three times over: context the target had no business
seeing, tokens for that context on every later turn, and a target inferring permissions from
a transcript that describes what the *source* was allowed to do.

A `Receiver` declares the Pydantic model it accepts as a `HandoffContract`.
`HandoffDesk.hand_off(to, reason=..., state=..., task=...)` validates the payload against it
and raises `HandoffContractError(reason="contract")` — naming the fields it got wrong —
before the target is invoked, so a refused handoff leaves the conversation exactly where it
was. Nothing else crosses unless the call passes `history` or `memory` explicitly.

Identity is not a parameter: `hand_off` takes no tenant and no user, both come from the
delegation, and the target runs holding the intersection of its own allowlist and the
source's. A target sharing no tool with the source is refused, as is a handoff made while
the source run is still in flight, which would leave an outstanding tool call owned by
nobody.

A person is a receiver like any other — a `HandoffQueue` is held to the same contract, and
`HandoffResult.queued` says the conversation is with someone rather than answered. The
payload passes the guardrail chain, redaction included, before it reaches the target, the
record or telemetry; each `Handoff` carries the run id and the path so a conversation that
has changed hands three times still reads as one trace.

`narrowed_to(agent, tools)` — the intersection helper the supervisor already used — moves to
`tesserix_adk.runtime.delegation` and is exported, since both delegation and handoff need
the same narrowing and a second copy would be a second set of rules.

**Stability:** additive. Documented in `docs/delegation.md`, exercised by
`examples/handoff.py`.
- surface: tesserix_adk.core.PlanValidationError, tesserix_adk.runtime.AgentPlanner, tesserix_adk.runtime.ExecutedPlan, tesserix_adk.runtime.InMemoryPlanStore, tesserix_adk.runtime.Plan, tesserix_adk.runtime.PlanExecutor, tesserix_adk.runtime.PlanRecord, tesserix_adk.runtime.PlanStep, tesserix_adk.runtime.PlanStore, tesserix_adk.runtime.Planner, tesserix_adk.runtime.StepResult, tesserix_adk.runtime.ToolContract
---

Planning is now two halves that cannot be collapsed back into one. `AgentPlanner` produces a
typed `Plan` and refuses at construction any agent that declares a tool or answers as
anything but a `Plan`, so a planner cannot dispatch what it is planning. `PlanExecutor`
validates the whole plan — registry, agent allowlist, delegated scope, argument schemas,
dependency graph — before the first step touches anything.

Nothing is repaired along the way. `PlanValidationError` carries the step, the tool, a
`reason` (`empty`, `too_long`, `unknown_tool`, `not_allowed`, `arguments`, `dependency`,
`cycle`, `replan`) and the raw planner payload; a `ToolContract` validates arguments
strictly, so `"2"` where the tool declared an integer is refused rather than coerced. A plan
longer than `max_steps` is refused rather than trimmed, and steps that wait on each other are
caught before they would deadlock.

Every step is cleared before any step runs. Where an `AutonomyLadder` classifies the tool the
grant decides, escalating to the approval gate or refusing outright; otherwise a contract
marked `irreversible` — or a tool on the agent's `approval_required_tools` — goes to a person.
A denial halfway down the plan therefore leaves nothing partially executed, and a step that
needs a person where no gate was configured is a `ConfigurationError` rather than a silent
pass.

`planned(planner, task)` asks again with the refusal as feedback, bounded by `max_replans`,
minting the next `revision` each attempt; past the allowance the last refusal is raised with
`reason="replan"` and `attempts`. Given a `PlanStore` the plan and its results are written
before the first step and after every one, and `resume()` revalidates against the contracts
as they are now — a tool's schema may have moved since the plan was made. Where an
`IdempotencyStore` is given, a repeated step returns its recorded outcome rather than running
again.

Plans persist through a `PlanStore` rather than the `CheckpointStore`: a checkpoint is
conversation shaped — messages, frontier, usage — and a plan is a graph of intended effects
with a different lifetime and a different reader. Four `RunEventKind` values are added —
`PLANNED`, `PLAN_REFUSED`, `REPLANNED`, `STEP_EXECUTED` — so a refused plan is as much a part
of the run's history as one that ran.

**Stability:** additive. Documented in `docs/planning.md`, exercised by `examples/planner.py`.
- surface: tesserix_adk.core.AggregationError, tesserix_adk.runtime.Aggregate, tesserix_adk.runtime.Aggregation, tesserix_adk.runtime.All, tesserix_adk.runtime.Branch, tesserix_adk.runtime.BranchOutcome, tesserix_adk.runtime.BranchResult, tesserix_adk.runtime.FirstSuccess, tesserix_adk.runtime.Quorum, tesserix_adk.runtime.Reduce, tesserix_adk.runtime.fan_out, tesserix_adk.runtime.Supervisor.tenant
---

`fan_out(supervisor, branches, into=..., max_concurrency=...)` runs several branches at once
through a `Supervisor` and adds them up under a rule that was declared. It adds no authority
of its own: every branch goes through `Supervisor.delegate`, so scope intersection, the one
shared ledger, the memory-key claim and the guardrail chain on the way back are unchanged.

The three properties `asyncio.gather` over a list of sub-agents loses are the three this has.
Concurrency is capped by a number somebody chose rather than by whatever the provider's rate
limiter allows, and `Aggregate.peak_in_flight` reports what the cap achieved rather than what
it permitted. Spend is attributed per branch in `Aggregate.spent`, including branches excluded
from the answer — work that was paid for and then left out is exactly the spend that goes
missing from a cost report. And an aggregate that cannot be formed is an `AggregationError`
carrying `strategy`, `reason`, `contributed` and `excluded`, never a smaller aggregate that
reads like a whole one.

Four strategies. `All` is the default and fails closed, because a branch missing from an
answer is usually a bug and a default that hides it is a default that ships it. `Quorum(n)`
forms from what answered provided enough did, and refuses a quorum nobody reached rather than
rounding it down. `FirstSuccess` takes the first branch **in declared order**, not the first
to finish: finishing order makes an answer depend on scheduling. `Reduce(fn)` hands the
contributing `BranchResult`s to a caller's own rule, so a reducer can attribute what it used.

Results are in declared order whatever order they arrived in, so two runs over the same
answers aggregate identically. Every branch is on `Aggregate.results` with a typed
`BranchOutcome` — `ok`, `failed`, `budget_exhausted`, `cancelled` — whether or not it is in
the answer.

A branch that exhausts a slice of its own is that branch's problem. A branch with no slice
that exhausts means the shared ledger has gone, so branches that had not started are refused
without running. A cancelled fan-out refuses rather than aggregating what happened to have
arrived, listing the branches that did answer among the excluded so the refusal does not read
as though it were about the others; branches still queued behind the cap never start.

`Supervisor.tenant` is exposed so a refusal raised on a supervisor's behalf is attributed the
same way its own are.

**Stability:** additive. Documented in `docs/parallel.md`, exercised by `examples/parallel.py`.
- **tesserix_adk.runtime.Delegation, tesserix_adk.runtime.DelegationLimits, tesserix_adk.runtime.DelegationScope, tesserix_adk.core.DelegationLimitError, tesserix_adk.core.ScopeEscalationError**: Multi-agent runs fail in two loud ways and one quiet one: they recurse until the budget is
gone, two agents pass the same task back and forth forever, or — the quiet one — a
sub-agent three levels down acts on the allowlist from its own configuration rather than
the narrowed scope of the caller it acts for. `Delegation` bounds the shape of a run and
narrows its scope on the way down. A delegation comes from `Delegation.root` or from
`parent.to(...)` and from nowhere else; the constructor raises `ConfigurationError`,
because one built by hand carries a scope nobody narrowed and a depth nobody counted.

`DelegationLimits` bounds three different things, because each catches a shape the others
miss: `max_depth` bounds one lineage, `max_fan_out` bounds one agent's children, and
`max_delegations` bounds their product, which is where a shallow, very wide tree would
otherwise escape both. Limits narrow downward only — a child declaring roomier ceilings
keeps its parent's, since a limit that can be raised from inside is not a limit. An agent
already on the current path is refused even where the depth ceiling would permit it, since
two agents alternating below the ceiling terminate only by exhausting the budget.

What a child holds is the intersection of what it asked for with what its parent holds.
Asking for a tool or mutation class the parent never held raises `ScopeEscalationError`
naming it, whether or not the child's own configuration would permit it. The tenant is not
a parameter of `to()` at all: a child inherits its parent's `TenantContext`, so crossing a
tenant boundary by delegation is unrepresentable rather than merely checked. A scope may
expire, read against a `Clock`; time that cannot be read fails closed, and a scope that
declares an expiry with no clock to read it against is refused at the root.

Every refusal happens before the child is created, so it spends nothing from the run's
allowance. `DelegationLimitError` carries a `reason` — `depth`, `fan_out`, `run`, `cycle`
or `expired` — and the path it happened on. Neither refusal is retryable: the same call
refused for the same reason refuses again, so a parent that retries it is looping rather
than recovering.

Budget narrowing is deliberately not part of this: a child inherits the run budget its
parent resolved, and per-delegation budgets belong with the spend ledger.

**Stability:** additive. Documented in `docs/delegation.md`, exercised by
`examples/delegation.py`.
- surface: tesserix_adk.core.AttributionError, tesserix_adk.observability.HEADER, tesserix_adk.observability.NODE_SPAN, tesserix_adk.observability.Node, tesserix_adk.observability.Pattern, tesserix_adk.observability.Rate, tesserix_adk.observability.RunTree, tesserix_adk.observability.TraceContext, tesserix_adk.observability.TreeTotals, tesserix_adk.observability.VERSION, tesserix_adk.observability.attributes_of_context, tesserix_adk.observability.node_of, tesserix_adk.observability.peer_node, tesserix_adk.observability.record_tree, tesserix_adk.observability.render, tesserix_adk.observability.totals_of, tesserix_adk.observability.tree, tesserix_adk.cli.Lookup, tesserix_adk.cli.inspect_main
---

A run made of a supervisor, three workers, a peer and a queued activity is five processes,
five traces and five cost figures, none of which answers what the run cost or which agent
spent it. `TraceContext` carries the link `Run` has no field for — root, parent, depth,
`Pattern` and branch — and `RunTree` totals the whole thing node by node.

The context crosses a process boundary as one `adk-trace` header, percent-encoded so an agent
named `x;tenant=globex` cannot rewrite the field beside it. A missing or unreadable header is
recorded as `broken` and carried forward to children rather than raised: dropping the work
because its trace went missing loses the spend as well as the trace. A `pattern` a newer sender
introduced reads as `activity` rather than severing the chain.

`tree` refuses anything that is not one run — `empty`, `no_root`, `two_roots`, `duplicate`,
`orphan` on `AttributionError.reason` — because each of those otherwise yields a tree that
silently drops a participant, and a dropped participant is dropped spend.

`TreeTotals` names what is outside the figure instead of folding it in. A worker that crashed
before reporting appears in `unattributed` and sets `lower_bound`; unknown spend counted as
zero is how a budget ceiling stops meaning anything. A peer billing in another currency reaches
the total only through a `Rate` somebody recorded, is listed in `converted`, and carries
`CostConfidence.ESTIMATED`; without a covering rate it stays unattributed rather than being
summed into a figure true in neither currency. A child whose clock ends before it starts is
marked `skewed` with a `latency_ms` of `0.0`.

`record_tree` emits one `adk.participant` span per participant and counters per participant
whatever the sampler did, so a wide fan-out cannot lose its cost to sampling — the money never
travels on a span. A participant with no tenant fails the export closed, before the first span
leaves.

`tesserix_adk.cli.inspect_main` draws an assembled tree with cost, tokens and latency per node;
the deployment supplies the lookup, the kit supplies the rendering and the exit codes.
`observability.totals_of` is now public, being what a per-step roll-up is built from.

**Stability:** additive. Documented in `docs/multi-agent-trace.md`, exercised by
`examples/multi_agent_trace.py`.
- **tesserix_adk.core.ApprovalTransport, tesserix_adk.core.ApprovalDeliveryError, tesserix_adk.runtime.TransportGate, tesserix_adk.runtime.self_granted, tesserix_adk.runtime.DEFAULT_APPROVAL_WAIT_SECONDS, tesserix_adk.runtime.TIMEOUT_IDENTITY, tesserix_adk.adapters.NatsApprovals, tesserix_adk.adapters.WebhookApprovals, tesserix_adk.adapters.ConsoleApprovals, tesserix_adk.adapters.MessagePublisher, tesserix_adk.adapters.HttpPoster**: Where an approval question is delivered and where the run waits for the answer are now two
things rather than one. `ApprovalTransport` has a single method, `deliver(record)`, which
returns the decision where the transport carries the answer back itself and nothing where the
answer will arrive out of band. `TransportGate` is the `ApprovalGate` over it: it holds the
call, `decide` settles it, and the delivery mechanism underneath is substitutable —
`NatsApprovals`, `WebhookApprovals`, `ConsoleApprovals`, or a callable of your own.

They are separated because they fail differently. A queue that is down is a delivery failure
and raises: nobody was asked, so nothing may proceed on the strength of it. An approver who is
asleep is a wait, and running out of patience is a denial decided by `system:timeout` rather
than a grant — never a person's name, because nobody decided it. An answer arriving after that,
or a second answer to a request already spent, settles nothing.

A *grant* whose `decided_by` names the agent that asked — bare, or written as `agent:`,
`service:`, `bot:` or `sa:` — is refused with the code `approval_self_granted`. An agent's own
service identity approving its own payment is not a second pair of eyes, and it is the shape an
over-broad token takes in practice.

The NATS subject carries the tenant (`adk.approvals.<tenant>`) so a subscriber can be authorised
for its own and no other, and a tenant that is not a plain subject token is refused rather than
published wider. The webhook body is signed over exactly the bytes sent and must be HTTPS; a
non-2xx answer, or a body that is not a decision about this request, is an
`ApprovalDeliveryError`.

**Stability:** additive. A runner given any existing gate behaves exactly as before, apart from
a self-granted approval now being refused. Documented in `docs/tool-approval.md`, exercised by
`examples/approval_transport.py`.
- **tesserix_adk.core.AutonomyLadder, tesserix_adk.core.AutonomyLevel, tesserix_adk.core.AutonomyOutcome, tesserix_adk.core.AutonomyGrant, tesserix_adk.core.AutonomyDecision, tesserix_adk.core.ActionClass, tesserix_adk.core.ActionRegistry, tesserix_adk.core.ActionRequest, tesserix_adk.core.Ceiling, tesserix_adk.core.GrantReader, tesserix_adk.core.GrantIssuer, tesserix_adk.core.CommitmentLedger, tesserix_adk.core.InMemoryGrants, tesserix_adk.core.RESERVED_ACTION_CLASS, tesserix_adk.core.AutonomyRefusedError, tesserix_adk.runtime.AutonomyGate, tesserix_adk.runtime.ReportLog, tesserix_adk.runtime.InMemoryReports, tesserix_adk.adapters.PostgresGrantStore, tesserix_adk.adapters.PostgresGrantSettings, tesserix_adk.adapters.GrantTables, tesserix_adk.adapters.DEFAULT_GRANT_TABLES, tesserix_adk.adapters.EXPECTED_GRANT_SCHEMA, tesserix_adk.adapters.GRANT_SCHEMA_VERSION**: How much an agent may do unattended is now a grant the runtime enforces rather than a number
in a config file. `AutonomyGrant` names who issued it, which action class it covers, up to
what ceiling, and when it expires — there is no non-expiring grant, and `act_within_limits`
without a ceiling is refused at construction because it is not a limit. `AutonomyLadder`
resolves one attempted action against the grants that cover it and answers `ACT`, `ESCALATE`
or `REFUSE`, naming the grant that decided even when it escalates: which grant was not enough
is the question an operator asks.

Everything unmatched falls to asking a human. A missing, expired or wrong-tenant grant, an
unregistered tool, an unreadable amount, a currency the grant is not in, or an amount over
the remaining headroom all escalate. Headroom is `Decimal` and is never rounded up to fit:
900 against 800 left escalates, and so does 900.01 against 900. A grant on `acme` does not
reach `acme/eu` unless it says so, since a grant that widened as the tenant tree grew would
be a grant nobody issued.

An agent cannot widen its own level. `AutonomyLadder` holds a `GrantReader`, which can only
read; issuance is a separate `GrantIssuer` the runtime is never given, so no object inside a
run has a path to a grant. The reserved class `autonomy.grant` is refused outright and
recorded as an attempted escalation rather than offered to a human.

`AutonomyGate` is what the loop consults, at the point a tool call would go out. `ESCALATE`
adds an approval requirement and records `autonomy_escalated`; `REFUSE` fails the run with
`AutonomyRefusedError` and records `autonomy_refused`. Autonomy only ever adds a gate — a
grant permitting unattended action never waives an approval the agent or the tool declared,
because the two answer different questions. `act_and_report` is enforced rather than trusted:
the gate records the obligation as it lets an action through, and the next action of that
class asks a human until the report is delivered.

`PostgresGrantStore` backs `GrantReader` and `GrantIssuer` with the database a deployment
already runs. It is append-only: an id already in use is refused rather than updated, so a
decision recorded against an id stays readable as what it permitted, and re-granting mints a
new id. Issuance is not retried — a retried insert that may already have landed is how one id
comes to mean two things — while reads retry a contended database and raise if they finally
fail, since a read that failed is not a read of no grants. The DDL stays the deployment's:
`EXPECTED_GRANT_SCHEMA` publishes the shape and `verify()` refuses any other at startup.

**Stability:** additive. The three levels are stable; a level added later is added beside
them. Documented in `docs/autonomy.md`, exercised by `examples/autonomy.py`.
- **tesserix_adk.core.CeilingLedger, tesserix_adk.core.InMemoryCeilingLedger, tesserix_adk.core.Hold, tesserix_adk.core.HoldState, tesserix_adk.core.Credit, tesserix_adk.core.exact, tesserix_adk.core.CeilingExceededError, tesserix_adk.core.InexactAmountError, tesserix_adk.adapters.PostgresCeilingLedger**: A ceiling is only a ceiling if it cannot be walked around, and it leaks in three standard
ways: two actions each read the same headroom and both fit, one action arrives as ten small
ones, and a timed-out action is retried onto fresh headroom on top of spend that already
happened. `CeilingLedger` answers all three the same way — headroom is taken before the
action and committed or released after it, keyed by tenant, action class, currency and
window, rather than read and then acted on.

What is held counts against the ceiling exactly as what is committed does, so a pending
escalation is not undercut by a parallel action spending the money a human is being asked
about. A hold nobody settles expires after `hold_seconds` and is reaped, because a process
that dies mid-call must not hold headroom until somebody notices.

The reservation is keyed by the call rather than the attempt, which is what makes a retry
of a call that may already have gone out ask about the same action. `AutonomyGate` takes
the headroom at decision time and the loop settles it after dispatch: a call a human
declined or one the batch never made gives it back, while a call that errored keeps it — a
tool that raised may still have moved the money, and a ceiling that assumes otherwise is
one a flaky tool walks through.

Arithmetic is `Decimal` throughout and amounts arrive through `exact`, which refuses a
float rather than rounding it: a limit built on `0.1 + 0.2` is off by whatever the hardware
felt like. Credits are recorded and never netted off, because subtracting a refund would
hand an agent fresh headroom nobody granted.

`PostgresCeilingLedger` is the cross-process story: the reserve is one `INSERT ... SELECT`
whose `WHERE` is the ceiling test, so a row lands under the limit or does not land, and
`idempotency_key` is a unique index rather than a lookup somebody could race. Amounts are
`numeric`. `EXPECTED_CEILING_SCHEMA` is the shape the adapter was written for, for the
migration repository to own.

**Stability:** additive. `AutonomyDecision` gains the `ceiling` that applied, and a gate or
ladder given no ledger behaves exactly as before. Documented in `docs/autonomy.md`,
exercised by `examples/ceiling.py`.
- **tesserix_adk.core.SuspendedRun, tesserix_adk.core.ApprovalToken, tesserix_adk.core.PendingDecision, tesserix_adk.core.TokenAttempt, tesserix_adk.core.SuspensionStore, tesserix_adk.core.TokenRedeemer, tesserix_adk.core.ApprovalTokenError, tesserix_adk.core.mint_token, tesserix_adk.core.digest_of_token, tesserix_adk.core.DEFAULT_SUSPENSION_SECONDS, tesserix_adk.core.RunState.SUSPENDED, tesserix_adk.core.RunEventKind.RUN_SUSPENDED, tesserix_adk.runtime.DeferringGate, tesserix_adk.runtime.MemorySuspensionStore, tesserix_adk.runtime.ApprovalDeferred, tesserix_adk.runtime.AgentRunner.resume_with_decision, tesserix_adk.cli.approvals_main, tesserix_adk.cli.Waiting, tesserix_adk.cli.Answering**: A run can now stop on a human decision for three days and carry on afterwards without
holding anything while it waits. `TransportGate` handles the minutes-long approval by
waiting; the working-week one it cannot, because a run that waits that long holds a worker,
a connection and a queue slot across two deploys and dies to the first of them.

`DeferringGate` puts the question on a transport, hands out a token, and raises
`ApprovalDeferred`. The loop writes the frontier at `CheckpointBoundary.BEFORE_APPROVAL`
with the held call on it, stores a `SuspendedRun`, records `RUN_SUSPENDED` and returns a run
in the new non-terminal `RunState.SUSPENDED` — no task, no connection, no in-memory state.
The signal carries the store as well as the token, so a gate keeps its suspensions wherever
it likes without the loop being told. A gate is refused the deferral outright where nothing
could carry the run on: no checkpointer, or a policy that does not write at that boundary.

`AgentRunner.resume_with_decision` is the other half. The token is redeemed once — spending
is atomic, and a second presentation raises `ApprovalTokenError` and executes nothing — and
the held call goes back through dispatch rather than round it. **A person saying yes is one
of the conditions, not all of them:** an autonomy window that closed while the run slept
closes it now, a withdrawn grant is still withdrawn, a tool whose schema moved refuses the
payload, and the approval ledger still binds the grant to the exact payload the approver was
shown. A token past its expiry resumes the run as a denial decided by `system:timeout`
rather than by the person who answered late. An agent that now names a different model is
refused unless `allow_model_drift=True`, because the answer was about what the first model
proposed.

Tokens are single-use, tenant-bound and expiring (three days by default), and carry the
digest of the arguments rather than the arguments. Stores keep the digest, never the value.
Every presentation is recorded as a `TokenAttempt` against the identity that made it,
accepted or not — a token presented twice is the shape of a replayed approval and is worth
more than a raised exception. `PendingDecision` is what a rota may show: what is asked, by
whom, why, when it closes, and the digest — never the account number.

`tesserix_adk.cli.approvals_main` answers one from a terminal — `list`, `approve`, `deny` —
over whatever store and resume the deployment supplies, since the kit cannot know either. It
prints digests rather than payloads, reports a token that bought nothing as exit code `1`
rather than a traceback, and installs no global `adk` binary to fight with the consumer's.

**Stability:** additive. `RunState.SUSPENDED` is the first non-terminal state a returned run
can be in, so a caller that assumes every returned run is finished should check for it;
nothing reaches that state without a gate that defers, and `TransportGate` never does.
Documented in `docs/suspension.md`, exercised by `examples/suspension.py`.
- **tesserix_adk.core.AuditEvent, tesserix_adk.core.AuditDecision, tesserix_adk.core.AuditSink, tesserix_adk.core.AuditUnavailableError, tesserix_adk.core.digest_of_arguments, tesserix_adk.core.pseudonym, tesserix_adk.runtime.AuditTrail, tesserix_adk.runtime.MemoryAuditSink, tesserix_adk.runtime.AutonomyGate.record, tesserix_adk.adapters.PostgresAuditSink, tesserix_adk.adapters.PostgresAuditSettings, tesserix_adk.adapters.AuditTables, tesserix_adk.adapters.JetStreamAudit, tesserix_adk.adapters.JetStreamPublisher, tesserix_adk.adapters.EXPECTED_AUDIT_SCHEMA, tesserix_adk.adapters.AUDIT_SCHEMA_VERSION, tesserix_adk.adapters.DEFAULT_AUDIT_TABLES, tesserix_adk.adapters.DEFAULT_AUDIT_SUBJECT**: Every autonomous decision is now written down, on a path telemetry cannot drop. The question
asked after an incident is "what did this agent do unattended, and what did it decline to
do?", and spans cannot answer it: they are sampled, dropped under load, and stripped of the
context that made the decision. Refusals left nothing behind at all, so nobody could show
that a ceiling had actually held.

An `AuditEvent` is one decision about one attempted call: the run and a monotonic per-run
sequence, tenant and user, agent and version, tool, action class, the autonomy level applied,
the grant that permitted it, ceiling headroom either side, who approved where a human did,
and an idempotency key so a retried activity writes one record rather than three. Four
decisions are recorded with equal weight — `executed`, `escalated`, `refused`, `revoked` —
which is the point: an escalation with `headroom_before` and `headroom_after` equal is the
evidence that the ceiling held.

`AutonomyGate` records what it stops; the run loop records `executed` at the single point a
call is cleared past every gate, so a call the ladder permitted but a tool-declared approval
later stopped is never recorded as executed, and a grant withdrawn while a run waited on a
person is recorded as `revoked`. Arguments are never stored: the payload goes through the
guardrails' redaction and then a digest, with `redact_patterns` for shapes only the
deployment knows.

**If the sink cannot take the record, the call does not go out.** `AuditTrail` raises
`AuditUnavailableError` and the loop fails the run before dispatch rather than performing an
action nobody can defend afterwards.

`PostgresAuditSink` is the queryable store — append-only, one row per decision, unique on
`(run_id, idempotency_key, decision)`, ordered so a run's decisions read back in the order
they were taken. `JetStreamAudit` publishes the same record on `adk.audit.<tenant>` for a
deployment that wants audit off the transaction path or in a second administrative domain.
`MemoryAuditSink` is for tests. An erasure request pseudonymises the person and keeps the
decision — deleting the row would take the evidence that the action was permitted with it —
and `EXPECTED_AUDIT_SCHEMA` grants that one non-insert statement to a role of its own.

**Stability:** additive. `AutonomyGate` takes `audit` as an optional keyword and behaves
exactly as before without it. Documented in `docs/audit.md`, exercised by
`examples/audit.py`.
- **tesserix_adk.core.Revocation, tesserix_adk.core.InFlightPolicy, tesserix_adk.core.GrantRevokedError, tesserix_adk.runtime.RevocationWatch, tesserix_adk.runtime.RevocationBroadcast**: Autonomy can now be taken back from work already under way. `Revocation` names one grant, or
a tenant, or a tenant and an action class, along with who withdrew it and when — enough to
stop a class of work across a fleet without knowing every id issued for it. A revocation that
names neither a grant nor a tenant is refused at construction, because it would either do
nothing or withdraw the world.

It lands on the very next action, not at the next deployment: grants are read from the store
per attempted action rather than once at run start, which is exactly what makes a withdrawal
prompt. Withdrawal is an append and never a delete, so a revoked grant cannot be reactivated
— re-granting mints a new id, and what was withdrawn stays readable as what it permitted
while it stood.

A run suspended on an approval was asleep while the authority behind it could have been taken
back, so the loop re-checks after the human decides and before anything goes out. It records
`grant_revoked` and then, per the gate's `revoked_runs`, either fails with
`GrantRevokedError` naming the grant and who withdrew it (`InFlightPolicy.CANCEL`, the
default) or proceeds on the approval it has while every later action of the class asks
(`InFlightPolicy.ASK_ALWAYS`).

`RevocationBroadcast` carries withdrawals to every process over the deployment's bus and
`RevocationWatch.follow` consumes them, but the bus is an accelerator and never the
authority: a missed message costs latency, and the store re-read is what refuses. The watch
fails closed on itself too — a view nobody has confirmed within `stale_after_seconds`
refuses unattended action rather than acting on what it last heard, since a process cut off
from the bus cannot know a grant is still live. It never fails open in the other direction: a
stale watch turns `act` into `refuse` and leaves an escalation as it was.

`PostgresGrantStore.revoke` appends to `adk_grant_revocations`, which the dispatch-path read
excludes; a repeated withdrawal is the same withdrawal rather than an error, because a caller
retrying one must not be told the authority is back. `GRANT_SCHEMA_VERSION` is 2.

**Stability:** additive. `revoked_runs` defaults to `CANCEL`, and a gate given no
`RevocationWatch` behaves exactly as before. Documented in `docs/autonomy.md`, exercised by
`examples/revocation.py`.
- **tesserix_adk.guardrails.Guard, tesserix_adk.guardrails.GuardrailPipeline, tesserix_adk.core.GuardResult, tesserix_adk.core.GuardStage, tesserix_adk.core.GuardVerdict, tesserix_adk.core.GuardrailError, tesserix_adk.core.GuardrailEvaluationError**: A safety check written inline in application code is invisible in the agent's definition:
the same agent is guarded in one product and unguarded in another, and nobody can tell by
reading either. An inline check that raises is usually swallowed too, so the run carries on
with nothing checking it — an unavailable guard silently becomes a permissive one.

`GuardrailPipeline` is the order a run's checks are asked in, applied by the loop at both
ends: input before the prompt is assembled, output before the answer is used. There is no
path to the provider that skips it and no per-agent wiring to forget. The order is the
order guards were declared, never registration timing, so where two guards disagree the
first block ends the pipeline and the more restrictive verdict is what the run acts on,
deterministically.

A guard answers with a `GuardResult`: `allow`, `redacted(content, code=…)` — continue on
this content, which the guards after it see and which is what comes back — or
`blocked(code=…)`, which stops it in any form. A block carries a machine-readable `code`
so a caller matches on why rather than on a sentence that will be reworded, and a `detail`
that is safe to log: never the offending content, which is the one thing an error carrying
it would spread into every log that catches it.

Failing closed is the point. A guard that raises, exceeds `timeout_seconds`, or answers
with something that is not a `GuardResult` raises `GuardrailEvaluationError` with a `reason`
of `raised`, `timeout` or `unreadable`; the content does not continue and the guards after
it are not asked. It shares a `GuardrailError` base with `GuardrailViolationError` — catch
the base to stop either way, the subclass to tell a decision from an outage. Neither is
retryable. Cancellation is not a verdict: `CancelledError` propagates rather than being
recorded as a refusal, because the caller withdrew the question.

In a run, a block or an evaluation failure ends it as `FAILED` with a `guardrail_refusal`
event naming the guard; a redaction is applied to what continues and records
`guardrail_redaction`. Each verdict is also a `GuardrailDecision` progress event carrying
the guard, the stage and what it decided — never the content.

`check_stream` buffers the whole answer before handing any of it on. That costs the latency
streaming was for, and the alternative is emitting the first half of something a guard was
about to block.

**Stability:** additive, alongside the `Guardrail` protocol change noted under breaking
changes. Documented in `docs/guardrails.md`, exercised by `examples/guardrails.py`.
- surface: tesserix_adk.core.TenantContext, tesserix_adk.core.MissingTenantContextError, tesserix_adk.core.TenantCrossingError, tesserix_adk.core.current_tenant, tesserix_adk.core.tenant_here, tesserix_adk.core.tenant_scope, tesserix_adk.core.bound
---

Tenant identity is now a property of the execution context rather than an argument somebody
remembers to pass. `tenant_scope` binds a `TenantContext` at a boundary — an HTTP handler, a
queue consumer, a run — and `current_tenant()` is what every egress point below it reads. The
binding is a contextvar, so it survives `await`, `asyncio.gather`, `TaskGroup`,
`asyncio.to_thread` and `create_task`; `bound` carries it into `loop.run_in_executor` and a
bare `ThreadPoolExecutor.submit`, which copy no context of their own.

`AgentRunner.run`, `resume` and `resume_with_decision` establish the scope for the whole run,
so a tool body reads the tenant it was never given. Two runs racing under different tenants
cannot observe each other, and a handler that has bound one tenant and starts a run for another
is refused with `TenantCrossingError` rather than quietly rebinding.

Absence raises. `current_tenant(where=...)` throws `MissingTenantContextError` naming the egress
point instead of falling back to a default tenant, because a default is one typo away from
being every tenant — an unscoped recall reads every tenant's memory and looks like an answer.
`MemoryScope.here()` builds a scope from the context, defaulting `user_id` to the acting
principal, and refuses outside one.

A crossing between tenants is legitimate and has to say why: `tenant_scope(other,
crossing="registry backfill")` records the reason on the bound context, and a crossing with no
reason is refused.

`TenantContext` moves to `tesserix_adk.core.tenancy` and gains `user`'s companions — `locale`,
`region`, `correlation_id`, `crossing` — plus `acting_as`. It stays importable from
`tesserix_adk.core` and `tesserix_adk.core.run`, keeps `tenant` and `user`, and is still frozen.

**Stability:** additive; `TenantContext`'s existing fields and both import paths are unchanged.
Documented in `docs/tenancy.md`, including the two known limitations — store types keep their
explicit tenant arguments, and an async generator holds its binding between yields, which the
crossing rule makes loud rather than silent. Exercised by `examples/tenancy.py`.
- surface: tesserix_adk.core.HEADER, tesserix_adk.core.PAYLOAD_KEY, tesserix_adk.core.VERSION, tesserix_adk.core.MAX_HEADER_BYTES, tesserix_adk.core.TenantContextError, tesserix_adk.core.TenantRefusal, tesserix_adk.core.header_of, tesserix_adk.core.carried, tesserix_adk.core.restored, tesserix_adk.core.in_payload, tesserix_adk.core.of_payload, tesserix_adk.core.arriving, tesserix_adk.testing.TenantPropagationConformance
---

The tenant now survives a hop that leaves the process, under one wire contract instead of a
field name per integration. `carried(current_tenant())` produces the headers to attach;
`arriving(headers)` binds what came back on the far side; `in_payload` / `of_payload` do the
same for carriers whose message is its input — a queued item, a durable workflow argument.
One header name (`adk-tenant`), one payload key (`adk_tenant`), one versioned encoding
(`adk/1 tenant=…;user=…`), percent-encoded so a value containing a separator cannot rewrite
the field beside it.

Ingress refuses rather than guesses, and each refusal names itself so a consumer branches on
`TenantContextError.reason` rather than on message text: `missing` where nothing was sent
(the worker's own tenant is never a default), `malformed` where no tenant is named, `version`
where the encoding is one this build does not read, `contradicted` where the message disagrees
with the caller's authenticated claim — the payload never outranks the credential — and
`oversized` where even the tenant alone exceeds the header ceiling, since half a tenant name
is a different tenant. A consumer that already holds a different tenant gets the existing
`TenantCrossingError` instead of a silent rebind.

Durable work carries the context as input rather than as ambient state, so a workflow replayed
on another worker reconstructs the tenant it started under, and a retry or dead-letter
redelivery reads the same bytes and lands on the same tenant. A transport with a header
ceiling sheds optional fields least-load-bearing first (`crossing`, `correlation_id`,
`region`, `locale`, `user`) and marks the result `partial`, so the far side can tell a field
that was absent from one that was lost — `TenantContext` gains that `partial` flag.

`TenantPropagationConformance` is published for transports to run against themselves: it
checks the whole context survives, that header case is not load-bearing, that a ceiling still
delivers an intact tenant, and that consecutive messages on one connection do not bleed.

**Stability:** additive; `TenantContext` gains one defaulted field and nothing changes shape.
Documented in `docs/tenant-propagation.md`, including the limitation that only the carriers
that exist are wired — `tesserix_adk.mcp`, `tesserix_adk.a2a` and `tesserix_adk.workflows` are
still placeholder modules, so the MCP metadata wrapper, A2A middleware and Temporal
interceptor adopt this contract when those modules land. Exercised by
`examples/tenant_propagation.py`.
- **tesserix_adk.models.gguf, tesserix_adk.models.providers**: Agents now run on a machine with no GPU. `LlamaCppProvider` puts `llama-server` behind the
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
instead of being OOM-killed mid-run. `quantization_for` picks a format and is never
heavier than `Q4_K_M`, the published trade-off point, because more bits past it buys little
quality and costs the memory bandwidth that is the whole budget on CPU.

**Stability:** additive. Documented in `docs/cpu-inference.md`, exercised by
`examples/cpu_inference.py`, `tests/test_gguf.py` and `tests/test_provider_llama_cpp.py`.
- **tesserix_adk.runtime.context**: `ContextWindow` decides what goes into the context and what leaves it when there is no
room, so a retrieval loop holds the most useful tokens rather than the most recent ones.
Admission is **keyed**: a `Segment` carrying a `key` — a chunk id, a document hash — is
refused if that key is already held in any layer, and `admit` returns `False` to say so.
Re-injecting a chunk the model already has is a rounding error on a GPU and seconds of
prefill per turn on CPU. `key=None` is never deduplicated, so two turns of conversation
that read identically both still happened.

`fit()` evicts until what is held fits `limit_tokens` and returns what left, in the order
it left, so a caller can log it or re-rank it for a later turn. Conversation goes first,
oldest first; then retrieval, lowest-scored first — it was ranked for this turn, so the
ranking is believed. The cacheable prefix — system, tools, pinned — is **never** evicted
even where dropping it would free the most tokens fastest: trimming it invalidates the
prefix and every downstream turn pays the refill, which is a cost dressed as a saving. A
prefix that cannot fit alone raises `ContextWindowExceededError` rather than being
trimmed. Eviction frees the key, because a chunk dropped for room is no longer content
the model has and is admissible again later.

Counting uses the tokenizer the window was given, defaulting to `approximate_tokens`;
pass the server's own where the boundary has to be exact. `texts(layer)` hands what
survived straight to `assemble_prompt`.

**Stability:** additive. Documented in `docs/context.md`, exercised by
`examples/context_window.py` and `tests/test_context.py`.
- **tesserix_adk.observability.metrics**: Prompt-cache hit ratio is now a first-class number at every level, because prefix stability
is unfalsifiable without one: every change to prompt assembly reads as an improvement if
nothing counts what the server re-evaluated. On CPU that is not a cost nicety — prefill
dominates, so a prefix that stopped being stable is a deployment that stopped being usable.

`Usage` gains `fresh_input_tokens` (input with cache reads taken out, never negative —
vendors disagree about whether a cache read sits inside the input count, and a negative
token count is believed by whatever divides it next), `cache_hit_ratio`, and `measured`.
`Totals` gains `cached_tokens`, `cache_write_tokens`, `hit_ratio` and `measured`, so
`totals_by` aggregates the cache question component-wise along the same dimensions as the
money. Cache writes total apart from reads because they are priced apart, often at a
premium: folding them together makes caching look free on the turn that is paying for it.

Nothing read is a ratio of **zero**, not a division error, and `measured` says which of the
two a reader is looking at — `hit_ratio == 0.0` with `measured is False` means nobody sent
anything, and with `measured is True` means the cache missed every time. A dashboard that
cannot tell those apart reports an outage as perfect behaviour.

`record_spend` emits two new counters, `adk.input_tokens` and `adk.cached_tokens`, under
the same dimensions as `adk.cost`. They are separate rather than one ratio because a ratio
cannot be re-aggregated: averaging the hit ratios of two series of different sizes gives a
number true of neither, while two counters divide correctly at any grouping.

**Stability:** additive. Documented in `docs/cost-attribution.md`, exercised by
`examples/cache_hit_ratio.py` and `tests/test_cache_metrics.py`.
- **tesserix_adk.core.ClaimCheckPolicy, tesserix_adk.core.ClaimCheckStore, tesserix_adk.core.ClaimTicket, tesserix_adk.core.claim_handle, tesserix_adk.core.HANDLE_PREFIX, tesserix_adk.core.ClaimUnavailableError, tesserix_adk.runtime.ClaimCheck, tesserix_adk.runtime.MemoryClaimCheckStore, tesserix_adk.tools.claim_check_tool, tesserix_adk.tools.DEFAULT_FETCH_CHARS**: A tool that returns a contract, a log file or a scraped page puts that content in the
conversation, and the conversation is re-sent on every iteration after it. The model reads
it once; the prefill pays for it every turn. `ClaimCheck` checks an oversized result in
instead: the content goes to a store, and what enters the conversation is an extractive
head and a handle. Bind it with `AgentRunner(..., claim_check=ClaimCheck(store=store))`.

The head is cut at a boundary the content itself provides — a blank line, a newline, the
end of a sentence — rather than mid-word, because a head that reads as damage gets fetched
whether or not it was needed, which is the cost this exists to avoid. `ClaimCheckPolicy`
sets the threshold below which nothing happens, how much stays, and how long the rest is
kept; a head no smaller than the threshold is refused at construction, since the
substitution would be as large as what it replaced. `per_tool` policies exist because a
tool returning a whole PDF and one returning a row count are not the same decision.

A handle is scoped to the tenant and the run that made it, and the scope is hashed into
the handle as well as checked on the lookup — a handle from another run is not merely
refused, it cannot be derived. Identical content within one run derives one handle, so a
tool called twice is stored once. A fetch outside that scope, past the retention window,
or for a handle nobody stored all answer the same way: `ClaimUnavailableError`, which the
retrieval tool turns into a `claim_unavailable` refusal. Distinguishing "gone" from "not
yours" would tell a caller which handles other runs hold.

`claim_check_tool(store)` builds the read-only `fetch_result(handle, offset=0)` tool that
redeems a handle. It returns a window, `DEFAULT_FETCH_CHARS` wide, not the document — a
fetch that returned the whole thing would put back into the conversation exactly what
checking it in took out, one tool call later. The content is never summarised and the head
is never paraphrased: what comes back is what the tool returned, because a model handed a
plausible substitute for a document it asked to read cannot tell it is reasoning about
nothing.

Checking in happens *after* the tool-result boundary, never instead of it, so what is
stored has already been validated against the tool's declared type and had structural
forgery neutralised. The run loop records a `tool_result_stored` event naming the size and
the handle, never the content. `MemoryClaimCheckStore` is the in-process default; a
deployment that must survive a failover binds its own `ClaimCheckStore`, whose `forget`
is where right-to-erasure reaches this content.

**Stability:** additive. Without a `claim_check` bound, an oversized result is still cut at
`max_tool_result_chars`. Documented in `docs/claim-check.md`.
- **tesserix_adk.core.Dispatch, tesserix_adk.core.DispatchNode, tesserix_adk.core.DispatchResult, tesserix_adk.core.NodeResult, tesserix_adk.core.NodeOutcome, tesserix_adk.core.DependencyCycleError**: Some work is neither a chain nor a flat fan-out: two lookups feed one comparison, three
retrievals feed one summary. Expressed as nested sequential and parallel steps, that shape
makes branches wait for each other for no reason, and the hand-written schedule goes stale
the first time a step is added. `Dispatch` takes the dependencies as declared on each
`DispatchNode` and derives the schedule from them, so independent branches run together
without anyone scheduling them, and a join starts when its inputs exist rather than when a
level finishes. It is given exactly what its dependencies returned, keyed by their names —
a node that reaches around the graph for its input has a dependency nobody declared.

A graph that could never run is refused where it is written. A cycle raises
`DependencyCycleError` naming the nodes in it, because a cycle discovered at runtime is a
set of tasks waiting on each other and that is indistinguishable from work that is merely
slow. A dependency no node declares, a name used twice, an empty graph and a width that
could never start anything are all `ConfigurationError` at construction. `order` reports the
derived grouping — what the graph permits to run together — while the run itself is finer
than that grouping.

A failure is contained rather than fatal. What depended on it is skipped rather than run
with a missing input, since a join over an absent branch produces an answer built on
nothing, and `blocked_by` names the failure rather than the skip in between. Branches that
did not depend on it still finish, because after a partial failure the useful question is
which parts of the answer exist. `failures` carries the exception itself, so a caller can
re-raise it or match on its type instead of parsing a message, and asking a failed or
skipped node for its value raises `KeyError` rather than returning `None` — a missing value
read as `None` is how a partial answer is mistaken for a whole one. Cancelling the run
withdraws the question rather than failing a node, so `CancelledError` propagates instead of
being recorded.

**Stability:** additive. Documented in `docs/dispatch.md`, exercised by `examples/dispatch.py`.
- **tesserix_adk.tools.Sandbox, tesserix_adk.tools.SandboxLimits, tesserix_adk.tools.SandboxResult, tesserix_adk.tools.SandboxArtifact, tesserix_adk.tools.SubprocessSandbox, tesserix_adk.tools.sandbox_tool, tesserix_adk.tools.DEFAULT_LIMITS, tesserix_adk.core.SandboxError, tesserix_adk.core.SandboxTimeoutError, tesserix_adk.core.SandboxMemoryError**: Model-generated code is untrusted by construction: whatever wrote it read tool output,
retrieved documents and user text, any of which an attacker may have supplied. Running it
in the agent's process hands an injected prompt that process's credentials, its network
position and its filesystem. `SubprocessSandbox` runs it somewhere else — a fresh
interpreter under `-I -S`, in a temporary workspace deleted when the call returns, with an
environment constructed rather than inherited, so nothing of the host's is in it. `socket`
is replaced before the code is compiled, so a connection raises rather than dials.

`SandboxLimits` bounds what it may spend. Wall time and processor time are different
diagnoses — elapsed time catches code waiting for something that will never arrive,
processor time catches code spinning, and four threads burn four processor seconds in one —
so hitting either raises `SandboxTimeoutError` with a `limit` that says which. `RLIMIT_AS`
is set on the child before any generated code runs, so an allocation that would starve the
host raises `SandboxMemoryError` inside the sandbox instead; Linux enforces it, macOS
refuses address-space ceilings outright and the time ceilings still bound the run there.
Output and artifacts are bounded as well, both being channels back into the conversation.

A non-zero exit is a `SandboxResult`, not an error: the caller wanted to know what the code
did and a traceback is what it did. Errors are reserved for the case where the sandbox took
the process away and there is no result to report, since a result object for a run that
produced none invites reading half an answer as a whole one. Files the code wrote come back
as `SandboxArtifact`s in name order, capped by count and by size; files the caller handed in
do not, having been paid for once already.

`sandbox_tool(sandbox)` is what an agent calls. It is not parallel-safe by declaration —
two runs in one workspace would see each other's files — carries its own ceilings so a tool
exposed to a model can be more careful than the backend it calls, and turns a fired ceiling
into a `sandbox_limit_exceeded` refusal, because running the same code again hits the same
ceiling.

This is defence in depth inside one process tree, not a virtual machine, and the docs say
so: code that reaches for `ctypes` is one kernel boundary from the host, and that boundary
is the container the agent already runs in. `Sandbox` is the seam for the stronger case — a
deployment binding gVisor, Kata or a remote executor changes nothing above it.

**Stability:** additive. Documented in `docs/sandbox.md`, exercised by `examples/sandbox.py`.
- **tesserix_adk.core.definition**: `AgentDefinition` states an agent as the artifact that was reviewed rather than as a
construction call site: the declaration itself plus the owner who answers for it, the
evaluation suite that checks it, the prompt entry it was written for, the memory policy it
runs under and the schema it answers in. Scattered across the places that build an agent,
none of that can be diffed in review or named by a run that has already happened.

`AgentDefinition.declared(..., known_tools=...)` refuses an allowlist naming a tool nobody
registered, at construction rather than at the first execution in production that happens
to call it. `Owner` refuses a contact nobody can be paged at.

The `revision` is a digest of everything the definition says, so an edit produces a new
revision rather than moving what an old run pointed at. `AgentRunner.run` and `run_sync`
now accept a definition in place of a bare agent and pin its revision onto `Run` as
`definition_revision` and onto every span as `adk.definition` — a past run names the exact
revision, not a version that may have been edited since. A run started from a bare agent
records `None`, attributed as `unknown`.
- **tesserix_adk.core.RunGrant, tesserix_adk.core.Run.grant, tesserix_adk.core.RunContext.grant, tesserix_adk.core.RunEventKind.SCOPE_REFUSED, tesserix_adk.runtime.handed_back**: A kit with two dispatch paths grows a control that covers one of them. Guardrails covering
tool calls but not delegation leave the cheapest bypass in the system open: hand the work
to a sub-agent that declared no guard, and every check the caller ran under is gone. This
closes that path, in the runtime rather than in a convention.

A run now states what it was allowed to do. `RunGrant` carries the tools it could call,
which of those a human had to clear first, and the guards it ran under, in order. It sits
on `Run.grant`, and `Run.context` carries it, so `runner.run(child, parent=parent.context)`
inherits it rather than falling back to the child agent's own configuration.

A delegated run is subject to every guard its caller was, in its caller's order, followed
by any of its own that were not already there — a child cannot drop one, and a guard named
twice is asked once. A guard the child's runner was never given is a `ConfigurationError`
at the boundary, since a sub-agent wired without its caller's check would run outside all
of them. A tool the caller never held is refused with `ScopeEscalationError`, recorded as
`SCOPE_REFUSED` and terminal, before a model is called — refused rather than silently
intersected, because the difference is a wiring mistake and a quiet intersection is how
nobody finds out about it. Narrowing holds at every depth: a grandchild cannot recover what
the level above it gave up. Approval requirements are inherited too, so a call a human had
to clear at the top is not cleared by being made one level down.

`handed_back(run)` is how a child's answer reaches its caller: the same untrusted-data
envelope a tool result crosses in, because a sub-agent's answer is model output that read
whatever the sub-agent read, and pasted in bare it is an instruction channel for whatever
wrote it. A run a guard stopped hands back the guard and its code rather than an empty
string, so a refusal in a child cannot reach the caller as an unexplained silence.

A `RunContext` built by hand outside the loop carries no grant, and narrows nothing: the
absence of a record is not a claim that the caller held nothing. Every context the loop
produces carries one.

**Stability:** additive. Documented in `docs/delegation.md`, exercised by
`examples/delegation.py`.
- **tesserix_adk.core.trust, tesserix_adk.core.routing, tesserix_adk.core.fallback**: Fallback can no longer trade a data-handling guarantee for an availability one.
`TrustBoundary` states where a model sits on three axes — `tier`, `hosting`, `residency` —
and `ModelSpec` carries it. A fallback is legal only between models that share a boundary:
routing drops the rest from the chain, records them in `RoutingDecision.excluded_by_boundary`
and in `rejected` with the axes that differ. Where the chain is spent and the only
alternatives left are outside, the run fails closed with `TrustBoundaryError` and no request
reaches the out-of-boundary provider.

A boundary nobody declared constrains nothing, so an existing deployment routes exactly as
it did; a boundary declared on one model is enough to protect it, because an undeclared
target is an unknown one and unknown is not equal.

Every model choice now records what produced it. `RoutingDecision` gains `required`,
`min_context_window_tokens` and `boundary` alongside the candidates it already listed, and
`explain()` reads them as one line. Everything in the rationale is drawn from a closed
vocabulary — refs, capability names, boundary axes, rule scope — so a trace carries no
prompt content and can be kept for a sealed matter.
- **tools.release**: The release version is now derived rather than remembered. `make release-plan` reads the
pending change fragments and the public API snapshot diff, applies the policy in
`docs/versioning.md` — before 1.0 the minor is the breaking channel, from 1.0 onward the
major is, and new surface is never a patch — and prints the version with every reason it
has to move. `make release VERSION=…` then folds the notes into the changelog, consumes
the fragments, and stops, printing the commit and tag commands rather than running them:
pushing the tag is the publish, and that stays a decision somebody makes.

Where a fragment and the snapshot disagree the snapshot wins, because a fragment is a
claim and the snapshot is evidence. A breaking fragment carrying no migration note is
refused outright — a breaking entry with no instructions is the failure the mechanism
exists to prevent, so it stops the release rather than the review.

The GitHub Release is now the channel that always ships. `publish` is gated on the
repository variable `PUBLISH_TO_PYPI`, matching how `alpha.yml` already gates alphas, and
`mirror` no longer waits on it — a skipped job skips everything behind it, so a tag pushed
before the trusted publisher exists would have produced no consumable release at all. A
`smoke-mirror` leg installs each extra from the release assets with `--find-links` and
runs `examples/getting_started.py`, so the path consumers actually use is exercised on
every release instead of only documented.

Folding the notes now replaces the hand-written `[Unreleased]` body rather than sitting
above it. Entries are written there by hand as each change merges and describe the same
work the fragments do, so the released section carried every entry twice — once derived,
once as it was typed. Earlier releases and the link definitions are untouched.

**Stability:** no runtime surface changes; this is release tooling and CI topology.
Documented in `docs/releasing.md`.
- **runtime**: enforce budgets mid-run and fail closed
- **release**: publish tesserix-adk from a release tag
- **core**: versioning policy with enforced deprecation window
- **core**: resolve configuration from code, env and file with provenance
- **core**: optional extras per integration behind require_extra
- **core**: declare the public API surface and gate it on a snapshot
- **testing**: version matrix, network isolation and bounded quarantine
- **core**: protocol boundaries with construction-time conformance checks

### Changed

- Typed run results: `Agent[TripPlan]` runs to a `Run[TripPlan]` and `run.output` is a
`TripPlan` rather than a dict. `Agent` and `Run` take one type parameter bound to
`BaseModel`, `AgentRunner.run` and `run_sync` carry it through, and `Run.with_output` takes
the instance. The parameter defaults to the new `NoOutput`, a model with no fields, so an
agent declaring `free_text=True` needs no annotation and every existing bare `Run` or
`Agent` annotation still reads unchanged. A run built by the loop is parameterised at
runtime as well, because a bare `Run` serialises a typed answer away to `{}` and the
checkpoint of a typed run would lose it. Rehydration names the type —
`Run[TripPlan].model_validate_json` gives back the instance and an unparameterised read is
refused rather than silently dropping the answer, since nothing on the wire says which type
it was. Inference is asserted rather than assumed: `tests/test_typed_results.py` states
what the checker must infer with `assert_type` and what it must keep rejecting with
`type: ignore`, both under `mypy --strict` in `make check`. Documented in `docs/typing.md`,
exercised by `examples/typed_results.py`.

**Stability:** breaking. `run.output` moves from `dict[str, Any] | None` to
`OutputT | None`, so a caller reading `run.output["nights"]` reads `run.output.nights`
instead, and `run.output == {...}` compares against the model. Nothing else changes shape:
the type parameter's default keeps every unannotated use of `Agent` and `Run` valid, and
the serialised form of a run is byte-identical.
- **tesserix_adk.testing.ScriptedProvider, tesserix_adk.testing.StallingProvider**: `ScriptedProvider.stream` and `StallingProvider.stream` replay the script instead of
raising `NotImplementedError`: word-wise `TextDelta`s, any reasoning, one `ToolCallDelta`
per call, `UsageDelta`, then `StreamEnd`. One script drives both paths, so a test asserting
that the streamed and buffered views of a run agree is asserting about the runtime rather
than about two fakes written to agree.

`MASK`, `SENSITIVE_SHAPES`, `looks_sensitive` and `scrub` moved to `tesserix_adk.core` so
the runtime can redact without importing observability; `tesserix_adk.observability` still
exports the same names. `scrub` masks the matching substring rather than the whole value,
which is what keeps a redacted tool argument renderable.
- **tesserix_adk.core.routing**: `RoutingDecision.explain()` now names the capability floor and the context floor the work
asked for, and the count the trust boundary excluded. A choice explained only by its rule
does not say what the rule was answering, which is the half of the question an operator
reading a bill actually has.
- **core**: skip rebinding an identical tenant and re-record the baseline

### Fixed

- **tesserix_adk.observability.attribution**: A model step whose provider reported no usage is now `Cost.unknown()` rather than a
counted zero. A call at a price nobody knows is not a call that was free, and recording it
as free is a false statement that totals up into a bill — the group it lands in now reads
`UNKNOWN` confidence instead of quietly understating. A tool step still has no price
rather than an unknown one, which is a different thing and stays `Cost.nothing()`.

**Stability:** `Totals.cost.confidence` changes for groups containing an unpriced model
step. The amount is unchanged; what changed is that the total stops claiming to be counted.
- **runtime**: keep the in-flight revocation policy on the autonomy gate
- **tools**: map a hard cpu-ceiling kill to the cpu timeout it is
- **tests**: name the redaction fixture so the secret scan reads it as one
- **ci**: scope the inventory check to the committed lock
- **ci**: start the full matrix when a pull request is labelled
- **ci**: stop the alpha and CI runs cancelling each other
- **core**: raise pyyaml floor to 6.0.2 for cp313 wheels

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
