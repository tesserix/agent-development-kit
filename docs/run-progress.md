# Watching a run

`AgentRunner.stream` reports a run while it happens, as typed events rather than as text
chunks. It drives the same run `run` drives — same loop, same guardrails, same record —
so a product does not choose between an experience and a correct answer.

```python
stream = runner.stream(agent, "Four nights near Kyoto.", tenant="acme")
async for event in stream:
    match event:
        case AnswerDelta(text=text):
            print(text, end="")
        case ToolCallStarted(tool=tool):
            print(f"\n[{tool}]")
run = stream.run
```

A worked run with a tool call, a guardrail and a truncated stream, no network:
`examples/run_progress.py`.

## The events

Every variant subclasses `ProgressEvent` and is discriminated by `kind`, so a consumer
switches on a value rather than on the shape of a payload.

| Kind | Carries | Emitted when |
|---|---|---|
| `run_started` | `agent`, `model`, `tenant` | Always first. |
| `iteration_started` | `iteration`, from one | Before each model call. |
| `answer_delta` | `text` | Free-text answer, in pieces. |
| `structured_delta` | `fragment` | Structured answer, as its JSON arrives. |
| `tool_call_started` | `call_id`, `tool`, `arguments` | A call cleared policy and is about to run. |
| `tool_call_finished` | `call_id`, `tool`, `truncated` | A tool returned. |
| `tool_call_failed` | `call_id`, `tool`, `error`, `detail` | A call was refused or raised. |
| `tool_call_indeterminate` | `call_id`, `tool`, `detail` | A call was stopped after dispatch. |
| `guardrail_decision` | `guardrail`, `allowed`, `detail` | A guardrail was asked. |
| `approval_required` | `call_id`, `tool`, `reason` | A call is held for a human. |
| `usage_updated` | `usage` | After each model response. |
| `run_completed` | `state`, `usage` | Terminal, good. |
| `run_failed` | `state`, `error`, `detail` | Terminal, bad. |
| `run_cancelled` | `state`, `reason`, `usage`, `last_sequence` | Terminal, stopped. |

Three properties hold whatever the run does.

**Exactly one terminal event, and it is last.** It is derived from the finished `Run`, not
emitted from inside the loop, so a stream that ends early cannot read as a finished answer.
Teardown enforces the ordering rather than trusting it: an event posted after the run ended
is dropped, not delivered behind the terminal one. A provider connection that drops
mid-response fails the run: accumulated text from a dropped connection is not an answer, and
the kit never presents it as one.

**Every event is numbered.** `run_id` and a gapless `sequence` from zero travel on every
event, so a multiplexed transport needs no envelope of its own and a consumer can tell a
slow stream from a lossy one. `SequenceCheck` does that counting:

```python
check = SequenceCheck()
if not check.accept(event):
    ...  # late or duplicate — rejected rather than reordered into place
print(check.missing)
```

**Redaction happens in the runtime.** `tool_call_started.arguments` is compact JSON with
secret shapes masked before the event is emitted, because a transport that redacts has
already handed the value to whatever it logs to. The answer itself is *not* scrubbed —
deltas that no longer reassemble to the answer are a corrupted answer, and a consumer that
must not see the content must not be given the run.

## Consuming a stream

Three patterns, all on the same object. Nothing starts until one of them does.

**Iterate then await** — progress while it happens, then the authoritative record.

```python
async with runner.stream(agent, "Four nights near Kyoto.", tenant="acme") as stream:
    async for event in stream:
        render(event)
run = await stream
```

**Await only** — the answer, no progress. The stream drains itself.

```python
run = await runner.stream(agent, "Four nights near Kyoto.", tenant="acme")
```

**Iterate and discard** — read until you have seen enough, and leave.

```python
async with runner.stream(agent, "Four nights near Kyoto.", tenant="acme") as stream:
    async for event in stream:
        if isinstance(event, ToolCallStarted):
            break
```

Leaving the block cancels a run nobody is reading any more, through the same cancellation
path a caller's own token uses; `stream.run` is then the cancelled record. A run left
driving in the background still calls providers and still bills. An exception in the loop
body takes the same exit, and the consumer's own exception is the one that propagates.

Awaiting the same stream from two places drives the run once and gives both the same `Run`.
Awaiting a stream that was abandoned raises `StreamInterruptedError` rather than handing
back what had accumulated — partial content returned as a result is a wrong answer that
looks right, and what arrived is on the error for a caller that deliberately wants it.

## Provisional is not final

Half a JSON object parses into something shaped exactly like the declared output type. A
consumer holding one cannot tell by inspection whether acting on it is safe, so it tells by
type: `stream.provisional` is a `Provisional[OutputT]`, which the checker refuses wherever
an `OutputT` is required.

```python
async for event in stream:
    draft = stream.provisional.snapshot()   # dict | None — never a TripPlan
    if draft is not None:
        preview(draft)
plan: TripPlan | None = (await stream).output
```

`snapshot` hands back a plain mapping, and `None` while the object is half-arrived — filling
in the missing half would be inventing content the model never sent. Only the run's own
`output` is schema-validated, and it exists only once the run reached a terminal event.

## What a consumer sees, and what it does not

Provider-level stream events — `TextDelta`, `ToolCallDelta`, `UsageDelta`, `StreamEnd` in
`tesserix_adk.core.streaming` — are the vendor's vocabulary and stay inside the runtime.
Tool-call arguments arriving in fragments are assembled first: a call is announced once,
whole, at `tool_call_started`. Half an argument object is not an argument object, and
rendering it says it is.

Where the provider cannot stream, or where nobody is watching, the answer is emitted as a
single delta. The event sequence is the same either way, so a consumer written against the
stream works against a provider that has no streaming at all.

## Version tolerance

Adding a variant is a minor release. `decode_progress` returns `None` for a `kind` this
version has never heard of, so a consumer pinned to an older kit skips it rather than
falling over. A variant it *does* know and cannot parse raises instead — a delta decoded by
guesswork renders as an answer nobody wrote. Removing or renaming a variant, or a field on
one, is breaking.

## Stopping a stream

A stop propagates in both directions: the client's stop reaches the run, and the run's
termination reaches the client as an event it can close its own view on.

Stopping goes through `CancellationToken` — the caller's own if one was passed to `stream`,
and over a transport through `RunBroker.cancel`, which authorises the tenant first. Leaving
an `async with` block, a websocket peer vanishing without a close frame, and a reader that
stopped reading all take the same path: a run nobody is watching still calls providers and
still bills.

```python
token = CancellationToken()
stream = runner.stream(agent, question, tenant="acme", cancellation=token)
...
token.cancel("the client pressed stop")
```

**The terminal event is reconcilable.** `run_cancelled` carries `reason`, `usage` accrued by
the time it stopped, and `last_sequence` — the sequence of the last event before it. A run
whose spend is knowable only on completion is unattributable exactly when it did not
complete, and a client that cannot tell where the stream ended cannot tell a stop from a
dropped connection.

**A stop racing a natural completion gives one outcome.** The run's own record decides: the
terminal event is derived from the state the loop reached, so a stop arriving after that does
not rewrite it. A client never sees both `run_completed` and `run_cancelled`.

**Teardown is idempotent.** Cancelling is a one-way switch and the first reason stands, so a
retrying client sending stop twice gets one cancelled run and one explanation rather than two
callers disagreeing about why it stopped.

**A tool caught in flight is indeterminate, not undone.** A tool stopped after dispatch emits
`tool_call_indeterminate`: whether its side effect landed cannot be known, and claiming it
was rolled back when nobody rolled it back is the worse answer. A tool the agent named in
`idempotent_tools` is reported failed and safe to retry instead. The stream says exactly what
the run record says — `tool_indeterminate` — because two accounts of one call is one too many.

**A client that is gone is still accounted for.** `RunBroker.cancel` drives the run to a
cancelled record even where nothing ever attached to it, so attribution does not depend on
someone being there to be told.

## When the consumer cannot keep up

The buffer is bounded, and the run never waits on it. Pass `backpressure=` to `stream`:

```python
stream = runner.stream(agent, question, tenant="acme", backpressure=Backpressure(high_water=64))
```

| Field | Default | Means |
|---|---|---|
| `high_water` | 256 | Events that may wait unread before deltas start merging. |
| `byte_budget` | 8 MiB | Bytes that may wait, for the same reason. |
| `stall_seconds` | 30 | How long a reader may hold an unread buffer before the run stops. |

**Deltas merge; nothing else does.** Above either mark, an arriving `AnswerDelta` or
`StructuredDelta` is concatenated onto the one already waiting and `coalesced` counts how
many were folded in. Every character a consumer would have rendered still arrives, in order.
Lifecycle, tool, approval, usage and terminal events are never merged and never dropped: a
run missing one of those is a run nobody can account for.

**The run never blocks on the buffer.** A queue that makes the run wait for its reader
deadlocks the run whose own tool result feeds that reader, and turns a slow client into a
slow answer for everyone. So pressure is answered by merging, not by waiting.

**A reader that stops reading stops the run.** The stall clock runs from the last read, and
only while something is waiting — a quiet run is not a stalled one. Past `stall_seconds` the
run is cancelled through the same cancellation path a caller's own token uses, so a dead
client that never disconnected stops costing provider spend. Await-only is not a slow
consumer: with no reader attached, events are numbered and discarded rather than buffered.

**Pressure is readable while it matters.** `stream.pressure` is a `Pressure` — `buffered`,
`peak`, `coalesced`, `oversize`, `stalled` — during the run, not only after it. An event
that alone exceeds the whole byte budget is admitted anyway and counted in `oversize`:
dropping it would lose a tool call, and growing for it is the unbounded case this replaces.

Sizing a process rather than a run: `Backpressure.shared(total_bytes=…, streams=…)` divides
an aggregate allowance into a per-run budget, because a per-run bound multiplied by however
many runs are in flight is not a bound on anything.

## Limits

- **One reader.** Events are consumed as they are read; iterating a stream twice raises
  rather than replaying a partial run.
- **Transports.** SSE and websocket helpers live in `tesserix_adk.adapters` —
  [`docs/transports.md`](transports.md).
- **Cassettes do not record streams.** `RecordingProvider` and `ReplayingProvider` still
  refuse `stream`; recorded streaming is #150.
