# Putting a run on the wire

`tesserix_adk.adapters` frames the typed event stream for a browser, over server-sent
events or a websocket. It depends on no web framework: an SSE helper is an async iterator of
strings, and the websocket bridge asks only for `send_text`, `receive_text` and `close`. Any
ASGI application already has both.

```python
broker = RunBroker[NoOutput]()
broker.register(runner.stream(agent, question, tenant=tenant), tenant=tenant)

# SSE, from any framework that can stream a response body.
return StreamingResponse(
    sse_events(broker.subscribe(run_id, tenant=tenant, after=last_event_id)),
    headers=SSE_HEADERS,
)
```

A worked run over both transports, with a reconnect and a refusal, no network:
`examples/transports.py`.

## The broker owns the run

A reconnect is a second reader of a run already in flight, so the run cannot belong to the
connection that started it. `RunBroker.register` takes the stream; the run starts when the
first transport attaches and is driven exactly once however many attach after it. A
registration nobody ever connects to costs nothing and misses nothing.

| Call | Does |
|---|---|
| `register(stream, tenant=…)` | Takes ownership, returns the run id. Starts nothing. |
| `subscribe(run_id, tenant=…, after=…)` | Missed events first, then live ones. |
| `cancel(run_id, tenant=…)` | Stops the run through the run's own cancellation path. |
| `run(run_id, tenant=…)` | The finished record, once the run reached a terminal event. |
| `authorise(run_id, tenant=…)` | Checks ownership before anything is framed. |

`history` bounds what is kept per run for a client that comes back.

## SSE framing

`sse_events` emits `retry:` first, then one frame per event:

```
event: answer_delta
id: 7
data: {"kind":"answer_delta","run_id":"run_1","sequence":7,"text":"Kyoto"}
```

`event:` is the event's own `kind`, so a browser dispatches with `addEventListener` rather
than parsing every payload to find out what it is. `id:` is the sequence number, which the
browser then sends back as `Last-Event-ID` on reconnect — hand it to `subscribe(after=…)`.

Serve it with `SSE_HEADERS`. An intermediary that buffers a streamed body turns a live run
into one lump delivered at the end; `X-Accel-Buffering: no` and `Cache-Control: no-transform`
tell the common ones not to. **Diagnosing a buffered stream:** the client receives nothing
for the length of the run and then everything at once, including every heartbeat in a batch.
The heartbeat count is the tell — a stream that is genuinely quiet delivers its heartbeats
one at a time, seconds apart.

Heartbeats keep an idle stream open through a proxy that closes a quiet connection. They are
comments (`: heartbeat`), so a client ignores them without any handling, and they never
interrupt the run: the pending read outlives the heartbeat rather than being cancelled by it.

## Websockets, and the control channel

`WebSocketBridge.serve` pumps the same payloads the SSE helper frames — a client switching
transport parses the same JSON — and reads control messages from the client:

| Message | Effect |
|---|---|
| `{"type": "cancel"}` | Cancels the run, through the cancellation path. |
| `{"type": "approval", "decision": {…}}` | Answers a gated tool call via `ApprovalInbox`. |
| anything else | Ignored. |

An unknown type is a newer client talking to an older server; tearing the connection down
for it would turn a forward-compatible change into an outage. Malformed JSON is ignored on
the same reasoning.

A peer that vanishes without a close frame is a consumer that has gone, but a run nobody is
watching keeps calling providers and keeps billing. The bridge cancels it.

`ApprovalInbox` is an `ApprovalGate` the runner can be given directly, so the person
deciding is the one on the other end of the socket. A decision is checked against the
tenant that owns the held call; a second decision for a call already answered is a race
between two reviewers, not an error, and the first one stands.

## Resuming

A client that reconnects presents its last sequence id. It receives either the events it
missed, in order, or — where it was away longer than the buffer is deep — a `StreamGap`
first, carrying how many events are gone and where the stream resumes. Silently closing the
gap is how a UI ends up showing a run that never happened.

A client that reconnects after the run has already ended receives the terminal event, not an
empty stream.

## The boundary fails closed

A run id from a client is a claim, not a fact. Every entry point — `subscribe`, `cancel`,
`run`, and `serve` before it frames anything — authorises the tenant first and raises
`TransportAuthorizationError` otherwise. An unknown run id and another tenant's run id give
the same refusal, because which ids exist is itself something one tenant should not learn
about another.

## What never reaches the wire

`wire_payload` is applied to every event both transports send:

- **Redaction.** Every string field is scrubbed with the kit's own redaction. The event may
  have arrived from a queue or another process, so it was scrubbed by nothing this code can
  see. Discriminators (`kind`, `run_id`) are left alone: masking one would turn a value the
  client switches on into prose.
- **Size.** An event above `payload_limit_bytes` is replaced by a `PayloadElided` carrying
  its kind, its size and a reference to fetch it by. Half a JSON document is not a smaller
  JSON document, so nothing is ever truncated into invalid JSON.
