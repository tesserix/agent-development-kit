# Determinism and replay

```python
recorder = RecordingProvider(provider, provider="anthropic", version="0.42.0")
run = await AgentRunner(provider=recorder, ids=SequentialIds(), clock=FakeClock()).run(...)
recorder.cassette.save(Path("tests/cassettes/trains.json"))

replay = ReplayingProvider(Cassette.load(Path("tests/cassettes/trains.json")))
run = await AgentRunner(provider=replay, ids=SequentialIds(), clock=FakeClock()).run(...)
```

Worked end to end, no network: `examples/determinism.py`.

Behaviour that cannot be regression-tested is behaviour nobody can change safely. Live
provider calls in CI are slow, costly and flaky, so teams stop running them — and then stop
noticing when a prompt edit moves behaviour. A cassette makes the second run of a recorded
run free.

## Everything ambient is injectable

The loop reads no clock, no id and no random source of its own:

| Source | Inject with | Fake |
|---|---|---|
| Time | `AgentRunner(clock=…)` | `FakeClock` |
| Identifiers | `AgentRunner(ids=…)` | `SequentialIds` |
| Retry jitter | `AgentRunner(jitter=Random(seed))` | any seeded `Random` |
| Prompt order | nothing to inject — assembly is deterministic and versioned |

A `uuid4` in the loop is a field no assertion can name: two runs of one agent on one input
differ, and the difference means nothing. Production passes nothing and gets random ids; a
test passes a factory and gets ids it can write down.

## The fingerprint is what "the same run" means

`fingerprint_of(request, hooks=…)` canonicalises everything that shapes a provider call —
the assembled prompt, the tool schemas the model was told about, the model, the output
schema and the hook chain that could rewrite any of them. Dict order is not meaning, so it
is normalised away; list order *is* meaning, so it is kept.

`RunFingerprint.diff` names the fields that moved. A replay that failed with "cassette
miss" and nothing else sends the reader off to diff two blobs by eye.

Hooks are in the fingerprint because they rewrite prompts: the same request under a
different chain is a different request, and their order matters because rewrites chain.

## A cassette records what the kit does not own

`RecordingProvider` wraps a real provider and keeps what it answered. `ReplayingProvider`
serves those recordings and does nothing else — there is no live provider behind it to fall
through to, because a replay that quietly reused the nearest response would be a green test
asserting nothing.

**Failures are recorded too.** A 503 and the retry that recovered from it both replay, with
the recorded status, so the recovery path is exercised rather than assumed.

**A cassette holds digests of the request, never its content,** and redacts secrets out of
what it does keep — argument keys that look like credentials, and token, bearer, email and
card-number shapes wherever they appear. A cassette is a file people commit, and a token in
a committed file outlives the run that used it.

**A cassette says what it was recorded against.** `expect_provider` and `expect_version`
refuse a recording made against a different provider or SDK version, and an unknown format
is refused on read. Replaying across an upgrade on trust is a green test that proves
nothing about the code now shipping.

## Comparing two runs

`assert_same_run(first, second)` compares state, id, output, usage, tool calls and the
event sequence with its names and details, and drops wall-clock instants: a slower machine
is not a behaviour change, but a different sequence of events is. It raises naming the
field or the event index that moved.

This is also how non-determinism *outside* the loop is caught. A hook that reads
`time.time()` or a random source produces a different rewrite each run, and the comparison
fails on the `hook_rewrite` event rather than passing quietly.

## What stays non-deterministic

- **Genuine model sampling.** A live provider at `temperature > 0` answers differently to
  the same request; that is the model, not the kit. Record it, or pin it at the provider.
- **External tool state.** A tool that reads a clock, a database or a third-party API is
  outside the fingerprint. Fake it — `FakeToolRegistry` — or accept that the run is only as
  reproducible as the tool is.
- **Hooks and guardrails that read ambient state.** Nothing stops one calling `time.time()`;
  what the kit offers is a comparison that catches it.

## Known limitations

- **Providers are recorded, tools are not.** A cassette holds provider exchanges only. Tool
  results come from the registry, so an offline test fakes the registry as it always did.
- **Streaming is not recorded.** `stream` raises `NotImplementedError` on both the
  recording and replaying providers until recorded streaming lands (#150).
- **Recording is per-provider-instance, not per-process.** `RecordingProvider.cassette` is
  what *that* wrapper saw; there is no ambient recorder collecting every call in a process.
- **A replay serves one run.** The provider tracks how much of the cassette it has served,
  so replaying a second run means constructing a second `ReplayingProvider`. Asking for
  more exchanges than were recorded is a miss, not a wrap-around.
- **Tool calls within a turn are dispatched one at a time.** Ordering is therefore the
  model's, and stable on replay. Bounded parallel dispatch (#44) will need normalising by
  tool-call id rather than by arrival order.
- **The fingerprint covers the request, not the sampling parameters.** `ModelRequest`
  carries no temperature or top-p yet; when the provider protocol (#49) lands them, they
  join the fingerprint, and cassettes recorded before that must be re-recorded.
- **Redaction is pattern-based.** It catches credential-shaped keys and values, not a
  secret that looks like ordinary prose. Review a cassette before committing it.
