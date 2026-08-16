# Recording provider traffic, and replaying it forever

Hand-written scripts drift from what providers actually return, so the unusual finish
reason, the tool-call id format and the truncated stream surface in production instead of
in CI. A cassette captures one real run and replays it deterministically, with no network
and no spend.

The recording and replay providers, the fingerprint they key on and what a cassette does
**not** hold are in [`docs/determinism.md`](determinism.md). This page is about the modes,
the marker and the ways a replay can be asked for something nobody recorded.

## Modes

`ADK_CASSETTE_MODE` selects one; `mode_from_env()` reads it.

| Mode | What it does |
|---|---|
| `replay` | **The default.** Serves the cassette. Never opens a socket |
| `record` | Records what is not on the cassette yet, and replays what is |
| `refresh` | Records over the cassette deliberately |

Replay is the default so that a suite cannot start spending because somebody forgot a
flag, and a value naming no mode is refused rather than quietly treated as replay — a
typo that means "replay" is a suite that never records again.

## The marker

```python
@pytest.mark.cassette("trains")
async def test_it_answers_from_the_timetable(cassettes):
    provider = cassettes.provider(live=OpenAiProvider())  # live is used only when recording
```

The `cassettes` fixture loads `tests/cassettes/trains.json` — move that with the
`cassette_dir` fixture — and writes back only what this run recorded. Marker keyword
arguments reach `CassetteHarness`, so `match=`, `provider=` and `version=` are set per
test. A test that asks for the fixture without the marker is a wiring error and says so.

## A miss is never papered over

`CassetteMismatchError` names the fields on which the request diverged from the nearest
recording and the command that re-records it. There is no fall-through to a live call and
no nearest-match response: either would be a green test asserting nothing.

The exhausted case has its own message — the run asked more times than it did when
recorded, which is usually a loop that no longer terminates rather than a stale cassette.

## Matching

`MatchOn` selects which parts of the request have to agree: `model`, `messages`, `tools`,
`output_schema`, `hooks`. All of them by default. Dropping one widens what a cassette
answers, which is how a test about tool wiring stops being re-recorded every time a word
in the prompt changes. A strategy that compares nothing is refused.

Non-deterministic fields never enter the key: a cassette holds digests of the request, so
timestamps and request ids are normalised out before anything is filed.

## Streaming

A recorded stream keeps its chunk boundaries and replays on the same ones, because a
consumer that renders per chunk behaves differently on different boundaries. A consumer
that stops part way still has what arrived recorded — the point it stopped at is usually
what the test was about.

## Nothing secret is committed

Redaction runs on write, and `Cassette.save` **refuses** to write a file in which anything
credential-shaped survived. That is deliberately a failure rather than a warning: nobody
reads a 4000-line JSON diff closely, and a token in a committed file outlives every run
that used it.

## Known limitations

- Provider exchanges only. Tool-side HTTP traffic is not recorded.
- A cassette recorded against a model that no longer exists replays happily; pin
  `expect_version` where that matters.
- Long multi-turn runs make large files. Prefer one cassette per behaviour over one per
  suite; a cassette nobody can read is a cassette nobody will re-record.
- Concurrent tests may share a cassette file: it is read-only in replay, and each harness
  keeps its own cursor. Two tests **recording** the same file is not supported.

## Related

- [`docs/determinism.md`](determinism.md) — the fingerprint, redaction and what a cassette holds.
- [`docs/fake-model-provider.md`](fake-model-provider.md) — scripting a provider rather than recording one.
