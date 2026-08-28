# Document and audio intake

A scanned contract and a recorded call are the same problem: text an agent needs, arriving
in a form it cannot read, and needing to stay traceable once it can. `tesserix_adk.tools.intake`
does both — OCR with page and region references, transcription with timestamps — so intake
is not rebuilt per product, and so provenance is not the part each rebuild leaves out.

## The backends are not dependencies

PaddleOCR and whisper.cpp are the CPU paths worth using, and neither is installed by this
kit, for the reason [`docs/local-embeddings.md`](local-embeddings.md) gives about
`onnxruntime`: they are large native installs, and a consumer who never reads a document
should not carry one. The kit takes two protocols and validates what crosses into them:

```python
class OcrBackend(Protocol):
    def pages(self, path: Path) -> AsyncIterator[OcrPage]: ...

class TranscriptionBackend(Protocol):
    def segments(self, path: Path) -> AsyncIterator[Segment]: ...
```

## Reading a document

```python
from tesserix_adk.tools import ocr_document

async for page in ocr_document(path, backend=paddle):
    page.number        # 3
    page.reference     # "p3"
    page.text          # the page in reading order
    page.regions[0].reference(page.number)  # "p3@0.10,0.42"
```

It yields a page at a time. A hundred-page contract costs one page of memory, and a caller
that wanted page three can stop after it.

`Region` carries the box it was read from as fractions of the page, so a reference survives
a re-render at a different resolution, plus the backend's own confidence and a `RegionKind`
from the layout pass — an answer built from a footer and one built from a clause are not
equally trustworthy, and only the layout knows which is which.

`page.scripts` names every writing system on the page, which is how a mixed-script document
announces itself rather than being discovered downstream.

## Reading a recording

```python
from tesserix_adk.tools import transcribe_audio

async for segment in transcribe_audio(path, backend=whisper):
    segment.reference()  # "00:01:23-00:01:29"
    segment.speaker      # where the backend labelled one
    segment.text
```

## What cannot be read says so

`MediaIntakeError` carries `reason`:

| `reason` | What happened |
|---|---|
| `unsupported` | The suffix is not one the backend reads. Refused before the file is opened. |
| `missing` | Nothing is there. |
| `empty` | The file is zero bytes, **or** the backend produced no pages or segments at all. |
| `corrupt` | The backend failed part-way. |

The second half of `empty` is the point of the module. Returning `""` for a document that
could not be read is indistinguishable from a blank page, and it ends with an agent
answering confidently from nothing. A backend that raises `MediaIntakeError` itself is
passed through untouched — it knew more about the failure than the wrapper does.

## The tools a model calls

```python
from tesserix_adk.tools import ocr_tool, transcribe_tool

read = ocr_tool(paddle, root=Path("/intake/acme"), max_chars=4096)
listen = transcribe_tool(whisper, root=Path("/intake/acme"))
```

A model names a file; what that is allowed to mean is decided by `root`. A name resolving
outside it is a `ToolRefusal` with code `path_not_permitted` — the check is on the resolved
path, so `../` does not help.

Output is windowed at `max_chars` for the same reason the claim-check tool windows: a tool
that returned a whole contract would put the document back into the context one call later.
Each page arrives prefixed `[p3]` and each span `[00:01:23-00:01:29]`, so what the model
quotes back can be cited. An unreadable file is a refusal — `unreadable_document` or
`unreadable_recording` — not a crash and not an empty string.

## Related

- [`docs/citations.md`](citations.md) — what a reference has to resolve to
- [`docs/claim-check.md`](claim-check.md) — the same windowing argument, for tool results
- [`examples/document_intake.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/document_intake.py)
