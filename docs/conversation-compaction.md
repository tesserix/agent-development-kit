# Conversation compaction

A conversation that runs long has to be made smaller, and the honest way to do it is to
replace the older turns with a summary. Prose is what a summary is allowed to lose.

Provenance is not. A span of turns that each cited a policy, replaced by three sentences
carrying no citations, reads as claims the agent made up itself — and the person holding
the answer cannot check any of them. That is not a smaller conversation, it is a conversation
whose sources were deleted.

## The shape

```python
from tesserix_adk.runtime import cited, compact_conversation

turn = cited(message, citation_ids)          # sources travel on the message

done = await compact_conversation(
    history,
    summarise=summariser,                    # writes the replacement message
    threshold_tokens=8_000,
    keep_recent=4,
)

done.history      # what to send
done.ran          # whether anything was folded
done.citations    # the ids carried across
done.event        # what to write down, or None
```

`compact_conversation` runs above the threshold and does nothing below it, so calling it
every turn is the intended use.

## Provenance is checked, not requested

`cited(message, ids)` puts the citation ids in the message's own metadata under
`adk.citations`, because a history is what gets persisted, replayed and handed to a
provider, and a parallel list of sources is what does not survive that. `citations_of`
reads them back.

The `Summariser` is handed the turns and returns the replacement message, so it can carry
across whatever else its messages hold. What is not left to it is the provenance: every id
carried by a folded turn must be on the message that replaces them. Where one is missing,
`compact_conversation` raises `ProvenanceLostError` naming the ids it would have dropped,
and the history is returned to the caller exactly as it was. There is no partial result and
no summary emitted with a lost source.

Summarisation quality is the summariser's business. This module has an opinion about
exactly one thing.

## The prefix is untouched

Compaction is a conversation-layer operation. The system prompt, the tool declarations and
the pinned context — the cacheable prefix — are not inputs to it, so `assemble_prompt`
produces the same `fingerprint` before and after and no cache downstream is refilled.
Dropping the prefix to save tokens is a cost dressed as a saving; see
[`docs/context-assembly.md`](context-assembly.md).

## Running it twice

The summary is marked `adk.compacted`. A second pass over an already-compacted conversation
folds nothing: where the only foldable span is a previous summary, there are no tokens to
win and a little fidelity to lose each time. Once new turns accumulate above the threshold,
the next pass folds the earlier summary in with them, and its citations carry forward like
any other turn's.

## What it records

`Compaction.event` is a `CompactionEvent` — the run, how many turns were folded, the ids
carried, and the token count before and after. `event.attributes()` gives the span
attributes under `adk.compaction.*`: counts only, never the conversation. A pass that folded
nothing records nothing, because an audit trail of non-events is one nobody reads.

## Known limitations

Summarisation quality, and the choice of which model writes it, are out of scope. So is
resolving a citation: this module carries ids and checks they survive — resolving one back
to a document version and span is [`docs/citations.md`](citations.md).
