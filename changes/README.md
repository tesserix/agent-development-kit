# Change fragments

One file per consumer-visible change, written in the pull request that makes the change,
while you still remember why. At release time they are assembled into the notes and
folded into `CHANGELOG.md`, then deleted.

Name the file `<issue>.<kind>.md`:

| Kind | Section | Notes |
|---|---|---|
| `breaking` | Breaking changes | Requires `migration:` |
| `removed` | Breaking changes | Requires `migration:` |
| `added` | Added | |
| `changed` | Changed | |
| `fixed` | Fixed | |
| `deprecated` | Deprecated | Pair it with a `@deprecate` record |
| `reverted` | Reverted | |
| `experimental` | Experimental | Carries no stability promise |

An optional header carries the fields:

```markdown
---
surface: tesserix_adk.core.load_config
migration: Call `resolve_config` and read `.config`; it also reports provenance.
---
`load_config` is replaced by `resolve_config`, which resolves once at startup and
reports which layer supplied each key.
```

A change with a conventional commit subject (`feat(core): …`) needs no fragment — the
subject is enough for the notes. A fragment is what you write when the subject is not
enough, and it is **required** for anything breaking, because that is where the migration
note lives.

Full policy: [`docs/releasing.md`](../docs/releasing.md#release-notes).
