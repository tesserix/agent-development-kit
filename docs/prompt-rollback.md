# Diffing and rolling back a prompt

Behaviour changed after a release. The first question is whether the prompt changed, and the
second is how to put it back. Today that means reading a code diff across repositories with
no view of the rendered result, and undoing it means an emergency deploy.

A prompt is versioned data. Moving an alias back to a version that already exists is not a
code change. `prompts_main` is the callable an embedding product can expose as
`adk prompt` during an incident.

```python
from tesserix_adk.cli import prompts_main

code = await prompts_main(argv, prompts=registry, aliases=book)
```

`prompts` is anything satisfying [`docs/prompt-registry.md`](prompt-registry.md)'s protocol.
`aliases` is the deployment's own alias store — the kit does not know where a project keeps
its moving labels, or what it lets an operator do to them.

## The commands

| Command | What it does |
|---|---|
| `list <name>` | Every version, its digest, the aliases on it, its last eval result |
| `show <name> --version 4` | One version in full, with the metadata around it |
| `diff <name> 4 5` | Body, declared variables, metadata and digest |
| `diff <name> 4 5 --values fixture.json` | The same, as the model would receive it |
| `rollback <name> --to 4 --by ada` | Repoint an alias at an earlier version |

`--json` on any of them prints one object, so an incident runbook and a CI step read the same
output a person does. Exit codes are `0` done, `1` refused, `2` unreadable command line.

The diff headline says whether the digest moved and whether the change was whitespace only —
a reordered line changes the digest and probably not the behaviour, and knowing which of the
two you are looking at is most of the triage. Long diffs are truncated with a count; `--full`
prints everything.

## What a rollback refuses

**A target the current call sites cannot render.** If `current` points at a version declaring
`city` and `budget`, and the rollback target declares only `city`, the command fails with
`IncompatiblePromptVersionError` naming `budget`, and nothing is repointed. A run that cannot
render its prompt is worse than the regression being rolled back.

**A target with no recorded eval result.** Rolling onto an unevaluated version is sometimes
exactly right during an incident, so it is possible — with `--force` and a `--reason`, which
is recorded. `--force` without a reason is refused: the point is the record, not the flag.

**An alias nobody declared.** A typo in `--alias` creates nothing.

Every accepted rollback is handed to the store with `expected` — where the alias pointed when
the checks ran. A store that compares it before writing turns two operators rolling back at
once into one winner and one refusal, rather than a state neither of them chose. Implementing
that comparison is the store's job, and worth doing.

## Known limitations

* `--values` renders with [`docs/prompt-templates.md`](prompt-templates.md), so it assumes
  `${name}` placeholders. A project templating some other way gets the raw diff.
* The eval result is whatever the deployment's store returns. The gate that produces it is a
  separate concern.
* Aliases are per registry. A prompt that exists in staging and not in production diffs in
  the environment you point the command at, and nowhere else.
