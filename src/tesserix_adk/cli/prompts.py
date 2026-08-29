"""`tesserix-adk prompt` — inspect and restore prompt versions.

When behaviour changes after a release the first question is whether the prompt changed and
how to undo it. Answering that today means reading a code diff across repositories with no
view of the rendered result, and undoing it means an emergency deploy. Both are the wrong
tools for an incident: the prompt is versioned data, and moving an alias back to a version
that already existed is not a code change.

So: `diff` shows what moved between two versions — body, declared variables, metadata and
digest — optionally as the model would receive it. `rollback` repoints an alias at an
earlier immutable version, refusing a target the current call sites cannot render, and
recording who did it and from where. `list` and `show` say what exists.

This application-wired command is not part of the self-contained dispatcher. Applications
may expose it under the project-qualified command. Where prompts and
aliases are kept is the deployment's business, so it supplies both and this supplies the
diffing, the safety checks and the exit codes.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from tesserix_adk.core.errors import (
    IncompatiblePromptVersionError,
    PromptNotFoundError,
    PromptRejectedError,
    TemplateError,
)
from tesserix_adk.core.rendering import PromptTemplate, Variable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

    from tesserix_adk.core.prompts import PromptDefinition

__all__ = ["Aliases", "Prompts", "main"]

OK = 0
REFUSED = 1
MISUSED = 2

_DIFF_LINES = 40


class Prompts(Protocol):
    """The little of a registry these commands read."""

    async def get(
        self, name: str, *, version: str = "", alias: str = "", tenant: str = ""
    ) -> PromptDefinition:
        """One published version, resolved by version or alias."""
        ...

    async def list_versions(self, name: str, *, tenant: str = "") -> tuple[str, ...]:
        """Every published version of `name`, oldest first."""
        ...


class Aliases(Protocol):
    """Where a deployment keeps its moving labels, and what it lets an operator do to them."""

    async def aliases(self, name: str, *, tenant: str = "") -> Mapping[str, str]:
        """Which version each alias points at."""
        ...

    async def evaluated(self, name: str, version: str, *, tenant: str = "") -> str:
        """The last recorded eval result for a version, empty where there is none."""
        ...

    async def repoint(
        self,
        name: str,
        *,
        alias: str,
        to: str,
        expected: str,
        by: str,
        reason: str,
        tenant: str = "",
    ) -> None:
        """Move `alias` to `to`, and record it.

        `expected` is where the alias pointed when the checks ran. A store that compares it
        before writing turns two operators rolling back at once into one winner and one
        refusal, rather than a partial state neither of them chose.
        """
        ...


async def main(
    argv: Sequence[str],
    *,
    prompts: Prompts,
    aliases: Aliases,
    out: TextIO | None = None,
) -> int:
    """Run one `prompt` command and return its exit code.

    Args:
        argv: Arguments after the program name, e.g. `["diff", "greeting", "4", "5"]`.
        prompts: Where published versions are read from.
        aliases: Where moving labels are kept, and moved.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the command did what it says, `1` where the registry or a safety check
        refused, and `2` for a command line this could not read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    handlers = {"list": _list, "show": _show, "diff": _diff, "rollback": _rollback}
    try:
        return await handlers[parsed.command](parsed, prompts, aliases, writer)
    except (PromptNotFoundError, PromptRejectedError) as refused:
        _write(writer, {"error": str(refused)}, f"refused: {refused}\n", parsed.json)
        return REFUSED


async def _list(
    parsed: argparse.Namespace, prompts: Prompts, aliases: Aliases, writer: TextIO
) -> int:
    """What exists: every version, its digest, the aliases on it and its last eval."""
    versions = await prompts.list_versions(parsed.name, tenant=parsed.tenant)
    pointing = await aliases.aliases(parsed.name, tenant=parsed.tenant)
    rows = []
    for version in versions:
        definition = await prompts.get(parsed.name, version=version, tenant=parsed.tenant)
        rows.append(
            {
                "version": version,
                "digest": definition.digest[:12],
                "owner": definition.owner,
                "aliases": sorted(alias for alias, at in pointing.items() if at == version),
                "evaluated": await aliases.evaluated(parsed.name, version, tenant=parsed.tenant),
            }
        )
    lines = "".join(
        f"{row['version']}  {row['digest']}  {row['owner'] or '-'}  "
        f"{','.join(row['aliases']) or '-'}  {row['evaluated'] or 'not evaluated'}\n"
        for row in rows
    )
    _write(writer, {"name": parsed.name, "versions": rows}, lines, parsed.json)
    return OK


async def _show(
    parsed: argparse.Namespace, prompts: Prompts, aliases: Aliases, writer: TextIO
) -> int:
    """One version, in full, with what a reviewer needs around it."""
    definition = await prompts.get(
        parsed.name, version=parsed.version, alias=parsed.alias, tenant=parsed.tenant
    )
    payload = {
        "name": definition.name,
        "version": definition.version,
        "digest": definition.digest,
        "owner": definition.owner,
        "task_class": definition.task_class,
        "variables": list(definition.variables),
        "estimated_tokens": definition.estimated_tokens,
        "evaluated": await aliases.evaluated(
            definition.name, definition.version, tenant=parsed.tenant
        ),
        "body": definition.body,
    }
    text = (
        f"{definition.name}@{definition.version}  {definition.digest[:12]}\n"
        f"owner {definition.owner or '-'}  variables {','.join(definition.variables) or '-'}\n"
        f"\n{definition.body}\n"
    )
    _write(writer, payload, text, parsed.json)
    return OK


async def _diff(
    parsed: argparse.Namespace, prompts: Prompts, aliases: Aliases, writer: TextIO
) -> int:
    """What moved between two versions, and whether it is likely to matter."""
    del aliases
    before = await prompts.get(parsed.name, version=parsed.old, tenant=parsed.tenant)
    after = await prompts.get(parsed.name, version=parsed.new, tenant=parsed.tenant)
    values = json.loads(parsed.values.read_text(encoding="utf-8")) if parsed.values else None
    left, right = (
        (_rendered(before, values), _rendered(after, values))
        if values is not None
        else (before.body, after.body)
    )
    body = list(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=f"{before.name}@{before.version}",
            tofile=f"{after.name}@{after.version}",
            lineterm="",
        )
    )
    shown = body if parsed.full else body[:_DIFF_LINES]
    payload = {
        "name": parsed.name,
        "from": before.version,
        "to": after.version,
        "digest_changed": before.digest != after.digest,
        "whitespace_only": left.split() == right.split() and left != right,
        "variables_added": sorted(set(after.variables) - set(before.variables)),
        "variables_removed": sorted(set(before.variables) - set(after.variables)),
        "metadata_changed": sorted(
            field
            for field in ("owner", "task_class", "requires")
            if getattr(before, field) != getattr(after, field)
        ),
        "rendered": values is not None,
        "body": shown,
        "truncated": len(body) - len(shown),
    }
    _write(writer, payload, _diff_text(payload), parsed.json)
    return OK


async def _rollback(
    parsed: argparse.Namespace, prompts: Prompts, aliases: Aliases, writer: TextIO
) -> int:
    """Repoint an alias at an earlier version, or refuse and change nothing."""
    target = await prompts.get(parsed.name, version=parsed.to, tenant=parsed.tenant)
    pointing = await aliases.aliases(parsed.name, tenant=parsed.tenant)
    if parsed.alias not in pointing:
        _write(
            writer,
            {"error": f"no alias {parsed.alias!r} on {parsed.name!r}"},
            f"refused: no alias {parsed.alias!r} on {parsed.name!r}\n",
            parsed.json,
        )
        return REFUSED
    current = await prompts.get(parsed.name, alias=parsed.alias, tenant=parsed.tenant)
    dropped = tuple(sorted(set(current.variables) - set(target.variables)))
    if dropped:
        refusal = IncompatiblePromptVersionError(
            f"{parsed.name}@{parsed.to} does not declare {', '.join(dropped)}, "
            f"which {parsed.name}@{current.version} does",
            name=parsed.name,
            version=parsed.to,
            variables=dropped,
        )
        _write(writer, {"error": str(refusal)}, f"refused: {refusal}\n", parsed.json)
        return REFUSED
    evaluated = await aliases.evaluated(parsed.name, parsed.to, tenant=parsed.tenant)
    if not evaluated and not parsed.force:
        message = (
            f"{parsed.name}@{parsed.to} has no recorded eval result; "
            f"pass --force with a --reason to roll back to it anyway"
        )
        _write(writer, {"error": message}, f"refused: {message}\n", parsed.json)
        return REFUSED
    if parsed.force and not parsed.reason:
        _write(
            writer,
            {"error": "--force needs a --reason"},
            "refused: --force needs a --reason\n",
            parsed.json,
        )
        return REFUSED
    await aliases.repoint(
        parsed.name,
        alias=parsed.alias,
        to=parsed.to,
        expected=pointing[parsed.alias],
        by=parsed.by,
        reason=parsed.reason,
        tenant=parsed.tenant,
    )
    payload = {
        "name": parsed.name,
        "alias": parsed.alias,
        "from": pointing[parsed.alias],
        "to": parsed.to,
        "by": parsed.by,
        "reason": parsed.reason,
        "forced": bool(parsed.force),
    }
    text = f"{parsed.name}: {parsed.alias} {pointing[parsed.alias]} -> {parsed.to} by {parsed.by}\n"
    _write(writer, payload, text, parsed.json)
    return OK


def _rendered(definition: PromptDefinition, values: Mapping[str, object]) -> str:
    """The prompt as the model would receive it, or a line saying why it would not."""
    template = PromptTemplate(
        name=definition.name,
        body=definition.body,
        variables=tuple(Variable(name=name) for name in definition.variables),
    )
    try:
        return template.render(values).text
    except TemplateError as refused:
        return f"<cannot render: {refused}>"


def _diff_text(payload: Mapping[str, object]) -> str:
    """The human form: the headline first, because that is what an incident reads."""
    headline = [
        f"{payload['name']} {payload['from']} -> {payload['to']}",
        "digest changed" if payload["digest_changed"] else "digest unchanged",
    ]
    if payload["whitespace_only"]:
        headline.append("whitespace only")
    for field in ("variables_added", "variables_removed", "metadata_changed"):
        changed = payload[field]
        if isinstance(changed, list) and changed:
            headline.append(f"{field.replace('_', ' ')}: {', '.join(changed)}")
    lines = payload["body"] if isinstance(payload["body"], list) else []
    tail = (
        f"\n… {payload['truncated']} more diff lines, pass --full\n"
        if isinstance(payload["truncated"], int) and payload["truncated"] > 0
        else ""
    )
    return "  |  ".join(headline) + "\n" + "".join(f"{line}\n" for line in lines) + tail


def _write(writer: TextIO, payload: Mapping[str, object], text: str, machine: bool) -> None:
    """One place decides which of the two forms a command emits."""
    writer.write(json.dumps(payload, sort_keys=True) + "\n" if machine else text)


def _parser() -> argparse.ArgumentParser:
    """The `prompt` command line."""
    parser = argparse.ArgumentParser(
        prog="tesserix-adk prompt", description="inspect and revert prompts"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, what in (
        ("list", "show every published version"),
        ("show", "show one version in full"),
        ("diff", "show what moved between two versions"),
        ("rollback", "repoint an alias at an earlier version"),
    ):
        command = commands.add_parser(name, help=what)
        command.add_argument("name", help="the prompt")
        command.add_argument("--tenant", default="", help="whose copy to read")
        command.add_argument("--json", action="store_true", help="machine-readable output")
        if name == "show":
            command.add_argument("--version", default="", help="which version")
            command.add_argument("--alias", default="", help="or which alias")
        if name == "diff":
            command.add_argument("old", help="the version to compare from")
            command.add_argument("new", help="the version to compare to")
            command.add_argument(
                "--values",
                type=Path,
                default=None,
                help="a JSON fixture of variable values, to diff what the model receives",
            )
            command.add_argument("--full", action="store_true", help="do not truncate the diff")
        if name == "rollback":
            command.add_argument("--to", required=True, help="the version to go back to")
            command.add_argument("--alias", default="current", help="which alias to move")
            command.add_argument("--by", required=True, help="who is rolling back")
            command.add_argument("--reason", default="", help="why")
            command.add_argument(
                "--force",
                action="store_true",
                help="roll back to a version with no recorded eval result",
            )
    return parser
