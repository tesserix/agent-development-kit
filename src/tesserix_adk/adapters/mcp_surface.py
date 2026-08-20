"""The tool surface an agent actually has, resolved once and readable by a person.

A model's tool list is its capability. Assembled from remote servers it is assembled from
someone else's list, and three things go wrong there. A server advertises a tool nobody
declared and it reaches the model anyway. Two servers both advertise `search`, and which
one a call lands on depends on discovery order. A server changes a tool's schema between a
review and a run, and nothing says so.

So the surface is declared, not discovered: a server contributes the tools its allowlist
names and nothing else, each under a name worked out the same way every time, and any two
tools that would answer to one name stop the run rather than shadowing each other. A
resolved surface can be pinned — written down as names and schema fingerprints — and a
later discovery that does not match the pin is a typed error naming what moved.

The sanitisation and truncation rule is part of the contract, not an implementation detail:
characters a provider's tool-name grammar rejects fold to `_`, and a name over the budget
keeps its head plus a digest of the whole, so two long names never become one.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from tesserix_adk.core import AdkError, AdkModel, ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from tesserix_adk.mcp import McpToolDescriptor

__all__ = [
    "MAX_NAME_LENGTH",
    "McpSurfaceDriftError",
    "McpToolConflictError",
    "SurfaceEntry",
    "SurfacePin",
    "ToolSurface",
    "fingerprint",
    "namespaced",
]

MAX_NAME_LENGTH = 64
"""The longest tool name every supported provider's grammar accepts."""

LOCAL = "local"
"""The origin a tool the process defines itself is reported under."""

_UNSUPPORTED = re.compile(r"[^A-Za-z0-9_-]")
_DIGEST_LENGTH = 8
_FINGERPRINT_LENGTH = 16


class McpToolConflictError(AdkError):
    """Two tools that would answer to one name, with both origins named.

    Args:
        name: The name they would both answer to.
        origins: Where each came from, as `server/tool`, in the order they were resolved.
    """

    def __init__(self, message: str, *, name: str, origins: tuple[str, ...]) -> None:
        self.name = name
        self.origins = origins
        super().__init__(message, details={"name": name, "origins": ", ".join(origins)})


class McpSurfaceDriftError(AdkError):
    """A server's tools are no longer what the pin says they were.

    Args:
        server: The server as configured here.
        tool: The tool that moved, under the server's own name for it.
        change: `added`, `removed` or `changed`.
    """

    def __init__(self, message: str, *, server: str, tool: str, change: str) -> None:
        self.server = server
        self.tool = tool
        self.change = change
        super().__init__(message, details={"server": server, "tool": tool, "change": change})


class SurfaceEntry(AdkModel):
    """One tool as it was resolved.

    Args:
        server: The server as configured here.
        tool: What that server calls it.
        name: What the model calls it, after prefixing, sanitisation and truncation.
        fingerprint: A digest of its schemas, so a changed contract is visible.
    """

    server: str
    tool: str
    name: str
    fingerprint: str = ""


class SurfacePin(AdkModel):
    """A resolved surface written down, so a later one can be held to it."""

    entries: tuple[SurfaceEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSurface:
    """The tools one agent has, and where each of them came from."""

    entries: tuple[SurfaceEntry, ...] = ()

    @classmethod
    def of(cls, entries: Iterable[SurfaceEntry], *, local: Collection[str] = ()) -> Self:
        """Resolve `entries` into a surface, refusing any two that share a name.

        Args:
            entries: The tools adopted, in the order they were resolved.
            local: Names the process already defines itself. A remote tool answering to one
                of them is a conflict, not something to let shadow the local tool.

        Raises:
            McpToolConflictError: Two tools would answer to one name.
        """
        taken = {name: f"{LOCAL}/{name}" for name in local}
        resolved = tuple(entries)
        for entry in resolved:
            origin = f"{entry.server}/{entry.tool}"
            held = taken.get(entry.name)
            if held is not None:
                raise McpToolConflictError(
                    f"{held} and {origin} would both answer to {entry.name}; "
                    f"give one of their servers a prefix",
                    name=entry.name,
                    origins=(held, origin),
                )
            taken[entry.name] = origin
        return cls(entries=resolved)

    @classmethod
    def merged(cls, *surfaces: ToolSurface, local: Collection[str] = ()) -> Self:
        """One surface out of several, with the same conflict rule across servers."""
        return cls.of(
            (entry for surface in surfaces for entry in surface.entries),
            local=local,
        )

    def names(self) -> tuple[str, ...]:
        """Every model-facing name on this surface, sorted."""
        return tuple(sorted(entry.name for entry in self.entries))

    def of_server(self, server: str) -> Self:
        """The part of this surface one server contributed."""
        return type(self)(entries=tuple(e for e in self.entries if e.server == server))

    def pin(self) -> SurfacePin:
        """This surface written down, ready to be committed next to the agent."""
        return SurfacePin(entries=self.entries)

    def check(self, pin: SurfacePin) -> None:
        """Hold this surface to `pin`.

        Raises:
            McpSurfaceDriftError: A tool was added, removed, or had its schemas changed.
        """
        expected = {(entry.server, entry.tool): entry for entry in pin.entries}
        found = {(entry.server, entry.tool): entry for entry in self.entries}
        for key in sorted(expected.keys() - found.keys()):
            raise self._drift(key, "removed", "no longer advertises")
        for key in sorted(found.keys() - expected.keys()):
            raise self._drift(key, "added", "now advertises")
        for key in sorted(found.keys() & expected.keys()):
            if found[key].fingerprint != expected[key].fingerprint:
                raise self._drift(key, "changed", "changed the schema of")

    def report(self) -> str:
        """The surface as a person reads it: server, its name, our name, fingerprint."""
        return "".join(
            f"{entry.server}  {entry.tool}  {entry.name}  {entry.fingerprint}\n"
            for entry in self.entries
        )

    def _drift(self, key: tuple[str, str], change: str, what: str) -> McpSurfaceDriftError:
        """One drift, named in terms of the server rather than of the digest."""
        server, tool = key
        return McpSurfaceDriftError(
            f"{server} {what} {tool}, which the pinned tool surface does not allow",
            server=server,
            tool=tool,
            change=change,
        )


def namespaced(tool: str, *, prefix: str = "", limit: int = MAX_NAME_LENGTH) -> str:
    """The name a model calls `tool` by, worked out the same way every time.

    Args:
        tool: The server's own name for it, in whatever grammar that server uses.
        prefix: What to namespace it under. Empty leaves the server's own name alone.
        limit: The longest name the provider accepts.

    Returns:
        A name of at most `limit` characters, in the grammar every provider accepts.

    Raises:
        ConfigurationError: The prefix leaves no room for a name to follow it.
    """
    room = limit - _DIGEST_LENGTH - 1
    if len(prefix) >= room:
        raise ConfigurationError(
            f"the MCP server prefix {prefix!r} leaves no room for a tool name within {limit} "
            f"characters"
        )
    cleaned = _UNSUPPORTED.sub("_", tool).strip("_-") or "tool"
    joined = f"{prefix}-{cleaned}" if prefix else cleaned
    if len(joined) <= limit:
        return joined
    return f"{joined[:room]}-{_digest(joined)[:_DIGEST_LENGTH]}"


def fingerprint(descriptor: McpToolDescriptor) -> str:
    """A digest of what a call to `descriptor` is held to, so a changed contract shows."""
    canonical = json.dumps(
        {"input": descriptor.input_schema, "output": descriptor.output_schema},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest(canonical)[:_FINGERPRINT_LENGTH]


def _digest(value: str) -> str:
    """A stable digest, used for disambiguation rather than for secrecy."""
    return hashlib.sha256(value.encode()).hexdigest()
