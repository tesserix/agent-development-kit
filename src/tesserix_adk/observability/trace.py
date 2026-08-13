"""One trace across every participant, and the context that carries it between them.

A run that spans a supervisor, four workers and a peer in another process produces, by
default, five traces that share nothing and a cost figure that covers whichever process
held the ledger. Nobody can say which agent spent the money or where the latency went.

What makes the difference is small: one root run id, a parent link, and enough about the
shape — pattern, branch, depth — to tell a fan-out branch from a handoff. That context has
to survive a process boundary, which is what `header` and `restored` are for, and a hop
that loses it is recorded as a break rather than quietly re-rooted.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote, unquote

from pydantic import Field

from tesserix_adk.core import AdkModel
from tesserix_adk.observability.attribution import ATTRIBUTE_PREFIX, UNKNOWN

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core import Run

__all__ = ["HEADER", "VERSION", "Pattern", "TraceContext", "attributes_of_context"]

HEADER = "adk-trace"
VERSION = "adk/1"

_FIELDS = ("root", "run", "agent", "tenant", "depth", "pattern", "branch", "broken")


class Pattern(StrEnum):
    """Why this participant was called, which is what a trace cannot be read without.

    Args:
        ROOT: The run somebody asked for.
        DELEGATION: A worker a supervisor handed a task to.
        HANDOFF: A typed transfer of the work itself to another agent.
        FAN_OUT: One branch of a parallel fan-out.
        PEER: An agent in another organisation's process, reached over A2A.
        ACTIVITY: A workflow activity or a queue consumer that picked the work up.
    """

    ROOT = "root"
    DELEGATION = "delegation"
    HANDOFF = "handoff"
    FAN_OUT = "fan_out"
    PEER = "peer"
    ACTIVITY = "activity"


class TraceContext(AdkModel):
    """Where one participant sits in a multi-agent run.

    Args:
        root_run_id: The run somebody asked for. Every participant carries the same one,
            which is what makes the trace one trace.
        run_id: This participant's own run.
        agent: Who ran.
        tenant: Whose run it is. Empty is not exportable — see `record_tree`.
        parent_run_id: Who called it. None only for the root.
        parent_agent: The calling agent, so a tree reads without a second lookup.
        depth: How far from the root. The root is zero.
        pattern: Why it was called.
        branch: Which branch of a fan-out, where it is one.
        broken: That the context arrived without a trace header and was re-rooted. A break
            is a fact about the trace, not a reason to drop the spend.
    """

    root_run_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent: str = ""
    tenant: str = ""
    parent_run_id: str | None = None
    parent_agent: str | None = None
    depth: int = Field(default=0, ge=0)
    pattern: Pattern = Pattern.ROOT
    branch: str | None = None
    broken: bool = False

    @classmethod
    def root_of(cls, run: Run[Any]) -> Self:
        """The context for the run somebody asked for.

        Args:
            run: The root run. Nothing on it is mutated.

        Example:
            >>> from tesserix_adk.core import Run
            >>> one = Run(id="r1", tenant="a", agent_name="planner", agent_version="1", model="m")
            >>> TraceContext.root_of(one).depth
            0
        """
        return cls(
            root_run_id=run.id,
            run_id=run.id,
            agent=run.agent_name or "",
            tenant=run.tenant or "",
        )

    def child(
        self,
        *,
        run_id: str,
        agent: str,
        pattern: Pattern = Pattern.DELEGATION,
        branch: str | None = None,
        tenant: str | None = None,
    ) -> Self:
        """The context for something this participant called.

        Args:
            run_id: The child's own run id.
            agent: Who is being called.
            pattern: Why. Defaults to delegation, the common case.
            branch: Which branch, for a fan-out.
            tenant: Only where the child runs for another tenant, which is not the same
                thing as a request made on another tenant's behalf. Defaults to this one.

        Returns:
            A context one level deeper, under the same root.
        """
        return type(self)(
            root_run_id=self.root_run_id,
            run_id=run_id,
            agent=agent,
            tenant=self.tenant if tenant is None else tenant,
            parent_run_id=self.run_id,
            parent_agent=self.agent,
            depth=self.depth + 1,
            pattern=pattern,
            branch=branch,
            broken=self.broken,
        )

    def header(self) -> str:
        """This context as one header value, for a hop that leaves the process.

        Every value is percent-encoded, so an agent name with a separator in it cannot
        rewrite somebody else's field on the way out.

        Example:
            >>> TraceContext(root_run_id="r1", run_id="r1", agent="planner").header()
            'adk/1 root=r1;run=r1;agent=planner;tenant=;depth=0;pattern=root;branch=;broken=0'
        """
        values = {
            "root": self.root_run_id,
            "run": self.run_id,
            "agent": self.agent,
            "tenant": self.tenant,
            "depth": str(self.depth),
            "pattern": str(self.pattern),
            "branch": self.branch or "",
            "broken": "1" if self.broken else "0",
        }
        stated = ";".join(f"{name}={quote(values[name], safe='')}" for name in _FIELDS)
        return f"{VERSION} {stated}"

    def carried(self) -> dict[str, str]:
        """The headers to put on the message, so the next process can re-attach."""
        return {HEADER: self.header()}

    @classmethod
    def restored(
        cls,
        headers: Mapping[str, str],
        *,
        run_id: str,
        agent: str,
        pattern: Pattern = Pattern.ACTIVITY,
        branch: str | None = None,
        tenant: str = "",
    ) -> Self:
        """Re-attach to the trace the headers came from, or record that they did not.

        Args:
            headers: What arrived with the message. Read case-insensitively, because a
                broker that normalises header case is not a broken trace.
            run_id: The run id this side is about to use.
            agent: Who is picking the work up.
            pattern: Why. Defaults to an activity, which is what a queue consumer is.
            branch: Which branch, where the hop is one branch of a fan-out.
            tenant: Whose work it is, where this side knows it better than the header did.

        Returns:
            A context under the header's root, or — where no readable header arrived — a
            new root with `broken` set. A break is recorded rather than raised: dropping
            the work because its trace went missing loses the spend as well as the trace.

        Example:
            >>> TraceContext.restored({}, run_id="r9", agent="worker").broken
            True
        """
        parent = _parsed(cls, headers)
        if parent is None:
            return cls(
                root_run_id=run_id,
                run_id=run_id,
                agent=agent,
                tenant=tenant,
                pattern=pattern,
                branch=branch,
                broken=True,
            )
        return parent.child(
            run_id=run_id,
            agent=agent,
            pattern=pattern,
            branch=branch,
            tenant=tenant or parent.tenant,
        )


def _parsed[ContextT: TraceContext](
    cls: type[ContextT], headers: Mapping[str, str]
) -> ContextT | None:
    """The context a header names, or None where nothing readable arrived."""
    found = next((v for k, v in headers.items() if k.lower() == HEADER), "")
    version, _, stated = found.partition(" ")
    if version != VERSION or not stated:
        return None
    fields = dict(part.partition("=")[::2] for part in stated.split(";"))
    if not {"root", "run"} <= fields.keys():
        return None
    root, run = unquote(fields["root"]), unquote(fields["run"])
    if not root or not run:
        return None
    return cls(
        root_run_id=root,
        run_id=run,
        agent=unquote(fields.get("agent", "")),
        tenant=unquote(fields.get("tenant", "")),
        depth=int(fields["depth"]) if fields.get("depth", "").isdigit() else 0,
        pattern=_pattern(unquote(fields.get("pattern", ""))),
        branch=unquote(fields.get("branch", "")) or None,
        broken=fields.get("broken") == "1",
    )


def _pattern(named: str) -> Pattern:
    """The pattern a header named, or `ACTIVITY` for one this version does not know."""
    return Pattern(named) if named in set(Pattern) else Pattern.ACTIVITY


def attributes_of_context(context: TraceContext) -> dict[str, str]:
    """The multi-agent attributes for one participant, ready to hang on a span.

    These extend the per-step set from `attributes_of` rather than replacing it: a span
    still says which tenant, agent and model burned the tokens, and these say where in the
    multi-agent run that happened.

    Args:
        context: The participant to describe.

    Returns:
        Attribute names under the `adk.` prefix, every value a string.
    """
    return {
        f"{ATTRIBUTE_PREFIX}trace_root": context.root_run_id,
        f"{ATTRIBUTE_PREFIX}run_id": context.run_id,
        f"{ATTRIBUTE_PREFIX}parent_run_id": context.parent_run_id or UNKNOWN,
        f"{ATTRIBUTE_PREFIX}parent_agent": context.parent_agent or UNKNOWN,
        f"{ATTRIBUTE_PREFIX}agent": context.agent or UNKNOWN,
        f"{ATTRIBUTE_PREFIX}tenant": context.tenant or UNKNOWN,
        f"{ATTRIBUTE_PREFIX}delegation_depth": str(context.depth),
        f"{ATTRIBUTE_PREFIX}pattern": str(context.pattern),
        f"{ATTRIBUTE_PREFIX}branch": context.branch or UNKNOWN,
        f"{ATTRIBUTE_PREFIX}trace_broken": "true" if context.broken else "false",
    }
