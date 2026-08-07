"""What a tool needs to know that the model must never be able to choose.

A model picks arguments. It does not pick the tenant whose data the tool may read, the
user the action is taken on behalf of, or the switch that stops the work — and a schema
that lists those as parameters is a schema the model can fill in. So they travel beside
the arguments rather than among them: injected by the runtime, never described, and never
accepted from a payload.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tesserix_adk.runtime.blocking import current_ambient

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.runtime.cancellation import CancellationToken

__all__ = ["ToolContext"]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """The run a tool call belongs to, handed to the tool and hidden from the model.

    A parameter annotated `ToolContext` is filled in by whoever invokes the tool and left
    out of the model-facing schema entirely, so no amount of prompting reaches it.

    Args:
        run_id: The run this call belongs to.
        tenant: The isolation boundary the work must not read or write outside of.
        user: The acting principal, where there is one.
        trace: Trace context to propagate onward, W3C keys included, so a call the tool
            makes lands under the same trace as the run that caused it.
        cancellation: The caller's switch. A tool that may run long checks it; a tool that
            never does is abandoned rather than interrupted.

    Example:
        >>> ToolContext(run_id="run-1", tenant="acme").tenant
        'acme'
    """

    run_id: str
    tenant: str
    user: str | None = None
    trace: Mapping[str, str] = field(default_factory=dict)
    cancellation: CancellationToken | None = None

    @classmethod
    def current(cls) -> ToolContext | None:
        """The context of the run in hand, or `None` outside one.

        Built from the ambient the runtime binds, so a tool invoked without an explicit
        context still gets the right tenant instead of a guessed one. `None` rather than
        an invented default: a tenant nobody set is the bug this makes visible.
        """
        ambient = current_ambient()
        if ambient is None:
            return None
        return cls(
            run_id=ambient.run_id,
            tenant=ambient.tenant,
            user=ambient.user,
            cancellation=ambient.cancellation,
        )

    def raise_if_cancelled(self) -> None:
        """Raise if the run has been cancelled, otherwise do nothing.

        Raises:
            CancelledError: If the caller's switch has been flipped. Nothing is raised
                where there is no switch: absence is not consent, it is nothing to check.
        """
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()
