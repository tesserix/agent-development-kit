"""What a run has been permitted, bound to exactly the payload a human was shown.

An approval that is not bound to its arguments is a licence: the repair loop that fixes a
malformed amount, the retry that re-sends the call, and the tool result that suggests a
larger refund all execute under a decision nobody made about them. The ledger holds a grant
to one payload, once, for one run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.errors import ApprovalBindingError
from tesserix_adk.core.hooks import digest_of

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from tesserix_adk.core.hooks import ApprovalRecord

__all__ = ["ApprovalLedger"]


class ApprovalLedger:
    """The grants one run is holding, and what each of them covers."""

    def __init__(self) -> None:
        self._granted: dict[str, str] = {}
        self._spent: set[str] = set()
        self._void = False

    def bind(self, record: ApprovalRecord) -> None:
        """Record that `record` was granted, for the payload it was raised over."""
        self._granted[record.id] = record.arguments_digest

    def spend(self, record: ApprovalRecord, arguments: Mapping[str, Any]) -> None:
        """Use the grant for `record` to execute `arguments`, exactly once.

        Raises:
            ApprovalBindingError: If the run was cancelled, if nothing granted this record,
                if the grant has already been used, or if the arguments are not the ones it
                was raised over. Every one of them fails closed and needs a fresh decision.
        """
        if self._void:
            raise ApprovalBindingError(
                f"approval for {record.tool_name!r} belongs to a run that is over; "
                f"a decision nobody is waiting on executes nothing"
            )
        granted = self._granted.get(record.id)
        if granted is None:
            raise ApprovalBindingError(f"approval for {record.tool_name!r} was never granted")
        if record.id in self._spent:
            raise ApprovalBindingError(
                f"approval for {record.tool_name!r} was already used; one decision is one "
                f"execution, so a replayed answer buys nothing"
            )
        if digest_of(arguments) != granted:
            raise ApprovalBindingError(
                f"the arguments for {record.tool_name!r} are not the ones that were "
                f"approved; permission is for one payload, not for the tool"
            )
        self._spent.add(record.id)

    def void(self) -> None:
        """Invalidate every grant, because the run they belong to has ended."""
        self._void = True
