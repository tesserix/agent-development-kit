"""Where a finished run goes once the runtime is done with it.

A recorder sees every terminal run and ships it somewhere: a trace backend, an audit
store, a test list. The runtime calls it after the terminal transition and never lets it
fail the run — a collector outage costs a trace, not an answer. The process-wide default
lets a deployment turn tracing on by installing one recorder at startup, so no product
has to thread a recorder through every runner it builds.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tesserix_adk.core.run import Run

__all__ = [
    "RunRecorder",
    "bound_session",
    "current_session",
    "default_recorder",
    "install_default_recorder",
]


@runtime_checkable
class RunRecorder(Protocol):
    """What the runtime hands every terminal run to.

    Recording fails open. An implementation swallows its own transport failures rather
    than propagating them, and `record` returns before any network round trip.
    """

    def record(self, run: Run[Any]) -> None:
        """Take one finished run. Must not raise and must not block on export."""
        ...

    def shutdown(self) -> None:
        """Flush whatever is queued. Called once, at process exit."""
        ...


_DEFAULT: RunRecorder | None = None
_SESSION: ContextVar[str | None] = ContextVar("adk_run_session", default=None)


def install_default_recorder(recorder: RunRecorder | None) -> RunRecorder | None:
    """Set the recorder every runner without its own uses. Returns the one it replaced."""
    global _DEFAULT
    previous = _DEFAULT
    _DEFAULT = recorder
    return previous


def default_recorder() -> RunRecorder | None:
    """The process-wide recorder, or None when nothing was installed."""
    return _DEFAULT


@contextlib.contextmanager
def bound_session(session_id: str | None) -> Iterator[None]:
    """Attach a conversation or job id to every run recorded inside the block."""
    token = _SESSION.set(session_id or None)
    try:
        yield
    finally:
        _SESSION.reset(token)


def current_session() -> str | None:
    """The session bound to this context, or None outside any."""
    return _SESSION.get()
