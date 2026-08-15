"""What a package needs of a credential, without knowing who mints it.

The integration packages sit beside `tesserix_adk.tools` in the layering rather than
under it, so what they need is stated here structurally. `tools.CredentialBroker`
satisfies both protocols without either side importing the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import SecretStr

    from tesserix_adk.core.identity import AgentIdentity

__all__ = ["CallCredential", "CredentialSource"]


@runtime_checkable
class CallCredential(Protocol):
    """The part of a minted credential a caller renders."""

    @property
    def token(self) -> SecretStr:
        """The secret itself, which only `headers` may render."""

    @property
    def scopes(self) -> frozenset[str]:
        """What was granted, which may be less than was asked for."""

    def headers(self) -> dict[str, str]:
        """The request headers carrying the credential and its attribution."""


@runtime_checkable
class CredentialSource(Protocol):
    """Whatever mints per-call credentials — `tesserix_adk.tools.CredentialBroker` does."""

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = ...,
    ) -> CallCredential:
        """Mint one credential for one caller, one audience and one set of scopes."""
