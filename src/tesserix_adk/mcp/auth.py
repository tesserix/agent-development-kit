"""What an MCP server is told about a caller, and how little authority it is handed.

A configured MCP server is one connection shared by every tenant in the process, so the
authority travels with the call rather than the connection: a credential is minted per
call for one tenant and one audience, narrowed to what that server declared it needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping  # noqa: TC003 — pydantic needs the runtime types
from dataclasses import dataclass, field

from pydantic import Field, field_validator

from tesserix_adk.core import (
    AdkModel,
    AgentIdentity,
    AuthorisationError,
    CallCredential,
    CredentialSource,
    McpAuthError,
    McpAuthReason,
    ScopeSet,
)

__all__ = [
    "META_PREFIX",
    "AuthorisedCall",
    "CallCredential",
    "CredentialSource",
    "McpAuthorizer",
    "McpServerAuth",
    "ServerSessions",
    "SessionLease",
]

META_PREFIX = "tesserix/adk"
"""The `_meta` namespace the kit's request metadata travels under."""


class McpServerAuth(AdkModel):
    """What one configured MCP server is allowed to be handed.

    Args:
        server: The configured name, used in refusals and spans.
        audience: What a credential for this server is minted for.
        scopes: The allowlist. A call presents the run's scopes narrowed to these, so a
            server cannot reach anything it was not configured for whatever it asks.
    """

    server: str = Field(min_length=1)
    audience: str
    scopes: tuple[str, ...] = ()

    @field_validator("audience")
    @classmethod
    def _addressed(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a configured mcp server names the audience its credential is for")
        return value

    @property
    def allowed(self) -> ScopeSet:
        """The allowlist as a set."""
        return ScopeSet.of(*self.scopes)


@dataclass(frozen=True, slots=True)
class AuthorisedCall:
    """One call's authority: a credential for the headers, metadata for everything else.

    The credential never appears in `meta` or `span_attributes`, because both are copied
    into places a token must not reach.
    """

    server: str
    tenant: str
    subject: str
    run_id: str
    agent: str
    scopes: frozenset[str]
    credential: CallCredential = field(repr=False)

    def headers(self) -> dict[str, str]:
        """The transport headers, the only rendering that carries the credential."""
        return self.credential.headers()

    def meta(self) -> dict[str, str]:
        """The MCP `_meta` entries naming who the call is for."""
        stated = {
            "tenant": self.tenant,
            "subject": self.subject,
            "run": self.run_id,
            "agent": self.agent,
            "scopes": " ".join(sorted(self.scopes)),
        }
        return {f"{META_PREFIX}/{key}": value for key, value in stated.items() if value}

    def span_attributes(self) -> dict[str, str]:
        """What a span may record: the shape of the authority, never the authority."""
        return {
            "mcp.server": self.server,
            "mcp.tenant": self.tenant,
            "mcp.scopes": " ".join(sorted(self.scopes)),
        }


@dataclass(frozen=True, slots=True)
class SessionLease:
    """A pooled session bound to the caller it was opened for."""

    key: tuple[str, str, str]
    server: str
    tenant: str
    subject: str

    def check(self, identity: AgentIdentity) -> None:
        """Refuse to hand this session to anyone else.

        Raises:
            McpAuthError: If another tenant or subject reached a session that is not
                theirs. A pooled connection that answers for whoever asks is a
                cross-tenant leak however scoped the credential on the call was.
        """
        principal = identity.principal
        if principal.tenant == self.tenant and principal.subject == self.subject:
            return
        raise McpAuthError(
            f"the {self.server} session belongs to {self.tenant}/{self.subject} and cannot "
            f"serve {principal.tenant}/{principal.subject}",
            reason=McpAuthReason.UNAUTHENTICATED,
            server=self.server,
        )


class ServerSessions:
    """Leases pooled MCP sessions, one per server, tenant and caller."""

    def __init__(self) -> None:
        self._leases: dict[tuple[str, str, str], SessionLease] = {}

    def lease(self, *, server: str, identity: AgentIdentity) -> SessionLease:
        """The lease for this caller on this server, opened if there is not one yet."""
        principal = identity.principal
        key = (server, principal.tenant, principal.subject)
        if key not in self._leases:
            self._leases[key] = SessionLease(
                key=key, server=server, tenant=principal.tenant, subject=principal.subject
            )
        return self._leases[key]

    @property
    def open(self) -> int:
        """How many leases are held."""
        return len(self._leases)


class McpAuthorizer:
    """Mints the authority for one MCP call, narrowed to one configured server."""

    def __init__(
        self, credentials: CredentialSource, *, servers: Mapping[str, McpServerAuth]
    ) -> None:
        self._credentials = credentials
        self._servers = dict(servers)

    def declared(self, server: str) -> McpServerAuth:
        """The configuration for `server`.

        Raises:
            McpAuthError: If the server was never configured. An unconfigured server gets
                a refusal rather than a default audience, which would be a credential
                minted for somewhere nobody named.
        """
        try:
            return self._servers[server]
        except KeyError:
            raise McpAuthError(
                f"{server!r} is not configured for this run, so no credential is minted for it",
                reason=McpAuthReason.UNAUTHENTICATED,
                server=server,
            ) from None

    def narrowed_for(
        self, *, server: str, identity: AgentIdentity, requested: Iterable[str]
    ) -> frozenset[str]:
        """What a server may have of what it asked for: the run's scopes, cut to its allowlist.

        A server asking during negotiation, or announcing a new capability later, moves
        this number down or leaves it alone. It can never move it up.
        """
        asked = ScopeSet.of(*requested)
        allowed = self.declared(server).allowed
        return frozenset(asked.narrowed_to(identity.effective).narrowed_to(allowed))

    async def authorise(
        self,
        *,
        server: str,
        identity: AgentIdentity,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
        strict: bool = False,
    ) -> AuthorisedCall:
        """The authority for one call to one server.

        Args:
            server: Which configured server the call goes to.
            identity: Who the run acts for and what it holds.
            needs: What this call wants.
            run_id: The run the call belongs to, carried downstream as attribution.
            agent_version: Which build of the agent is calling.
            strict: Refuse when the run does not hold everything `needs` asked for,
                rather than narrowing. For a call that is wrong rather than smaller
                without its full set.

        Returns:
            The call's credential and metadata.

        Raises:
            AuthorisationError: If `strict` and the run lacks a scope, or if narrowing
                left nothing — a credential for no scopes authorises nothing and would
                fail at the server as an opaque 403.
            McpAuthError: If the server is not configured, or the credential could not be
                minted. Nothing half-authorised is returned for a session to reuse.
        """
        declared = self.declared(server)
        if strict:
            identity.check(needs, where=f"mcp:{server}")
        granted = self.narrowed_for(server=server, identity=identity, requested=needs)
        if not granted:
            held = ", ".join(identity.effective) or "nothing"
            raise AuthorisationError(
                f"a call to {server} would present nothing: it declares "
                f"{', '.join(declared.scopes) or 'no scopes'} and the run holds {held}",
                agent=identity.agent,
                subject=identity.principal.subject,
                where=f"mcp:{server}",
            )
        try:
            credential = await self._credentials.for_tool(
                identity=identity,
                audience=declared.audience,
                needs=sorted(granted),
                run_id=run_id,
                agent_version=agent_version,
            )
        except (AuthorisationError, McpAuthError):
            raise
        except Exception as failure:
            raise McpAuthError(
                f"no credential could be minted for {server}, so the call was not made",
                reason=McpAuthReason.UNAUTHENTICATED,
                server=server,
                scopes=tuple(sorted(granted)),
            ) from failure
        return AuthorisedCall(
            server=server,
            tenant=identity.principal.tenant,
            subject=identity.principal.subject,
            run_id=run_id,
            agent=identity.agent,
            scopes=frozenset(credential.scopes),
            credential=credential,
        )
