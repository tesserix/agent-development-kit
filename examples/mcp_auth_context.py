"""Two tenants, one process, one MCP server that can tell them apart.

Run it with `uv run python examples/mcp_auth_context.py`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from tesserix_adk.adapters import (
    AuthorisingSession,
    CallerContext,
    HttpTransport,
    TenantAuthority,
    TransportSession,
    arriving_call,
    redacted,
)
from tesserix_adk.core import McpAuthError, Principal, TenantContext, tenant_scope
from tesserix_adk.core.config import McpServerConfig
from tesserix_adk.core.identity import AgentIdentity
from tesserix_adk.mcp import GatewayToolResult, McpAuthorizer, McpServerAuth
from tesserix_adk.tools.credentials import Credential, CredentialBroker

if TYPE_CHECKING:
    from tesserix_adk.tools.credentials import CredentialRequest

HANDBOOK = McpServerConfig(name="handbook", endpoint="https://handbook.internal/mcp")


class Clock:
    """A clock the example holds still."""

    def now(self) -> float:
        """The reading everything in this example is measured against."""
        return 0.0

    async def sleep(self, seconds: float) -> None:
        """Nothing waits here."""
        del seconds


class Mint:
    """A credential provider that mints one short-lived token per tenant and audience."""

    def __init__(self) -> None:
        self.minted = 0

    async def issue(self, request: CredentialRequest) -> Credential:
        """One credential, named after the tenant it was minted for."""
        self.minted += 1
        return Credential(
            token=SecretStr(f"tok-{request.attribution.tenant}-{self.minted}"),
            audience=request.audience,
            scopes=request.scopes,
            expires_at=600.0,
            attribution=request.attribution,
        )


async def server(request: httpx.Request) -> httpx.Response:
    """A server that scopes its answer to the caller and echoes the credential back at it."""
    params = json.loads(request.read().decode()).get("params", {})
    arrived = arriving_call(headers=dict(request.headers), meta=params.get("_meta", {}))
    with arrived.bound() as here:
        answer = f"leave policy for {here.tenant}, asked by {arrived.subject}"
    echoed = request.headers.get("authorization", "")
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": f"{answer} (you sent {echoed})"}]},
        },
    )


def identity(tenant: str, subject: str) -> AgentIdentity:
    """The authority one run holds, resolved the way the runtime resolves it."""
    return AgentIdentity.resolve(
        agent="desk",
        declared=("hb:read",),
        principal=Principal(subject=subject, tenant=tenant, scopes=frozenset({"hb:read"})),
    )


async def asked(authority: TenantAuthority, client: httpx.AsyncClient) -> GatewayToolResult:
    """One tool call, carrying whatever the bound tenant makes it able to carry."""
    transport = HttpTransport(HANDBOOK, client=client, authority=authority)
    session = AuthorisingSession(
        TransportSession(transport, config=HANDBOOK), authority=authority, server="handbook"
    )
    return await session.call_tool("search", {"query": "leave"}, meta={}, timeout_seconds=5.0)


async def main() -> None:
    """Two tenants through one authorizer, then a call with nothing bound."""
    clock = Clock()
    mint = Mint()
    authorizer = McpAuthorizer(
        CredentialBroker(mint, clock=clock),
        servers={
            "handbook": McpServerAuth(
                server="handbook", audience="handbook.svc", scopes=("hb:read",)
            )
        },
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))

    for tenant, subject in (("acme", "ada"), ("globex", "bo")):
        who = identity(tenant, subject)
        run = f"run-{tenant}"
        authority = TenantAuthority(
            authorizer,
            caller=lambda who=who, run=run: CallerContext.current(identity=who, run_id=run),
            clock=clock,
        )
        with tenant_scope(TenantContext(tenant=tenant, user=subject)):
            answered = await asked(authority, client)
        print(tenant, "->", answered.content[0]["text"])  # noqa: T201

    unscoped = TenantAuthority(
        authorizer,
        caller=lambda: CallerContext.current(identity=identity("acme", "ada"), run_id="run-x"),
        clock=clock,
    )
    try:
        await asked(unscoped, client)
    except McpAuthError as refused:
        print("no tenant bound ->", refused)  # noqa: T201

    echoed = GatewayToolResult(content=({"type": "text", "text": "you sent tok-acme-1"},))
    print("redacted ->", redacted(echoed, secrets=("tok-acme-1",)).content[0]["text"])  # noqa: T201

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
