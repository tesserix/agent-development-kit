# ADR 0002 — AgentGateway is the MCP data-plane boundary

**Status:** Accepted
**Date:** 2026-08-19

## Context

The Agentic Registry already owns MCP server discovery and renders AgentGateway routes.
AgentGateway and its route-sync job already own the live data plane. The ADK has per-call
MCP credential narrowing and tenant-scoped session leases, but consumers still have to
assemble URLs, sessions, discovery, namespacing, and tool registration themselves.

The first consumers advertise at most 40 tools from fewer than 16 MCP servers. A tool call
is normally below 64 KiB of arguments and 256 KiB of result data. The client-side routing
objective is p99 below 5 ms excluding network time; the default end-to-end deadline is 15
seconds. The library stores no data, so 12- and 36-month data volume are both zero. Reads
and writes share the same path, but writes are never retried automatically.

## Decision

The ADK provides a first-class AgentGateway MCP router with these boundaries:

- the registry remains the control plane and is never consulted during a tool call;
- every route is operator configured, every exposed tool is explicitly allowlisted, and
  tool names are namespaced as `<server>__<tool>`;
- trusted descriptions, required scopes, and approval policy come from ADK configuration,
  never from an MCP server's untrusted annotations;
- credentials are minted per discovery or tool call and narrowed to the run, route, and
  tool; authorization leases remain keyed by server, tenant, and subject and are not MCP
  transport sessions;
- discovery is pinned into one immutable tool set for a run, with a default ceiling of 40;
- arguments and responses are bounded, gateway responses remain untrusted tool results,
  and an MCP error becomes a typed ADK failure;
- the optional `mcp` extra owns the Streamable HTTP transport. The router itself speaks a
  small protocol so tests and alternate transports do not import vendor types.

As of MCP `2026-07-28`, every AgentGateway operation first uses the official SDK's
`server/discover` path and then sends self-contained requests. It does not call legacy
`initialize`, does not retain `Mcp-Session-Id`, and may reach a different healthy backend
replica on the next operation. The in-process `McpSession.initialize()` abstraction remains
for transport-neutral and legacy compatibility tests; it is not the production HTTP path.

When AgentGateway is down, discovery and calls fail closed with a typed error. There is no
direct-to-backend fallback because that would bypass the single-egress policy. When the
registry is down, already-rendered AgentGateway routes and already-pinned run tool sets keep
working. The router performs no automatic retries: callers may retry a read-only operation
through their existing idempotency policy, while effectful calls remain at-most-once from
the ADK's point of view.

## Alternatives

Querying the registry on every call was rejected because it adds a serial dependency and
makes a control-plane outage a data-plane outage. Connecting agents directly to each MCP
server was rejected because it bypasses gateway auth, policy, telemetry, and rate limits.
Trusting remote tool descriptions and annotations was rejected because they enter the model
prompt and cannot grant authority or disable approval.

## Consequences

An operator must configure each exposed tool once, including its trusted description and
least-privilege scopes. A server adding a tool does not silently widen an agent. Tool changes
take effect when a new tool set is built, not halfway through a run. The extra opens a fresh
bounded MCP session per operation initially; pooling can be added behind the transport
protocol without changing the public router or weakening tenant leases.

The change is additive. Rollback means removing the router from consumer composition; the
existing MCP authorization API and direct transports continue to work unchanged. Rolling
the transport back to handshake-era MCP is not an approved production rollback because it
would restore session affinity and pod-loss coupling.
