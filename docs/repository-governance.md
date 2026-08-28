# Repository governance

The source is public and forkable, but the canonical `main` branch is not directly
writable. Public visibility grants read and fork access; it does not grant upstream push
access.

## Ownership

`.github/CODEOWNERS` assigns every path to at least two maintainers and repeats ownership
for workflow, dependency, security, and lockfile changes. A matching code owner is
automatically requested on a pull request. Ownership is review responsibility, not a
direct-push exception.

## Enforced `main` contract

The active GitHub ruleset for the default branch requires:

- every change to arrive through a pull request;
- at least one code-owner approval, with stale approvals dismissed after a push;
- approval from someone other than the person who made the latest push;
- the pull request branch to be current with `main`;
- all named CI, documentation, security, and CodeQL checks to succeed;
- every review conversation to be resolved;
- squash merging and linear history; and
- deletion and force-push prevention.

The `main` ruleset has **no bypass actors**, including repository administrators. An
urgent correction is a small reviewed pull request that reverts or fixes the change; it
is not a direct push. Changing or disabling the ruleset is an organization-level
administrative action and should be treated as a security event with an audit trail.

Release tags matching `v*` have a separate rule: only repository administrators may
create them, and existing release tags cannot be moved or deleted through the normal
contributor path. A tag triggers the reviewed release workflow; it does not bypass its
version, build, provenance, or smoke-install gates.

## Contributor path

1. Fork the repository or create a non-protected branch.
2. Add the failing test, the scoped implementation, documentation, and a change fragment
   where required.
3. Run the commands in [Contributing](contributing.md).
4. Open a pull request using the template and allow workflows that require no repository
   secret to run against the contribution.
5. Address review and wait for a code owner and every required check.

Outside contributors never receive repository secrets. Release publication uses GitHub
OIDC and a protected environment rather than a long-lived upload token.

## Maintainer path

Maintainers follow the same pull-request path. Dependency update bots propose changes;
they do not merge them. Releases are cut only from a reviewed commit already on `main`,
using the tag-driven process in [Releasing](releasing.md).

Ruleset drift should be checked after ownership, plan, or workflow changes. The source
files establish intended ownership and checks; the GitHub API is the evidence that the
hosted enforcement remains active.
