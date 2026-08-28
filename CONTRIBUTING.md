# Contributing to Tesserix ADK

Thank you for improving Tesserix ADK. Changes should preserve its typed boundaries,
provider neutrality, tenant isolation, offline testability, and lean base installation.

## Before opening a change

1. Search existing issues and open or link an engineering story for non-trivial work.
2. Read the detailed [engineering guide](docs/contributing.md).
3. For a new integration or architecture, document the contract, trust boundary, failure
   behavior, and ownership before implementation.
4. For a security vulnerability, do not open a public issue; follow
   [SECURITY.md](SECURITY.md).

## Set up

```bash
git clone https://github.com/tesserix/agent-development-kit.git
cd agent-development-kit
uv sync --frozen --all-groups
uv run python examples/getting_started.py
```

The repository pins Python and `uv`. Do not resolve the environment with another package
manager and commit the resulting drift.

## Develop

- Keep the change scoped and add a failing test before implementation.
- Preserve public protocol boundaries; vendor SDK types do not travel inward.
- Keep optional SDKs behind their named extra.
- Declare capabilities instead of probing by failure.
- Treat authentication, tenant identity, secrets, untrusted input, and side effects as
  explicit boundaries.
- Add or update public documentation for consumer-visible behavior.
- Add a change fragment under `changes/` when a commit subject will not provide adequate
  release notes; breaking changes always require a migration fragment.
- Regenerate `docs/api-surface.txt` after a deliberate public API addition.

## Verify

Run focused tests while developing, then:

```bash
make check
make audit
make secrets
make licences
make docs-check
uv build
```

Report the commands actually run in the pull request. A test requiring live
infrastructure belongs in the integration lane; the default suite must remain offline.

## Pull requests

- Do not push directly to `main`, including as a maintainer. Use a branch or fork and
  open a pull request.
- `CODEOWNERS` requests the responsible maintainers. At least one code owner other than
  the last pusher must approve; a new push dismisses stale approval.
- Bring the branch up to date, resolve every review conversation, and wait for all
  required CI, documentation, and security checks.
- Merge by squash after the ruleset permits it. Force-pushes and deletion of `main` are
  prohibited.
- Use a conventional commit-style title under 72 characters.
- Explain the user-visible outcome, contract changes, risks, and rollback.
- Keep unrelated formatting or refactoring out of the diff.
- Update tests, documentation, dependency admissions, schemas, and compatibility
  snapshots in the same change when applicable.
- Do not include credentials, customer content, production endpoints, or generated
  environment dumps.

Contributions are accepted under the [Apache License 2.0](LICENSE).
Repository ownership and enforced settings are documented in
[Repository governance](docs/repository-governance.md).
