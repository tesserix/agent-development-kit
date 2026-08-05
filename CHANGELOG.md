# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

### Added

- Optional extras per integration (`mcp`, `temporal`, `graphiti`, `redis`, `postgres`,
  and `all` as their union), with `tesserix_adk.core.require_extra` and
  `MissingExtraError` naming the extra and its install command. Base requirements are
  `pydantic`, `httpx` and `opentelemetry-api` only, held there by a transitive package
  ceiling and a per-extra CI leg.
- Public API surface snapshot (`docs/api-surface.txt`) with a CI gate, a leak check
  rejecting vendor and concrete implementation types in public signatures, and an
  explicit allowlist for third-party re-exports.
- `tesserix_adk.testing.pytest_plugin`: network isolation for unit suites and a
  quarantine marker requiring owner, expiry and reason.
- `tesserix_adk.core`: protocols, error hierarchy, `verify_conformance`, and shipped
  conformance suites and fakes in `tesserix_adk.testing`.

### Notes

- `tesserix_adk.experimental` carries no stability promise and is deliberately excluded
  from the snapshot. Promoting a symbol out of it requires an entry here and a
  stability statement in the same pull request.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
