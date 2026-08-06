# Changelog

Notable changes to `tesserix-adk`. Format follows [Keep a Changelog]; the project
follows semantic versioning once it reaches 0.1.0.

Any pull request that changes `docs/api-surface.txt` must add an entry here and state
the stability decision behind it. The `api-surface` CI job stays red until it does.

## [Unreleased]

### Added

- Signed releases with build provenance (`docs/verifying.md`): every artefact, on the
  release and alpha channels alike, is signed keyless by workflow identity and carries a
  SLSA provenance attestation naming the repository, workflow, commit and build inputs.
  `sbom.cdx.json` is attested by the same run, uploads carry a PEP 740 attestation to
  PyPI, and the bundles are attached to the GitHub Release so a mirrored or air-gapped
  install can verify offline. Both the release smoke job and the downstream consumer job
  verify before installing and fail closed. There is no signing key to steal or rotate.
- A bill of materials published with every release (`tools/sbom.py`): a CycloneDX 1.6
  document built from `uv.lock` inside the release job, so it describes the graph that was
  published rather than a later re-resolution, attached to the GitHub Release as
  `sbom.cdx.json`. Each component carries its purl, licence, source hash, the install
  profile that reaches it (`base` or `extra:<name>`) and every wheel with its platform tag
  and hash; development-only packages are excluded. A diff against the previous release's
  document goes into the release notes.
- A licence policy across the whole graph (`tools/licences.py`, `security/licences.toml`),
  checked on every pull request and again in the release before anything irreversible. An
  undeclared or disallowed licence blocks, `A AND B` needs both halves allowed, `A OR B`
  needs a recorded decision naming an owner, and a licence off the allow list can be
  accepted for one named package and one named licence only.
- Dependency and secret scanning (`docs/security.md`, `.github/workflows/security.yml`):
  advisories audited against the frozen lock on every pull request and daily, rated from
  OSV with an unrated advisory blocking, and reported with the first fixed version and the
  blast radius derived from the lockfile — a finding reachable only through an extra still
  blocks and names the extra. Credentials are matched by shape across the tree and by a
  pinned `gitleaks` across the history, with recorded provider traffic also checked for
  personal identifiers and every matched value truncated in the report. Suppressions live
  in `security/policy.toml` and require an owner, a reason and an expiry of at most 90
  days; an expired suppression fails the build. Neither job uses a repository secret, so
  both run on a pull request from a fork.
- Pre-release alpha channel (`docs/stability.md`, `.github/workflows/alpha.yml`): every
  merge to `main` publishes a pre-release through the same gates and trusted publisher as
  a release, numbered by `tools/alpha.py` from the next minor after the last stable.
  Consumers opt in explicitly — a stable specifier never resolves a pre-release — and the
  per-subpackage stability matrix states what each one promises. Promotion is
  alpha → rc → stable through the one pipeline, retention is reported rather than
  guessed, and a downstream job runs the earliest consumer's suite against the last
  stable and the alpha, failing only on a regression the alpha introduced.
- Release notes assembled from the repository (`tools/release_notes.py`): change
  fragments in `changes/`, conventional commit subjects, the public API snapshot diff and
  the live deprecation records. A change with neither a fragment nor a readable subject
  blocks the release, as does a breaking change with no migration note. The
  `release-notes` CI job renders the notes on every pull request, and the release body is
  the assembled notes rather than generated commit titles.
- Release path (`docs/releasing.md`): the git tag is the sole source of the version
  (`hatch-vcs` derives it, `tesserix_adk.__version__` reads it back from the installed
  distribution), a guard refuses a tag off `main`, in the wrong format, or naming a
  version already on the index, and publishing goes to PyPI by trusted publishing with
  no upload token in existence. Artefacts are mirrored to GitHub Release assets, a
  divergence between the two fails the release, and a per-extra smoke job installs the
  published wheel from the index and runs `examples/getting_started.py`.
- Versioning and deprecation policy (`docs/versioning.md`), with `@deprecate` recording
  the alternative and removal version, `AdkDeprecationWarning` warned once per call site
  and attributed to the caller's frame, `TESSERIX_ADK_DEPRECATIONS_AS_ERRORS` for consumers
  preparing an upgrade, a generated `docs/deprecations.md`, and a release check that
  refuses a removal with no deprecation record or a version bump too small for it.
- `tesserix_adk.core.config`: one frozen `AdkConfig` resolved once at startup from code,
  `TESSERIX_ADK_*` environment variables, `adk.toml` or `[tool.tesserix-adk]`, in that
  precedence. `resolve_config` additionally returns per-key provenance and `explain()`;
  `ConfigError` reports every problem at once with the layer that supplied each. Secrets
  are environment-only and masked everywhere they could be rendered. Adding a key is a
  minor release; renaming or removing one goes through the deprecation policy.
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
