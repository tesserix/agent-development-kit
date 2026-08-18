.DEFAULT_GOAL := help
.PHONY: help sync lock hooks lint format typecheck test cov api-snapshot api-check typing-gate replay-check deprecations deprecations-check release-check notes notes-check release-plan release alpha alpha-retention audit secrets licences sbom deps admissions admissions-check disclosure disclosure-check check clean

help: ## Show available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

sync: ## Install the frozen dependency set
	uv sync --frozen --all-groups

lock: ## Regenerate uv.lock after editing pyproject.toml
	uv lock

hooks: ## Install the git pre-commit hooks (same tools as CI, via uv run)
	uv run pre-commit install

lint: ## Lint, check formatting and enforce the RFC 0001 layering contracts
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports

format: ## Apply formatting and safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy in strict mode
	uv run mypy --strict src tests tools

test: ## Run the test suite
	uv run pytest

cov: ## Run the test suite with coverage enforcement
	uv run pytest --cov --cov-report=term-missing

bench: ## Measure the benchmark suite against the committed baseline
	uv run python -m tools.benchmark

bench-quick: ## The same, cut down for a fast local check before opening a change
	uv run python -m tools.benchmark --rounds 3 --iterations 10

bench-record: ## Record a new baseline, which belongs in a reviewed commit of its own
	uv run python -m tools.benchmark --write --only tokens,peak_bytes,time_to_first_token,tokens_per_second,cache_hit_ratio

api-snapshot: ## Regenerate docs/api-surface.txt after a deliberate surface change
	uv run python -m tools.api_surface --write

api-check: ## Fail if the public surface differs from the committed snapshot
	uv run python -m tools.api_surface

typing-gate: ## Fail on an undeclared typing escape hatch, or a policy entry the code lost
	uv run python -m tools.typing_gate

replay-check: ## Fail on workflow code that cannot replay, or a history that no longer does
	uv run python -m tools.replay_check

deprecations: ## Regenerate docs/deprecations.md from the @deprecate records
	uv run python -m tools.deprecations --write

deprecations-check: ## Fail if the deprecations page is out of date or a removal is overdue
	uv run python -m tools.deprecations

release-check: ## Fail if this release breaks the versioning policy against the last tag
	@planned_version=$$(uv run python -m tools.release --print-version); \
		uv run python -m tools.release_check --version "$$planned_version"

# The version only labels a preview; the real one comes from the tag at release time.
VERSION ?= next

notes: ## Preview the release notes assembled from the change fragments
	uv run python -m tools.release_notes --version $(VERSION) --dry-run

notes-check: ## Fail if a change since the last tag has no note fragment or readable subject
	uv run python -m tools.release_notes --version $(VERSION) --dry-run > /dev/null

release-plan: ## Show the version the pending changes require, and why
	uv run python -m tools.release

release: ## Fold the notes, consume the fragments and print the commands that cut VERSION
	uv run python -m tools.release --apply --version $(VERSION)

alpha: ## Show the pre-release version the next merge to main would publish
	uv run python -m tools.alpha

alpha-retention: ## List the pre-releases that should be yanked from the index
	uv run python -m tools.alpha --retention

audit: ## Fail on a blocking advisory against the locked dependency set
	uv run python -m tools.audit

secrets: ## Fail on a credential shape in the tree that the policy does not declare
	uv run python -m tools.secret_scan

licences: ## Fail on a dependency licence the policy does not allow
	uv run python -m tools.licences

sbom: ## Write the bill of materials for VERSION to sbom.cdx.json
	uv run python -m tools.sbom --version $(VERSION) --output sbom.cdx.json

disclosure: ## Regenerate the generated tables in SECURITY.md
	uv run python -m tools.disclosure --write

disclosure-check: ## Fail if a disclosure target was missed or SECURITY.md is out of date
	uv run python -m tools.disclosure

deps: ## Fail if a published requirement carries an unrecorded floor or cap
	uv run python -m tools.dependency_policy

admissions: ## Regenerate security/inventory.toml from the lock
	uv run python -m tools.admissions --write

admissions-check: ## Fail if a dependency a consumer inherits has no decision record
	uv run python -m tools.admissions

check: lint typecheck deps admissions-check disclosure-check api-check typing-gate replay-check deprecations-check release-check notes-check cov ## Everything CI runs

clean: ## Remove build and cache artefacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
