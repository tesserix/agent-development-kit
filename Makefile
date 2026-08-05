.DEFAULT_GOAL := help
.PHONY: help sync lock hooks lint format typecheck test cov api-snapshot api-check deprecations deprecations-check release-check check clean

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

api-snapshot: ## Regenerate docs/api-surface.txt after a deliberate surface change
	uv run python -m tools.api_surface --write

api-check: ## Fail if the public surface differs from the committed snapshot
	uv run python -m tools.api_surface

deprecations: ## Regenerate docs/deprecations.md from the @deprecate records
	uv run python -m tools.deprecations --write

deprecations-check: ## Fail if the deprecations page is out of date or a removal is overdue
	uv run python -m tools.deprecations

release-check: ## Fail if this release breaks the versioning policy against the last tag
	uv run python -m tools.release_check

check: lint typecheck api-check deprecations-check release-check cov ## Everything CI runs

clean: ## Remove build and cache artefacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
