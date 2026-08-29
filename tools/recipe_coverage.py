"""Map every declared public symbol to a runnable, docs-included recipe."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.api_surface import collect_surface

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "docs" / "cookbook" / "recipes.toml"
MAPPING = ROOT / "docs" / "cookbook" / "symbols.json"


class CoverageError(Exception):
    """Raised when a coverage route cannot provide the recipe it promises."""


@dataclass(frozen=True)
class RecipeRoute:
    """One public-name prefix and the runnable recipe that teaches it."""

    prefix: str
    recipe: str
    page: str
    rule: str
    id: str = ""
    extra: str = ""


def routes(path: Path = ROUTES) -> tuple[RecipeRoute, ...]:
    """Read and validate the committed recipe routes."""
    document: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_routes = document.get("route")
    if not isinstance(raw_routes, list):
        raise CoverageError(f"{path} must declare [[route]] entries")
    declared = tuple(_route(item, path) for item in raw_routes)
    identifiers = [route.id for route in declared]
    repeated = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if repeated:
        raise CoverageError(f"duplicate recipe ids: {', '.join(repeated)}")
    if not any(route.prefix == "tesserix_adk" for route in declared):
        raise CoverageError("recipe routes need a tesserix_adk fallback")
    return declared


def _route(value: object, path: Path) -> RecipeRoute:
    """Validate one untrusted TOML table as a route."""
    if not isinstance(value, dict):
        raise CoverageError(f"{path} contains a route that is not a table")
    required = ("id", "prefix", "recipe", "page", "rule")
    missing = [key for key in required if not isinstance(value.get(key), str) or not value[key]]
    if missing:
        raise CoverageError(f"recipe route is missing text fields: {', '.join(missing)}")
    return RecipeRoute(
        id=str(value["id"]),
        prefix=str(value["prefix"]),
        recipe=str(value["recipe"]),
        page=str(value["page"]),
        rule=str(value["rule"]),
        extra=str(value.get("extra", "")),
    )


def mapped(
    surface: Mapping[str, str], declared: Sequence[RecipeRoute]
) -> dict[str, dict[str, str]]:
    """Map every symbol to the most specific valid route."""
    for route in declared:
        _verify(route)
    ordered = sorted(enumerate(declared), key=lambda item: (-len(item[1].prefix), item[0]))
    coverage: dict[str, dict[str, str]] = {}
    for symbol in sorted(surface):
        selected = next((item for _, item in ordered if symbol.startswith(item.prefix)), None)
        if selected is None:
            raise CoverageError(f"no runnable recipe route covers {symbol}")
        coverage[symbol] = {
            "id": selected.id,
            "recipe": selected.recipe,
            "page": selected.page,
            "rule": selected.rule,
            **({"extra": selected.extra} if selected.extra else {}),
        }
    return coverage


def _verify(route: RecipeRoute) -> None:
    """Refuse a route whose example or docs inclusion is missing."""
    recipe = ROOT / route.recipe
    page = ROOT / route.page
    if recipe.suffix != ".py" or not recipe.is_file():
        raise CoverageError(f"recipe {route.recipe} does not exist as an executable Python file")
    if not page.is_file():
        raise CoverageError(f"cookbook page {route.page} does not exist")
    included = f'--8<-- "{route.recipe}"'
    if included not in page.read_text(encoding="utf-8"):
        raise CoverageError(f"{route.recipe} is not included from {route.page}")


def render(coverage: Mapping[str, Mapping[str, str]]) -> str:
    """Render a reviewable, deterministic symbol-to-recipe contract."""
    recipes: dict[str, dict[str, str]] = {}
    symbols: dict[str, str] = {}
    for symbol, details in sorted(coverage.items()):
        identifier = details["id"]
        symbols[symbol] = identifier
        record = {key: value for key, value in details.items() if key != "id"}
        if identifier in recipes and recipes[identifier] != record:
            raise CoverageError(f"recipe id {identifier} resolves to conflicting routes")
        recipes[identifier] = record
    return (
        json.dumps({"format": 1, "recipes": recipes, "symbols": symbols}, indent=2, sort_keys=True)
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Check the committed mapping, or regenerate it with ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the mapping")
    parsed = parser.parse_args(argv)
    try:
        current = render(mapped(collect_surface(), routes()))
    except CoverageError as failure:
        sys.stderr.write(f"recipe coverage refused: {failure}\n")
        return 1
    if parsed.write:
        MAPPING.write_text(current, encoding="utf-8")
        return 0
    committed = MAPPING.read_text(encoding="utf-8") if MAPPING.exists() else ""
    if current == committed:
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(), current.splitlines(), "committed", "current", lineterm=""
    )
    sys.stderr.write("\n".join(diff) + "\nrun `make recipe-coverage` and review the mapping.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
