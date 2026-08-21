"""Generate, publish and diff the event contracts consumers subscribe to.

Run `python -m tools.event_schemas --write` (or `make event-schemas`) to regenerate
`docs/event-schemas.json`. CI runs the same generation and fails on any difference, and on
any difference that a consumer written against the committed version could not absorb —
naming the event, the version and the field, because a build that fails without saying
which field gets worked around rather than fixed.

The file is a release artefact: a downstream team pins the copy from the release it
integrated against and diffs its own consumers against the next one.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from tesserix_adk.core.event_schema import compatibility_breaks, event_schemas

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "event-schemas.json"


def render(schemas: dict[str, dict[str, object]]) -> str:
    """The schemas as the committed file: sorted, indented, one contract per key."""
    return json.dumps(schemas, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Check the committed contracts, or rewrite them with `--write`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the schemas")
    args = parser.parse_args(argv)

    current = event_schemas()
    rendered = render(current)
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered, encoding="utf-8")
        return 0

    committed = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else "{}"
    breaks = compatibility_breaks(json.loads(committed), current)
    if breaks:
        sys.stderr.write("incompatible event contract change:\n  " + "\n  ".join(breaks) + "\n")
        sys.stderr.write(
            "\nwithin a major an event may only gain optional fields and enum members."
            " Publish a new version of the event type instead, and emit both for one minor.\n"
        )
        return 1
    if rendered != committed:
        diff = difflib.unified_diff(
            committed.splitlines(), rendered.splitlines(), "committed", "current", lineterm=""
        )
        sys.stderr.write("\n".join(diff) + "\n")
        sys.stderr.write("\nevent contracts changed: run `make event-schemas`.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
