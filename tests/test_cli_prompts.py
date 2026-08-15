"""Inspecting and reverting a prompt from a terminal, during the incident it caused."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.cli import prompts_main
from tesserix_adk.core import FilePromptRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

ITINERARY = """
owner = "platform"
task_class = "planning"

[aliases]
current = "5"

[versions.4]
body = "Plan an itinerary for ${city}.\\nKeep it to three days."
variables = ["city"]

[versions.5]
body = "Plan an itinerary for ${city} within ${budget}.\\nKeep it to three days."
variables = ["city", "budget"]
"""


class Book:
    """A deployment's alias store, recording what an operator did to it."""

    def __init__(self, *, evaluated: Mapping[str, str] | None = None) -> None:
        self.pointing = {"current": "5"}
        self.results = dict(evaluated or {"5": "pass", "4": "pass"})
        self.moves: list[dict[str, str]] = []

    async def aliases(self, name: str, *, tenant: str = "") -> Mapping[str, str]:
        del name, tenant
        return dict(self.pointing)

    async def evaluated(self, name: str, version: str, *, tenant: str = "") -> str:
        del name, tenant
        return self.results.get(version, "")

    async def repoint(
        self,
        name: str,
        *,
        alias: str,
        to: str,
        expected: str,
        by: str,
        reason: str,
        tenant: str = "",
    ) -> None:
        del tenant
        if self.pointing[alias] != expected:
            message = "alias moved under the rollback"
            raise RuntimeError(message)
        self.pointing[alias] = to
        self.moves.append(
            {"name": name, "alias": alias, "from": expected, "to": to, "by": by, "reason": reason}
        )


@pytest.fixture
def registry(tmp_path: Path) -> FilePromptRegistry:
    """One prompt at two versions, with `current` on the newer."""
    (tmp_path / "itinerary_system.toml").write_text(ITINERARY, encoding="utf-8")
    return FilePromptRegistry(tmp_path)


async def run(argv: Sequence[str], *, registry: FilePromptRegistry, book: Book) -> tuple[int, str]:
    """One command, its exit code and what it printed."""
    out = io.StringIO()
    code = await prompts_main(argv, prompts=registry, aliases=book, out=out)
    return code, out.getvalue()


class TestSeeingWhatExists:
    """The first question in an incident is which versions there are and what is live."""

    async def test_list_shows_versions_digests_aliases_and_evals(
        self, registry: FilePromptRegistry
    ) -> None:
        code, written = await run(["list", "itinerary_system"], registry=registry, book=Book())

        assert code == 0
        assert "4" in written
        assert "current" in written
        assert "platform" in written

    async def test_a_version_never_evaluated_says_so(self, registry: FilePromptRegistry) -> None:
        book = Book(evaluated={"5": "pass"})

        _, written = await run(["list", "itinerary_system"], registry=registry, book=book)

        assert "not evaluated" in written

    async def test_show_prints_one_version_in_full(self, registry: FilePromptRegistry) -> None:
        code, written = await run(
            ["show", "itinerary_system", "--version", "4"], registry=registry, book=Book()
        )

        assert code == 0
        assert "Keep it to three days." in written

    async def test_a_prompt_this_registry_does_not_have_is_refused(
        self, registry: FilePromptRegistry
    ) -> None:
        code, written = await run(
            ["show", "elsewhere", "--version", "1"], registry=registry, book=Book()
        )

        assert code == 1
        assert "refused" in written


class TestSeeingWhatMoved:
    """A diff of prompt text, and of everything around it that changes behaviour."""

    async def test_the_diff_names_body_variables_and_the_digest(
        self, registry: FilePromptRegistry
    ) -> None:
        code, written = await run(
            ["diff", "itinerary_system", "4", "5"], registry=registry, book=Book()
        )

        assert code == 0
        assert "digest changed" in written
        assert "variables added: budget" in written
        assert "+Plan an itinerary for ${city} within ${budget}." in written

    async def test_a_whitespace_only_change_is_called_out(self, tmp_path: Path) -> None:
        (tmp_path / "greeting.toml").write_text(
            '[versions.1]\nbody = "Greet them."\n[versions.2]\nbody = "Greet   them."\n',
            encoding="utf-8",
        )

        _, written = await run(
            ["diff", "greeting", "1", "2"], registry=FilePromptRegistry(tmp_path), book=Book()
        )

        assert "whitespace only" in written

    async def test_a_long_diff_is_summarised_unless_asked_for_in_full(self, tmp_path: Path) -> None:
        old = "\\n".join(f"line {index}" for index in range(80))
        new = "\\n".join(f"changed {index}" for index in range(80))
        (tmp_path / "long.toml").write_text(
            f'[versions.1]\nbody = "{old}"\n[versions.2]\nbody = "{new}"\n', encoding="utf-8"
        )
        registry = FilePromptRegistry(tmp_path)

        _, truncated = await run(["diff", "long", "1", "2"], registry=registry, book=Book())
        _, full = await run(["diff", "long", "1", "2", "--full"], registry=registry, book=Book())

        assert "more diff lines" in truncated
        assert "more diff lines" not in full
        assert len(full) > len(truncated)

    async def test_a_fixture_diffs_what_the_model_would_receive(
        self, registry: FilePromptRegistry, tmp_path: Path
    ) -> None:
        values = tmp_path / "values.json"
        values.write_text(json.dumps({"city": "Kyoto", "budget": "AUD 2000"}), encoding="utf-8")

        _, written = await run(
            ["diff", "itinerary_system", "4", "5", "--values", str(values)],
            registry=registry,
            book=Book(),
        )

        assert "Plan an itinerary for Kyoto within AUD 2000." in written
        assert "${city}" not in written

    async def test_a_version_that_cannot_render_the_fixture_says_so(
        self, registry: FilePromptRegistry, tmp_path: Path
    ) -> None:
        values = tmp_path / "values.json"
        values.write_text(json.dumps({"city": "Kyoto"}), encoding="utf-8")

        _, written = await run(
            ["diff", "itinerary_system", "4", "5", "--values", str(values)],
            registry=registry,
            book=Book(),
        )

        assert "cannot render" in written


class TestPuttingItBack:
    """Rolling an alias back is not a code change, and is recorded like one."""

    async def test_an_alias_moves_to_the_earlier_version(self, tmp_path: Path) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        book = Book()

        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada"],
            registry=FilePromptRegistry(tmp_path),
            book=book,
        )

        assert code == 0
        assert book.pointing["current"] == "4"
        assert "current 5 -> 4" in written

    async def test_the_rollback_records_who_when_from_and_to(self, tmp_path: Path) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        book = Book()

        await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada", "--reason", "PLAT-91"],
            registry=FilePromptRegistry(tmp_path),
            book=book,
        )

        assert book.moves == [
            {
                "name": "itinerary_system",
                "alias": "current",
                "from": "5",
                "to": "4",
                "by": "ada",
                "reason": "PLAT-91",
            }
        ]

    async def test_a_target_dropping_a_variable_in_use_is_refused(
        self, registry: FilePromptRegistry
    ) -> None:
        book = Book()

        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada"],
            registry=registry,
            book=book,
        )

        assert code == 1
        assert "budget" in written
        assert book.pointing["current"] == "5"

    async def test_a_target_never_evaluated_needs_forcing(self, tmp_path: Path) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        registry = FilePromptRegistry(tmp_path)
        book = Book(evaluated={"5": "pass"})

        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada"],
            registry=registry,
            book=book,
        )

        assert code == 1
        assert "no recorded eval result" in written
        assert book.pointing["current"] == "5"

    async def test_forcing_it_needs_a_reason(self, tmp_path: Path) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        book = Book(evaluated={"5": "pass"})

        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada", "--force"],
            registry=FilePromptRegistry(tmp_path),
            book=book,
        )

        assert code == 1
        assert "needs a --reason" in written

    async def test_a_forced_rollback_with_a_reason_goes_through(self, tmp_path: Path) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        book = Book(evaluated={"5": "pass"})

        code, _ = await run(
            [
                "rollback",
                "itinerary_system",
                "--to",
                "4",
                "--by",
                "ada",
                "--force",
                "--reason",
                "regression PLAT-91, eval backlog",
            ],
            registry=FilePromptRegistry(tmp_path),
            book=book,
        )

        assert code == 0
        assert book.pointing["current"] == "4"

    async def test_an_alias_nobody_declared_is_refused(self, registry: FilePromptRegistry) -> None:
        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada", "--alias", "canary"],
            registry=registry,
            book=Book(),
        )

        assert code == 1
        assert "canary" in written

    async def test_the_store_is_told_where_the_alias_was_so_it_can_compare_and_set(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        book = Book()
        book.pointing["current"] = "5"

        await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada"],
            registry=FilePromptRegistry(tmp_path),
            book=book,
        )

        assert book.moves[0]["from"] == "5"


class TestRunningItFromARunbook:
    """The same commands have to be readable by a script, not only by a person."""

    async def test_every_command_can_answer_in_json(self, registry: FilePromptRegistry) -> None:
        _, listed = await run(
            ["list", "itinerary_system", "--json"], registry=registry, book=Book()
        )
        _, diffed = await run(
            ["diff", "itinerary_system", "4", "5", "--json"], registry=registry, book=Book()
        )

        assert json.loads(listed)["versions"][0]["version"] == "4"
        assert json.loads(diffed)["variables_added"] == ["budget"]

    async def test_a_refusal_is_machine_readable_too(self, registry: FilePromptRegistry) -> None:
        code, written = await run(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada", "--json"],
            registry=registry,
            book=Book(),
        )

        assert code == 1
        assert "budget" in json.loads(written)["error"]

    async def test_a_command_line_this_cannot_read_is_not_a_refusal(
        self, registry: FilePromptRegistry
    ) -> None:
        code, _ = await run(["rollback", "itinerary_system"], registry=registry, book=Book())

        assert code == 2
