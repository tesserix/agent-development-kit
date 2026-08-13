"""Answering a three-day approval from a terminal.

The failures this file exists to prevent are an operator being shown an account number they
have no business with, and a command that says it approved something when it did not.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from tesserix_adk.cli import approvals_main
from tesserix_adk.core import ApprovalRecord, ApprovalTokenError, SuspendedRun, ToolCall, mint_token

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

NOW = 1_000.0
TENANT = "acme"
IBAN = "GB33BUKB20201555555555"


class Rota:
    """Whatever is waiting, in the tenant it is waiting in."""

    def __init__(self, *held: SuspendedRun) -> None:
        self._held = held

    async def pending(self, *, tenant: str) -> tuple[SuspendedRun, ...]:
        return tuple(one for one in self._held if one.tenant == tenant)


class Answered:
    """A resume that records what it was told, or refuses like a spent token would."""

    def __init__(self, *, refusing: bool = False) -> None:
        self.taken: list[tuple[str, bool, str, str]] = []
        self._refusing = refusing

    async def __call__(self, token: str, granted: bool, decided_by: str, reason: str) -> None:
        if self._refusing:
            raise ApprovalTokenError(
                "the decision for run_1 was already taken, so this token buys nothing",
                run_id="run_1",
                presented_by=decided_by,
            )
        self.taken.append((token, granted, decided_by, reason))


class TestSeeingWhatIsWaiting:
    async def test_it_names_the_run_the_tool_and_who_asked(self) -> None:
        written = io.StringIO()

        code = await _run(["list", "--tenant", TENANT], rota=Rota(_suspended()), out=written)

        assert code == 0
        assert "run_1" in written.getvalue()
        assert "wire_funds" in written.getvalue()
        assert "planner" in written.getvalue()

    async def test_it_shows_a_digest_rather_than_the_account_number(self) -> None:
        """A terminal is read over shoulders and scrolls into somebody's session log."""
        written = io.StringIO()

        await _run(["list", "--tenant", TENANT], rota=Rota(_suspended()), out=written)

        assert IBAN not in written.getvalue()

    async def test_it_says_why_a_person_is_being_asked(self) -> None:
        written = io.StringIO()

        await _run(["list", "--tenant", TENANT], rota=Rota(_suspended()), out=written)

        assert "declared to require approval" in written.getvalue()

    async def test_an_empty_rota_says_so_rather_than_printing_nothing(self) -> None:
        written = io.StringIO()

        code = await _run(["list", "--tenant", TENANT], rota=Rota(), out=written)

        assert code == 0
        assert written.getvalue() == f"nothing is waiting on {TENANT}\n"

    async def test_another_tenant_s_rota_is_not_this_one(self) -> None:
        written = io.StringIO()

        await _run(["list", "--tenant", "globex"], rota=Rota(_suspended()), out=written)

        assert "run_1" not in written.getvalue()


class TestAnsweringFromATerminal:
    async def test_approving_carries_the_token_and_who_decided(self) -> None:
        answering = Answered()

        code = await _run(["approve", "--token", "t-1", "--by", "ada"], answering=answering)

        assert code == 0
        assert answering.taken == [("t-1", True, "ada", "")]

    async def test_denying_says_it_denied(self) -> None:
        answering = Answered()
        written = io.StringIO()

        code = await _run(
            ["deny", "--token", "t-1", "--by", "ada", "--reason", "not this quarter"],
            answering=answering,
            out=written,
        )

        assert code == 0
        assert answering.taken == [("t-1", False, "ada", "not this quarter")]
        assert written.getvalue() == "denied by ada\n"

    async def test_a_token_that_buys_nothing_is_reported_rather_than_raised(self) -> None:
        written = io.StringIO()

        code = await _run(
            ["approve", "--token", "t-1", "--by", "mallory"],
            answering=Answered(refusing=True),
            out=written,
        )

        assert code == 1
        assert "already taken" in written.getvalue()


class TestACommandLineNobodyCouldRead:
    async def test_an_unknown_command_is_a_misuse_rather_than_a_crash(self) -> None:
        assert await _run(["shrug"]) == 2

    async def test_approving_without_saying_who_is_a_misuse(self) -> None:
        assert await _run(["approve", "--token", "t-1"]) == 2

    async def test_listing_without_a_tenant_is_a_misuse(self) -> None:
        assert await _run(["list"]) == 2


class TestWritingSomewhere:
    async def test_it_writes_to_stdout_when_nowhere_else_is_named(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        await approvals_main(["list", "--tenant", TENANT], waiting=Rota(), answering=Answered())

        assert "nothing is waiting" in capsys.readouterr().out


async def _run(
    argv: Sequence[str],
    *,
    rota: Rota | None = None,
    answering: Answered | None = None,
    out: io.StringIO | None = None,
) -> int:
    return await approvals_main(
        argv,
        waiting=rota if rota is not None else Rota(),
        answering=answering if answering is not None else Answered(),
        out=out if out is not None else io.StringIO(),
    )


def _suspended() -> SuspendedRun:
    record = ApprovalRecord.for_call(
        run_id="run_1",
        tenant=TENANT,
        agent_name="planner",
        tool_name="wire_funds",
        arguments={"amount": 500, "iban": IBAN},
        reason="wire_funds is declared to require approval",
        requested_at=NOW,
    )
    return SuspendedRun(
        run_id="run_1",
        tenant=TENANT,
        agent_name="planner",
        record=record,
        call=ToolCall(id="c1", name="wire_funds", arguments={"amount": 500}),
        token_digest=mint_token(record).digest,
        suspended_at=NOW,
        expires_at=NOW + 259_200.0,
    )
