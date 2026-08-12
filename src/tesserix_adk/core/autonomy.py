"""How much an agent may do unattended, declared as a grant the runtime enforces.

Asking a human before every action makes an agent useless and letting it act freely makes
it dangerous, so products settle on a number in a config file — with no expiry, no audit
and nobody's name against it. Here autonomy is a grant: issued by someone, for one class
of action, up to a ceiling, until a moment. Anything an issued grant does not cover
escalates to a human, including everything nobody thought about.

Grant issuance lives behind `GrantIssuer`, which the runtime is never given. The ladder
holds a `GrantReader` and can only read, so no agent, tool or model output has a path to
widen what it may do.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "RESERVED_ACTION_CLASS",
    "ActionClass",
    "ActionRegistry",
    "ActionRequest",
    "AutonomyDecision",
    "AutonomyGrant",
    "AutonomyLadder",
    "AutonomyLevel",
    "AutonomyOutcome",
    "Ceiling",
    "CommitmentLedger",
    "GrantIssuer",
    "GrantReader",
    "InMemoryGrants",
]

RESERVED_ACTION_CLASS = "autonomy.grant"
"""Issuing autonomy is itself an action class, and it is the one the ladder always refuses."""


class AutonomyLevel(StrEnum):
    """How far a grant lets an agent go before a human is involved.

    The default everywhere is `ASK_ALWAYS`, and it is what an unmatched action falls to.
    """

    ASK_ALWAYS = "ask_always"
    ACT_WITHIN_LIMITS = "act_within_limits"
    ACT_AND_REPORT = "act_and_report"


class AutonomyOutcome(StrEnum):
    """What the ladder decided about one attempted action."""

    ACT = "act"
    ESCALATE = "escalate"
    REFUSE = "refuse"


class ActionClass(AdkModel):
    """A named class of side effect, and where in a call's arguments its money is.

    Classes rather than tools, because a grant is about what an action does to the world:
    three refund tools are one thing to the person deciding how much may be refunded.

    Args:
        name: What the class is called, `payment.refund` style.
        irreversible: Whether an action of this class can be taken back. An irreversible
            class may never be granted without a ceiling.
        amount_field: Which validated argument carries the amount, where one does.
        currency_field: Which carries its currency.
    """

    name: str = Field(min_length=1)
    irreversible: bool = False
    amount_field: str | None = None
    currency_field: str | None = None

    @property
    def priced(self) -> bool:
        """Whether an action of this class carries an amount to check a ceiling against."""
        return self.amount_field is not None


class ActionRegistry:
    """Which action class each tool belongs to.

    A tool that is in no class is unknown, and unknown asks a human — including a tool
    added after the grants were issued, which is the whole reason the default is that way
    round.
    """

    def __init__(self, tools: Mapping[str, ActionClass]) -> None:
        self._tools = dict(tools)

    @property
    def tools(self) -> Mapping[str, ActionClass]:
        """What is registered, for a consumer building a wider registry from this one."""
        return dict(self._tools)

    def of(self, tool: str) -> ActionClass | None:
        """The class `tool` belongs to, or nothing where it belongs to none."""
        return self._tools.get(tool)

    def named(self, action_class: str) -> ActionClass | None:
        """The class called `action_class`, where any tool is registered under it."""
        return next((known for known in self._tools.values() if known.name == action_class), None)


class Ceiling(AdkModel):
    """How much a grant covers, in what, over what window.

    Args:
        amount: The most that may be committed in the window. Decimal, because a ceiling
            held as a float is a ceiling that is occasionally a hundredth wrong.
        currency: ISO 4217. A grant in one currency does not cover an action in another.
        window_seconds: The rolling window the amount applies over.
    """

    amount: Decimal = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    window_seconds: float = Field(gt=0)


class AutonomyGrant(AdkModel):
    """Permission to act unattended, issued by somebody, expiring by itself.

    Args:
        id: Identity of this grant. A revoked grant is never reactivated; re-granting
            mints a new id.
        tenant: Who it was issued to.
        action_class: What it covers.
        level: How far it goes.
        granted_by: The human or system identity that issued it. Recorded on every
            decision it answers.
        issued_at: Unix seconds. Where two grants cover one action, the later one decides.
        expires_at: Unix seconds. There is no non-expiring grant.
        user: The single principal it covers, where it covers one rather than the tenant.
        ceiling: How much. Required for `ACT_WITHIN_LIMITS`.
        includes_subtenants: Whether `acme` also covers `acme/eu`. Off by default: a
            grant that silently widened as the tenant tree grew would be a grant nobody
            issued.
    """

    id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    action_class: str = Field(min_length=1)
    level: AutonomyLevel
    granted_by: str = Field(min_length=1)
    issued_at: float
    expires_at: float
    user: str | None = None
    ceiling: Ceiling | None = None
    includes_subtenants: bool = False

    @model_validator(mode="after")
    def _is_issuable(self) -> Self:
        """Refuse a grant that could not be enforced as written."""
        if self.expires_at <= self.issued_at:
            raise ValueError("a grant expires after it is issued, and every grant expires")
        if self.level is AutonomyLevel.ACT_WITHIN_LIMITS and self.ceiling is None:
            raise ValueError("act_within_limits without a ceiling is not a limit")
        return self

    def covers(self, *, tenant: str, user: str | None, now: float) -> bool:
        """Whether this grant is live and speaks for `tenant` and `user`."""
        if now >= self.expires_at:
            return False
        if self.user is not None and self.user != user:
            return False
        if tenant == self.tenant:
            return True
        return self.includes_subtenants and tenant.startswith(f"{self.tenant}/")


class ActionRequest(AdkModel):
    """One action an agent wants to take, and what the runtime knows about it.

    Args:
        tool: What it wants to call.
        tenant: On whose behalf.
        arguments: The validated arguments, which is where an amount is read from.
        user: The acting principal, where there is one.
        committed: What this tenant has already committed against this class in the
            window. Left unset, the ladder asks its `CommitmentLedger`, and a ladder with
            no ledger sees nothing committed.
        reports_outstanding: Whether an `act_and_report` report is owed and undelivered.
    """

    tool: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    user: str | None = None
    committed: Decimal | None = None
    reports_outstanding: bool = False

    @field_validator("committed")
    @classmethod
    def _is_an_amount(cls, committed: Decimal | None) -> Decimal | None:
        if committed is not None and committed < 0:
            raise ValueError("a negative commitment is headroom nobody granted")
        return committed


class AutonomyDecision(AdkModel):
    """What the ladder decided, and what has to be recorded about it.

    Args:
        outcome: Act unattended, escalate to a human, or refuse outright.
        action_class: The class the tool resolved to, or `unknown`.
        level: The level that applied. `ASK_ALWAYS` where nothing matched.
        reason: Why, in a form an audit reader can act on.
        grant_id: Which grant answered, where one did. Recorded on escalations too: the
            question of which grant was not enough is the one an operator asks.
        headroom: What was left under the ceiling, where there was a ceiling.
        reports: Whether acting obliges a report before the next action of this class.
    """

    outcome: AutonomyOutcome
    action_class: str = Field(min_length=1)
    level: AutonomyLevel
    reason: str = Field(min_length=1)
    grant_id: str | None = None
    headroom: Decimal | None = None
    reports: bool = False

    @property
    def unattended(self) -> bool:
        """Whether the runtime may go ahead without asking anybody."""
        return self.outcome is AutonomyOutcome.ACT


@runtime_checkable
class GrantReader(Protocol):
    """What the runtime is allowed to know about grants: what is live, and nothing more."""

    async def grants_for(self, *, tenant: str, action_class: str) -> Sequence[AutonomyGrant]:
        """Every unexpired grant that could speak for `tenant` about `action_class`."""
        ...


@runtime_checkable
class CommitmentLedger(Protocol):
    """What a tenant has already committed against an action class in a window."""

    async def committed(self, *, tenant: str, action_class: str, window_seconds: float) -> Decimal:
        """The total committed in the last `window_seconds`, exactly."""
        ...


@runtime_checkable
class GrantIssuer(Protocol):
    """Issuing and withdrawing autonomy, which only a human or system identity does.

    Deliberately a second protocol. The ladder holds a `GrantReader`, so there is no
    object inside a run that could issue a grant even if a model asked it to.
    """

    async def issue(self, grant: AutonomyGrant) -> AutonomyGrant:
        """Record `grant`, and return it as stored."""
        ...


class InMemoryGrants:
    """Grants in a dict, for tests and single-process deployments."""

    def __init__(self, grants: Iterable[AutonomyGrant] = ()) -> None:
        self._grants: dict[str, AutonomyGrant] = {held.id: held for held in grants}

    async def grants_for(self, *, tenant: str, action_class: str) -> Sequence[AutonomyGrant]:
        """Every grant for `action_class` that names `tenant` or a tenant above it."""
        return [
            held
            for held in self._grants.values()
            if held.action_class == action_class
            and (held.tenant == tenant or tenant.startswith(f"{held.tenant}/"))
        ]

    async def issue(self, grant: AutonomyGrant) -> AutonomyGrant:
        """Record `grant`, refusing an id that already exists.

        Raises:
            ConfigurationError: If the id is in use. A grant is never rewritten, because a
                decision recorded against an id must stay readable as what it permitted.
        """
        if grant.id in self._grants:
            raise ConfigurationError(
                f"grant {grant.id!r} already exists; re-granting mints a new id so that a "
                f"decision recorded against an id stays readable as what it permitted"
            )
        self._grants[grant.id] = grant
        return grant

    def all_grants(self) -> Sequence[AutonomyGrant]:
        """Everything issued, for a startup check over the whole set."""
        return list(self._grants.values())


class AutonomyLadder:
    """Resolves one attempted action against the grants that cover it.

    Reading only. There is no path from here to issuing a grant, which is what makes
    "a model cannot widen its own level" structural rather than a rule someone enforces.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        *,
        grants: GrantReader,
        clock: Clock,
        commitments: CommitmentLedger | None = None,
        reserved: str = RESERVED_ACTION_CLASS,
    ) -> None:
        self._registry = registry
        self._grants = grants
        self._clock = clock
        self._commitments = commitments
        self._reserved = reserved

    def classify(self, tool: str) -> str | None:
        """The action class `tool` belongs to, or nothing where it belongs to none."""
        known = self._registry.of(tool)
        return known.name if known else None

    async def decide(self, request: ActionRequest) -> AutonomyDecision:
        """What may happen about `request`: act, escalate to a human, or refuse."""
        known = self._registry.of(request.tool)
        if known is None:
            return self._ask("unknown", f"{request.tool} is in an unregistered action class")
        if known.name == self._reserved:
            return AutonomyDecision(
                outcome=AutonomyOutcome.REFUSE,
                action_class=known.name,
                level=AutonomyLevel.ASK_ALWAYS,
                reason=f"{request.tool} would issue autonomy; attempted escalation",
            )
        held = self._answering(
            await self._grants.grants_for(tenant=request.tenant, action_class=known.name), request
        )
        if held is None:
            return self._ask(known.name, f"no live grant covers {known.name} for {request.tenant}")
        return await self._under(held, known, request)

    def validate_grants(self) -> None:
        """Refuse a set of grants that is unenforceable as issued.

        Called at startup where the store can enumerate. A `GrantReader` that cannot is
        checked one grant at a time as it answers.

        Raises:
            ConfigurationError: If an irreversible action class is granted without a
                ceiling — an uncapped refund is a grant nobody could have meant to issue.
        """
        listing = getattr(self._grants, "all_grants", None)
        for held in listing() if callable(listing) else ():
            self._is_enforceable(held)

    def _is_enforceable(self, held: AutonomyGrant) -> None:
        """Refuse one grant that could not be enforced against its class."""
        known = self._registry.named(held.action_class)
        if known is not None and known.irreversible and held.ceiling is None:
            raise ConfigurationError(
                f"grant {held.id!r} leaves the irreversible class {held.action_class!r} "
                f"uncapped; an irreversible action is granted with a ceiling or not at all"
            )

    def _answering(
        self, grants: Sequence[AutonomyGrant], request: ActionRequest
    ) -> AutonomyGrant | None:
        """The grant that decides, which is the most recently issued one that covers this."""
        now = self._clock.now()
        covering = [
            held
            for held in grants
            if held.covers(tenant=request.tenant, user=request.user, now=now)
        ]
        return max(covering, key=lambda held: (held.issued_at, held.id), default=None)

    async def _under(
        self, held: AutonomyGrant, known: ActionClass, request: ActionRequest
    ) -> AutonomyDecision:
        """What `held` permits about this request."""
        self._is_enforceable(held)
        if held.level is AutonomyLevel.ASK_ALWAYS:
            return self._ask(known.name, "the grant is ask_always", held=held)
        reports = held.level is AutonomyLevel.ACT_AND_REPORT
        if reports and request.reports_outstanding:
            return self._ask(known.name, "a report for an earlier action is undelivered", held=held)
        if held.ceiling is None:
            return self._act(known.name, held, "within an uncapped grant", reports=reports)
        return await self._within(held.ceiling, held, known, request, reports=reports)

    async def _committed(
        self, ceiling: Ceiling, request: ActionRequest, known: ActionClass
    ) -> Decimal:
        """What is already committed: what the caller said, or what the ledger says."""
        if request.committed is not None:
            return request.committed
        if self._commitments is None:
            return Decimal(0)
        return await self._commitments.committed(
            tenant=request.tenant,
            action_class=known.name,
            window_seconds=ceiling.window_seconds,
        )

    async def _within(
        self,
        ceiling: Ceiling,
        held: AutonomyGrant,
        known: ActionClass,
        request: ActionRequest,
        *,
        reports: bool,
    ) -> AutonomyDecision:
        """Whether the amount this call carries fits under what is left of the ceiling."""
        headroom = ceiling.amount - await self._committed(ceiling, request, known)
        amount = _amount_in(request.arguments, known.amount_field)
        if amount is None:
            return self._ask(
                known.name,
                f"no readable amount in {known.amount_field!r}, and a ceiling cannot be "
                f"checked against an amount nobody can read",
                held=held,
                headroom=headroom,
            )
        if _currency_in(request.arguments, known.currency_field) != ceiling.currency:
            return self._ask(
                known.name,
                f"the grant is in {ceiling.currency} and this action is not; a currency "
                f"mismatch is a question for a human, never an implicit conversion",
                held=held,
                headroom=headroom,
            )
        if amount > headroom:
            return self._ask(
                known.name,
                f"{amount} is over the {headroom} left under the grant's ceiling",
                held=held,
                headroom=headroom,
            )
        return self._act(
            known.name, held, f"{amount} of {headroom}", reports=reports, headroom=headroom
        )

    def _act(
        self,
        action_class: str,
        held: AutonomyGrant,
        reason: str,
        *,
        reports: bool,
        headroom: Decimal | None = None,
    ) -> AutonomyDecision:
        """A decision to go ahead unattended, naming the grant that permitted it."""
        return AutonomyDecision(
            outcome=AutonomyOutcome.ACT,
            action_class=action_class,
            level=held.level,
            reason=reason,
            grant_id=held.id,
            headroom=headroom,
            reports=reports,
        )

    def _ask(
        self,
        action_class: str,
        reason: str,
        *,
        held: AutonomyGrant | None = None,
        headroom: Decimal | None = None,
    ) -> AutonomyDecision:
        """A decision to put the action in front of a human, and why."""
        return AutonomyDecision(
            outcome=AutonomyOutcome.ESCALATE,
            action_class=action_class,
            level=AutonomyLevel.ASK_ALWAYS,
            reason=reason,
            grant_id=held.id if held else None,
            headroom=headroom,
        )


def _amount_in(arguments: Mapping[str, Any], field: str | None) -> Decimal | None:
    """The amount `field` carries, exactly, or nothing where it carries no number."""
    if field is None:
        return None
    value = arguments.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float | str | Decimal):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _currency_in(arguments: Mapping[str, Any], field: str | None) -> str | None:
    """The currency `field` carries, where it carries a string."""
    value = arguments.get(field) if field is not None else None
    return value if isinstance(value, str) else None
