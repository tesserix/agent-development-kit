"""The table an operator writes to answer a task class.

`core.routing` names the vocabulary; this is the one implementation of it that reads a
table. Rules are matched most-specific-first — a rule naming a tenant and an agent beats
one naming a tenant, which beats the general rule — and within a rule the candidates are
tried in the order they were written, because that order is the operator's preference and
reordering it silently is choosing on their behalf.

Nothing here falls back. A class with no rule, a rule whose candidates cannot do the work,
and a pin the table does not know are all `NoEligibleModelError`: a downgrade nobody asked
for is a bill nobody can explain and, worse, an answer produced by a model the run record
does not name.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

from pydantic import model_validator

from tesserix_adk.core.capabilities import ModelRef, ModelSpec
from tesserix_adk.core.errors import ConfigurationError, NoEligibleModelError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.routing import (
    ModelRequirements,
    RejectedCandidate,
    RoutingDecision,
    TaskClass,
)
from tesserix_adk.models.catalogue import known_models, model_card

__all__ = [
    "ROUTING_TABLE_ENV",
    "TABLE_VERSION",
    "RoutingRule",
    "RoutingTable",
    "TableRouter",
    "routing_table",
]

TABLE_VERSION = 1
ROUTING_TABLE_ENV = "ADK_ROUTING_TABLE"
_LACKS = "lacks "


class RoutingRule(AdkModel):
    """One task class, optionally narrowed to who is asking, and what may answer it.

    Args:
        task_class: The kind of work this rule answers.
        candidates: What may answer it, in preference order.
        tenant: Narrows the rule to one tenant, where a deployment routes them apart.
        agent: Narrows the rule to one agent, for escalating a single step.
    """

    task_class: TaskClass
    candidates: tuple[ModelSpec, ...] = ()
    tenant: str = ""
    agent: str = ""

    @property
    def scope(self) -> str:
        """This rule's identity in a decision record, as `class@tenant/agent`."""
        who = "/".join(part for part in (self.tenant, self.agent) if part)
        return f"{self.task_class}@{who}" if who else str(self.task_class)

    @property
    def specificity(self) -> int:
        """How many things the rule names, so the narrower rule is tried first."""
        return bool(self.tenant) + bool(self.agent)

    def answers(self, task_class: TaskClass, tenant: str | None, agent: str | None) -> bool:
        """Whether this rule covers work of `task_class` asked by `tenant` and `agent`."""
        return (
            self.task_class == task_class
            and self.tenant in ("", tenant)
            and self.agent in ("", agent)
        )


class RoutingTable(AdkModel):
    """Every rule a deployment routes by, validated when it is built.

    Built at boot from configuration, so a table naming an impossible requirement or two
    rules at one scope fails there rather than on the first request that needed either.

    Args:
        version: The schema the table was written against.
        rules: The rules, in any order — matching sorts them by specificity.
    """

    version: int = TABLE_VERSION
    rules: tuple[RoutingRule, ...] = ()

    @model_validator(mode="after")
    def _refuse_a_table_that_would_fail_later(self) -> Self:
        if self.version != TABLE_VERSION:
            raise ConfigurationError(
                f"routing table version {self.version} is not one this kit reads "
                f"(expected {TABLE_VERSION})"
            )
        if not self.rules:
            raise ConfigurationError(
                "the routing table has no rules, and a table that routes nothing would "
                "refuse every task class on the first request instead of at boot"
            )
        self._refuse_duplicate_scopes()
        self._refuse_a_rule_nothing_can_answer()
        return self

    def _refuse_duplicate_scopes(self) -> None:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.scope in seen:
                raise ConfigurationError(
                    f"{rule.scope} is routed twice, so which candidates answer it depends "
                    f"on rule order rather than on anything an operator wrote"
                )
            seen.add(rule.scope)

    def _refuse_a_rule_nothing_can_answer(self) -> None:
        for rule in self.rules:
            if not rule.candidates:
                raise ConfigurationError(f"{rule.scope} has no candidates")
            for candidate in rule.candidates:
                if not candidate.capabilities.declared:
                    raise ConfigurationError(
                        f"{rule.scope} offers {candidate.ref}, which declares no "
                        f"capabilities and so satisfies no requirement anyone states"
                    )
                _refuse_a_model_the_vendor_stopped_serving(rule, candidate)

    def matching(
        self, task_class: TaskClass, tenant: str | None, agent: str | None
    ) -> RoutingRule | None:
        """The narrowest rule covering this work, or nothing where none does."""
        covering = [rule for rule in self.rules if rule.answers(task_class, tenant, agent)]
        return max(covering, key=lambda rule: rule.specificity, default=None)


def _refuse_a_model_the_vendor_stopped_serving(rule: RoutingRule, candidate: ModelSpec) -> None:
    """Catch a decommissioned id at boot, for the vendors the catalogue covers.

    A provider the catalogue says nothing about — a self-hosted endpoint, a proxy — is
    left alone: absence there is absence of a card, not evidence the model is gone.
    """
    if not known_models(candidate.provider):
        return
    try:
        model_card(candidate.ref)
    except ConfigurationError as unknown:
        raise ConfigurationError(
            f"{rule.scope} offers {candidate.ref}, which is not in the model catalogue. "
            f"A vendor that retired an id answers every request for it with an error"
        ) from unknown


class TableRouter:
    """A `ModelRouter` that reads a `RoutingTable`.

    A reload builds a new router with `reloaded`, so a run holding this one keeps the
    table it started against rather than changing model halfway through.

    Args:
        table: The rules to route by.
        entitlements: Model refs each tenant may be routed to, where a deployment
            restricts them. A tenant with no entry is unrestricted; a tenant with an
            empty entry is entitled to nothing, which is a thing an operator can mean.
    """

    def __init__(
        self, table: RoutingTable, *, entitlements: Mapping[str, frozenset[str]] | None = None
    ) -> None:
        self._table = table
        self._entitlements = dict(entitlements or {})

    @property
    def table(self) -> RoutingTable:
        """The table this router reads."""
        return self._table

    def reloaded(self, table: RoutingTable) -> TableRouter:
        """Return a router reading `table`, leaving this one as it was."""
        return type(self)(table, entitlements=self._entitlements)

    def _entitled(self, candidate: ModelSpec, tenant: str | None) -> bool:
        allowed = self._entitlements.get(tenant) if tenant is not None else None
        return allowed is None or str(candidate.ref) in allowed

    def resolve(
        self,
        task_class: TaskClass,
        *,
        requirements: ModelRequirements | None = None,
        tenant: str | None = None,
        agent: str | None = None,
        pinned: ModelRef | None = None,
    ) -> RoutingDecision:
        """Choose the model for this piece of work.

        Raises:
            NoEligibleModelError: If the class is not routed, if nothing the rule offers
                meets the requirements, or if a pinned model is unknown or ineligible.
        """
        needed = requirements or ModelRequirements()
        rule = self._table.matching(task_class, tenant, agent)
        if rule is None:
            raise NoEligibleModelError(
                f"nothing routes {task_class}, and falling through to another class's "
                f"models would bill this work as work nobody asked for",
                task_class=task_class,
                tenant=tenant,
            )
        if pinned is not None:
            return self._pin(task_class, rule, needed, pinned, tenant)
        return self._first_eligible(task_class, rule, needed, tenant)

    def _first_eligible(
        self,
        task_class: TaskClass,
        rule: RoutingRule,
        needed: ModelRequirements,
        tenant: str | None,
    ) -> RoutingDecision:
        rejected: list[RejectedCandidate] = []
        eligible: list[ModelSpec] = []
        for candidate in rule.candidates:
            reason = self._ineligible(candidate, needed, tenant)
            if reason:
                rejected.append(RejectedCandidate(ref=str(candidate.ref), reason=reason))
            else:
                eligible.append(candidate)
        if not eligible:
            raise _nothing_eligible(task_class, rule, needed, rejected, tenant)
        chosen, *alternatives = eligible
        legal = [spec for spec in alternatives if chosen.trust.admits(spec.trust)]
        outside = [spec for spec in alternatives if spec not in legal]
        rejected.extend(
            RejectedCandidate(ref=str(spec.ref), reason=_outside(chosen, spec)) for spec in outside
        )
        return RoutingDecision(
            task_class=task_class,
            chosen=chosen.ref,
            considered=tuple(str(spec.ref) for spec in rule.candidates),
            chain=(str(chosen.ref), *(str(spec.ref) for spec in legal)),
            rejected=tuple(rejected),
            excluded_by_boundary=tuple(str(spec.ref) for spec in outside),
            required=needed.named,
            min_context_window_tokens=needed.min_context_window_tokens or 0,
            boundary=chosen.trust,
            rule=rule.scope,
        )

    def _ineligible(
        self, candidate: ModelSpec, needed: ModelRequirements, tenant: str | None
    ) -> str:
        """Why this candidate cannot answer, or an empty string where it can."""
        if not self._entitled(candidate, tenant):
            return f"not entitled: {tenant} may not be routed to it"
        unsatisfied = needed.unsatisfied_by(candidate.capabilities)
        return f"lacks {', '.join(unsatisfied)}" if unsatisfied else ""

    def _pin(
        self,
        task_class: TaskClass,
        rule: RoutingRule,
        needed: ModelRequirements,
        pinned: ModelRef,
        tenant: str | None,
    ) -> RoutingDecision:
        candidate = self._known(pinned)
        if candidate is None:
            raise NoEligibleModelError(
                f"{pinned} is not in the routing table and not in the model catalogue, "
                f"so nothing says what it can do. A pin is a choice between known models "
                f"rather than a way past the checks",
                task_class=task_class,
                tenant=tenant,
            )
        reason = self._ineligible(candidate, needed, tenant)
        if reason:
            raise _nothing_eligible(
                task_class,
                rule,
                needed,
                [RejectedCandidate(ref=str(candidate.ref), reason=reason)],
                tenant,
            )
        return RoutingDecision(
            task_class=task_class,
            chosen=candidate.ref,
            considered=(str(candidate.ref),),
            chain=(str(candidate.ref),),
            required=needed.named,
            min_context_window_tokens=needed.min_context_window_tokens or 0,
            boundary=candidate.trust,
            rule=rule.scope,
            pinned=True,
        )

    def _known(self, pinned: ModelRef) -> ModelSpec | None:
        """The pinned model's capability record, from the table or from the catalogue.

        The table wins: a deployment that narrowed a model's record — a self-hosted
        endpoint serving smaller weights — knows something the published card does not.
        """
        configured = {str(spec.ref): spec for rule in self._table.rules for spec in rule.candidates}
        if str(pinned) in configured:
            return configured[str(pinned)]
        try:
            card = model_card(pinned)
        except ConfigurationError:
            return None
        return ModelSpec(
            provider=pinned.provider, model=pinned.model, capabilities=card.capabilities
        )


def _nothing_eligible(
    task_class: TaskClass,
    rule: RoutingRule,
    needed: ModelRequirements,
    rejected: list[RejectedCandidate],
    tenant: str | None,
) -> NoEligibleModelError:
    unsatisfied = sorted({name for candidate in rejected for name in _named(candidate)})
    return NoEligibleModelError(
        f"nothing routed for {rule.scope} can do the work: "
        f"{'; '.join(f'{c.ref} {c.reason}' for c in rejected)}. Requirements: {needed!r}",
        task_class=task_class,
        unsatisfied=unsatisfied,
        rejected=[(candidate.ref, candidate.reason) for candidate in rejected],
        tenant=tenant,
    )


def _named(candidate: RejectedCandidate) -> tuple[str, ...]:
    """The requirement names inside a rejection, where the rejection was about one."""
    if not candidate.reason.startswith(_LACKS):
        return ()
    return tuple(part.strip() for part in candidate.reason.removeprefix(_LACKS).split(","))


def routing_table(
    path: Path | str | None = None, *, env: Mapping[str, str] | None = None
) -> RoutingTable:
    """Read a routing table from TOML, validated as it is read.

    Precedence: the path argument, then `ADK_ROUTING_TABLE` in the environment. Nothing
    is discovered by convention — a deployment routing by a file nobody named is one
    where the answer to "which models is this billing" is on somebody's laptop.

    Args:
        path: The file to read.
        env: The environment to read `ADK_ROUTING_TABLE` from. Defaults to the process's.

    Returns:
        The table, already validated: an unroutable file fails here rather than on the
        first request that needed the rule it got wrong.

    Raises:
        ConfigurationError: If no file was named, the named file is absent or is not
            readable TOML, or the table it holds would fail later.
    """
    environ = os.environ if env is None else env
    named = path if path is not None else environ.get(ROUTING_TABLE_ENV)
    if named is None:
        raise ConfigurationError(
            f"no routing table was named: pass a path or set {ROUTING_TABLE_ENV}"
        )
    try:
        return RoutingTable.model_validate(_read(Path(named)), strict=False)
    except ValidationError as invalid:
        raise ConfigurationError(
            f"the routing table {named} is not a table: {invalid}"
        ) from invalid


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"the routing table {path} is not there")
    try:
        with path.open("rb") as handle:
            loaded: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as broken:
        raise ConfigurationError(
            f"the routing table {path} is not readable TOML: {broken}"
        ) from broken
    return loaded


def _outside(chosen: ModelSpec, candidate: ModelSpec) -> str:
    """Why a capable candidate is still not a legal fallback."""
    axes = ", ".join(chosen.trust.differs_from(candidate.trust))
    return f"outside the trust boundary: {axes}"
