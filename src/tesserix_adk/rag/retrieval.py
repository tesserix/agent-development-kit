"""Finding the passages that answer a question, by meaning and by the words themselves.

Semantic search misses the booking reference and the policy code, because an identifier has
no meaning to embed. Keyword search misses the paraphrase, because the answer shares no
words with the question. Products pick one, find the gap in production, and bolt the other
on in application code with no shared scoring and no way to fake it in a test.

So there is one `Retriever` protocol and a `HybridRetriever` that runs both branches and
fuses them, and every hit says which branch found it and where it ranked. The tenant
predicate is put into the store's own query rather than applied to what comes back: a
filter in Python is a filter that runs after the rows have already left the tenant they
belong to.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence  # noqa: TC003 — pydantic needs the runtime types
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core import (
    AdkModel,
    ConfigurationError,
    RetrievalDegradedError,
    SchemaViolationError,
    current_tenant,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from tesserix_adk.rag.embedding import Embedder, Vector

__all__ = [
    "Branch",
    "BranchScore",
    "Fusion",
    "Hit",
    "HybridRetriever",
    "IndexQuery",
    "IndexRetriever",
    "ReciprocalRankFusion",
    "RetrievalResult",
    "RetrievalScope",
    "Retriever",
    "SearchIndex",
    "WeightedSum",
]

TENANT_KEY = "tenant"
"""The predicate no caller may supply: it is the store's, and it is not negotiable."""


class Branch(StrEnum):
    """Which way a passage was found."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class BranchScore(AdkModel):
    """One branch's opinion of one passage.

    Args:
        branch: Which branch scored it.
        score: What that branch scored it, in that branch's own units. Not comparable
            across branches, which is why fusion ranks rather than adds them by default.
        rank: Where it came in that branch, counting from 1.
    """

    branch: Branch
    score: float
    rank: int = Field(gt=0)


class Hit(AdkModel):
    """One retrieved passage, and the case for it.

    Args:
        chunk_id: The chunk, unique within the collection.
        document_id: What it was chunked from.
        text: The passage itself.
        score: The fused score, in the fusion's units.
        metadata: What the store holds beside the text, for filtering and for display.
        contributions: One entry per branch that found it. A hit with one entry was found
            one way only, which is worth knowing before trusting its rank.
    """

    chunk_id: str = Field(min_length=1)
    document_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: Mapping[str, str] = Field(default_factory=dict)
    contributions: tuple[BranchScore, ...] = ()

    def found_by(self, branch: Branch) -> bool:
        """Whether `branch` contributed to this hit."""
        return any(contribution.branch is branch for contribution in self.contributions)


class RetrievalScope(AdkModel):
    """Where to look. The tenant is not here: it comes from the context, not the caller.

    Args:
        collection: The collection to search.
    """

    collection: str = Field(min_length=1)


class IndexQuery(AdkModel):
    """What a store is asked, with the tenant already in it.

    Args:
        tenant: Whose data may be searched. Set from the tenant context by the retriever,
            never from caller input, and stores must apply it inside their own query.
        collection: Which collection.
        branch: Which branch this query serves, so a store can answer both.
        k: The most hits to return.
        text: The query text, for a keyword branch.
        vector: The query embedding, for a semantic branch.
        predicates: Equality predicates over metadata, each satisfied by any of its
            values. Pushed into the store's query, not applied to its answer.
    """

    tenant: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    branch: Branch
    k: int = Field(gt=0)
    text: str = ""
    vector: tuple[float, ...] | None = None
    predicates: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_what_its_branch_needs(self) -> Self:
        """A semantic query with no vector would silently degrade to a scan of the table."""
        if self.branch is Branch.SEMANTIC and self.vector is None:
            raise ValueError("a semantic query needs a vector")
        if self.branch is Branch.KEYWORD and not self.text:
            raise ValueError("a keyword query needs text")
        return self


class Admissible(AdkModel):
    """A retrieval request that has already been checked, shared by every branch.

    Settled once, before any branch runs: a caller's mistake must read as a refusal from
    the retriever, not as a branch that happened to fail and left a degraded result.

    Args:
        tenant: Whose data may be searched, taken from the tenant context.
        collection: Which collection.
        predicates: The caller's filters, in the form a store pushes down.
        k: The most hits to return.
    """

    tenant: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    predicates: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    k: int = Field(gt=0)


class RetrievalResult(AdkModel):
    """What one retrieval found, and whether it is the whole of what there was.

    Args:
        query: The question as asked.
        hits: Best first.
        branches: The branches that answered.
        partial: Whether a branch was asked and did not answer. A caller that reads a
            narrower result set as a complete answer is the reason this is not silent.
    """

    query: str
    hits: tuple[Hit, ...] = ()
    branches: tuple[Branch, ...] = ()
    partial: bool = False

    def __bool__(self) -> bool:
        """Whether anything was found, so an empty result reads as one at a call site."""
        return bool(self.hits)


@runtime_checkable
class SearchIndex(Protocol):
    """A store that can answer one branch's query, with the filters inside the query."""

    @property
    def name(self) -> str:
        """What this store is, for errors and metrics."""
        ...

    def supports(self, branch: Branch) -> bool:
        """Whether it can answer `branch` at all."""
        ...

    async def search(self, query: IndexQuery) -> Sequence[Hit]:
        """Answer `query`, applying its tenant and predicates within the store's own query.

        Returns at most `query.k` hits, best first, each scored in this branch's units.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """The one shape an agent retrieves through, whatever is behind it."""

    async def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        filters: Mapping[str, Sequence[str]] | None = None,
        k: int = 10,
    ) -> RetrievalResult:
        """Find the passages that bear on `query`.

        Raises:
            MissingTenantContextError: Where no tenant is in force.
            SchemaViolationError: Where a filter tries to name the tenant.
            RetrievalDegradedError: Where a required branch did not answer.
        """
        ...


class Fusion(Protocol):
    """How two branches' rankings become one."""

    def fuse(self, ranked: Mapping[Branch, Sequence[Hit]], k: int) -> Sequence[Hit]:
        """Combine `ranked`, keeping at most `k`, best first."""
        ...


class ReciprocalRankFusion:
    """Fuses on rank alone: 1 / (k0 + rank), summed across the branches that found it.

    The default, because branch scores are not comparable. A cosine similarity of 0.82 and
    a BM25 score of 11.4 have no exchange rate, and inventing one by normalising a batch
    makes every score depend on what else happened to be in it.

    Args:
        k0: The rank offset. Larger flattens the curve, so a first place counts for less
            over a second; 60 is the value the literature settled on.
    """

    def __init__(self, k0: int = 60) -> None:
        if k0 < 1:
            raise ConfigurationError("a rank offset below 1 divides by zero at rank 1")
        self._k0 = k0

    def fuse(self, ranked: Mapping[Branch, Sequence[Hit]], k: int) -> Sequence[Hit]:
        """Score each hit by the ranks it earned, and keep the best `k`."""
        return _best(ranked, k, lambda contribution: 1 / (self._k0 + contribution.rank))


class WeightedSum:
    """Fuses on the branch scores themselves, weighted.

    For a deployment whose two branches are on a comparable scale, or which has calibrated
    them. It ranks by magnitude rather than by position, so one branch with a wide score
    range decides the order; that is the trade, and it is why it is not the default.

    Args:
        weights: Per branch. A branch with no weight contributes nothing.
    """

    def __init__(self, weights: Mapping[Branch, float]) -> None:
        if not weights or any(weight < 0 for weight in weights.values()):
            raise ConfigurationError("weights must be given and may not be negative")
        self._weights = dict(weights)

    def fuse(self, ranked: Mapping[Branch, Sequence[Hit]], k: int) -> Sequence[Hit]:
        """Score each hit by its weighted branch scores, and keep the best `k`."""
        return _best(
            ranked,
            k,
            lambda contribution: self._weights.get(contribution.branch, 0.0) * contribution.score,
        )


class IndexRetriever:
    """One branch of retrieval: embed where the branch is semantic, then ask the store.

    Args:
        index: The store.
        branch: Which branch it serves here. A store may back both, asked twice.
        embedder: How the query becomes a vector. Required for a semantic branch.
    """

    def __init__(
        self,
        index: SearchIndex,
        *,
        branch: Branch = Branch.SEMANTIC,
        embedder: Embedder | None = None,
    ) -> None:
        if branch is Branch.SEMANTIC and embedder is None:
            raise ConfigurationError("a semantic branch needs an embedder to ask with")
        if not index.supports(branch):
            raise ConfigurationError(f"{index.name} does not answer the {branch} branch")
        self._index = index
        self._branch = branch
        self._embedder = embedder

    @property
    def branch(self) -> Branch:
        """Which branch this retriever is."""
        return self._branch

    @property
    def name(self) -> str:
        """The store behind it."""
        return self._index.name

    async def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        filters: Mapping[str, Sequence[str]] | None = None,
        k: int = 10,
    ) -> RetrievalResult:
        """Search one branch, and rank what comes back.

        Raises:
            MissingTenantContextError: Where no tenant is in force.
            SchemaViolationError: Where a filter names the tenant, or the query is empty.
        """
        hits = await self.branch_hits(query, asked=_admissible(scope, filters, k))
        return RetrievalResult(query=query, hits=tuple(hits), branches=(self._branch,))

    async def branch_hits(self, query: str, *, asked: Admissible) -> Sequence[Hit]:
        """The hits for this branch, each carrying its rank in it."""
        query_for_store = IndexQuery(
            tenant=asked.tenant,
            collection=asked.collection,
            branch=self._branch,
            k=asked.k,
            text=query,
            vector=await self._vector(query),
            predicates=asked.predicates,
        )
        return [
            hit.model_copy(
                update={
                    "contributions": (BranchScore(branch=self._branch, score=hit.score, rank=rank),)
                }
            )
            for rank, hit in enumerate(await self._index.search(query_for_store), start=1)
        ][: asked.k]

    async def _vector(self, query: str) -> Vector | None:
        if self._branch is not Branch.SEMANTIC or self._embedder is None:
            return None
        return await self._embedder.embed_query(query)


class HybridRetriever:
    """Both branches at once, fused, and honest about the one that did not answer.

    Args:
        branches: The branch retrievers to run, at most one per branch.
        fusion: How their rankings are combined.
        require: The branches whose failure is a failure. Empty means a partial result is
            acceptable and will be flagged; naming a branch makes its absence an error,
            because a narrower result set reads downstream as a complete answer.
        timeout_seconds: How long each branch is given, or None for no limit.
    """

    def __init__(
        self,
        *branches: IndexRetriever,
        fusion: Fusion | None = None,
        require: Sequence[Branch] = (),
        timeout_seconds: float | None = None,
    ) -> None:
        if not branches:
            raise ConfigurationError("a hybrid retriever with no branches retrieves nothing")
        by_branch = {retriever.branch: retriever for retriever in branches}
        if len(by_branch) != len(branches):
            raise ConfigurationError("two retrievers for one branch is a fusion of one branch")
        missing = [branch for branch in require if branch not in by_branch]
        if missing:
            raise ConfigurationError(f"required branch {missing[0]} has no retriever")
        self._branches = by_branch
        self._fusion = fusion or ReciprocalRankFusion()
        self._require = tuple(require)
        self._timeout = timeout_seconds

    async def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        filters: Mapping[str, Sequence[str]] | None = None,
        k: int = 10,
    ) -> RetrievalResult:
        """Run every branch, fuse what answered, and say what did not.

        Raises:
            MissingTenantContextError: Where no tenant is in force.
            SchemaViolationError: Where a filter names the tenant, or the query is empty.
            RetrievalDegradedError: Where a required branch did not answer in time.
        """
        asked = _admissible(scope, filters, k)
        _asked(query)
        answered = await asyncio.gather(
            *(
                self._branch_hits(retriever, query, asked=asked)
                for retriever in self._branches.values()
            ),
            return_exceptions=True,
        )

        ranked: dict[Branch, Sequence[Hit]] = {}
        failed: dict[Branch, BaseException] = {}
        for branch, answer in zip(self._branches, answered, strict=True):
            if isinstance(answer, asyncio.CancelledError):
                raise answer
            if isinstance(answer, BaseException):
                failed[branch] = answer
            else:
                ranked[branch] = answer

        self._checked(failed)
        return RetrievalResult(
            query=query,
            hits=tuple(self._fusion.fuse(ranked, k)),
            branches=tuple(ranked),
            partial=bool(failed),
        )

    async def _branch_hits(
        self, retriever: IndexRetriever, query: str, *, asked: Admissible
    ) -> Sequence[Hit]:
        if self._timeout is None:
            return await retriever.branch_hits(query, asked=asked)
        async with asyncio.timeout(self._timeout):
            return await retriever.branch_hits(query, asked=asked)

    def _checked(self, failed: Mapping[Branch, BaseException]) -> None:
        required = [branch for branch in self._require if branch in failed]
        if required:
            raise RetrievalDegradedError(
                f"the {required[0]} branch did not answer and the caller requires it",
                missing=tuple(str(branch) for branch in required),
                answered=tuple(str(branch) for branch in self._branches if branch not in failed),
            ) from failed[required[0]]
        if len(failed) == len(self._branches):
            raise RetrievalDegradedError(
                "no branch answered",
                missing=tuple(str(branch) for branch in failed),
                answered=(),
            ) from next(iter(failed.values()))


def _asked(query: str) -> str:
    """The query, or a refusal: an empty query retrieves the collection at random."""
    if not query.strip():
        raise SchemaViolationError(
            "an empty query has nothing to retrieve on",
            model="IndexQuery",
            paths=("query",),
        )
    return query


def _admissible(
    scope: RetrievalScope, filters: Mapping[str, Sequence[str]] | None, k: int
) -> Admissible:
    """Settle what every branch will be asked, before any of them is.

    Raises:
        MissingTenantContextError: Where no tenant is in force.
        SchemaViolationError: Where a filter names the tenant.
        ConfigurationError: Where fewer than one hit was asked for.
    """
    if k < 1:
        raise ConfigurationError("retrieving fewer than one hit retrieves nothing")
    return Admissible(
        tenant=current_tenant(where="retrieval").tenant,
        collection=scope.collection,
        predicates=_pushdown(filters),
        k=k,
    )


def _pushdown(filters: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    """The caller's filters as store predicates, with the tenant refused rather than merged.

    Merging would let a caller widen the scope by supplying every tenant it can name, and
    silently dropping it would teach nobody that the filter did nothing.
    """
    if not filters:
        return {}
    if TENANT_KEY in filters:
        raise SchemaViolationError(
            "the tenant predicate comes from the tenant context and may not be filtered on",
            model="IndexQuery",
            paths=(f"filters.{TENANT_KEY}",),
        )
    return {key: tuple(values) for key, values in filters.items() if values}


def _best(
    ranked: Mapping[Branch, Sequence[Hit]],
    k: int,
    weigh: Callable[[BranchScore], float],
) -> list[Hit]:
    """Merge the branches on chunk id, sum the weighed contributions, and keep the best k."""
    merged: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for hits in ranked.values():
        for hit in hits:
            held = merged.get(hit.chunk_id)
            merged[hit.chunk_id] = (
                hit
                if held is None
                else held.model_copy(
                    update={"contributions": held.contributions + hit.contributions}
                )
            )
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + sum(
                weigh(contribution) for contribution in hit.contributions
            )
    best = sorted(merged.values(), key=lambda hit: (-scores[hit.chunk_id], hit.chunk_id))
    return [hit.model_copy(update={"score": scores[hit.chunk_id]}) for hit in best[:k]]
