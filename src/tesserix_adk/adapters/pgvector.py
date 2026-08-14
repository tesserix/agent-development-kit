"""Chunks and their embeddings in PostgreSQL, with both branches over one table.

The point of this adapter is where the tenant predicate lives: in the `WHERE`, bound as a
parameter, alongside the `ORDER BY` that picks the nearest neighbours. A store that fetches
`k` rows and filters them in Python returns fewer than `k` of the tenant's own chunks
whenever a neighbour belongs to somebody else — and silently returns nothing at all when
the top `k` are all theirs. Filtering after the fact is not a slower way to be correct.

Metadata predicates are bound too, keys included: `metadata ->> $n` takes a parameter, so
nothing a caller supplies is ever interpolated. Only the table name is, and `ChunkTables`
refuses anything that is not a plain identifier.

The DDL is not here. `verify()` reads the schema version the platform's bootstrap applied
and refuses a shape this adapter was not written for; the expected shape is
`EXPECTED_CHUNK_SCHEMA`, for the migration repository to own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters.sql_state import PostgresStoreSettings, SqlSession
from tesserix_adk.adapters.state import _text
from tesserix_adk.core.errors import ConfigurationError, ProviderUnavailableError
from tesserix_adk.rag.retrieval import Branch, Hit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.rag.retrieval import IndexQuery

__all__ = [
    "CHUNK_SCHEMA_VERSION",
    "DEFAULT_CHUNK_TABLES",
    "EXPECTED_CHUNK_SCHEMA",
    "ChunkTables",
    "PgvectorIndex",
    "PgvectorSettings",
]

CHUNK_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class ChunkTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        chunks: Chunk text, embedding and metadata, one row per chunk.
        schema: Where the migration recorded which shape it applied.
    """

    chunks: str = "adk_chunks"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.chunks, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_CHUNK_TABLES = ChunkTables()


class PgvectorSettings(PostgresStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise.

    Args:
        schema_version: The shape the adapter expects. `verify()` refuses anything else.
        text_search_config: The `regconfig` the keyword branch parses with. It must be the
            one the stored `tsvector` was built with, or the branch matches nothing.
    """

    schema_version: int = CHUNK_SCHEMA_VERSION
    text_search_config: str = "english"


class PgvectorIndex:
    """A `SearchIndex` over one chunk table, answering either branch or both.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        settings: Connection, timeout and text-search policy.
        tables: Where the rows live.
        branches: The branches this deployment backs. A table with no `tsvector` column
            should declare the semantic branch only, so a hybrid retriever knows not to
            ask rather than learning from an error.
        name: What this store calls itself in a degraded result.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        settings: PgvectorSettings,
        tables: ChunkTables = DEFAULT_CHUNK_TABLES,
        branches: Sequence[Branch] = (Branch.SEMANTIC, Branch.KEYWORD),
        name: str = "pgvector",
    ) -> None:
        self._sql = executor
        self._settings = settings
        self._tables = tables
        self._branches = tuple(branches)
        self._name = name

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PgvectorIndex:  # noqa: ANN401 — the constructor's own keywords
        """Construct the index and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        index = cls(executor, **kwargs)
        await index.verify()
        return index

    @property
    def name(self) -> str:
        """What this store is."""
        return self._name

    def supports(self, branch: Branch) -> bool:
        """Whether this deployment backs `branch`."""
        return branch in self._branches

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per call.

        Raises:
            ConfigurationError: If the shape has moved, or a statement could run forever.
        """
        rows = await self._fetch(SCHEMA_OF.format(schema=self._tables.schema), "chunks")
        found = int(rows[0][0]) if rows else 0
        if found != self._settings.schema_version:
            raise ConfigurationError(
                f"chunk schema is version {found}, and this adapter reads version"
                f" {self._settings.schema_version}; a column that moved is a query against"
                f" the wrong shape"
            )
        timeout = await self._fetch(TIMEOUT)
        if _text(timeout[0][0]) in {"0", "0ms"}:
            raise ConfigurationError(
                "this connection has no statement_timeout; one held statement would hold a"
                f" pooled connection until the process restarts. Set it to about"
                f" {self._settings.statement_timeout_ms}ms on the role or the pool"
            )

    async def search(self, query: IndexQuery) -> Sequence[Hit]:
        """The best `k` chunks for `query`, filtered by tenant inside the statement.

        Raises:
            ConfigurationError: If asked for a branch this deployment does not back.
            ProviderUnavailableError: If the database could not answer.
        """
        if not self.supports(query.branch):
            raise ConfigurationError(f"{self._name} does not back the {query.branch.value} branch")
        statement, args = self._statement(query)
        rows = await self._fetch(statement, *args)
        return [self._read(row) for row in rows]

    def _statement(self, query: IndexQuery) -> tuple[str, list[Any]]:
        """The SQL and its bound arguments, tenant and predicates included."""
        if query.branch is Branch.SEMANTIC:
            args: list[Any] = [_literal(query.vector or ()), query.tenant, query.collection]
        else:
            args = [
                self._settings.text_search_config,
                query.text,
                query.tenant,
                query.collection,
            ]
        predicates = _predicates(query.predicates, args)
        args.append(query.k)
        template = SEMANTIC if query.branch is Branch.SEMANTIC else KEYWORD
        return (
            template.format(
                chunks=self._tables.chunks, predicates=predicates, limit=f"${len(args)}"
            ),
            args,
        )

    def _read(self, row: Sequence[Any]) -> Hit:
        """One row as the hit it records."""
        return Hit(
            chunk_id=_text(row[0]),
            document_id=_text(row[1]),
            text=_text(row[2]),
            score=float(row[3]),
            metadata=_metadata(row[4]),
        )

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        try:
            return await self._sql.fetch(statement, *args)
        except Exception as failure:
            raise ProviderUnavailableError(f"{self._name} could not answer") from failure


def _predicates(predicates: Mapping[str, tuple[str, ...]], args: list[Any]) -> str:
    """The metadata predicates as bound SQL, appending their arguments to `args`."""
    clauses = []
    for key, values in predicates.items():
        args.extend((key, list(values)))
        clauses.append(f" AND metadata ->> ${len(args) - 1} = ANY(${len(args)})")
    return "".join(clauses)


def _literal(vector: Sequence[float]) -> str:
    """A vector as the text literal pgvector casts, so no driver codec is required."""
    return f"[{','.join(repr(float(value)) for value in vector)}]"


def _metadata(value: Any) -> Mapping[str, str]:  # noqa: ANN401 — jsonb arrives as a mapping or as text
    """The metadata column, whichever of the two shapes the driver hands back."""
    holding = json.loads(value) if isinstance(value, str | bytes) else value
    return {str(key): _text(held) for key, held in (holding or {}).items()}


# The tenant is in the WHERE, not applied to the answer: filtering k rows afterwards
# returns fewer than k of this tenant's chunks whenever a neighbour is somebody else's.
SEMANTIC = """
SELECT chunk_id, document_id, chunk_text, 1 - (embedding <=> $1::vector) AS score, metadata
FROM {chunks}
WHERE tenant = $2 AND collection = $3{predicates}
ORDER BY embedding <=> $1::vector
LIMIT {limit}
"""

KEYWORD = """
SELECT chunk_id, document_id, chunk_text,
       ts_rank(search, websearch_to_tsquery($1::regconfig, $2)) AS score, metadata
FROM {chunks}
WHERE tenant = $3 AND collection = $4
  AND search @@ websearch_to_tsquery($1::regconfig, $2){predicates}
ORDER BY score DESC, chunk_id
LIMIT {limit}
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = "SHOW statement_timeout"

EXPECTED_CHUNK_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('chunks', 1);

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. The embedding's width is fixed by the model the corpus was built
-- with; a second model means a second table, not a nullable second column.
CREATE TABLE adk_chunks (
    chunk_id    text PRIMARY KEY,
    document_id text NOT NULL,
    tenant      text NOT NULL,
    collection  text NOT NULL,
    chunk_text  text NOT NULL,
    embedding   vector(1536) NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    search      tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
);

-- Tenant and collection lead both indexes: they are in every WHERE this adapter writes,
-- and an ANN index scanned without them returns neighbours the caller may not read.
CREATE INDEX adk_chunks_semantic ON adk_chunks
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX adk_chunks_scope ON adk_chunks (tenant, collection);
CREATE INDEX adk_chunks_keyword ON adk_chunks USING gin (search);
CREATE INDEX adk_chunks_metadata ON adk_chunks USING gin (metadata jsonb_path_ops);
"""
