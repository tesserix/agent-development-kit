"""A cache store shared by every replica, which is where cached answers stop being local.

An in-process cache is one replica's memory of its own work. A shared one is a store full
of customers' answers that outlives the process that wrote them, so the key carries the
tenant and the versions rather than only a digest: an erasure request is then a key
pattern, not a scan of every value looking for whose it is.

The model's own reasoning is dropped before anything is written. It is marked sensitive
because it is not the answer and was never shown to anyone; persisting it into a shared
store is how it ends up somewhere nobody meant to keep it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.models.cache import CacheEntry, CacheKey

if TYPE_CHECKING:
    from tesserix_adk.adapters.ledger import RedisClient

__all__ = ["DEFAULT_NAMESPACE", "RedisCacheStore"]

DEFAULT_NAMESPACE = "adk:cache"

# Server-side because a scan driven from the client is a round trip per batch, and a
# purge that takes a hundred of them is a purge somebody cancels half way through.
_LOOKUP = """
local cursor = "0"
repeat
  local page = redis.call('SCAN', cursor, 'MATCH', ARGV[1], 'COUNT', 500)
  cursor = page[1]
  for _, key in ipairs(page[2]) do
    local held = redis.call('GET', key)
    if held then return held end
  end
until cursor == "0"
return nil
"""

_WRITE = "redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])"

_PURGE = """
local cursor = "0"
local removed = 0
repeat
  local page = redis.call('SCAN', cursor, 'MATCH', ARGV[1], 'COUNT', 500)
  cursor = page[1]
  for _, key in ipairs(page[2]) do
    removed = removed + redis.call('DEL', key)
  end
until cursor == "0"
return removed
"""


class RedisCacheStore:
    """A `CacheStore` over Redis, one Lua script per operation.

    Keys are `<namespace>:<tenant>:<prompt version>:<model version>:<digest>`, so every
    criterion a purge takes is a segment of the key and every purge is one pattern.

    Args:
        client: Anything that can run a Lua script server-side.
        namespace: The key prefix, for a deployment sharing a Redis with something else.
        redact_reasoning: Drop the model's reasoning before writing. On by default: it is
            sensitive, it is never replayed, and a cache is not a place to keep it.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        redact_reasoning: bool = True,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._redact_reasoning = redact_reasoning

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry stored under `digest`, or `None` where there is none.

        A value that cannot be read as an entry is a miss rather than an error: a cache
        that fails a run because an old release wrote a different shape is a cache that
        turns a deploy into an outage.

        Raises:
            Exception: Whatever the client raises. The caller degrades to a live call.
        """
        held = await self._client.eval(_LOOKUP, 0, self._pattern(digest=digest))
        if held is None:
            return None
        return _read(held.decode() if isinstance(held, bytes) else str(held))

    async def put(self, entry: CacheEntry) -> None:
        """Store `entry` for whatever life it has left.

        An entry that has already expired is not written at all, since a zero TTL means
        "no expiry" to Redis and an immortal stale answer is the worst outcome here.

        Raises:
            ValueError: If a key part contains the key separator or a glob character,
                which would let one tenant's name match another tenant's pattern.
            Exception: Whatever the client raises.
        """
        remaining = entry.expires_at - entry.stored_at
        if remaining <= 0:
            return
        await self._client.eval(
            _WRITE,
            1,
            self._key_for(entry.key),
            _written(entry, self._redact_reasoning),
            f"{int(remaining * 1000)}",
        )

    async def purge(self, **by: str | None) -> int:
        """Remove everything matching every criterion given, and return how many.

        Raises:
            Exception: Whatever the client raises. Erasure that failed must be seen.
        """
        removed = await self._client.eval(_PURGE, 0, self._pattern(**by))
        return int(removed or 0)

    def _key_for(self, key: CacheKey) -> str:
        """The full key for one entry, with every part checked before it is joined."""
        parts = {
            "tenant": key.tenant,
            "prompt_version": key.prompt_version,
            "model_version": key.model_version,
        }
        for name, value in parts.items():
            _safe(name, value)
        return ":".join(
            (
                self._namespace,
                key.tenant,
                key.prompt_version or "-",
                key.model_version or "-",
                key.digest,
            )
        )

    def _pattern(self, **by: str | None) -> str:
        """A glob over the key, with a wildcard for every criterion nobody gave."""
        for name, value in by.items():
            if value is not None:
                _safe(name, value)
        return ":".join(
            (
                self._namespace,
                by.get("tenant") or "*",
                by.get("prompt_version") or "*",
                by.get("model_version") or "*",
                by.get("digest") or "*",
            )
        )


def _safe(name: str, value: str) -> None:
    """Refuse a key part that could be read as a separator or as a wildcard."""
    if ":" in value or "*" in value or "?" in value:
        raise ValueError(f"{name} must not contain ':', '*' or '?', got {value!r}")


def _written(entry: CacheEntry, redact_reasoning: bool) -> str:
    """The entry as one JSON document, minus anything that must not be persisted."""
    response = entry.response.model_dump(mode="json")
    if redact_reasoning:
        response["reasoning"] = ""
    return json.dumps(
        {
            "key": {
                "tenant": entry.key.tenant,
                "model": entry.key.model,
                "prompt": entry.key.prompt,
                "tools": entry.key.tools,
                "output_schema": entry.key.output_schema,
                "parameters": entry.key.parameters,
                "prompt_version": entry.key.prompt_version,
                "model_version": entry.key.model_version,
            },
            "response": response,
            "stored_at": entry.stored_at,
            "expires_at": entry.expires_at,
            "embedding_model": entry.embedding_model,
            "threshold": entry.threshold,
        },
        sort_keys=True,
    )


def _read(document: str) -> CacheEntry | None:
    """Rebuild an entry, or `None` where the value is not one."""
    try:
        held: dict[str, Any] = json.loads(document)
        return CacheEntry(
            key=CacheKey(**held["key"]),
            # Not strict: JSON has no enums and no tuples, so what was written back is
            # loose by the time it is read, however strict the model is in the process.
            response=ModelResponse.model_validate(held["response"], strict=False),
            stored_at=held["stored_at"],
            expires_at=held["expires_at"],
            embedding_model=held["embedding_model"],
            threshold=held["threshold"],
        )
    except (ValueError, KeyError, TypeError):
        return None
