from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import redis


@dataclass
class CachedQA:
    query: str
    answer: str


class RedisCache:
    def __init__(
        self,
        url: str | None = None,
        prefix: str = "qa:",
        ttl_seconds: int = 60 * 60 * 24,
    ) -> None:
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.client = redis.Redis.from_url(self.url, decode_responses=True)

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return f"{self.prefix}{digest}"

    def _index_key(self) -> str:
        return f"{self.prefix}index"

    def get(self, query: str) -> str | None:
        return self.client.get(self._key(query))

    def set(self, query: str, answer: str) -> None:
        key = self._key(query)
        payload = json.dumps({"query": query.strip(), "answer": answer}, ensure_ascii=False)
        pipe = self.client.pipeline()
        pipe.setex(key, self.ttl_seconds, answer)
        pipe.sadd(self._index_key(), key)
        pipe.setex(f"{key}:payload", self.ttl_seconds, payload)
        pipe.execute()

    def get_cached_items(self, limit: int = 1000) -> list[CachedQA]:
        keys = list(self.client.smembers(self._index_key()))[:limit]
        if not keys:
            return []

        payload_keys = [f"{key}:payload" for key in keys]
        payloads = self.client.mget(payload_keys)
        items: list[CachedQA] = []
        expired_keys = []

        for key, payload in zip(keys, payloads):
            if not payload:
                expired_keys.append(key)
                continue
            data = json.loads(payload)
            items.append(CachedQA(query=str(data["query"]), answer=str(data["answer"])))

        if expired_keys:
            self.client.srem(self._index_key(), *expired_keys)
        return items
