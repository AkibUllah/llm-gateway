"""Per-key request and token limits over one-minute windows (memory or Redis)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

from llm_gateway.errors import GatewayError

WINDOW_S = 60


class CounterStore(Protocol):
    async def incr(self, key: str, amount: int, ttl_s: int) -> int: ...

    async def get(self, key: str) -> int: ...


class MemoryCounters:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def incr(self, key: str, amount: int, ttl_s: int) -> int:
        async with self._lock:
            # Windows are part of the key; drop stale ones opportunistically.
            if len(self._values) > 10_000:
                self._values.clear()
            self._values[key] = self._values.get(key, 0) + amount
            return self._values[key]

    async def get(self, key: str) -> int:
        return self._values.get(key, 0)


class RedisCounters:
    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def incr(self, key: str, amount: int, ttl_s: int) -> int:
        pipe = self.redis.pipeline()
        pipe.incrby(key, amount)
        pipe.expire(key, ttl_s)
        value, _ = await pipe.execute()
        return int(value)

    async def get(self, key: str) -> int:
        value = await self.redis.get(key)
        return int(value or 0)


class KeyRateLimiter:
    def __init__(self, store: CounterStore, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.clock = clock

    def _window(self) -> tuple[int, int]:
        now = self.clock()
        window = int(now // WINDOW_S)
        retry_after = max(1, math.ceil((window + 1) * WINDOW_S - now))
        return window, retry_after

    async def check(self, key_id: str, *, rpm: int | None, tpm: int | None, tokens: int) -> None:
        window, retry_after = self._window()
        headers = {"Retry-After": str(retry_after)}
        if tpm is not None:
            used = await self.store.get(f"llmgw:tpm:{key_id}:{window}")
            if used + tokens > tpm:
                raise GatewayError(
                    429,
                    f"token rate limit reached ({used}/{tpm} tokens this minute)",
                    type="rate_limit_error",
                    code="tokens_per_minute",
                    headers=headers,
                )
        if rpm is not None:
            count = await self.store.incr(f"llmgw:rpm:{key_id}:{window}", 1, WINDOW_S * 2)
            if count > rpm:
                raise GatewayError(
                    429,
                    f"request rate limit reached ({rpm} requests per minute)",
                    type="rate_limit_error",
                    code="requests_per_minute",
                    headers=headers,
                )

    async def add_tokens(self, key_id: str, tokens: int) -> None:
        if tokens > 0:
            window, _ = self._window()
            await self.store.incr(f"llmgw:tpm:{key_id}:{window}", tokens, WINDOW_S * 2)
