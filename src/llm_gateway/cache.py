"""Exact-match response cache keyed by a normalised request hash (memory or Redis, with TTL).

"Normalised" means the key ignores differences that do not change the meaning of the request:
repeated whitespace, Unicode compatibility forms, key order in tool schemas and, optionally,
letter case. Anything that can change the output (model, sampling parameters, tools, stop
sequences, response format, seed) is part of the key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from llm_gateway.config import CacheConfig
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

KEY_PREFIX = "llmgw:cache:"


def normalise_text(text: str, *, case_insensitive: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text.casefold() if case_insensitive else text


def cache_key(request: ChatCompletionRequest, *, case_insensitive: bool = False) -> str:
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        entry: dict[str, Any] = {
            "role": m.role,
            "content": normalise_text(m.text(), case_insensitive=case_insensitive),
        }
        if isinstance(m.content, list):
            entry["parts"] = [p for p in m.content if p.get("type") != "text"]
        if m.tool_calls:
            entry["tool_calls"] = [c.model_dump() for c in m.tool_calls]
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.name:
            entry["name"] = m.name
        messages.append(entry)
    material = {
        "model": request.model,
        "messages": messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.output_limit,
        "stop": request.stop_list,
        "seed": request.seed,
        "tools": [t.model_dump() for t in request.tools or []],
        "tool_choice": request.tool_choice,
        "response_format": request.response_format,
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def is_cacheable(request: ChatCompletionRequest, config: CacheConfig) -> bool:
    if not config.enabled:
        return False
    return not (config.only_deterministic and request.temperature not in (0, 0.0))


class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_s: int) -> None: ...

    async def clear(self) -> int: ...


class MemoryCache:
    """LRU with per-entry expiry; safe for one process."""

    def __init__(
        self, max_entries: int = 5000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.max_entries = max_entries
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires <= self.clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        async with self._lock:
            self._data[key] = (self.clock() + ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._data)
            self._data.clear()
            return count


class RedisCache:
    """Shared cache for multiple gateway replicas."""

    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def get(self, key: str) -> str | None:
        value = await self.redis.get(KEY_PREFIX + key)
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        await self.redis.set(KEY_PREFIX + key, value, ex=ttl_s)

    async def clear(self) -> int:
        count = 0
        async for key in self.redis.scan_iter(match=KEY_PREFIX + "*", count=500):
            await self.redis.delete(key)
            count += 1
        return count


class ResponseCache:
    def __init__(self, backend: CacheBackend, config: CacheConfig) -> None:
        self.backend = backend
        self.config = config

    def key_for(self, request: ChatCompletionRequest) -> str | None:
        if not is_cacheable(request, self.config):
            return None
        return cache_key(request, case_insensitive=self.config.case_insensitive)

    async def get(self, key: str) -> ChatCompletionResponse | None:
        raw = await self.backend.get(key)
        return ChatCompletionResponse.model_validate_json(raw) if raw else None

    async def put(self, key: str, response: ChatCompletionResponse) -> None:
        await self.backend.set(key, response.model_dump_json(), self.config.ttl_s)
