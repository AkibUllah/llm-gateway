"""Cache, rate limits, token counting, pricing, PII redaction and settings."""

from __future__ import annotations

import json
import logging

import fakeredis
import pytest

from llm_gateway.cache import MemoryCache, RedisCache, ResponseCache, cache_key, is_cacheable
from llm_gateway.config import CacheConfig, Settings
from llm_gateway.errors import GatewayError
from llm_gateway.logs import JsonFormatter, redact, request_id_var
from llm_gateway.pricing import Price, PriceTable
from llm_gateway.ratelimit import KeyRateLimiter, MemoryCounters, RedisCounters
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
)
from llm_gateway.tokens import count_messages, count_text, heuristic_tokens


def req(content: str = "hello", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="chat", messages=[{"role": "user", "content": content}], **kwargs
    )


# -- cache --------------------------------------------------------------------------------


def test_cache_key_normalises_whitespace_and_unicode_forms() -> None:
    base = cache_key(req("What is   the bKash\nfee?", temperature=0))
    assert cache_key(req("  What is the bKash fee? ", temperature=0)) == base
    assert cache_key(req("What is the ｂKash fee?", temperature=0)) == base  # NFKC full-width
    assert cache_key(req("what is the bkash fee?", temperature=0)) != base
    assert cache_key(
        req("what is the bkash fee?", temperature=0), case_insensitive=True
    ) == cache_key(req("WHAT is the bKash fee?", temperature=0), case_insensitive=True)


def test_cache_key_includes_everything_that_changes_output() -> None:
    base = cache_key(req(temperature=0))
    assert cache_key(req(temperature=0, max_tokens=10)) != base
    assert cache_key(req(temperature=0, stop=["x"])) != base
    assert cache_key(req(temperature=0, seed=1)) != base
    tools = [{"type": "function", "function": {"name": "f"}}]
    assert cache_key(req(temperature=0, tools=tools)) != base


def test_only_deterministic_requests_are_cacheable() -> None:
    config = CacheConfig()
    assert is_cacheable(req(temperature=0), config)
    assert not is_cacheable(req(), config)
    assert not is_cacheable(req(temperature=0.7), config)
    assert is_cacheable(req(temperature=0.7), CacheConfig(only_deterministic=False))
    assert not is_cacheable(req(temperature=0), CacheConfig(enabled=False))


@pytest.mark.anyio
async def test_memory_cache_ttl_and_lru() -> None:
    now = [0.0]
    cache = MemoryCache(max_entries=2, clock=lambda: now[0])
    await cache.set("a", "1", ttl_s=10)
    await cache.set("b", "2", ttl_s=10)
    assert await cache.get("a") == "1"  # touch a so b is least recently used
    await cache.set("c", "3", ttl_s=10)
    assert await cache.get("b") is None and await cache.get("c") == "3"
    now[0] = 11
    assert await cache.get("a") is None


@pytest.mark.anyio
async def test_redis_cache_round_trip() -> None:
    redis = fakeredis.FakeAsyncRedis()
    cache = ResponseCache(RedisCache(redis), CacheConfig(ttl_s=30))
    response = ChatCompletionResponse(
        model="m", choices=[Choice(message=ResponseMessage(content="hi"))]
    )
    key = cache.key_for(req(temperature=0))
    assert key is not None
    await cache.put(key, response)
    assert (await cache.get(key)).choices[0].message.content == "hi"
    assert 0 < await redis.ttl("llmgw:cache:" + key) <= 30
    assert await cache.backend.clear() == 1


# -- rate limits --------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("store", ["memory", "redis"])
async def test_requests_per_minute(store: str) -> None:
    now = [120.0]
    counters = MemoryCounters() if store == "memory" else RedisCounters(fakeredis.FakeAsyncRedis())
    limiter = KeyRateLimiter(counters, clock=lambda: now[0])
    for _ in range(3):
        await limiter.check("k", rpm=3, tpm=None, tokens=10)
    with pytest.raises(GatewayError) as info:
        await limiter.check("k", rpm=3, tpm=None, tokens=10)
    assert info.value.status == 429 and info.value.headers["Retry-After"] == "60"
    now[0] = 180.0  # next window
    await limiter.check("k", rpm=3, tpm=None, tokens=10)


@pytest.mark.anyio
async def test_tokens_per_minute() -> None:
    limiter = KeyRateLimiter(MemoryCounters(), clock=lambda: 30.0)
    await limiter.check("k", rpm=None, tpm=100, tokens=60)
    await limiter.add_tokens("k", 80)
    with pytest.raises(GatewayError) as info:
        await limiter.check("k", rpm=None, tpm=100, tokens=30)
    assert info.value.code == "tokens_per_minute" and info.value.headers["Retry-After"] == "30"


# -- tokens and pricing -------------------------------------------------------------------


def test_heuristic_token_counts_are_reasonable() -> None:
    assert heuristic_tokens("") == 0
    assert 8 <= heuristic_tokens("The quick brown fox jumps over the lazy dog.") <= 14
    assert heuristic_tokens("আমার সোনার বাংলা") >= 12  # non-Latin scripts cost more tokens
    msgs = ChatCompletionRequest(
        model="m",
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
    ).messages
    assert count_messages(msgs) == 3 + 2 * 4 + count_text("be brief") + count_text("hi")


def test_price_table_exact_glob_and_unknown(caplog: pytest.LogCaptureFixture) -> None:
    table = PriceTable(
        {
            "gpt-4o-mini": Price(input=0.15, output=0.6),
            "claude-haiku-4-5*": Price(input=1, output=5),
        }
    )
    assert table.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert table.cost("claude-haiku-4-5-20251001", 2000, 1000) == pytest.approx(0.007)
    with caplog.at_level(logging.WARNING):
        assert table.cost("mystery", 100, 100) == 0
        table.cost("mystery", 100, 100)
    assert sum("mystery" in r.message for r in caplog.records) == 1


def test_bundled_price_file_loads() -> None:
    from pathlib import Path

    table = PriceTable.load(Path(__file__).resolve().parents[1] / "config" / "pricing.yaml")
    assert table.lookup("claude-sonnet-4-5") is not None and table.lookup("llama3.2").input == 0


# -- logging ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail rahim.uddin@example.com now", "email"),
        ("call +8801711223344 today", "phone"),
        ("bkash 01811-223344", "phone"),
        ("card 4111 1111 1111 1111 exp", "card"),
        ("key sk-proj-abcdefghijklmnop1234", "api_key"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer"),
    ],
)
def test_redaction_patterns(text: str, kind: str) -> None:
    assert f"[REDACTED:{kind}]" in redact(text)


def test_redaction_leaves_non_card_numbers_alone() -> None:
    assert redact("order 1234567890123 total 450") == "order 1234567890123 total 450"


def test_json_formatter_adds_request_id_extras_and_redacts() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "user %s wrote", ("a@b.com",), None)
    record.messages = [{"content": "my number is +8801911000111"}]
    token = request_id_var.set("req_1")
    try:
        line = json.loads(JsonFormatter(redact_pii=True).format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "req_1"
    assert line["msg"] == "user [REDACTED:email] wrote"
    assert "[REDACTED:phone]" in line["messages"][0]["content"]


def test_settings_normalise_database_urls() -> None:
    assert (
        Settings(_env_file=None, database_url="postgres://u:p@db/x").database_url
        == "postgresql+asyncpg://u:p@db/x"
    )
    assert (
        Settings(_env_file=None, database_url="sqlite:///./x.db").database_url
        == "sqlite+aiosqlite:///./x.db"
    )
    assert Settings(_env_file=None, budget_alert_thresholds="0.8, 0.5,1").alert_thresholds == [
        0.5,
        0.8,
        1.0,
    ]
