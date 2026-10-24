"""Routing: weighted balancing, retries with backoff, fallbacks, circuit breaker, timeouts."""

from __future__ import annotations

import asyncio
import random
from collections import Counter

import pytest

from llm_gateway.config import GatewayConfig, ProviderConfig
from llm_gateway.errors import GatewayError
from llm_gateway.providers.fake import FakeProvider
from llm_gateway.routing.breaker import CircuitBreaker, State
from llm_gateway.routing.router import Router
from llm_gateway.schemas import ChatCompletionRequest

pytestmark = pytest.mark.anyio


def req(model: str = "chat", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model, messages=[{"role": "user", "content": "hello"}], **kwargs
    )


def build(config: GatewayConfig, sleeps, **provider_overrides) -> Router:
    providers = {}
    for name, cfg in config.providers.items():
        if cfg.type != "fake":
            continue
        cfg = cfg.model_copy(update=provider_overrides.get(name, {}))
        providers[name] = FakeProvider(name, cfg)
    unavailable = {n: "not configured" for n in config.providers if n not in providers}
    return Router(config, providers, unavailable=unavailable, rng=random.Random(42), sleep=sleeps)


async def test_weighted_balancing_follows_weights(gateway_config, sleeps) -> None:
    router = build(gateway_config, sleeps)
    picks = Counter()
    for _ in range(400):
        _, info = await router.chat(req())
        picks[info.deployment.provider.name] += 1
    assert 0.68 < picks["primary"] / 400 < 0.82  # weight 3:1
    assert picks["primary"] + picks["secondary"] == 400


async def test_retries_with_exponential_backoff_then_success(gateway_config, sleeps) -> None:
    gateway_config.models["chat"].targets = gateway_config.models["chat"].targets[:1]
    router = build(gateway_config, sleeps, primary={"fail_first": 2, "failure_status": 503})
    response, info = await router.chat(req())
    assert info.deployment.provider.name == "primary" and len(info.attempts) == 3
    assert len(sleeps.calls) == 2
    assert (
        0.05 <= sleeps.calls[0] <= 0.1 and 0.1 <= sleeps.calls[1] <= 0.2
    )  # 100ms, 200ms with jitter
    assert response.choices[0].message.content


async def test_falls_back_to_next_target_then_fallback_alias(gateway_config, sleeps) -> None:
    router = build(
        gateway_config,
        sleeps,
        primary={"failure_rate": 1.0, "failure_status": 500},
        secondary={"failure_rate": 1.0, "failure_status": 502},
    )
    _, info = await router.chat(req())
    assert info.deployment.provider.name == "backup" and info.fallback_used
    failed = [a.deployment for a in info.attempts if a.error]
    assert failed.count("primary/fake-large") == 3 and failed.count("secondary/fake-large") == 3


async def test_auth_errors_skip_retries_but_fall_back(gateway_config, sleeps) -> None:
    router = build(
        gateway_config,
        sleeps,
        primary={"failure_rate": 1.0, "failure_status": 401},
        secondary={"failure_rate": 1.0, "failure_status": 403},
    )
    _, info = await router.chat(req())
    assert info.deployment.provider.name == "backup"
    assert len(info.attempts) == 3 and sleeps.calls == []


async def test_bad_request_fails_fast_without_fallback(gateway_config, sleeps) -> None:
    router = build(
        gateway_config,
        sleeps,
        primary={"failure_rate": 1.0, "failure_status": 400},
        secondary={"failure_rate": 1.0, "failure_status": 400},
    )
    with pytest.raises(GatewayError) as info:
        await router.chat(req())
    assert info.value.status == 400 and info.value.code == "upstream_bad_request"
    assert router.providers["backup"].calls == 0


async def test_all_rate_limited_returns_429(gateway_config, sleeps) -> None:
    gateway_config.models["chat"].fallbacks = []
    router = build(
        gateway_config,
        sleeps,
        primary={"failure_rate": 1.0, "failure_status": 429},
        secondary={"failure_rate": 1.0, "failure_status": 429},
    )
    with pytest.raises(GatewayError) as info:
        await router.chat(req())
    assert info.value.status == 429 and info.value.code == "upstream_rate_limited"


async def test_circuit_breaker_opens_and_skips_provider(gateway_config, sleeps) -> None:
    gateway_config.routing.max_retries = 0
    gateway_config.models["chat"].targets = gateway_config.models["chat"].targets[:1]
    router = build(gateway_config, sleeps, primary={"failure_rate": 1.0, "failure_status": 503})
    for _ in range(3):  # threshold is 3 consecutive failures
        await router.chat(req())
    assert router.breakers["primary"].state is State.OPEN
    calls_before = router.providers["primary"].calls
    _, info = await router.chat(req())
    assert router.providers["primary"].calls == calls_before  # not even tried
    assert info.attempts[0].error == "circuit open" and info.deployment.provider.name == "backup"
    health = {row["name"]: row for row in router.health()}
    assert health["primary"]["state"] == "open" and health["anthropic"]["available"] is False


def test_breaker_half_open_probe_and_recovery() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=10, clock=lambda: now[0])
    breaker.record_failure("x")
    breaker.record_failure("x")
    assert breaker.state is State.OPEN and not breaker.allow()
    now[0] = 10.5
    assert breaker.allow() and breaker.state is State.HALF_OPEN
    assert not breaker.allow()  # one probe at a time
    breaker.record_failure("still down")
    assert breaker.state is State.OPEN
    now[0] = 21
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state is State.CLOSED and breaker.allow()


async def test_timeouts_count_as_failures(gateway_config, sleeps) -> None:
    gateway_config.routing.request_timeout_s = 0.05
    gateway_config.routing.max_retries = 0
    router = build(
        gateway_config, sleeps, primary={"latency_ms": 500}, secondary={"latency_ms": 500}
    )
    _, info = await router.chat(req())
    assert info.deployment.provider.name == "backup"
    assert all("no response" in a.error for a in info.attempts[:-1])


async def test_retry_after_is_honoured_when_short(gateway_config) -> None:
    router = Router(gateway_config, {}, rng=random.Random(0))
    assert router._backoff(0, retry_after=0.7) == 0.7
    assert router._backoff(0, retry_after=60) <= 0.1  # longer than the cap: use backoff instead


async def test_unknown_and_unavailable_models(gateway_config, sleeps) -> None:
    router = build(gateway_config, sleeps)
    with pytest.raises(GatewayError) as unknown:
        await router.chat(req("nope"))
    with pytest.raises(GatewayError) as unavailable:
        await router.chat(req("gpt"))
    assert unknown.value.status == 404 and unknown.value.code == "model_not_found"
    assert unavailable.value.status == 503
    assert "gpt" not in router.aliases() and "claude" in router.aliases()  # claude has a fallback


async def test_stream_falls_back_before_first_chunk(gateway_config, sleeps) -> None:
    router = build(
        gateway_config, sleeps, primary={"failure_rate": 1.0}, secondary={"failure_rate": 1.0}
    )
    handle = await router.stream(req(stream=True))
    chunks = [handle.first] + [c async for c in handle.rest]
    assert handle.info.deployment.provider.name == "backup"
    assert chunks[-1].usage is not None


def test_fallback_cycles_are_safe() -> None:
    config = GatewayConfig.model_validate(
        {
            "providers": {"a": {"type": "fake"}, "b": {"type": "fake"}},
            "models": {
                "x": {"targets": [{"provider": "a", "model": "m"}], "fallbacks": ["y"]},
                "y": {"targets": [{"provider": "b", "model": "m"}], "fallbacks": ["x"]},
            },
        }
    )
    router = Router(config, {n: FakeProvider(n, ProviderConfig(type="fake")) for n in ("a", "b")})
    assert [d.label for d in router.resolve("x")] == ["a/m", "b/m"]


def test_config_rejects_unknown_references() -> None:
    with pytest.raises(ValueError):
        GatewayConfig.model_validate(
            {"providers": {}, "models": {"x": {"targets": [{"provider": "p", "model": "m"}]}}}
        )


async def test_parallel_requests_share_breakers(gateway_config, sleeps) -> None:
    router = build(gateway_config, sleeps)
    results = await asyncio.gather(*(router.chat(req()) for _ in range(20)))
    assert len(results) == 20 and router.breakers["primary"].state is State.CLOSED
