"""Latency-aware routing: rolling windows, effective weights, traffic shifting and recovery."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from llm_gateway.config import GatewayConfig, LatencyAwareConfig
from llm_gateway.providers.fake import FakeProvider
from llm_gateway.routing.latency import LatencyTracker, p95
from llm_gateway.routing.router import Router
from llm_gateway.schemas import ChatCompletionRequest
from tests.conftest import admin

pytestmark = pytest.mark.anyio


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def tracker(clock: Clock, **overrides) -> LatencyTracker:
    config = {
        "enabled": True,
        "window": 10,
        "min_samples": 3,
        "max_age_s": 60,
        "tolerance_ms": 0,
        **overrides,
    }
    return LatencyTracker(LatencyAwareConfig(**config), clock)


def feed(t: LatencyTracker, label: str, *samples: float) -> None:
    for ms in samples:
        t.record(label, ms)


def test_p95_is_nearest_rank() -> None:
    assert p95([5]) == 5
    assert p95(range(1, 101)) == 95
    assert p95([10, 20, 30, 40, 1000]) == 1000  # 5 samples: the slowest is the 95th percentile
    with pytest.raises(ValueError):
        p95([])


def test_window_keeps_the_last_samples_and_drops_stale_ones() -> None:
    clock = Clock()
    t = tracker(clock, window=3)
    feed(t, "a", 900, 10, 20, 30)
    assert t.stats("a") == (30, 3)  # 900 fell out of the window
    clock.now += 61
    assert t.stats("a") == (None, 0)
    assert t.stats("never-seen") == (None, 0)


def test_slow_deployment_is_scaled_by_fastest_over_its_p95() -> None:
    clock = Clock()
    t = tracker(clock)
    feed(t, "fast", 100, 100, 100)
    feed(t, "slow", 400, 400, 400)
    weights = t.weights([("fast", 3.0), ("slow", 3.0)])
    assert weights["fast"].effective_weight == 3.0
    assert weights["slow"].effective_weight == 0.75  # 4x slower -> a quarter of its weight
    assert weights["slow"].p95_ms == 400 and weights["slow"].samples == 3


def test_penalty_and_floor() -> None:
    clock = Clock()
    harsh = tracker(clock, penalty=2)
    feed(harsh, "fast", 100, 100, 100)
    feed(harsh, "slow", 200, 200, 200)
    assert harsh.weights([("fast", 1), ("slow", 1)])["slow"].effective_weight == 0.25

    floored = tracker(clock, min_weight_ratio=0.1)
    feed(floored, "fast", 1, 1, 1)
    feed(floored, "slow", 5000, 5000, 5000)
    assert floored.weights([("fast", 1), ("slow", 2)])["slow"].effective_weight == 0.2


def test_small_gaps_are_treated_as_noise() -> None:
    clock = Clock()
    t = tracker(clock, tolerance_ms=25)
    feed(t, "a", 300, 300, 300)
    feed(t, "b", 320, 320, 320)
    feed(t, "c", 600, 600, 600)
    weights = t.weights([("a", 1), ("b", 1), ("c", 1)])
    assert weights["b"].effective_weight == 1  # 20 ms slower: within tolerance
    assert weights["c"].effective_weight == 0.5


def test_weights_unchanged_without_enough_evidence_or_when_disabled() -> None:
    clock = Clock()
    t = tracker(clock)
    feed(t, "fast", 100, 100, 100)
    feed(t, "slow", 900, 900)  # below min_samples: not judged yet
    assert t.weights([("fast", 1), ("slow", 1)])["slow"].effective_weight == 1
    assert t.weights([("fast", 1)])["fast"].effective_weight == 1  # nothing to compare with

    off = tracker(clock, enabled=False)
    feed(off, "fast", 100, 100, 100)
    feed(off, "slow", 900, 900, 900)
    info = off.weights([("fast", 1), ("slow", 1)])["slow"]
    assert info.effective_weight == 1 and info.p95_ms == 900  # still measured, just not applied


def test_min_samples_cannot_exceed_window() -> None:
    with pytest.raises(ValueError):
        LatencyAwareConfig(window=5, min_samples=10)


def req() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="chat", messages=[{"role": "user", "content": "hello"}])


def latency_router(config: GatewayConfig, clock: Clock, slow_ms: int) -> Router:
    config.models["chat"].fallbacks = []
    config.routing.latency_aware = LatencyAwareConfig(
        enabled=True, window=20, min_samples=5, max_age_s=60, tolerance_ms=15
    )
    providers = {
        "primary": FakeProvider(
            "primary", config.providers["primary"].model_copy(update={"latency_ms": slow_ms})
        ),
        "secondary": FakeProvider("secondary", config.providers["secondary"]),
    }
    return Router(config, providers, rng=random.Random(7), clock=clock)


async def share(router: Router, n: int) -> float:
    picks = Counter()
    for _ in range(n):
        _, info = await router.chat(req())
        picks[info.deployment.provider.name] += 1
    return picks["primary"] / n


async def test_traffic_shifts_away_from_a_slow_deployment_and_recovers(gateway_config) -> None:
    clock = Clock()
    router = latency_router(gateway_config, clock, slow_ms=40)
    for _ in range(80):  # warm up until both deployments have min_samples
        await router.chat(req())
        if all(router.latency.stats(f"{p}/fake-large")[1] >= 5 for p in ("primary", "secondary")):
            break

    weights = {row["alias"]: row for row in router.routing_weights()["primary"]}["chat"]
    assert weights["weight"] == 3 and weights["effective_weight"] < 1
    assert weights["p95_ms"] >= 40
    assert await share(router, 200) < 0.4  # was 75% by static weight

    # The slow deployment gets fast again: once its old samples age out, its share comes back.
    router.providers["primary"].config = router.providers["primary"].config.model_copy(
        update={"latency_ms": 0}
    )
    clock.now += 61
    assert await share(router, 200) > 0.6


async def test_static_weights_when_latency_aware_is_off(gateway_config) -> None:
    clock = Clock()
    router = latency_router(gateway_config, clock, slow_ms=20)
    router.config.routing.latency_aware.enabled = False
    assert await share(router, 120) > 0.6
    row = {r["alias"]: r for r in router.routing_weights()["primary"]}["chat"]
    assert row["effective_weight"] == row["weight"] == 3 and row["samples"] > 0


async def test_admin_providers_shows_effective_weights(client) -> None:
    response = await client.get("/admin/providers", headers=admin())
    rows = {row["name"]: row for row in response.json()}
    chat = {d["alias"]: d for d in rows["primary"]["deployments"]}["chat"]
    assert chat == {
        "alias": "chat",
        "model": "fake-large",
        "weight": 3.0,
        "effective_weight": 3.0,
        "p95_ms": None,
        "samples": 0,
    }
    assert "deployments" not in rows["anthropic"]  # not configured, so no routing data
