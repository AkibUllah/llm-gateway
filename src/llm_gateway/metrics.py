"""Prometheus metrics, one registry per app instance (keeps tests isolated)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "llmgw_requests_total",
            "Gateway requests by model alias, provider and outcome",
            ["endpoint", "model", "provider", "status", "cache"],
            registry=r,
        )
        self.latency = Histogram(
            "llmgw_request_duration_seconds",
            "End-to-end request latency",
            ["endpoint", "model"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "llmgw_tokens_total", "Tokens by model and kind", ["model", "kind"], registry=r
        )
        self.cost = Counter("llmgw_cost_usd_total", "Spend in USD by model", ["model"], registry=r)
        self.retries = Counter(
            "llmgw_upstream_retries_total", "Retries per provider", ["provider"], registry=r
        )
        self.fallbacks = Counter(
            "llmgw_fallbacks_total", "Requests served by a fallback alias", ["alias"], registry=r
        )
        self.breaker_state = Gauge(
            "llmgw_circuit_state",
            "Circuit breaker state per provider (0 closed, 1 half-open, 2 open)",
            ["provider"],
            registry=r,
        )
        self.cache_events = Counter(
            "llmgw_cache_events_total", "Cache hits and misses", ["event"], registry=r
        )
        self.rejections = Counter(
            "llmgw_rejections_total", "Requests rejected before routing", ["reason"], registry=r
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
