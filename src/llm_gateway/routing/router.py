"""Model alias resolution, weighted load balancing, retries, fallbacks and circuit breaking."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from llm_gateway.config import GatewayConfig
from llm_gateway.errors import GatewayError, ProviderError
from llm_gateway.providers.base import Provider
from llm_gateway.routing.breaker import CircuitBreaker
from llm_gateway.routing.latency import LatencyTracker
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)

if TYPE_CHECKING:
    from llm_gateway.metrics import Metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Deployment:
    alias: str
    provider: Provider
    model: str
    weight: float = 1.0

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model}"


@dataclass
class Attempt:
    deployment: str
    error: str | None = None
    status: int | None = None
    duration_ms: float = 0.0


@dataclass
class RouteInfo:
    alias: str
    deployment: Deployment | None = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fallback_used(self) -> bool:
        return self.deployment is not None and self.deployment.alias != self.alias

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


@dataclass
class StreamHandle:
    info: RouteInfo
    first: ChatCompletionChunk
    rest: AsyncIterator[ChatCompletionChunk]


class Router:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        *,
        unavailable: dict[str, str] | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        metrics: Metrics | None = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self.unavailable = unavailable or {}
        self.rng = rng or random.Random()
        self.sleep = sleep
        self.metrics = metrics
        breaker = config.routing.breaker
        self.breakers = {
            name: CircuitBreaker(breaker.failure_threshold, breaker.recovery_s, clock)
            for name in providers
        }
        self.latency = LatencyTracker(config.routing.latency_aware, clock)

    # -- resolution -------------------------------------------------------------------------

    def aliases(self) -> list[str]:
        """Aliases that have at least one usable deployment (directly or through fallbacks)."""
        return [alias for alias in self.config.models if self._chain(alias)]

    def _deployments(self, alias: str) -> list[Deployment]:
        route = self.config.models[alias]
        return [
            Deployment(alias, self.providers[t.provider], t.model, t.weight)
            for t in route.targets
            if t.provider in self.providers
        ]

    def _chain(self, alias: str) -> list[Deployment]:
        """Weighted primary pick, the alias's other targets, then fallback aliases depth-first."""
        ordered: list[Deployment] = []
        seen_aliases: set[str] = set()

        def visit(name: str, weighted: bool) -> None:
            if name in seen_aliases or name not in self.config.models:
                return
            seen_aliases.add(name)
            group = self._deployments(name)
            if weighted and len(group) > 1:
                group = self._weighted_order(group)
            else:
                group = sorted(group, key=lambda d: -d.weight)
            for deployment in group:
                if all(d.label != deployment.label for d in ordered):
                    ordered.append(deployment)
            for fallback in self.config.models[name].fallbacks:
                visit(fallback, weighted=False)

        visit(alias, weighted=True)
        return ordered

    def _effective_weights(self, group: list[Deployment]) -> dict[str, float]:
        if not self.config.routing.latency_aware.enabled:
            return {d.label: d.weight for d in group}
        weights = self.latency.weights([(d.label, d.weight) for d in group])
        return {label: w.effective_weight for label, w in weights.items()}

    def _weighted_order(self, group: list[Deployment]) -> list[Deployment]:
        weights = self._effective_weights(group)
        remaining = list(group)
        ordered: list[Deployment] = []
        while remaining:
            pick = self.rng.choices(remaining, weights=[weights[d.label] for d in remaining])[0]
            ordered.append(pick)
            remaining.remove(pick)
        return ordered

    def resolve(self, alias: str) -> list[Deployment]:
        if alias not in self.config.models:
            raise GatewayError(
                404, f"model {alias!r} does not exist", code="model_not_found", param="model"
            )
        chain = self._chain(alias)
        if not chain:
            reasons = {
                t.provider: self.unavailable.get(t.provider, "unknown")
                for t in self.config.models[alias].targets
            }
            raise GatewayError(
                503,
                f"model {alias!r} has no available provider: {reasons}",
                type="service_unavailable",
                code="no_provider",
            )
        return chain

    # -- execution --------------------------------------------------------------------------

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        routing = self.config.routing
        cap = routing.backoff_max_ms / 1000
        if retry_after is not None and retry_after <= cap:
            return retry_after
        base = routing.backoff_base_ms / 1000 * (2**attempt)
        return min(cap, base) * (0.5 + self.rng.random() / 2)

    async def _execute(
        self, alias: str, call: Callable[[Deployment], Awaitable[T]]
    ) -> tuple[T, RouteInfo]:
        info = RouteInfo(alias=alias)
        timeout = self.config.routing.request_timeout_s
        last_error: ProviderError | None = None
        skipped_open = 0

        for deployment in self.resolve(alias):
            breaker = self.breakers[deployment.provider.name]
            for attempt in range(self.config.routing.max_retries + 1):
                if not breaker.allow():
                    skipped_open += 1
                    info.attempts.append(Attempt(deployment.label, "circuit open"))
                    break
                started = time.perf_counter()
                try:
                    result = await asyncio.wait_for(call(deployment), timeout=timeout)
                except asyncio.TimeoutError:
                    self.latency.record(deployment.label, timeout * 1000)
                    error = ProviderError(deployment.provider.name, f"no response in {timeout}s")
                except ProviderError as exc:
                    error = exc
                else:
                    breaker.record_success()
                    self._breaker_metric(deployment)
                    duration_ms = _ms_since(started)
                    self.latency.record(deployment.label, duration_ms)
                    info.attempts.append(Attempt(deployment.label, duration_ms=duration_ms))
                    info.deployment = deployment
                    if info.fallback_used and self.metrics:
                        self.metrics.fallbacks.labels(alias=alias).inc()
                    return result, info

                info.attempts.append(
                    Attempt(deployment.label, error.message, error.status, _ms_since(started))
                )
                last_error = error
                if not error.fallback:
                    breaker.record_success()  # the provider answered; the request was the problem
                    raise GatewayError(
                        error.status or 400,
                        f"upstream rejected the request: {error.message}",
                        type="invalid_request_error",
                        code="upstream_bad_request",
                    ) from error
                breaker.record_failure(error.message)
                self._breaker_metric(deployment)
                logger.warning(
                    "upstream failure",
                    extra={
                        "deployment": deployment.label,
                        "status": error.status,
                        "error": error.message,
                    },
                )
                if not error.retryable or attempt == self.config.routing.max_retries:
                    break
                if self.metrics:
                    self.metrics.retries.labels(provider=deployment.provider.name).inc()
                await self.sleep(self._backoff(attempt, error.retry_after))

        summary = "; ".join(f"{a.deployment}: {a.error}" for a in info.attempts)
        if last_error is None and skipped_open:
            raise GatewayError(
                503,
                f"all providers for {alias!r} are unavailable (circuit open)",
                type="service_unavailable",
                code="circuit_open",
                headers={"Retry-After": str(int(self.config.routing.breaker.recovery_s))},
            )
        if (
            last_error is not None
            and last_error.is_rate_limit
            and all(a.status == 429 or a.error == "circuit open" for a in info.attempts)
        ):
            raise GatewayError(
                429,
                f"all upstreams rate limited: {summary}",
                type="rate_limit_error",
                code="upstream_rate_limited",
            )
        raise GatewayError(
            502,
            f"all upstream providers failed: {summary}",
            type="api_error",
            code="upstream_error",
        )

    def _breaker_metric(self, deployment: Deployment) -> None:
        if self.metrics:
            from llm_gateway.routing.breaker import STATE_VALUE

            breaker = self.breakers[deployment.provider.name]
            self.metrics.breaker_state.labels(provider=deployment.provider.name).set(
                STATE_VALUE[breaker.state]
            )

    async def chat(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionResponse, RouteInfo]:
        return await self._execute(request.model, lambda d: d.provider.chat(request, d.model))

    async def embed(self, request: EmbeddingRequest) -> tuple[EmbeddingResponse, RouteInfo]:
        return await self._execute(request.model, lambda d: d.provider.embed(request, d.model))

    async def stream(self, request: ChatCompletionRequest) -> StreamHandle:
        """Open a stream with retries/fallback up to the first chunk.

        Once a chunk has been produced, the client has started receiving bytes, so later errors
        are surfaced in-band instead of being retried elsewhere.
        """

        async def open_stream(
            d: Deployment,
        ) -> tuple[ChatCompletionChunk, AsyncIterator[ChatCompletionChunk]]:
            iterator = d.provider.stream(request, d.model).__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration as exc:
                raise ProviderError(d.provider.name, "stream ended before any data") from exc
            return first, iterator

        (first, rest), info = await self._execute(request.model, open_stream)
        return StreamHandle(info=info, first=first, rest=rest)

    def routing_weights(self) -> dict[str, list[dict[str, object]]]:
        """Configured and effective weight, p95 and sample count per deployment, by provider."""
        by_provider: dict[str, list[dict[str, object]]] = {}
        for alias in self.config.models:
            group = self._deployments(alias)
            if not group:
                continue
            effective = self._effective_weights(group)
            stats = self.latency.weights([(d.label, d.weight) for d in group])
            for d in group:
                by_provider.setdefault(d.provider.name, []).append(
                    {
                        "alias": alias,
                        "model": d.model,
                        "weight": d.weight,
                        "effective_weight": effective[d.label],
                        "p95_ms": stats[d.label].p95_ms,
                        "samples": stats[d.label].samples,
                    }
                )
        return by_provider

    def health(self) -> list[dict[str, object]]:
        deployments = self.routing_weights()
        rows: list[dict[str, object]] = []
        for name, cfg in self.config.providers.items():
            row: dict[str, object] = {"name": name, "type": cfg.type}
            if name in self.providers:
                row.update({"available": True, **self.breakers[name].snapshot()})
                row["deployments"] = deployments.get(name, [])
            else:
                row.update({"available": False, "reason": self.unavailable.get(name)})
            rows.append(row)
        return rows

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
