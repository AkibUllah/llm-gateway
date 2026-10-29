"""Rolling latency windows per deployment and latency-aware effective weights.

Each deployment keeps its last ``window`` samples (successes and timeouts, measured to the
response or, for streams, to the first chunk). Samples older than ``max_age_s`` are dropped, so a
deployment that was slow an hour ago is not punished forever. When a group has at least two
deployments with ``min_samples`` fresh samples, each one's weight is scaled by
``(fastest p95 / its p95) ** penalty``, so a deployment twice as slow as the fastest one gets half
its configured share (with the default penalty of 1). Gaps up to ``tolerance_ms`` count as noise,
and the weight never drops below ``min_weight_ratio`` of its configured weight: a slow deployment
keeps a trickle of traffic, which is how its window refills with fresh samples once it speeds up
again.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from llm_gateway.config import LatencyAwareConfig


def p95(samples: Iterable[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("p95 of an empty sample")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


@dataclass
class LatencyWindow:
    size: int
    samples: deque[tuple[float, float]] = field(default_factory=deque)

    def add(self, at: float, ms: float) -> None:
        self.samples.append((at, ms))
        while len(self.samples) > self.size:
            self.samples.popleft()

    def fresh(self, now: float, max_age_s: float) -> list[float]:
        while self.samples and now - self.samples[0][0] > max_age_s:
            self.samples.popleft()
        return [ms for _, ms in self.samples]


@dataclass(frozen=True)
class WeightInfo:
    weight: float
    effective_weight: float
    p95_ms: float | None
    samples: int


class LatencyTracker:
    def __init__(
        self, config: LatencyAwareConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self.clock = clock
        self._windows: dict[str, LatencyWindow] = {}

    def record(self, label: str, ms: float) -> None:
        window = self._windows.setdefault(label, LatencyWindow(self.config.window))
        window.add(self.clock(), ms)

    def stats(self, label: str) -> tuple[float | None, int]:
        window = self._windows.get(label)
        if window is None:
            return None, 0
        fresh = window.fresh(self.clock(), self.config.max_age_s)
        return (round(p95(fresh), 2) if fresh else None), len(fresh)

    def weights(self, group: list[tuple[str, float]]) -> dict[str, WeightInfo]:
        """Effective weight for each ``(label, configured weight)`` in one routing group."""
        stats = {label: self.stats(label) for label, _ in group}
        measured = {
            label: value
            for label, (value, count) in stats.items()
            if value is not None and count >= self.config.min_samples
        }
        fastest = min(measured.values()) if len(measured) >= 2 else None
        result: dict[str, WeightInfo] = {}
        for label, weight in group:
            effective = weight
            slow = measured.get(label)
            if (
                self.config.enabled
                and fastest is not None
                and slow is not None
                and slow - fastest > self.config.tolerance_ms
            ):
                scale = (max(fastest, 0.001) / slow) ** self.config.penalty
                effective = weight * max(self.config.min_weight_ratio, scale)
            value, count = stats[label]
            result[label] = WeightInfo(weight, round(effective, 4), value, count)
        return result
