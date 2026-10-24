"""Per-provider circuit breaker: closed -> open after N consecutive failures -> half-open probe."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


STATE_VALUE = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: State = State.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    total_failures: int = 0
    total_successes: int = 0
    last_error: str | None = None
    _probe_in_flight: bool = field(default=False, repr=False)

    def allow(self) -> bool:
        """Should a request be sent? In half-open state only one probe is let through."""
        if self.state is State.OPEN:
            assert self.opened_at is not None
            if self.clock() - self.opened_at < self.recovery_s:
                return False
            self.state = State.HALF_OPEN
            self._probe_in_flight = False
        if self.state is State.HALF_OPEN:
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self.total_successes += 1
        self.consecutive_failures = 0
        self.state = State.CLOSED
        self.opened_at = None
        self._probe_in_flight = False

    def record_failure(self, error: str | None = None) -> None:
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_error = error
        if self.state is State.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = State.OPEN
            self.opened_at = self.clock()
        self._probe_in_flight = False

    def snapshot(self) -> dict[str, object]:
        retry_in = None
        if self.state is State.OPEN and self.opened_at is not None:
            retry_in = round(max(0.0, self.recovery_s - (self.clock() - self.opened_at)), 2)
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_error": self.last_error,
            "retry_in_s": retry_in,
        }
