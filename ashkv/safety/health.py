"""Health monitor — tracks system health across decode steps.

The health monitor aggregates signals from the safety layer:
- circuit breaker states
- pressure states over time
- migration failure rates
- fallback frequencies

It produces a single HealthState enum that the integration layer
can use to decide whether to admit new requests, drain, or shut down.

The monitor is cold-path for reads (the integration layer polls it
between decode steps). It does NOT run on the per-page hot path.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from ..contracts.results import PressureState


class HealthState(IntEnum):
    """Overall system health. Ordered by severity."""
    HEALTHY = 0       # everything fine
    DEGRADED = 1      # some failures, but serving
    UNHEALTHY = 2     # many failures, should drain
    CRITICAL = 3      # system failing, must restart


@dataclass(slots=True)
class HealthMonitor:
    """Tracks system health over time.

    Not thread-safe. Poll between decode steps (cold path).
    """

    window_size: int = 1000  # number of events to track

    _pressure_states: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _migration_results: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _fallback_results: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _tripped_breakers: int = 0
    _last_check: float = field(default_factory=time.monotonic, repr=False)

    def record_pressure(self, state: PressureState) -> None:
        """Record a pressure observation. Called between decode steps."""
        self._pressure_states.append(int(state))

    def record_migration(self, status_int: int) -> None:
        """Record a migration result status. Called after each migrate()."""
        self._migration_results.append(status_int)

    def record_fallback(self, level_int: int) -> None:
        """Record a fallback result level."""
        self._fallback_results.append(level_int)

    def record_tripped_breaker(self) -> None:
        """Record that a circuit breaker tripped."""
        self._tripped_breakers += 1

    def compute_health(self) -> HealthState:
        """Compute current system health.

        Decision rules (evaluated in order, first match wins):
        1. If >10% of recent migrations are CORRUPT → CRITICAL
        2. If >25% of recent migrations are FAILURE → UNHEALTHY
        3. If any breaker is tripped OR >25% recent pressure is CRITICAL+ → DEGRADED
        4. Otherwise → HEALTHY
        """
        # Empty system is healthy
        n = len(self._migration_results)
        if n == 0:
            return HealthState.HEALTHY

        # Compute migration failure rates (use actual count, not maxlen)
        if n >= 50:
            corrupt_count = sum(1 for s in self._migration_results if s == 4)  # CORRUPT
            failure_count = sum(1 for s in self._migration_results if s == 2)  # FAILURE

            corrupt_rate = corrupt_count / n
            failure_rate = failure_count / n

            # Rule 1: high corruption rate → CRITICAL
            if corrupt_rate >= 0.10:
                return HealthState.CRITICAL

            # Rule 2: high failure rate → UNHEALTHY
            if failure_rate >= 0.25:
                return HealthState.UNHEALTHY

        # Rule 3: pressure or breakers → DEGRADED
        if self._tripped_breakers > 0:
            return HealthState.DEGRADED

        if self._pressure_states:
            recent = list(self._pressure_states)
            critical_or_worse = sum(1 for s in recent if s >= 2)  # CRITICAL, SATURATED
            if critical_or_worse / len(recent) > 0.25:
                return HealthState.DEGRADED

        # Rule 4: healthy
        return HealthState.HEALTHY

    def stats(self) -> dict:
        """Return a snapshot of health statistics. For telemetry."""
        n = len(self._migration_results)
        return {
            "total_migrations": n,
            "corrupt_count": sum(1 for s in self._migration_results if s == 4),
            "failure_count": sum(1 for s in self._migration_results if s == 2),
            "tripped_breakers": self._tripped_breakers,
            "health_state": int(self.compute_health()),
        }

    def reset(self) -> None:
        """Reset all tracking. For testing or admin ops."""
        self._pressure_states.clear()
        self._migration_results.clear()
        self._fallback_results.clear()
        self._tripped_breakers = 0
