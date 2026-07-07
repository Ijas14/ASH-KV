"""Plain counters for ASH-KV telemetry.

No metric objects, no histograms, no transitive deps. Just integers
that get incremented on the hot path (via the telemetry closure)
and read by the metrics exporter (cold path).

The hot path calls the telemetry_fn closure (compiled at startup).
That closure increments these counters. The exporter reads them
on a scrape interval.

This is the no-op binding pattern: if telemetry is disabled, the
closure is a no-op and these counters are never touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class Counters:
    """All ASH-KV telemetry counters.

    Thread-safe via a single lock. The lock is only contended on
    the scrape interval (cold path), not on the hot path — the
    hot-path closure increments without locking (see note below).
    """

    # Migration counters
    migrations_ok: int = 0
    migrations_skipped: int = 0
    migrations_failure: int = 0
    migrations_fallback: int = 0
    migrations_corrupt: int = 0

    # Tier counters (pages per tier)
    pages_in_bf16: int = 0
    pages_in_fp8: int = 0
    pages_in_int4: int = 0
    pages_in_archive: int = 0
    pages_in_cpu: int = 0
    pages_in_disk: int = 0

    # Pressure counters (how many decode steps at each state)
    pressure_normal: int = 0
    pressure_elevated: int = 0
    pressure_critical: int = 0
    pressure_saturated: int = 0

    # Fallback counters
    bf16_recoveries: int = 0
    pages_quarantined: int = 0

    # Circuit breaker counters
    breakers_tripped: int = 0

    _lock: Lock = field(default_factory=Lock, repr=False)

    def snapshot(self) -> dict[str, int]:
        """Return a snapshot of all counters. Thread-safe.

        Called by the exporter on scrape. Returns a dict of name → value.
        """
        with self._lock:
            return {
                "migrations_ok": self.migrations_ok,
                "migrations_skipped": self.migrations_skipped,
                "migrations_failure": self.migrations_failure,
                "migrations_fallback": self.migrations_fallback,
                "migrations_corrupt": self.migrations_corrupt,
                "pages_in_bf16": self.pages_in_bf16,
                "pages_in_fp8": self.pages_in_fp8,
                "pages_in_int4": self.pages_in_int4,
                "pages_in_archive": self.pages_in_archive,
                "pages_in_cpu": self.pages_in_cpu,
                "pages_in_disk": self.pages_in_disk,
                "pressure_normal": self.pressure_normal,
                "pressure_elevated": self.pressure_elevated,
                "pressure_critical": self.pressure_critical,
                "pressure_saturated": self.pressure_saturated,
                "bf16_recoveries": self.bf16_recoveries,
                "pages_quarantined": self.pages_quarantined,
                "breakers_tripped": self.breakers_tripped,
            }

    def record_migration(self, status_int: int) -> None:
        """Increment the appropriate migration counter.

        Called by the telemetry closure on the hot path. Uses the
        lock for safety — the overhead is minimal because migrate()
        is called at most a few times per decode step.
        """
        with self._lock:
            if status_int == 0:    # OK
                self.migrations_ok += 1
            elif status_int == 1:  # SKIPPED
                self.migrations_skipped += 1
            elif status_int == 2:  # FAILURE
                self.migrations_failure += 1
            elif status_int == 3:  # FALLBACK
                self.migrations_fallback += 1
            elif status_int == 4:  # CORRUPT
                self.migrations_corrupt += 1

    def record_pressure(self, state_int: int) -> None:
        """Increment the appropriate pressure counter."""
        with self._lock:
            if state_int == 0:
                self.pressure_normal += 1
            elif state_int == 1:
                self.pressure_elevated += 1
            elif state_int == 2:
                self.pressure_critical += 1
            elif state_int == 3:
                self.pressure_saturated += 1

    def record_tier_change(self, old_tier: int, new_tier: int) -> None:
        """Update tier counters when a page changes tier."""
        if old_tier == new_tier:
            return
        with self._lock:
            self._decrement_tier(old_tier)
            self._increment_tier(new_tier)

    def _increment_tier(self, tier: int) -> None:
        if tier == 0:
            self.pages_in_bf16 += 1
        elif tier == 1:
            self.pages_in_fp8 += 1
        elif tier == 2:
            self.pages_in_int4 += 1
        elif tier == 3:
            self.pages_in_archive += 1
        elif tier == 4:
            self.pages_in_cpu += 1
        elif tier == 5:
            self.pages_in_disk += 1

    def _decrement_tier(self, tier: int) -> None:
        if tier == 0:
            self.pages_in_bf16 = max(0, self.pages_in_bf16 - 1)
        elif tier == 1:
            self.pages_in_fp8 = max(0, self.pages_in_fp8 - 1)
        elif tier == 2:
            self.pages_in_int4 = max(0, self.pages_in_int4 - 1)
        elif tier == 3:
            self.pages_in_archive = max(0, self.pages_in_archive - 1)
        elif tier == 4:
            self.pages_in_cpu = max(0, self.pages_in_cpu - 1)
        elif tier == 5:
            self.pages_in_disk = max(0, self.pages_in_disk - 1)

    def reset(self) -> None:
        """Reset all counters. For testing."""
        with self._lock:
            self.migrations_ok = 0
            self.migrations_skipped = 0
            self.migrations_failure = 0
            self.migrations_fallback = 0
            self.migrations_corrupt = 0
            self.pages_in_bf16 = 0
            self.pages_in_fp8 = 0
            self.pages_in_int4 = 0
            self.pages_in_archive = 0
            self.pages_in_cpu = 0
            self.pages_in_disk = 0
            self.pressure_normal = 0
            self.pressure_elevated = 0
            self.pressure_critical = 0
            self.pressure_saturated = 0
            self.bf16_recoveries = 0
            self.pages_quarantined = 0
            self.breakers_tripped = 0


# Singleton counters instance. Imported by the compiler and the exporter.
counters = Counters()
