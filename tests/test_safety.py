"""Tests for the safety layer: circuit breaker, pressure guard, fallback, health."""
from __future__ import annotations

import time
import pytest

from ashkv.contracts import (
    ASHKVConfig,
    MigrationResult,
    MigrationStatus,
    PageTable,
    PressureState,
    Tier,
)
from ashkv.safety import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    FallbackLevel,
    HealthMonitor,
    HealthState,
    attempt_bf16_recovery,
    classify_pressure,
    handle_migration_failure,
    should_admit_new_request,
    should_demote_aggressively,
    should_offload_to_cpu,
)


# --- Circuit Breaker ---

class TestCircuitBreaker:
    def test_starts_not_tripped(self) -> None:
        cb = CircuitBreaker()
        assert not cb.is_tripped

    def test_trips_after_threshold(self) -> None:
        cb = CircuitBreaker(threshold=3, window_seconds=60.0, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_tripped
        cb.record_failure()
        assert cb.is_tripped

    def test_record_success_does_not_reset_trip(self) -> None:
        cb = CircuitBreaker(threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped
        cb.record_success()
        assert cb.is_tripped  # still tripped during cooldown

    def test_resets_after_cooldown(self) -> None:
        cb = CircuitBreaker(threshold=2, window_seconds=60.0, cooldown_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped
        time.sleep(0.15)
        assert not cb.is_tripped  # cooldown elapsed

    def test_prunes_old_failures(self) -> None:
        cb = CircuitBreaker(threshold=3, window_seconds=0.05, cooldown_seconds=60.0)
        cb.record_failure()
        time.sleep(0.06)
        cb.record_failure()
        cb.record_failure()
        # First failure is outside the window, so only 2 recent
        assert not cb.is_tripped

    def test_reset_clears_state(self) -> None:
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped

    def test_failure_count(self) -> None:
        cb = CircuitBreaker(threshold=10, window_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 3


class TestCircuitBreakerRegistry:
    def test_get_or_create(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = reg.get_or_create("fp8")
        cb2 = reg.get_or_create("fp8")
        assert cb1 is cb2

    def test_unknown_codec_is_available(self) -> None:
        reg = CircuitBreakerRegistry()
        assert reg.is_codec_available("unknown")

    def test_tripped_codec_unavailable(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get_or_create("fp8")
        cb.threshold = 2
        reg.record_codec_failure("fp8")
        reg.record_codec_failure("fp8")
        assert not reg.is_codec_available("fp8")

    def test_reset_all(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get_or_create("fp8")
        cb.threshold = 1
        reg.record_codec_failure("fp8")
        assert not reg.is_codec_available("fp8")
        reg.reset_all()
        assert reg.is_codec_available("fp8")


# --- Pressure Guard ---

class TestPressureGuard:
    def _config(self) -> ASHKVConfig:
        return ASHKVConfig()

    def test_normal_pressure(self) -> None:
        assert classify_pressure(0.5, self._config()) == PressureState.NORMAL

    def test_elevated_pressure(self) -> None:
        assert classify_pressure(0.87, self._config()) == PressureState.ELEVATED

    def test_critical_pressure(self) -> None:
        # p_emergency = 0.95
        assert classify_pressure(0.96, self._config()) == PressureState.CRITICAL

    def test_saturated_pressure(self) -> None:
        assert classify_pressure(0.995, self._config()) == PressureState.SATURATED

    def test_negative_pressure_clamps(self) -> None:
        assert classify_pressure(-0.5, self._config()) == PressureState.NORMAL

    def test_high_pressure_clamps(self) -> None:
        assert classify_pressure(2.0, self._config()) == PressureState.SATURATED

    def test_nan_pressure_treated_as_worst(self) -> None:
        state = classify_pressure(float("nan"), self._config())
        assert state == PressureState.SATURATED

    def test_should_admit_new_request(self) -> None:
        assert should_admit_new_request(PressureState.NORMAL)
        assert should_admit_new_request(PressureState.ELEVATED)
        assert not should_admit_new_request(PressureState.CRITICAL)
        assert not should_admit_new_request(PressureState.SATURATED)

    def test_should_demote_aggressively(self) -> None:
        assert not should_demote_aggressively(PressureState.NORMAL)
        assert should_demote_aggressively(PressureState.ELEVATED)
        assert should_demote_aggressively(PressureState.CRITICAL)
        assert should_demote_aggressively(PressureState.SATURATED)

    def test_should_offload_to_cpu(self) -> None:
        assert not should_offload_to_cpu(PressureState.NORMAL)
        assert not should_offload_to_cpu(PressureState.ELEVATED)
        assert should_offload_to_cpu(PressureState.CRITICAL)
        assert should_offload_to_cpu(PressureState.SATURATED)


# --- Fallback ---

class TestFallback:
    def test_recover_page_to_bf16(self) -> None:
        from ashkv.runtime import MockAllocator

        pt = PageTable(capacity=64)
        source_bytes = b"hello world " * 100
        pid = pt.add(0, 0, 256, hash(source_bytes) & 0xFFFFFFFFFFFFFFFF, 1000)
        pt.apply_tier_transition(pid, Tier.FP8, 999)  # page at FP8, "corrupt"

        allocator = MockAllocator()
        result = attempt_bf16_recovery(
            pid, pt, allocator,
            bf16_source_lookup=lambda _: source_bytes,
            bf16_size_lookup=lambda _: len(source_bytes),
        )
        assert result.recovered
        assert result.level == FallbackLevel.BF16_RECONSTRUCTED
        # Page should now be at BF16
        assert pt.get_tier(pid) == int(Tier.BF16)

    def test_recover_missing_page_quarantines(self) -> None:
        from ashkv.runtime import MockAllocator

        pt = PageTable(capacity=64)
        allocator = MockAllocator()
        result = attempt_bf16_recovery(
            999, pt, allocator,
            bf16_source_lookup=lambda _: b"",
            bf16_size_lookup=lambda _: 0,
        )
        assert not result.recovered
        assert result.level == FallbackLevel.QUARANTINED

    def test_recover_no_source_quarantines(self) -> None:
        from ashkv.runtime import MockAllocator

        pt = PageTable(capacity=64)
        pid = pt.add(0, 0, 256, 12345, 1000)
        allocator = MockAllocator()

        result = attempt_bf16_recovery(
            pid, pt, allocator,
            bf16_source_lookup=lambda _: None,
            bf16_size_lookup=lambda _: 100,
        )
        assert not result.recovered
        assert result.level == FallbackLevel.QUARANTINED

    def test_handle_migration_failure_triggers_recovery(self) -> None:
        from ashkv.runtime import MockAllocator

        pt = PageTable(capacity=64)
        source_bytes = b"hello world " * 100
        pid = pt.add(0, 0, 256, hash(source_bytes) & 0xFFFFFFFFFFFFFFFF, 1000)

        allocator = MockAllocator()
        # Simulate a corrupt migration result
        corrupt_result = MigrationResult(
            status=MigrationStatus.CORRUPT,
            page_id=pid,
            from_tier=Tier.BF16,
            to_tier=Tier.FP8,
            duration_us=42,
            error="checksum mismatch",
        )
        fallback = handle_migration_failure(
            corrupt_result, pt, allocator,
            bf16_source_lookup=lambda _: source_bytes,
            bf16_size_lookup=lambda _: len(source_bytes),
        )
        assert fallback.recovered
        assert fallback.level == FallbackLevel.BF16_RECONSTRUCTED
        assert pt.get_tier(pid) == int(Tier.BF16)

    def test_handle_migration_skipped_no_recovery(self) -> None:
        from ashkv.runtime import MockAllocator

        pt = PageTable(capacity=64)
        allocator = MockAllocator()
        skipped_result = MigrationResult(
            status=MigrationStatus.SKIPPED,
            page_id=0,
            from_tier=Tier.BF16,
            to_tier=Tier.FP8,
            duration_us=1,
            error="pinned",
        )
        fallback = handle_migration_failure(
            skipped_result, pt, allocator,
            bf16_source_lookup=lambda _: b"",
            bf16_size_lookup=lambda _: 0,
        )
        assert fallback.recovered  # SKIPPED is not a failure
        assert fallback.level == FallbackLevel.STAYED


# --- Health Monitor ---

class TestHealthMonitor:
    def test_empty_is_healthy(self) -> None:
        hm = HealthMonitor()
        assert hm.compute_health() == HealthState.HEALTHY

    def test_all_ok_is_healthy(self) -> None:
        hm = HealthMonitor()
        for _ in range(100):
            hm.record_migration(0)  # OK
        assert hm.compute_health() == HealthState.HEALTHY

    def test_high_corruption_is_critical(self) -> None:
        hm = HealthMonitor()
        for _ in range(90):
            hm.record_migration(0)  # OK
        for _ in range(10):
            hm.record_migration(4)  # CORRUPT
        assert hm.compute_health() == HealthState.CRITICAL

    def test_high_failure_is_unhealthy(self) -> None:
        hm = HealthMonitor()
        for _ in range(75):
            hm.record_migration(0)  # OK
        for _ in range(25):
            hm.record_migration(2)  # FAILURE
        assert hm.compute_health() == HealthState.UNHEALTHY

    def test_tripped_breaker_is_degraded(self) -> None:
        hm = HealthMonitor()
        for _ in range(100):
            hm.record_migration(0)  # OK
        hm.record_tripped_breaker()
        assert hm.compute_health() == HealthState.DEGRADED

    def test_high_critical_pressure_is_degraded(self) -> None:
        hm = HealthMonitor()
        for _ in range(100):
            hm.record_migration(0)  # OK
        for _ in range(50):
            hm.record_pressure(int(PressureState.CRITICAL))
        for _ in range(50):
            hm.record_pressure(int(PressureState.NORMAL))
        # 50% critical in recent history
        assert hm.compute_health() == HealthState.DEGRADED

    def test_stats(self) -> None:
        hm = HealthMonitor()
        hm.record_migration(0)
        hm.record_migration(2)
        hm.record_migration(4)
        hm.record_tripped_breaker()
        stats = hm.stats()
        assert stats["total_migrations"] == 3
        assert stats["failure_count"] == 1
        assert stats["corrupt_count"] == 1
        assert stats["tripped_breakers"] == 1
