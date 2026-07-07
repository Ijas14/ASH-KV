"""Safety layer — fault tolerance for ASH-KV.

This package implements the safety ladder:
- circuit_breaker: per-codec failure tracking
- pressure_guard: pressure state classification
- fallback: BF16 fallback for page-level recovery
- health: system health monitoring

All modules are pure Python, importable from contracts/ and runtime/.
None of them run on the per-page hot path — they're checked between
decode steps or on migration results.
"""
from __future__ import annotations

from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from .fallback import (
    FallbackLevel,
    FallbackResult,
    attempt_bf16_recovery,
    handle_migration_failure,
)
from .health import HealthMonitor, HealthState
from .pressure_guard import (
    classify_pressure,
    should_admit_new_request,
    should_demote_aggressively,
    should_offload_to_cpu,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "FallbackLevel",
    "FallbackResult",
    "attempt_bf16_recovery",
    "handle_migration_failure",
    "HealthMonitor",
    "HealthState",
    "classify_pressure",
    "should_admit_new_request",
    "should_demote_aggressively",
    "should_offload_to_cpu",
]
