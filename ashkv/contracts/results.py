"""Typed results for hot-path operations.

The hot path NEVER raises. It returns one of these. Every runtime
function either returns a typed result or escalates to the safety
layer (which also returns a typed result, never raises).

Severity is determined by blast radius, not by exception class:
- Soft  : one page, transient   -> retry, then stay
- Hard  : one page, persistent  -> fallback to BF16 for that page
- Critical : many pages         -> reject request, preserve server
- Fatal : server health         -> drain and shut down
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .tiers import Tier


class MigrationStatus(IntEnum):
    """Outcome of a migrate() call. Ordered by severity."""

    OK = 0          # migration completed, page now at target tier
    SKIPPED = 1     # no-op: pinned, breaker tripped, or already at target
    FAILURE = 2     # codec failed, page unchanged at original tier
    FALLBACK = 3    # fell back to BF16 for page-level recovery
    CORRUPT = 4     # checksum mismatch detected, page quarantined


class PressureState(IntEnum):
    """Allocator pressure classification. Ordered by severity."""

    NORMAL = 0      # pressure < 0.85
    ELEVATED = 1    # 0.85 <= pressure < p_emergency
    CRITICAL = 2    # p_emergency <= pressure < 0.99
    SATURATED = 3   # pressure >= 0.99


# Thresholds for the ELEVATED and SATURATED bands.
# CRITICAL's lower bound is config.p_emergency; these two are fixed.
PRESSURE_ELEVATED_THRESHOLD: float = 0.85
PRESSURE_SATURATED_THRESHOLD: float = 0.99


@dataclass(slots=True, frozen=True)
class MigrationResult:
    """Result of a single migrate() call. Never raised, always returned."""

    status: MigrationStatus
    page_id: int
    from_tier: Tier
    to_tier: Tier
    duration_us: int
    error: str = ""   # populated only on non-OK statuses


@dataclass(slots=True, frozen=True)
class PressureReport:
    """Allocator pressure snapshot. The only allocator signal the
    controller is allowed to see."""

    pressure: float       # scalar in [0, 1]
    state: PressureState
