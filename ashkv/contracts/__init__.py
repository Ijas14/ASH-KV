"""ASH-KV contracts.

Frozen at Phase 0. Changes here are coordinated across all three
splits (Core, Codecs, Integration). Do not extend this surface
without a design review.

This module is the only thing Split 2 and Split 3 import from Split 1
at the contract level. The compiler and runtime are imported only by
the integration layer.
"""
from __future__ import annotations

from .config import ASHKVConfig
from .page import PAGE_DTYPE, PageHandle, PageTable
from .protocols import Allocator, Codec
from .results import (
    MigrationResult,
    MigrationStatus,
    PressureReport,
    PressureState,
    PRESSURE_ELEVATED_THRESHOLD,
    PRESSURE_SATURATED_THRESHOLD,
)
from .tiers import (
    COLDEST_TIER,
    HOTTEST_TIER,
    NUM_TIERS,
    Tier,
    is_colder,
    is_hotter,
    next_colder,
    next_hotter,
    tier_distance,
)

__all__ = [
    # tiers
    "Tier",
    "NUM_TIERS",
    "HOTTEST_TIER",
    "COLDEST_TIER",
    "is_hotter",
    "is_colder",
    "next_hotter",
    "next_colder",
    "tier_distance",
    # page
    "PageTable",
    "PageHandle",
    "PAGE_DTYPE",
    # config
    "ASHKVConfig",
    # results
    "MigrationStatus",
    "PressureState",
    "MigrationResult",
    "PressureReport",
    "PRESSURE_ELEVATED_THRESHOLD",
    "PRESSURE_SATURATED_THRESHOLD",
    # protocols
    "Codec",
    "Allocator",
]
