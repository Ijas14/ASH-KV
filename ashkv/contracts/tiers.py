"""Tier definitions for ASH-KV.

Tiers form an ordered hierarchy from hottest (BF16) to coldest (Disk).
The integer ordering is the only thing the controller knows about
tiers — it does not know what codec backs a tier or what device it
lives on. Device and codec mapping is a compiler concern.

Invariants:
- Tier integer values are stable and must never change after release.
- Lower integer = hotter = higher fidelity.
- The controller treats tiers as an ordered list, nothing more.
"""
from __future__ import annotations

from enum import IntEnum


class Tier(IntEnum):
    """Ordered KV residency tiers.

    Ordering is stable and must never change after release. Adding a
    new tier appends to the end (cold side); existing tiers keep their
    integer values.
    """

    BF16 = 0      # exact hot state
    FP8 = 1       # first compression lever
    INT4 = 2      # cold GPU tier
    ARCHIVE = 3   # reconstructive / low-rank
    CPU = 4       # host memory offload
    DISK = 5      # persistent cold storage


NUM_TIERS: int = len(Tier)
HOTTEST_TIER: Tier = Tier.BF16
COLDEST_TIER: Tier = Tier.DISK


def is_hotter(a: Tier, b: Tier) -> bool:
    """True if tier a is hotter than tier b."""
    return int(a) < int(b)


def is_colder(a: Tier, b: Tier) -> bool:
    """True if tier a is colder than tier b."""
    return int(a) > int(b)


def next_hotter(t: Tier) -> Tier | None:
    """Return the next hotter tier, or None if already hottest."""
    if t == HOTTEST_TIER:
        return None
    return Tier(int(t) - 1)


def next_colder(t: Tier) -> Tier | None:
    """Return the next colder tier, or None if already coldest."""
    if t == COLDEST_TIER:
        return None
    return Tier(int(t) + 1)


def tier_distance(a: Tier, b: Tier) -> int:
    """Absolute distance in the tier hierarchy. 0 = same tier."""
    return abs(int(a) - int(b))
