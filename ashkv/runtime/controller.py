"""Hot-path controller: decide target tier from score and pressure.

Pure vectorized numpy. No config access. No exceptions.

The controller takes a score array and a pressure scalar and returns
an array of target tiers. The migration engine then acts on the
diff between current and target.

Two-tier policy:
    R >= theta_high + delta   =>  BF16
    R <= theta_low - delta    =>  INT4 (or colder, if pressure forces)
    otherwise                 =>  FP8

Pressure escalation:
    When pressure >= p_emergency, demote more aggressively: the
    effective theta_low rises, pushing more pages into INT4.

Hysteresis:
    Demotion requires R <= theta_low - delta.
    Promotion requires R >= theta_high + delta.
    The current tier is taken into account so a page at BF16 with
    R = 0.7 (between theta_low and theta_high) stays at BF16, not
    demoted to FP8.
"""
from __future__ import annotations

import numpy as np

from ..contracts.tiers import Tier


def desired_tiers(
    R: np.ndarray,
    current_tiers: np.ndarray,
    pressure: float,
    theta_high: float,
    theta_low: float,
    delta: float,
    p_emergency: float,
) -> np.ndarray:
    """Compute target tier for each page, vectorized.

    Parameters
    ----------
    R : float32 array, shape (N,)
        Score for each page, in [0, 1].
    current_tiers : int8 array, shape (N,)
        Current tier of each page (Tier enum values).
    pressure : float
        Allocator pressure scalar in [0, 1].
    theta_high, theta_low, delta, p_emergency : floats
        Controller parameters (captured from config at compile time).

    Returns
    -------
    int8 array, shape (N,)
        Target tier for each page.

    Never raises. Mismatched shapes => returns current_tiers unchanged.
    """
    n = len(R)
    if n == 0 or len(current_tiers) != n:
        return current_tiers.copy() if hasattr(current_tiers, "copy") else np.array([], dtype=np.int8)

    # Effective thresholds shift under pressure.
    # Under CRITICAL pressure, the demotion band widens (theta_low rises),
    # pushing more pages into colder tiers.
    pressure_boost = 0.0
    if pressure >= p_emergency:
        pressure_boost = (pressure - p_emergency) * 2.0  # mild escalation
        pressure_boost = min(pressure_boost, 0.15)        # cap at 0.15

    eff_low = theta_low + pressure_boost
    eff_high = theta_high  # high boundary unchanged

    # Hysteresis bands
    promote_threshold = eff_high + delta
    demote_threshold = eff_low - delta

    # Default target = current (no migration unless a band is hit)
    targets = current_tiers.copy()

    # Promotion: only if currently colder than BF16 and R is high enough.
    # A page at FP8 with R >= promote_threshold => BF16.
    # A page at INT4 with R >= promote_threshold => FP8 (one tier up).
    high_mask = R >= promote_threshold
    not_hottest = current_tiers > int(Tier.BF16)
    promote_mask = high_mask & not_hottest
    # Promote by one tier (gradual, not jump-to-BF16).
    if promote_mask.any():
        targets[promote_mask] = current_tiers[promote_mask] - 1

    # Demotion: only if currently hotter than INT4 and R is low enough.
    # A page at BF16 with R <= demote_threshold => FP8 (one tier down).
    # A page at FP8 with R <= demote_threshold => INT4.
    low_mask = R <= demote_threshold
    not_coldest_gpu = current_tiers < int(Tier.INT4)
    demote_mask = low_mask & not_coldest_gpu
    if demote_mask.any():
        targets[demote_mask] = current_tiers[demote_mask] + 1

    return targets
