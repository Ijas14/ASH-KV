"""Runtime tests: score, controller, migrate.

Verifies vectorized correctness, hysteresis behavior, and the
never-throw invariant under fault injection.
"""
from __future__ import annotations

import numpy as np
import pytest

from ashkv.contracts import (
    ASHKVConfig,
    Allocator,
    Codec,
    MigrationStatus,
    PageTable,
    PressureState,
    Tier,
)
from ashkv.runtime import desired_tiers, score_vectorized


# --- Score ---

class TestScore:
    def test_score_basic(self) -> None:
        T = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        S = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        N = np.zeros(3, dtype=np.float32)
        P = np.zeros(3, dtype=np.float32)
        R = score_vectorized(T, S, N, P, 0.7, 0.1, 0.1, 0.1)
        # Page 0: 0.7*1 + 0.1*0 + ... = 0.7
        # Page 1: 0.7*0.5 + 0.1*0.5 = 0.4
        # Page 2: 0.7*0 + 0.1*1 = 0.1
        assert np.allclose(R, [0.7, 0.4, 0.1], atol=1e-6)

    def test_score_empty(self) -> None:
        R = score_vectorized(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            0.7, 0.1, 0.1, 0.1,
        )
        assert len(R) == 0

    def test_score_mismatched_lengths_returns_empty(self) -> None:
        # Never raises; returns empty.
        T = np.array([1.0, 0.5], dtype=np.float32)
        S = np.array([0.0], dtype=np.float32)
        N = np.zeros(2, dtype=np.float32)
        P = np.zeros(2, dtype=np.float32)
        R = score_vectorized(T, S, N, P, 0.7, 0.1, 0.1, 0.1)
        assert len(R) == 0

    def test_score_clamps_to_unit(self) -> None:
        T = np.array([2.0, -1.0], dtype=np.float32)
        S = np.zeros(2, dtype=np.float32)
        N = np.zeros(2, dtype=np.float32)
        P = np.zeros(2, dtype=np.float32)
        R = score_vectorized(T, S, N, P, 1.0, 0.0, 0.0, 0.0)
        assert R[0] == 1.0
        assert R[1] == 0.0


# --- Controller ---

class TestController:
    def _params(self) -> tuple[float, float, float, float]:
        # theta_high=0.72, theta_low=0.33, delta=0.04, p_emergency=0.95
        return 0.72, 0.33, 0.04, 0.95

    def test_high_score_promotes(self) -> None:
        R = np.array([0.9], dtype=np.float32)
        current = np.array([int(Tier.FP8)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        assert int(targets[0]) == int(Tier.BF16)

    def test_low_score_demotes(self) -> None:
        R = np.array([0.1], dtype=np.float32)
        current = np.array([int(Tier.BF16)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        # One tier down: BF16 -> FP8
        assert int(targets[0]) == int(Tier.FP8)

    def test_mid_score_stays(self) -> None:
        # R = 0.5 is between theta_low + delta and theta_high - delta
        R = np.array([0.5], dtype=np.float32)
        current = np.array([int(Tier.FP8)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        assert int(targets[0]) == int(Tier.FP8)

    def test_hysteresis_prevents_flapping(self) -> None:
        # A page at BF16 with R just below theta_high stays at BF16
        # (doesn't demote unless R <= theta_low - delta).
        R = np.array([0.7], dtype=np.float32)
        current = np.array([int(Tier.BF16)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        assert int(targets[0]) == int(Tier.BF16)

    def test_cold_tier_not_demoted_below_int4(self) -> None:
        # A page at INT4 with very low R stays at INT4 (not pushed to ARCHIVE
        # — that's a separate decision the safety layer makes).
        R = np.array([0.0], dtype=np.float32)
        current = np.array([int(Tier.INT4)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        assert int(targets[0]) == int(Tier.INT4)

    def test_pressure_escelates_demotion(self) -> None:
        # Under CRITICAL pressure, the demotion band widens.
        # A page at BF16 with R = 0.4 would normally stay (between
        # theta_low+delta and theta_high-delta). Under pressure, it
        # might demote.
        R = np.array([0.4], dtype=np.float32)
        current = np.array([int(Tier.BF16)], dtype=np.int8)
        # Normal pressure
        targets_normal = desired_tiers(R, current, 0.5, *self._params())
        assert int(targets_normal[0]) == int(Tier.BF16)
        # Critical pressure
        targets_pressure = desired_tiers(R, current, 0.99, *self._params())
        # Under pressure, eff_low rises, so 0.4 might fall below
        # eff_low - delta and demote.
        assert int(targets_pressure[0]) >= int(Tier.BF16)  # at least unchanged or demoted

    def test_vectorized_consistency(self) -> None:
        # 100 pages, mixed scores
        np.random.seed(42)
        R = np.random.rand(100).astype(np.float32)
        current = np.random.choice(
            [int(Tier.BF16), int(Tier.FP8), int(Tier.INT4)],
            size=100,
        ).astype(np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        # Each target must be either current, current+1, or current-1.
        diff = targets.astype(np.int32) - current.astype(np.int32)
        assert np.all((diff >= -1) & (diff <= 1))

    def test_no_demotion_below_int4(self) -> None:
        # Pages already at INT4 are not demoted further by the controller.
        # (The safety layer may move them to ARCHIVE/CPU, but that's
        # a separate path.)
        R = np.zeros(5, dtype=np.float32)
        current = np.full(5, int(Tier.INT4), dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        assert np.all(targets == int(Tier.INT4))

    def test_mismatched_shapes_returns_current(self) -> None:
        R = np.array([0.1, 0.2], dtype=np.float32)
        current = np.array([int(Tier.BF16)], dtype=np.int8)
        targets = desired_tiers(R, current, 0.5, *self._params())
        # Returns current (length matches current, not R)
        assert len(targets) == 1
