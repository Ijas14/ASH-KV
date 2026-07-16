"""Tests for the contract layer.

Verifies:
1. ASHKVConfig validation (the 8-number surface)
2. Tier ordering and helpers
3. PageTable lifecycle: add, snapshot, transition, pin, remove
4. MigrationResult / PressureState immutability
5. Codec and Allocator protocol structural checks
"""
from __future__ import annotations

import numpy as np
import pytest

from ashkv.contracts import (
    ASHKVConfig,
    Allocator,
    Codec,
    COLDEST_TIER,
    HOTTEST_TIER,
    MigrationResult,
    MigrationStatus,
    PAGE_DTYPE,
    PageTable,
    PRESSURE_ELEVATED_THRESHOLD,
    PRESSURE_SATURATED_THRESHOLD,
    PressureReport,
    PressureState,
    Tier,
    is_colder,
    is_hotter,
    next_colder,
    next_hotter,
    tier_distance,
)


# --- Tier ordering ---

class TestTierOrdering:
    def test_bf16_is_hottest(self) -> None:
        assert HOTTEST_TIER == Tier.BF16
        assert int(Tier.BF16) == 0

    def test_disk_is_coldest(self) -> None:
        assert COLDEST_TIER == Tier.DISK

    def test_ordering_monotonic(self) -> None:
        tiers = list(Tier)
        for i in range(len(tiers) - 1):
            assert is_hotter(tiers[i], tiers[i + 1])
            assert is_colder(tiers[i + 1], tiers[i])

    def test_next_hotter(self) -> None:
        assert next_hotter(Tier.FP8) == Tier.BF16
        assert next_hotter(Tier.BF16) is None

    def test_next_colder(self) -> None:
        assert next_colder(Tier.BF16) == Tier.FP8
        assert next_colder(Tier.DISK) is None

    def test_tier_distance(self) -> None:
        assert tier_distance(Tier.BF16, Tier.BF16) == 0
        assert tier_distance(Tier.BF16, Tier.DISK) == 5
        assert tier_distance(Tier.DISK, Tier.BF16) == 5

    def test_six_tiers(self) -> None:
        assert len(Tier) == 6


# --- Config validation ---

class TestASHKVConfig:
    def test_default_valid(self) -> None:
        c = ASHKVConfig()
        assert c.w_T == 0.7
        assert c.theta_high == 0.72
        assert c.theta_low == 0.33
        assert c.delta == 0.04
        assert c.p_emergency == 0.95

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ASHKVConfig(w_T=0.5, w_S=0.1, w_N=0.1, w_P=0.1)

    def test_weight_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="w_T"):
            ASHKVConfig(w_T=1.5, w_S=-0.5, w_N=0.5, w_P=0.5)

    def test_theta_low_must_be_below_high(self) -> None:
        with pytest.raises(ValueError, match="theta_low"):
            ASHKVConfig(theta_low=0.8, theta_high=0.7)

    def test_theta_equal_rejected(self) -> None:
        with pytest.raises(ValueError, match="theta_low"):
            ASHKVConfig(theta_low=0.5, theta_high=0.5)

    def test_p_emergency_must_be_in_open_interval(self) -> None:
        with pytest.raises(ValueError, match="p_emergency"):
            ASHKVConfig(p_emergency=1.0)
        with pytest.raises(ValueError, match="p_emergency"):
            ASHKVConfig(p_emergency=0.0)

    def test_delta_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            ASHKVConfig(delta=-0.1)

    def test_frozen(self) -> None:
        c = ASHKVConfig()
        with pytest.raises(Exception):
            c.w_T = 0.5  # type: ignore[misc]

    def test_slots(self) -> None:
        c = ASHKVConfig()
        # frozen + slots rejects new attributes; the exact exception
        # type varies by Python version (AttributeError or TypeError).
        with pytest.raises((AttributeError, TypeError)):
            c.extra_field = 1  # type: ignore[attr-defined]

    def test_as_dict_roundtrip(self) -> None:
        c = ASHKVConfig(w_T=0.6, w_S=0.2, w_N=0.1, w_P=0.1)
        d = c.as_dict()
        assert d["w_T"] == 0.6
        assert len(d) == 8
        assert set(d.keys()) == {
            "w_T", "w_S", "w_N", "w_P",
            "theta_high", "theta_low", "delta", "p_emergency",
        }

    def test_custom_valid_config(self) -> None:
        c = ASHKVConfig(
            w_T=0.4, w_S=0.3, w_N=0.2, w_P=0.1,
            theta_high=0.8, theta_low=0.2,
            delta=0.05, p_emergency=0.9,
        )
        assert c.w_T == 0.4


# --- PageTable ---

class TestPageTable:
    def test_add_and_size(self) -> None:
        pt = PageTable(capacity=1024)
        pid = pt.add(
            layer_id=3, token_start=0, token_end=256,
            bf16_checksum=12345, creation_time=1000,
        )
        assert pid == 0
        assert pt.size == 1
        assert len(pt) == 1

    def test_add_increments_page_id(self) -> None:
        pt = PageTable(capacity=1024)
        p0 = pt.add(0, 0, 256, 1, 1000)
        p1 = pt.add(0, 256, 512, 2, 1000)
        assert p0 == 0
        assert p1 == 1

    def test_new_page_starts_at_bf16(self) -> None:
        pt = PageTable(capacity=1024)
        pt.add(0, 0, 256, 1, 1000)
        snap = pt.snapshot()
        assert snap["tier"][0] == int(Tier.BF16)

    def test_new_page_has_max_temporal(self) -> None:
        pt = PageTable(capacity=1024)
        pt.add(0, 0, 256, 1, 1000)
        snap = pt.snapshot()
        assert snap["T"][0] == 1.0

    def test_snapshot_is_copy(self) -> None:
        pt = PageTable(capacity=1024)
        pt.add(0, 0, 256, 1, 1000)
        snap = pt.snapshot()
        snap["T"] = 99.0
        # Original unchanged
        assert pt.snapshot()["T"][0] == 1.0

    def test_apply_tier_transition(self) -> None:
        pt = PageTable(capacity=1024)
        pid = pt.add(0, 0, 256, 1, 1000)
        ok = pt.apply_tier_transition(pid, Tier.FP8, new_checksum=999)
        assert ok is True
        snap = pt.snapshot()
        assert snap["tier"][0] == int(Tier.FP8)
        assert int(snap["current_checksum"][0]) == 999

    def test_pinned_page_cannot_migrate(self) -> None:
        pt = PageTable(capacity=1024)
        pid = pt.add(0, 0, 256, 1, 1000)
        pt.pin(pid)
        ok = pt.apply_tier_transition(pid, Tier.FP8, 999)
        assert ok is False
        snap = pt.snapshot()
        assert snap["tier"][0] == int(Tier.BF16)

    def test_pin_unpin_round_trip(self) -> None:
        pt = PageTable(capacity=1024)
        pid = pt.add(0, 0, 256, 1, 1000)
        pt.pin(pid)
        pt.pin(pid)
        assert int(pt.snapshot()["pin_count"][0]) == 2
        pt.unpin(pid)
        assert int(pt.snapshot()["pin_count"][0]) == 1
        pt.unpin(pid)
        assert int(pt.snapshot()["pin_count"][0]) == 0
        # Unpin below zero is a no-op
        pt.unpin(pid)
        assert int(pt.snapshot()["pin_count"][0]) == 0

    def test_remove_compacts(self) -> None:
        pt = PageTable(capacity=1024)
        p0 = pt.add(0, 0, 256, 1, 1000)
        p1 = pt.add(0, 256, 512, 2, 1000)
        p2 = pt.add(0, 512, 768, 3, 1000)
        assert pt.size == 3
        pt.remove(p1)
        assert pt.size == 2
        snap = pt.snapshot()
        ids = set(int(x) for x in snap["page_id"])
        assert p0 in ids
        assert p2 in ids
        assert p1 not in ids

    def test_remove_missing_is_noop(self) -> None:
        pt = PageTable(capacity=1024)
        pt.remove(999)  # no-op, no raise
        assert pt.size == 0

    def test_transition_missing_page(self) -> None:
        pt = PageTable(capacity=1024)
        ok = pt.apply_tier_transition(999, Tier.FP8, 0)
        assert ok is False

    def test_touch_updates_access(self) -> None:
        pt = PageTable(capacity=1024)
        pid = pt.add(0, 0, 256, 1, 1000)
        pt.touch(pid, 2000)
        snap = pt.snapshot()
        assert int(snap["last_access"][0]) == 2000
        assert int(snap["access_count"][0]) == 1

    def test_update_score_inputs_batch(self) -> None:
        pt = PageTable(capacity=1024)
        pids = [pt.add(i, 0, 256, i, 1000) for i in range(10)]
        pids_arr = np.array(pids, dtype=np.int64)
        T = np.full(10, 0.5, dtype=np.float32)
        S = np.full(10, 0.3, dtype=np.float32)
        N = np.full(10, 0.2, dtype=np.float32)
        P = np.full(10, 0.1, dtype=np.float32)
        pt.update_score_inputs(pids_arr, T, S, N, P)
        snap = pt.snapshot()
        # Use approx comparison: float32(0.3) != float64(0.3) exactly.
        assert np.allclose(snap["T"], 0.5, atol=1e-6)
        assert np.allclose(snap["S"], 0.3, atol=1e-6)
        assert np.allclose(snap["N"], 0.2, atol=1e-6)
        assert np.allclose(snap["P"], 0.1, atol=1e-6)

    def test_update_score_inputs_returns_silently_on_mismatch(self) -> None:
        pt = PageTable(capacity=1024)
        p0 = pt.add(0, 0, 256, 1, 1000)
        pids_arr = np.array([p0, 9999], dtype=np.int64)
        T = np.array([0.5], dtype=np.float32)  # mismatched length
        S = np.array([0.3, 0.99], dtype=np.float32)
        N = np.array([0.2, 0.99], dtype=np.float32)
        P = np.array([0.1, 0.99], dtype=np.float32)
        
        # Must return silently
        pt.update_score_inputs(pids_arr, T, S, N, P)
        
        # Verify page 0 was not updated
        snap = pt.snapshot()
        assert snap["T"][0] == 1.0  # Still at initial max temporal

    def test_update_score_inputs_skips_missing(self) -> None:
        pt = PageTable(capacity=1024)
        p0 = pt.add(0, 0, 256, 1, 1000)
        # Mix valid and invalid page ids
        pids_arr = np.array([p0, 9999], dtype=np.int64)
        T = np.array([0.5, 0.99], dtype=np.float32)
        S = np.array([0.3, 0.99], dtype=np.float32)
        N = np.array([0.2, 0.99], dtype=np.float32)
        P = np.array([0.1, 0.99], dtype=np.float32)
        pt.update_score_inputs(pids_arr, T, S, N, P)
        snap = pt.snapshot()
        assert float(snap["T"][0]) == 0.5  # valid updated
        # Invalid page was skipped, no crash

    def test_table_full_raises(self) -> None:
        pt = PageTable(capacity=2)
        pt.add(0, 0, 256, 1, 1000)
        pt.add(0, 256, 512, 2, 1000)
        with pytest.raises(RuntimeError, match="full"):
            pt.add(0, 512, 768, 3, 1000)

    def test_page_dtype_has_all_fields(self) -> None:
        expected = {
            "page_id", "layer_id", "token_start", "token_end",
            "tier", "pin_count", "last_access", "access_count",
            "creation_time", "bf16_checksum", "current_checksum",
            "T", "S", "N", "P",
        }
        assert set(PAGE_DTYPE.names) == expected


# --- Result types ---

class TestResults:
    def test_migration_result_is_frozen(self) -> None:
        r = MigrationResult(
            status=MigrationStatus.OK,
            page_id=0,
            from_tier=Tier.BF16,
            to_tier=Tier.FP8,
            duration_us=42,
        )
        with pytest.raises(Exception):
            r.status = MigrationStatus.FAILURE  # type: ignore[misc]

    def test_migration_result_defaults(self) -> None:
        r = MigrationResult(
            status=MigrationStatus.OK,
            page_id=0,
            from_tier=Tier.BF16,
            to_tier=Tier.FP8,
            duration_us=42,
        )
        assert r.error == ""

    def test_pressure_state_ordering(self) -> None:
        assert int(PressureState.NORMAL) < int(PressureState.ELEVATED)
        assert int(PressureState.ELEVATED) < int(PressureState.CRITICAL)
        assert int(PressureState.CRITICAL) < int(PressureState.SATURATED)

    def test_pressure_thresholds(self) -> None:
        assert PRESSURE_ELEVATED_THRESHOLD == 0.85
        assert PRESSURE_SATURATED_THRESHOLD == 0.99

    def test_pressure_report_is_frozen(self) -> None:
        r = PressureReport(pressure=0.5, state=PressureState.NORMAL)
        with pytest.raises(Exception):
            r.pressure = 0.9  # type: ignore[misc]


# --- Protocol structural checks ---

class TestProtocols:
    def test_codec_protocol_is_runtime_checkable(self) -> None:
        # A class that implements encode/decode/checksum should satisfy.
        class DummyCodec:
            def encode(self, source_bytes: bytes) -> bytes:
                return source_bytes

            def decode(self, target_bytes: bytes) -> bytes:
                return target_bytes

            def checksum(self, raw_bytes: bytes) -> int:
                return 0

        assert isinstance(DummyCodec(), Codec)

    def test_allocator_protocol_is_runtime_checkable(self) -> None:
        class DummyAllocator:
            def alloc(self, tier: Tier, size_bytes: int) -> int:
                return 0

            def free(self, handle: int) -> None:
                pass

            def read(self, handle: int) -> bytes:
                return b""

            def write(self, handle: int, data: bytes) -> None:
                pass

            def pressure(self) -> float:
                return 0.0

        assert isinstance(DummyAllocator(), Allocator)

    def test_incomplete_codec_not_accepted(self) -> None:
        class Incomplete:
            def encode(self, b: bytes) -> bytes:
                return b
            # missing decode and checksum

        assert not isinstance(Incomplete(), Codec)
