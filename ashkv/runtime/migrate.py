"""Hot-path migration engine.

The migration engine is the ONLY thing that moves pages between tiers.
It is single-path: one migrate() function, dispatched via a codec
table. There is no bf16_to_fp8(), no fp8_to_int4(). Those would grow
as N(N-1) special cases. This grows as N codecs.

Contract (Principle: never throw on the hot path):
- migrate() returns a MigrationResult. It never raises.
- The page is never in an invalid state. Either migration completed
  (tier swapped, old buffer freed) or it didn't (page unchanged).
- Codec failures => FAILURE, page unchanged.
- Checksum mismatch => CORRUPT, page unchanged (caller may quarantine).
- Pinned pages => SKIPPED, page unchanged.

The migrate() function does NOT decide what to migrate. The
controller does that. migrate() only executes a single transition
and reports the outcome.
"""
from __future__ import annotations

import time
from typing import Callable

from ..contracts.page import PageTable
from ..contracts.protocols import Allocator, Codec
from ..contracts.results import MigrationResult, MigrationStatus
from ..contracts.tiers import Tier


# Type alias: a codec table maps (from_tier, to_tier) -> Codec.
CodecTable = dict[tuple[int, int], Codec]


def migrate(
    page_id: int,
    target_tier: Tier,
    page_table: PageTable,
    allocator: Allocator,
    codec_table: CodecTable,
    page_handle_lookup: Callable[[int], int],
    page_size_lookup: Callable[[int], int],
    breaker_registry: Any = None,
) -> MigrationResult:
    """Migrate a single page from its current tier to target_tier.

    Parameters
    ----------
    page_id : int
        The page to migrate.
    target_tier : Tier
        Desired destination tier.
    page_table : PageTable
        Metadata store. Used to read current tier and commit transition.
    allocator : Allocator
        Memory allocator for raw bytes.
    codec_table : CodecTable
        Compiled codec dispatch dict, keyed by (from_tier, to_tier).
    page_handle_lookup : callable
        Returns the allocator handle for a page_id.
    page_size_lookup : callable
        Returns the BF16 byte size of a page_id.

    Returns
    -------
    MigrationResult
        Never raises.

    Safety ladder (per-page):
    1. Pin check           -> SKIPPED if pinned
    2. No-op check         -> SKIPPED if already at target
    3. Codec lookup        -> SKIPPED if no codec registered
    4. Read source         -> FAILURE if read fails
    5. Encode              -> FAILURE if codec raises
    6. Allocate target     -> FAILURE if alloc fails
    7. Write target        -> FAILURE if write fails
    8. Verify checksum     -> CORRUPT if mismatch (page unchanged)
    9. Commit transition   -> OK on success
    """
    t_start = time.perf_counter_ns()

    def _done(status: MigrationStatus, from_t: Tier, to_t: Tier, err: str = "") -> MigrationResult:
        t_end = time.perf_counter_ns()
        return MigrationResult(
            status=status,
            page_id=page_id,
            from_tier=from_t,
            to_tier=to_t,
            duration_us=int((t_end - t_start) // 1000),
            error=err,
        )

    # --- 1. Snapshot current state via direct dict lookup ---
    # O(1) page_id -> row, never raises.
    current_tier_raw = page_table.get_tier(page_id)
    if current_tier_raw < 0:
        return _done(MigrationStatus.SKIPPED, Tier.BF16, target_tier, "page not found")
    current_tier = Tier(current_tier_raw)

    # --- 2. Pin check ---
    pin_count = page_table.get_pin_count(page_id)
    if pin_count < 0:
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "page not found")
    if pin_count > 0:
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "pinned")

    # --- 3. No-op check ---
    if current_tier == target_tier:
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "already at target")

    # --- 4. Codec lookup ---
    key = (int(current_tier), int(target_tier))
    codec = codec_table.get(key)
    if codec is None:
        # No direct codec. Try via BF16 hop? For now: SKIPPED.
        # The compiler is responsible for ensuring the codec table
        # has all required (from, to) pairs. If a pair is missing,
        # that's a compile-time bug, not a runtime failure.
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "no codec")

    codec_name = f"{current_tier.name}_to_{target_tier.name}"
    if breaker_registry is not None and not breaker_registry.is_codec_available(codec_name):
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "breaker tripped")

    # --- 5. Read source bytes ---
    src_handle = _safe_call(page_handle_lookup, page_id, default=-1)
    if src_handle < 0:
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, "no handle")
    src_bytes = _safe_read(allocator, src_handle)
    if src_bytes is None:
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, "read failed")

    # --- 6. Encode ---
    try:
        target_bytes = codec.encode(src_bytes)
    except Exception as e:
        if breaker_registry is not None:
            breaker_registry.record_codec_failure(codec_name)
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, f"encode: {e}")

    # --- 7. Allocate target buffer ---
    target_handle = _safe_alloc(allocator, target_tier, len(target_bytes))
    if target_handle < 0:
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, "alloc failed")

    # --- 8. Write target ---
    if not _safe_write(allocator, target_handle, target_bytes):
        _safe_free(allocator, target_handle)
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, "write failed")

    # --- 9. Verify checksum (round-trip) ---
    try:
        reconstructed = codec.decode(target_bytes)
    except Exception as e:
        _safe_free(allocator, target_handle)
        if breaker_registry is not None:
            breaker_registry.record_codec_failure(codec_name)
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, f"verify decode: {e}")

    try:
        new_checksum = codec.checksum(reconstructed)
    except Exception as e:
        _safe_free(allocator, target_handle)
        if breaker_registry is not None:
            breaker_registry.record_codec_failure(codec_name)
        return _done(MigrationStatus.FAILURE, current_tier, target_tier, f"verify checksum: {e}")

    bf16_checksum = page_table.get_bf16_checksum(page_id)
    if new_checksum != bf16_checksum:
        # CORRUPT: the round-trip did not reproduce the original.
        # Quarantine by NOT committing. The caller decides what to do
        # (typically: trip the codec's circuit breaker, fall back to BF16).
        _safe_free(allocator, target_handle)
        if breaker_registry is not None:
            breaker_registry.record_codec_failure(codec_name)
        return _done(MigrationStatus.CORRUPT, current_tier, target_tier, "checksum mismatch")

    # --- 10. Commit transition ---
    committed = page_table.apply_tier_transition(page_id, target_tier, new_checksum)
    if not committed:
        # Someone pinned it between our snapshot and commit. Roll back.
        _safe_free(allocator, target_handle)
        return _done(MigrationStatus.SKIPPED, current_tier, target_tier, "commit rejected (pin race)")

    # --- 11. Free source buffer ---
    _safe_free(allocator, src_handle)

    if breaker_registry is not None:
        breaker_registry.record_codec_success(codec_name)

    return _done(MigrationStatus.OK, current_tier, target_tier)


# --- Helpers: every allocator operation is wrapped to never throw ---

def _safe_call(fn: Callable, arg, default):
    try:
        return fn(arg)
    except Exception:
        return default


def _safe_read(allocator: Allocator, handle: int) -> bytes | None:
    try:
        return allocator.read(handle)
    except Exception:
        return None


def _safe_write(allocator: Allocator, handle: int, data: bytes) -> bool:
    try:
        allocator.write(handle, data)
        return True
    except Exception:
        return False


def _safe_alloc(allocator: Allocator, tier: Tier, size: int) -> int:
    try:
        return allocator.alloc(tier, size)
    except Exception:
        return -1


def _safe_free(allocator: Allocator, handle: int) -> None:
    try:
        allocator.free(handle)
    except Exception:
        pass
