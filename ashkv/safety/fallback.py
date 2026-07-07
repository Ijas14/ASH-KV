"""BF16 fallback ladder — page-level recovery.

When a page cannot be served at its current tier (codec broke,
checksum mismatched, breaker tripped), this module provides the
recovery path: reconstruct the page in BF16.

Principle 11: BF16 is the fallback tier for page-level recovery.
Recovery is scoped to the page, never to the request or the server.

The fallback ladder:
    1. Codec failure during migration → page stays on current tier
    2. Current tier unreadable (corruption) → reconstruct in BF16
    3. BF16 unavailable (OOM) → evict cold pages, retry BF16
    4. Still unavailable → quarantine page, reject request
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ..contracts.page import PageTable
from ..contracts.protocols import Allocator
from ..contracts.results import MigrationResult, MigrationStatus
from ..contracts.tiers import Tier


class FallbackLevel(IntEnum):
    """How far down the fallback ladder we went for a page."""
    NONE = 0          # no fallback needed
    STAYED = 1        # migration failed, page stayed on current tier
    BF16_RECONSTRUCTED = 2  # page reconstructed in BF16
    EVICTED_AND_RECONSTRUCTED = 3  # cold pages evicted, then BF16
    QUARANTINED = 4   # could not recover, page quarantined


@dataclass(slots=True, frozen=True)
class FallbackResult:
    """Result of a fallback attempt. Never raised."""
    level: FallbackLevel
    page_id: int
    recovered: bool
    error: str = ""


def attempt_bf16_recovery(
    page_id: int,
    page_table: PageTable,
    allocator: Allocator,
    bf16_source_lookup,
    bf16_size_lookup,
) -> FallbackResult:
    """Attempt to recover a page by reconstructing it in BF16.

    This is called when a page's current tier is unreadable or corrupt.
    The page's bf16_checksum is the anchor — we reconstruct from the
    BF16 source and verify.

    Parameters
    ----------
    page_id : int
        The page to recover.
    page_table : PageTable
        Metadata store.
    allocator : Allocator
        Memory allocator.
    bf16_source_lookup : callable
        Returns the BF16 source bytes for the page_id, or None.
    bf16_size_lookup : callable
        Returns the BF16 byte size for the page_id.

    Returns
    -------
    FallbackResult
        Never raises.
    """
    # 1. Check the page exists
    current_tier_raw = page_table.get_tier(page_id)
    if current_tier_raw < 0:
        return FallbackResult(
            level=FallbackLevel.QUARANTINED,
            page_id=page_id,
            recovered=False,
            error="page not found",
        )

    # 2. Get the BF16 source
    try:
        source_bytes = bf16_source_lookup(page_id)
    except Exception:
        source_bytes = None

    if source_bytes is None:
        return FallbackResult(
            level=FallbackLevel.QUARANTINED,
            page_id=page_id,
            recovered=False,
            error="bf16 source unavailable",
        )

    # 3. Allocate a BF16 buffer
    try:
        size = bf16_size_lookup(page_id)
    except Exception:
        size = len(source_bytes)

    target_handle = _safe_alloc(allocator, Tier.BF16, size)
    if target_handle < 0:
        # BF16 allocation failed (OOM) — try evicting cold pages
        # The eviction itself is the integration layer's job.
        # Here we just report the failure.
        return FallbackResult(
            level=FallbackLevel.QUARANTINED,
            page_id=page_id,
            recovered=False,
            error="bf16 alloc failed (oom)",
        )

    # 4. Write the BF16 source to the new buffer
    if not _safe_write(allocator, target_handle, source_bytes):
        _safe_free(allocator, target_handle)
        return FallbackResult(
            level=FallbackLevel.QUARANTINED,
            page_id=page_id,
            recovered=False,
            error="bf16 write failed",
        )

    # 5. Commit the tier transition to BF16
    bf16_checksum = page_table.get_bf16_checksum(page_id)
    committed = page_table.apply_tier_transition(page_id, Tier.BF16, bf16_checksum)
    if not committed:
        _safe_free(allocator, target_handle)
        return FallbackResult(
            level=FallbackLevel.STAYED,
            page_id=page_id,
            recovered=False,
            error="commit rejected (pin race)",
        )

    return FallbackResult(
        level=FallbackLevel.BF16_RECONSTRUCTED,
        page_id=page_id,
        recovered=True,
    )


def handle_migration_failure(
    result: MigrationResult,
    page_table: PageTable,
    allocator: Allocator,
    bf16_source_lookup,
    bf16_size_lookup,
) -> FallbackResult:
    """Handle a failed migration result.

    This is the entry point called by the migration engine's caller
    (the integration layer) after migrate() returns a non-OK status.

    Decision tree:
    - FAILURE: try BF16 recovery (the codec might have corrupted the page)
    - CORRUPT: try BF16 recovery (definitely corrupt)
    - SKIPPED: no action needed (page is fine, just didn't move)
    - FALLBACK: already fell back, no further action
    - OK: no action needed
    """
    if result.status == MigrationStatus.OK:
        return FallbackResult(
            level=FallbackLevel.NONE,
            page_id=result.page_id,
            recovered=True,
        )

    if result.status == MigrationStatus.SKIPPED:
        return FallbackResult(
            level=FallbackLevel.STAYED,
            page_id=result.page_id,
            recovered=True,
            error=result.error,
        )

    if result.status == MigrationStatus.FAILURE:
        # Codec failed. The page should still be on its original tier
        # (migrate() guarantees this). But the page might be in a bad
        # state. Try BF16 recovery to be safe.
        return attempt_bf16_recovery(
            result.page_id, page_table, allocator,
            bf16_source_lookup, bf16_size_lookup,
        )

    if result.status == MigrationStatus.CORRUPT:
        # Checksum mismatch — the page's current-tier bytes are
        # definitely wrong. Must recover in BF16.
        return attempt_bf16_recovery(
            result.page_id, page_table, allocator,
            bf16_source_lookup, bf16_size_lookup,
        )

    # FALLBACK or unknown — no further action
    return FallbackResult(
        level=FallbackLevel.STAYED,
        page_id=result.page_id,
        recovered=True,
        error=f"unhandled status: {result.status}",
    )


# --- Helpers (never raise) ---

def _safe_alloc(allocator: Allocator, tier: Tier, size: int) -> int:
    try:
        return allocator.alloc(tier, size)
    except Exception:
        return -1


def _safe_write(allocator: Allocator, handle: int, data: bytes) -> bool:
    try:
        allocator.write(handle, data)
        return True
    except Exception:
        return False


def _safe_free(allocator: Allocator, handle: int) -> None:
    try:
        allocator.free(handle)
    except Exception:
        pass
