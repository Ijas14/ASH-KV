"""Page and PageTable.

PageTable is intentionally minimal: metadata storage, lookup, and
state transitions (tier changes). It contains NO policy.

The controller does not read from PageTable directly during the hot
path. It reads arrays via PageTable.snapshot() and writes back via
apply_tier_transition(). All vectorized score/controller logic runs
on the snapshot.

Invariants:
- Tier transitions only happen via apply_tier_transition().
- pin_count > 0 implies the page cannot be migrated.
- snapshot() returns a copy; mutating it does not affect the table.
- All mutator methods are non-raising on the hot path. Cold-path
  errors (table full, etc.) raise RuntimeError.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiers import Tier


# Structured dtype for the numpy-backed page table.
# This is the hot-path representation: vectorized score/controller
# operate directly on arrays of this dtype.
#
# Field ordering is stable. Adding a field appends to the end; existing
# fields keep their position.
PAGE_DTYPE = np.dtype([
    ("page_id", np.int64),
    ("layer_id", np.int32),
    ("token_start", np.int32),
    ("token_end", np.int32),
    ("tier", np.int8),
    ("pin_count", np.int32),
    ("last_access", np.int64),
    ("access_count", np.int64),
    ("creation_time", np.int64),
    ("bf16_checksum", np.uint64),     # immutable source checksum
    ("current_checksum", np.uint64),  # checksum of current-tier bytes
    ("T", np.float32),                # temporal score input
    ("S", np.float32),                # saliency score input
    ("N", np.float32),                # novelty score input
    ("P", np.float32),                # prefix affinity score input
])


@dataclass(slots=True, frozen=True)
class PageHandle:
    """Stable, immutable reference to a page.

    Use page_id to look up the row in PageTable. Carrying the handle
    instead of the row avoids stale references after compaction.
    """

    page_id: int


class PageTable:
    """Numpy-backed page metadata store.

    The hot path uses snapshot() + apply_tier_transition(). Cold path
    uses add/remove/pin/unpin/touch/update_score_inputs.

    Thread-safety: NOT thread-safe. Callers must hold the GIL and
    serialize hot-path operations (which they will, since the
    controller runs single-threaded per decode step).
    """

    __slots__ = ("_arr", "_next_id", "_index", "_capacity", "_size")

    def __init__(self, capacity: int = 1 << 20) -> None:
        self._capacity = int(capacity)
        self._arr = np.zeros(self._capacity, dtype=PAGE_DTYPE)
        self._arr["page_id"] = -1  # mark all rows as empty
        self._next_id = 0
        self._size = 0
        self._index: dict[int, int] = {}  # page_id -> row index

    # --- Cold path: lifecycle ---

    def add(
        self,
        layer_id: int,
        token_start: int,
        token_end: int,
        bf16_checksum: int,
        creation_time: int,
    ) -> int:
        """Register a new page. Returns page_id.

        New pages start at BF16 with T=1.0 (newest). Cold path only;
        raises RuntimeError if the table is full.
        """
        if self._size >= self._capacity:
            raise RuntimeError(
                f"PageTable full (capacity={self._capacity})"
            )
        idx = self._size
        page_id = self._next_id
        self._next_id += 1
        self._size += 1

        row = self._arr[idx]
        row["page_id"] = page_id
        row["layer_id"] = layer_id
        row["token_start"] = token_start
        row["token_end"] = token_end
        row["tier"] = int(Tier.BF16)
        row["pin_count"] = 0
        row["last_access"] = creation_time
        row["access_count"] = 0
        row["creation_time"] = creation_time
        row["bf16_checksum"] = bf16_checksum
        row["current_checksum"] = bf16_checksum
        row["T"] = 1.0
        row["S"] = 0.0
        row["N"] = 0.0
        row["P"] = 0.0

        self._index[page_id] = idx
        return page_id

    def remove(self, page_id: int) -> None:
        """Remove a page. Compacts by swapping with the last row.

        Never raises. Missing page_id is a no-op.
        """
        idx = self._index.pop(page_id, None)
        if idx is None:
            return
        last = self._size - 1
        if idx != last:
            self._arr[idx] = self._arr[last]
            moved_id = int(self._arr[idx]["page_id"])
            self._index[moved_id] = idx
        self._size -= 1

    # --- Hot path: read ---

    def snapshot(self) -> np.ndarray:
        """Return a copy of active rows for vectorized score/controller.

        The returned array is a COPY. Mutating it does not affect the
        table. Score and controller run on this copy. The controller
        then calls apply_tier_transition() to commit decisions.
        """
        return self._arr[: self._size].copy()

    def find(self, page_id: int) -> int:
        """Return the row index for page_id, or -1 if not found.

        Hot-path-safe: O(1) dict lookup, never raises.
        """
        return self._index.get(int(page_id), -1)

    def get_tier(self, page_id: int) -> int:
        """Return the current tier of a page, or -1 if not found.

        Returns the raw int8 tier value (cast to int). Never raises.
        """
        idx = self._index.get(int(page_id), -1)
        if idx < 0:
            return -1
        return int(self._arr[idx]["tier"])

    def get_pin_count(self, page_id: int) -> int:
        """Return the pin count of a page, or -1 if not found."""
        idx = self._index.get(int(page_id), -1)
        if idx < 0:
            return -1
        return int(self._arr[idx]["pin_count"])

    def get_bf16_checksum(self, page_id: int) -> int:
        """Return the immutable BF16 checksum of a page, or 0 if not found."""
        idx = self._index.get(int(page_id), -1)
        if idx < 0:
            return 0
        return int(self._arr[idx]["bf16_checksum"])

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    # --- Hot path: state transitions (no policy) ---

    def apply_tier_transition(
        self,
        page_id: int,
        new_tier: Tier,
        new_checksum: int,
    ) -> bool:
        """Atomically transition a page to a new tier.

        Returns True on success, False if rejected (pinned or missing).
        NEVER raises. This is the only method that mutates the tier
        field on the hot path.
        """
        idx = self._index.get(page_id)
        if idx is None:
            return False
        if self._arr[idx]["pin_count"] > 0:
            return False
        self._arr[idx]["tier"] = int(new_tier)
        self._arr[idx]["current_checksum"] = new_checksum
        return True

    # --- Cold path: metadata updates ---

    def touch(self, page_id: int, time: int) -> None:
        """Mark a page as accessed. No-op if missing."""
        idx = self._index.get(page_id)
        if idx is None:
            return
        self._arr[idx]["last_access"] = time
        self._arr[idx]["access_count"] += 1

    def pin(self, page_id: int) -> None:
        """Increment pin count. No-op if missing."""
        idx = self._index.get(page_id)
        if idx is None:
            return
        self._arr[idx]["pin_count"] += 1

    def unpin(self, page_id: int) -> None:
        """Decrement pin count. No-op if missing or already 0."""
        idx = self._index.get(page_id)
        if idx is None:
            return
        cnt = int(self._arr[idx]["pin_count"])
        if cnt > 0:
            self._arr[idx]["pin_count"] = cnt - 1

    def update_score_inputs(
        self,
        page_ids: np.ndarray,
        T: np.ndarray,
        S: np.ndarray,
        N: np.ndarray,
        P: np.ndarray,
    ) -> None:
        """Batch update score inputs. Vectorized.

        All arrays must be the same length. Pages not in the table
        are silently skipped. Cold path — called between decode steps
        by the integration layer.
        """
        n = len(page_ids)
        if not (len(T) == len(S) == len(N) == len(P) == n):
            return

        # Resolve page_ids to row indices. Missing -> -1.
        idx = np.empty(n, dtype=np.int64)
        index_get = self._index.get
        for i in range(n):
            idx[i] = index_get(int(page_ids[i]), -1)

        valid = idx >= 0
        if not valid.any():
            return
        vidx = idx[valid]
        # NOTE: must assign to self._arr["T"][vidx] (view) rather than
        # self._arr[vidx]["T"] (copy). NumPy structured arrays return a
        # copy from the second form, so the assignment would silently
        # not stick.
        self._arr["T"][vidx] = T[valid]
        self._arr["S"][vidx] = S[valid]
        self._arr["N"][vidx] = N[valid]
        self._arr["P"][vidx] = P[valid]
