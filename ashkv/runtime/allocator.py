"""Mock allocator for testing the integration without real GPU memory.

Implements the Allocator protocol from contracts.protocols. Uses a
Python dict to simulate GPU memory. Useful for:
- Testing the migration engine end-to-end
- Testing the safety layer (fallback, circuit breakers)
- Testing the compiler and runtime without SGLang

NOT for production. The real allocator wraps SGLang's block manager.
"""
from __future__ import annotations

from typing import Dict

from ..contracts.tiers import Tier


class MockAllocator:
    """In-memory mock allocator for testing.

    Implements the full Allocator protocol. Never raises (returns -1
    or empty bytes on failure, as per the contract).

    Simulates pressure via a configurable budget. When used bytes
    exceed the budget, pressure() approaches 1.0.
    """

    __slots__ = (
        "_buffers",
        "_next_handle",
        "_budget",
        "_used",
        "_per_tier_used",
    )

    def __init__(self, budget_bytes: int = 10 * 1024 * 1024 * 1024) -> None:
        """Initialize with a memory budget (default 10 GB)."""
        self._buffers: Dict[int, bytes] = {}
        self._next_handle: int = 1
        self._budget: int = budget_bytes
        self._used: int = 0
        self._per_tier_used: Dict[int, int] = {}

    def alloc(self, tier: Tier, size_bytes: int) -> int:
        """Allocate a buffer. Returns handle or -1 on failure."""
        if size_bytes <= 0:
            return -1

        # Simulate OOM: if allocation would exceed budget, fail
        if self._used + size_bytes > self._budget:
            return -1

        handle = self._next_handle
        self._next_handle += 1
        self._buffers[handle] = b"\x00" * size_bytes
        self._used += size_bytes
        self._per_tier_used[int(tier)] = (
            self._per_tier_used.get(int(tier), 0) + size_bytes
        )
        return handle

    def free(self, handle: int) -> None:
        """Free a buffer. No-op on invalid handle."""
        buf = self._buffers.pop(handle, None)
        if buf is not None:
            self._used -= len(buf)
            # Note: we don't track which tier the buffer was, so
            # _per_tier_used may drift. This is acceptable for a mock.

    def read(self, handle: int) -> bytes:
        """Read buffer contents. Returns b'' on invalid handle."""
        return self._buffers.get(handle, b"")

    def write(self, handle: int, data: bytes) -> None:
        """Write to a buffer. No-op on invalid handle."""
        if handle in self._buffers:
            old = self._buffers[handle]
            self._used -= len(old)
            self._buffers[handle] = data
            self._used += len(data)

    def pressure(self) -> float:
        """Return memory pressure as a scalar in [0, 1]."""
        if self._budget == 0:
            return 1.0
        return min(1.0, self._used / self._budget)

    # --- Test utilities (not part of the Allocator protocol) ---

    def reset(self) -> None:
        """Reset the allocator to empty state. For testing."""
        self._buffers.clear()
        self._next_handle = 1
        self._used = 0
        self._per_tier_used.clear()

    @property
    def used_bytes(self) -> int:
        """Current bytes in use."""
        return self._used

    @property
    def budget_bytes(self) -> int:
        """Total budget."""
        return self._budget

    def set_budget(self, budget: int) -> None:
        """Adjust the budget. For testing pressure scenarios."""
        self._budget = budget

    def preload(self, handle: int, data: bytes, tier: Tier = Tier.BF16) -> None:
        """Pre-load data at a specific handle. For test setup."""
        self._buffers[handle] = data
        self._used += len(data)
        self._per_tier_used[int(tier)] = (
            self._per_tier_used.get(int(tier), 0) + len(data)
        )
        if handle >= self._next_handle:
            self._next_handle = handle + 1
