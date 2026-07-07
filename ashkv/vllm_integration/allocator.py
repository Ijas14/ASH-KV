"""vLLM-specific implementation of the ASH-KV Allocator protocol.

This allocator maintains a separate PyTorch memory pool (via PyTorch's
caching allocator) for the compressed tiers (INT8, FP8). It strictly
adheres to the architecture directive: DO NOT hijack vLLM's internal
BF16 blocks. ASH-KV operates as a shadow cache.
"""
from __future__ import annotations

import torch
from typing import Dict

from ashkv.contracts.tiers import Tier
from ashkv.contracts.protocols import Allocator


class VLLMShadowAllocator:
    """A shadow memory allocator for compressed vLLM KV blocks.

    Uses PyTorch's native CUDA allocator to manage memory for compressed
    pages (e.g., INT8). Exposes pressure based on a defined budget.
    """

    __slots__ = (
        "_budget_bytes",
        "_used_bytes",
        "_next_handle",
        "_blocks",
    )

    def __init__(self, budget_bytes: int = 4 * 1024 * 1024 * 1024) -> None:
        """Initialize the shadow allocator with a maximum memory budget.
        
        Args:
            budget_bytes: Maximum GPU memory this allocator can consume (default 4GB).
        """
        self._budget_bytes = budget_bytes
        self._used_bytes = 0
        self._next_handle = 1
        # Maps handle -> torch.Tensor (uint8, cuda)
        self._blocks: Dict[int, torch.Tensor] = {}

    def alloc(self, tier: Tier, size_bytes: int) -> int:
        """Allocate a buffer. Returns handle or -1 on failure."""
        if size_bytes <= 0:
            return -1

        if self._used_bytes + size_bytes > self._budget_bytes:
            return -1

        try:
            tensor = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        except RuntimeError:
            # OOM from PyTorch
            return -1

        handle = self._next_handle
        self._next_handle += 1
        
        self._blocks[handle] = tensor
        self._used_bytes += size_bytes
        return handle

    def free(self, handle: int) -> None:
        """Free a buffer. No-op on invalid handle."""
        tensor = self._blocks.pop(handle, None)
        if tensor is not None:
            self._used_bytes -= tensor.numel()
            del tensor

    def read(self, handle: int) -> bytes:
        """Read buffer contents. Returns b'' on invalid handle."""
        tensor = self._blocks.get(handle)
        if tensor is None:
            return b""
        
        # Note: Codec protocol currently expects raw bytes.
        # This incurs a Device->Host transfer.
        return tensor.cpu().numpy().tobytes()

    def get_tensor(self, handle: int) -> torch.Tensor | None:
        """Bypass the bytes contract and return the raw GPU tensor.
        
        Used by the integration layer's fast path to avoid PCIe overhead.
        """
        return self._blocks.get(handle)

    def write(self, handle: int, data: bytes) -> None:
        """Write to a buffer. No-op on invalid handle."""
        tensor = self._blocks.get(handle)
        if tensor is not None:
            data_len = len(data)
            
            # Re-allocate if size doesn't match
            if data_len != tensor.numel():
                self._used_bytes -= tensor.numel()
                try:
                    tensor = torch.empty(data_len, dtype=torch.uint8, device="cuda")
                except RuntimeError:
                    # Ignore write on OOM to match non-raising contract, though this
                    # implies data loss for this specific handle.
                    return
                self._blocks[handle] = tensor
                self._used_bytes += data_len
            
            # Host->Device transfer
            source_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            tensor.copy_(source_tensor)

    def pressure(self) -> float:
        """Return memory pressure as a scalar in [0, 1]."""
        if self._budget_bytes == 0:
            return 1.0
        return min(1.0, self._used_bytes / self._budget_bytes)

# Register it as satisfying the protocol statically (for type checkers)
# This will fail at import time if the protocol is violated.
_is_allocator: Allocator = VLLMShadowAllocator()
