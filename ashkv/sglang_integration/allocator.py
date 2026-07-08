import torch
import gc

class SGLangShadowAllocator:
    """A standalone PyTorch memory pool for the ASH-KV INT8 shadow cache.
    
    Operates strictly outside of SGLang's internal TokenToKVPool. This ensures
    that SGLang's RadixAttention kernel continues to operate exclusively on
    native `bfloat16` data without memory corruption.
    """
    
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.allocated_bytes = 0
        self.pool = {}
        self.next_handle = 1
        
    def alloc(self, tier, size_bytes: int) -> int:
        """Allocate a contiguous GPU byte buffer for a compressed node.
        
        Args:
            tier: The target compression tier (e.g. Tier.INT8)
            size_bytes: Required size in bytes
            
        Returns:
            A unique integer handle for the allocated buffer, or -1 if OOM.
        """
        if self.allocated_bytes + size_bytes > self.max_bytes:
            # Simple emergency cleanup attempt
            torch.cuda.empty_cache()
            gc.collect()
            if self.allocated_bytes + size_bytes > self.max_bytes:
                return -1
                
        # Allocate flat byte tensor on GPU
        # We use uint8 internally for raw byte storage
        tensor = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        
        handle = self.next_handle
        self.next_handle += 1
        
        self.pool[handle] = tensor
        self.allocated_bytes += size_bytes
        return handle
        
    def free(self, handle: int) -> None:
        """Free the shadow buffer associated with the handle."""
        if handle in self.pool:
            tensor = self.pool.pop(handle)
            self.allocated_bytes -= tensor.numel()
            del tensor
            
    def get_tensor(self, handle: int) -> torch.Tensor | None:
        """Retrieve the raw GPU tensor for direct Triton kernel operations."""
        return self.pool.get(handle)
        
    def get_utilization(self) -> float:
        """Return the current utilization ratio (0.0 to 1.0)."""
        if self.max_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.max_bytes

    def read(self, handle: int) -> bytes:
        """Read bytes from the shadow buffer."""
        tensor = self.pool.get(handle)
        if tensor is None:
            return b""
        # Ensure we return cpu bytes
        return tensor.cpu().numpy().tobytes()

    def write(self, handle: int, data: bytes) -> None:
        """Write bytes to the shadow buffer."""
        tensor = self.pool.get(handle)
        if tensor is not None:
            # Create a tensor from bytes and copy it over
            import numpy as np
            byte_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).to("cuda")
            tensor.copy_(byte_tensor)

    def pressure(self) -> float:
        """Return the allocator pressure."""
        return self.get_utilization()
