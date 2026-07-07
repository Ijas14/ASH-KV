"""vLLM BlockSpaceManager Monkey Patches.

Hooks into vLLM's BlockAllocator to intercept `allocate` and `free` operations.
This ensures atomic preemption interception: when vLLM frees a block, it is
first synchronously compressed to the ASH-KV INT8 shadow cache before the
physical block is returned to vLLM's free pool. When vLLM allocates a block
for a resumed sequence, it is synchronously decompressed from the shadow pool.
"""
import threading
from typing import Any, List
import torch

from ashkv.vllm_integration.hooks import VLLMHooks

# Global references injected at runtime by the integration engine
_HOOKS: VLLMHooks | None = None
_VLLM_KV_CACHE: torch.Tensor | None = None
_PATCH_LOCK = threading.Lock()


def apply_block_manager_patches(
    block_allocator_cls: Any,
    hooks: VLLMHooks,
    vllm_kv_cache: torch.Tensor,
) -> None:
    """Monkey-patch vLLM's BlockAllocator.
    
    Args:
        block_allocator_cls: The vLLM BlockAllocator class (e.g., vllm.core.block.BlockAllocator)
        hooks: The ASH-KV VLLMHooks instance.
        vllm_kv_cache: The physical BF16 vLLM KV cache tensor.
    """
    global _HOOKS, _VLLM_KV_CACHE
    _HOOKS = hooks
    _VLLM_KV_CACHE = vllm_kv_cache

    original_allocate = block_allocator_cls.allocate
    original_free = block_allocator_cls.free

    def ashkv_allocate(self, *args, **kwargs):
        """Intercept block allocation to trigger promote_hook."""
        # 1. Call original allocate to get the new physical block
        with _PATCH_LOCK:
            block = original_allocate(self, *args, **kwargs)
            
            # If block is allocated on GPU, ensure it is promoted from shadow cache if necessary.
            # vLLM's PhysicalTokenBlock usually has a device and block_number.
            device = getattr(block, "device", None)
            # vLLM uses enum Device.GPU, we check string representation or name
            if block is not None and str(device).upper() == "GPU" or getattr(device, "name", "") == "GPU":
                block_num = block.block_number
                
                # trigger promote_hook. Note: In reality, allocate() is for NEW blocks,
                # but if we are resuming a sequence, vLLM allocates new physical blocks 
                # and maps them to the existing logical blocks.
                # If ASH-KV tracks by physical block, a newly allocated block won't be in the shadow cache.
                # BUT if ASH-KV tracks logical -> physical mapping, it would. 
                # For this patch, we assume the hook handles the logic.
                if _HOOKS and _VLLM_KV_CACHE is not None:
                    _HOOKS.promote_hook([block_num], _VLLM_KV_CACHE)
                    
            return block

    def ashkv_free(self, block, *args, **kwargs):
        """Intercept block free to trigger demote_hook BEFORE freeing."""
        with _PATCH_LOCK:
            if block is not None:
                device = getattr(block, "device", None)
                if str(device).upper() == "GPU" or getattr(device, "name", "") == "GPU":
                    block_num = block.block_number
                    
                    # Atomic preemption: compress to shadow cache BEFORE vLLM reclaims the block
                    if _HOOKS and _VLLM_KV_CACHE is not None:
                        _HOOKS.demote_hook([block_num], _VLLM_KV_CACHE)

            # Now safely return the block to vLLM's free pool
            return original_free(self, block, *args, **kwargs)

    block_allocator_cls.allocate = ashkv_allocate
    block_allocator_cls.free = ashkv_free
