"""SGLang Integration Hooks (Node-Level).

Unlike vLLM which manages blocks independently, SGLang manages KV cache
via a Radix Tree. These hooks operate directly on `RadixNode` instances.
When a node is evicted by SGLang, it is demoted to INT8 instead of being deleted.
When a node is hit by a prefix match, it is promoted back to BF16.
"""
from __future__ import annotations

import torch
from typing import Any

from ashkv.contracts.tiers import Tier
from ashkv.contracts.page import PageTable
from ashkv.compiler.registry import codec_registry
from ashkv.sglang_integration.allocator import SGLangShadowAllocator
from ashkv.codecs.int8 import _get_kernels


class SGLangHooks:
    """Manages the boundary between SGLang's RadixCache and ASH-KV."""

    def __init__(
        self,
        page_table: PageTable,
        shadow_allocator: SGLangShadowAllocator,
        codec_name: str = "int8_default",
    ):
        self.page_table = page_table
        self.allocator = shadow_allocator
        self.codec = codec_registry.get_codec(codec_name)
        if self.codec is None:
            raise ValueError(f"Codec {codec_name} not found in registry.")

    def promote_hook(self, node: Any, sglang_kv_cache: torch.Tensor, memory_pool: Any) -> None:
        """Pre-decode: Restore a compressed RadixNode to BF16.
        
        Args:
            node: SGLang RadixNode instance.
            sglang_kv_cache: SGLang's underlying bfloat16 KV cache tensor.
            memory_pool: SGLang's TokenToKVPool for allocating fresh slots.
        """
        # 1. Check if node is compressed
        if not getattr(node, "is_compressed", False) or not hasattr(node, "shadow_handle"):
            return

        handle = node.shadow_handle
        shadow_tensor = self.allocator.get_tensor(handle)
        if shadow_tensor is None:
            # Shadow cache miss / corrupted state
            node.is_compressed = False
            return

        # 2. Allocate fresh physical slots from SGLang's memory pool
        # Depending on the SGLang version, this might be pool.allocate(len(node))
        # We assume the caller or this hook handles mapping `node` to a number of tokens.
        # SGLang nodes typically have a `length` or `num_tokens` attribute.
        num_tokens = getattr(node, "length", 0)
        if num_tokens == 0 and hasattr(node, "value"):
             num_tokens = len(node.value)
             
        # Request slots from SGLang
        new_indices = memory_pool.allocate(num_tokens)
        
        # 3. Decode INT8 -> BF16
        hidden_dim = sglang_kv_cache.shape[-1]
        scale_bytes = num_tokens * 2
        
        scale_factors = shadow_tensor[:scale_bytes].view(torch.bfloat16)
        int8_vals = shadow_tensor[scale_bytes:].view(torch.int8).view(num_tokens, hidden_dim)
        
        bf16_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
        
        _, decode_kernel = _get_kernels()
        grid = (num_tokens,)
        decode_kernel[grid](
            int8_vals,
            scale_factors,
            bf16_vals,
            num_tokens,
            hidden_dim,
        )
        
        # 4. Write back to SGLang's fresh slots
        # sglang_kv_cache shape is [num_total_slots, hidden_dim]
        # new_indices is a tensor of slot indices
        sglang_kv_cache[new_indices] = bf16_vals
        
        # 5. Update Node state
        node.kv_indices = new_indices
        node.is_compressed = False
        del node.shadow_handle
        
        # 6. Free shadow memory
        self.allocator.free(handle)

    def demote_hook(self, node: Any, sglang_kv_cache: torch.Tensor, memory_pool: Any) -> bool:
        """Post-decode: Compress a RadixNode to INT8 instead of deleting it.
        
        Args:
            node: SGLang RadixNode instance about to be evicted.
            sglang_kv_cache: SGLang's underlying bfloat16 KV cache tensor.
            memory_pool: SGLang's TokenToKVPool for returning slots.
            
        Returns:
            bool: True if successfully compressed and intercepted eviction, False otherwise.
        """
        if getattr(node, "is_compressed", False):
            return True # Already compressed

        if not hasattr(node, "kv_indices") or node.kv_indices is None:
            return False
            
        kv_indices = node.kv_indices
        num_tokens = len(kv_indices)
        hidden_dim = sglang_kv_cache.shape[-1]
        
        # 1. Gather BF16 values from scattered indices
        # arr_gpu shape: [num_tokens, hidden_dim]
        arr_gpu = sglang_kv_cache[kv_indices]
        
        # 2. Encode BF16 -> INT8
        scale_factors = torch.empty((num_tokens,), dtype=torch.bfloat16, device="cuda")
        int8_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.int8, device="cuda")
        
        encode_kernel, _ = _get_kernels()
        grid = (num_tokens,)
        encode_kernel[grid](
            arr_gpu,
            int8_vals,
            scale_factors,
            num_tokens,
            hidden_dim,
        )
        
        # 3. Store in Shadow Allocator
        scale_bytes = scale_factors.numel() * 2
        int8_bytes = int8_vals.numel()
        total_bytes = scale_bytes + int8_bytes
        
        handle = self.allocator.alloc(Tier.INT8, total_bytes)
        if handle < 0:
            return False # Shadow cache OOM, let SGLang evict natively
            
        shadow_tensor = self.allocator.get_tensor(handle)
        shadow_tensor[:scale_bytes].copy_(scale_factors.view(torch.uint8))
        shadow_tensor[scale_bytes:].copy_(int8_vals.view(torch.uint8))
        
        # 4. Update Node State
        node.is_compressed = True
        node.shadow_handle = handle
        node.kv_indices = None # Drop physical reference
        
        # 5. Free the physical slots back to SGLang
        memory_pool.free(kv_indices)
        
        return True
