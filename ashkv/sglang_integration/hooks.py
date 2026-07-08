"""SGLang Integration Hooks (Node-Level).

Unlike earlier serving engines which manage blocks independently, SGLang manages KV cache
via a Radix Tree. These hooks operate directly on `RadixNode` instances.
When a node is evicted by SGLang, it is demoted to INT8 instead of being deleted.
When a node is hit by a prefix match, it is promoted back to BF16.
"""
from __future__ import annotations

import torch
from typing import Any, List

from ashkv.contracts.tiers import Tier
from ashkv.contracts.page import PageTable
from ashkv.compiler.registry import codec_registry
from ashkv.sglang_integration.allocator import SGLangShadowAllocator
from ashkv.codecs.int8 import _get_kernels
from ashkv.sglang_integration.layer_type_filter import get_compressible_layers
from ashkv.safety.circuit_breaker import CircuitBreakerRegistry


def _get_layer_tensor(sglang_kv_cache: Any, layer_idx: int, kv_indices: torch.Tensor) -> torch.Tensor:
    """Safely index into the SGLang KV cache, handling lists or multi-dimensional tensors."""
    if isinstance(sglang_kv_cache, (list, tuple)):
        # List of layer tensors, each [num_slots, ...]
        return sglang_kv_cache[layer_idx][kv_indices]
    else:
        # Single tensor [num_layers, num_slots, ...]
        return sglang_kv_cache[layer_idx, kv_indices]

def _set_layer_tensor(sglang_kv_cache: Any, layer_idx: int, kv_indices: torch.Tensor, values: torch.Tensor) -> None:
    """Safely set values into the SGLang KV cache."""
    if isinstance(sglang_kv_cache, (list, tuple)):
        sglang_kv_cache[layer_idx][kv_indices] = values
    else:
        sglang_kv_cache[layer_idx, kv_indices] = values


class SGLangHooks:
    """Manages the boundary between SGLang's RadixCache and ASH-KV."""

    def __init__(
        self,
        page_table: PageTable,
        shadow_allocator: SGLangShadowAllocator,
        model_config: Any,
        codec_name: str = "int8_default",
    ):
        self.page_table = page_table
        self.allocator = shadow_allocator
        self.codec_name = codec_name
        self.codec = codec_registry.get(codec_name)
        if self.codec is None:
            raise ValueError(f"Codec {codec_name} not found in registry.")
            
        self.compressible_layers = get_compressible_layers(model_config)
        self.circuit_breaker = CircuitBreakerRegistry()

    def promote_hook(self, node: Any, sglang_kv_cache: Any, memory_pool: Any) -> None:
        """Pre-decode: Restore a compressed RadixNode to BF16.
        
        Args:
            node: SGLang RadixNode instance.
            sglang_kv_cache: SGLang's underlying bfloat16 KV cache tensor(s).
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
        num_tokens = getattr(node, "length", 0)
        if num_tokens == 0 and hasattr(node, "value"):
             num_tokens = len(node.value)
             
        # Request slots from SGLang
        new_indices = memory_pool.allocate(num_tokens)
        
        if isinstance(sglang_kv_cache, (list, tuple)):
            hidden_dim = sglang_kv_cache[0].shape[-1]
        else:
            hidden_dim = sglang_kv_cache.shape[-1]
            
        # 3. Decode INT8 -> BF16 for all layers
        scale_bytes_per_layer = num_tokens * 2
        int8_vals_per_layer = num_tokens * hidden_dim
        layer_stride = scale_bytes_per_layer + int8_vals_per_layer
        
        _, decode_kernel = _get_kernels()
        grid = (num_tokens,)
        
        offset = 0
        for layer_idx in self.compressible_layers:
            scale_factors = shadow_tensor[offset : offset + scale_bytes_per_layer].view(torch.bfloat16)
            int8_vals = shadow_tensor[offset + scale_bytes_per_layer : offset + layer_stride].view(torch.int8).view(num_tokens, hidden_dim)
            
            bf16_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
            
            decode_kernel[grid](
                int8_vals,
                scale_factors,
                bf16_vals,
                num_tokens,
                hidden_dim,
            )
            
            # Write back to SGLang's fresh slots
            _set_layer_tensor(sglang_kv_cache, layer_idx, new_indices, bf16_vals)
            
            offset += layer_stride
        
        # 4. Update Node state
        node.kv_indices = new_indices
        node.is_compressed = False
        del node.shadow_handle
        
        # 5. Free shadow memory
        self.allocator.free(handle)

    def demote_hook(self, node: Any, sglang_kv_cache: Any, memory_pool: Any) -> bool:
        """Post-decode: Compress a RadixNode to INT8 instead of deleting it.
        
        Args:
            node: SGLang RadixNode instance about to be evicted.
            sglang_kv_cache: SGLang's underlying bfloat16 KV cache tensor(s).
            memory_pool: SGLang's TokenToKVPool for returning slots.
            
        Returns:
            bool: True if successfully compressed and intercepted eviction, False otherwise.
        """
        if hasattr(node, "is_compressed") and node.is_compressed:
            return True
            
        physical_indices = getattr(node, "kv_indices", getattr(node, "value", None))
        if physical_indices is None:
            return False
            
        num_tokens = getattr(node, "length", 0)           
        if not self.circuit_breaker.is_codec_available(self.codec_name):
            print(f"[HOOKS] Circuit breaker unavailable for {self.codec_name}")
            return False # Circuit breaker tripped, fall back to native eviction
            
        kv_indices = physical_indices
        
        if isinstance(sglang_kv_cache, (list, tuple)):
            hidden_dim = sglang_kv_cache[0].shape[-1]
        else:
            hidden_dim = sglang_kv_cache.shape[-1]
        
        scale_bytes_per_layer = num_tokens * 2
        int8_vals_per_layer = num_tokens * hidden_dim
        layer_stride = scale_bytes_per_layer + int8_vals_per_layer
        total_bytes = layer_stride * len(self.compressible_layers)
        
        handle = self.allocator.alloc(Tier.FP8, total_bytes)
        if handle < 0:
            return False # Shadow cache OOM, let SGLang evict natively
            
        try:
            shadow_tensor = self.allocator.get_tensor(handle)
            encode_kernel, decode_kernel = _get_kernels()
            grid = (num_tokens,)
            
            offset = 0
            for layer_idx in self.compressible_layers:
                arr_gpu = _get_layer_tensor(sglang_kv_cache, layer_idx, kv_indices)
                
                scale_factors = torch.empty((num_tokens,), dtype=torch.bfloat16, device="cuda")
                int8_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.int8, device="cuda")
                
                encode_kernel[grid](
                    arr_gpu,
                    int8_vals,
                    scale_factors,
                    num_tokens,
                    hidden_dim,
                )
                
                # Checksum verification round-trip (GPU)
                bf16_verify = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
                decode_kernel[grid](
                    int8_vals,
                    scale_factors,
                    bf16_verify,
                    num_tokens,
                    hidden_dim,
                )
                
                if not torch.allclose(arr_gpu, bf16_verify, atol=1e-2):
                    print(f"[HOOKS] allclose failed! arr_gpu: {arr_gpu.sum()}, bf16_verify: {bf16_verify.sum()}")
                    self.circuit_breaker.record_codec_failure(self.codec_name)
                    self.allocator.free(handle)
                    return False
                    
                self.circuit_breaker.record_codec_success(self.codec_name)
                
                # Store in Shadow Allocator
                shadow_tensor[offset : offset + scale_bytes_per_layer].copy_(scale_factors.view(torch.uint8).flatten())
                shadow_tensor[offset + scale_bytes_per_layer : offset + layer_stride].copy_(int8_vals.view(torch.uint8).flatten())
                
                offset += layer_stride
            
            # Update Node State
            node.is_compressed = True
            node.shadow_handle = handle
            node.kv_indices = None # Drop physical reference
            
            # Free the physical slots back to SGLang
            memory_pool.free(kv_indices)
            
            return True

        except Exception as e:
            print(f"[HOOKS] Exception occurred: {e}")
            self.allocator.free(handle)
            return False
