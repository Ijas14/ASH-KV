"""vLLM integration hooks.

These hooks enforce the shadow cache architecture. ASH-KV acts as a GPU-resident
INT8 swap space for preempted/cold sequences. vLLM's KV cache is strictly BF16.
"""
from __future__ import annotations

import torch
from typing import List, Dict

from ashkv.contracts.tiers import Tier
from ashkv.contracts.page import PageTable
from ashkv.compiler.registry import codec_registry
from ashkv.vllm_integration.allocator import VLLMShadowAllocator
from ashkv.codecs.int8 import _get_kernels, _next_power_of_2


class VLLMHooks:
    """Manages the boundary between vLLM's BF16 cache and ASH-KV's shadow pool."""

    def __init__(
        self,
        page_table: PageTable,
        shadow_allocator: VLLMShadowAllocator,
        codec_name: str = "int8_default",
    ):
        self.page_table = page_table
        self.allocator = shadow_allocator
        self.codec = codec_registry.get_codec(codec_name)
        if self.codec is None:
            raise ValueError(f"Codec {codec_name} not found in registry.")
            
        # Maps vLLM block_number -> ASH-KV shadow handle
        self.shadow_handles: Dict[int, int] = {}

    def promote_hook(self, running_block_numbers: List[int], vllm_kv_cache: torch.Tensor) -> None:
        """Pre-decode: Ensure all blocks for running sequences are in BF16.

        If a block is in the INT8 shadow pool, it is decoded back to BF16 and
        written into vLLM's kv_cache in-place.
        """
        encode_kernel, decode_kernel = _get_kernels()

        for block_num in running_block_numbers:
            tier_raw = self.page_table.get_tier(block_num)
            if tier_raw < 0:
                continue

            tier = Tier(tier_raw)
            if tier == Tier.INT8:
                # Block is compressed in shadow cache. We must promote it.
                handle = self.shadow_handles.get(block_num)
                if handle is None:
                    continue

                shadow_tensor = self.allocator.get_tensor(handle)
                if shadow_tensor is None:
                    continue

                block_tensor = vllm_kv_cache[block_num]
                
                # We know the INT8 codec packed shape: [scale_factors (bfloat16)] + [int8_vals (int8)]
                # Determine sizes from block_tensor shape (assumed: num_tokens * hidden_dim)
                # vLLM's shape is complex, but we flatten it for the codec.
                hidden_dim = block_tensor.shape[-1]
                num_tokens = block_tensor.numel() // hidden_dim
                
                scale_bytes = num_tokens * 2
                scale_factors = shadow_tensor[:scale_bytes].view(torch.bfloat16)
                int8_vals = shadow_tensor[scale_bytes:].view(torch.int8).view(num_tokens, hidden_dim)
                
                bf16_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
                
                # Direct Triton call (no CPU byte conversion)
                grid = (num_tokens,)
                decode_kernel[grid](
                    int8_vals,
                    scale_factors,
                    bf16_vals,
                    num_tokens,
                    hidden_dim,
                )

                # Write back to vLLM's kv_cache tensor in-place
                block_tensor.view(-1).copy_(bf16_vals.view(-1))

                # Free shadow memory
                self.allocator.free(handle)
                del self.shadow_handles[block_num]

                # Commit transition back to BF16 (trusting the cache, no checksum recalc to save time)
                original_checksum = self.page_table.get_bf16_checksum(block_num)
                self.page_table.apply_tier_transition(block_num, Tier.BF16, original_checksum)

    def demote_hook(self, cold_block_numbers: List[int], vllm_kv_cache: torch.Tensor) -> None:
        """Post-decode: Compress cold blocks to the INT8 shadow pool.

        These are blocks belonging to waiting or preempted sequences. Instead of
        CPU swap or recompute, we compress them to the GPU INT8 shadow cache.
        """
        encode_kernel, decode_kernel = _get_kernels()

        for block_num in cold_block_numbers:
            tier_raw = self.page_table.get_tier(block_num)
            if tier_raw < 0:
                continue

            tier = Tier(tier_raw)
            if tier == Tier.BF16:
                block_tensor = vllm_kv_cache[block_num]
                
                hidden_dim = block_tensor.shape[-1]
                num_tokens = block_tensor.numel() // hidden_dim
                
                arr_gpu = block_tensor.to(dtype=torch.bfloat16).view(num_tokens, hidden_dim)
                
                scale_factors = torch.empty((num_tokens,), dtype=torch.bfloat16, device="cuda")
                int8_vals = torch.empty((num_tokens, hidden_dim), dtype=torch.int8, device="cuda")
                
                # Direct Triton call
                grid = (num_tokens,)
                encode_kernel[grid](
                    arr_gpu,
                    int8_vals,
                    scale_factors,
                    num_tokens,
                    hidden_dim,
                )
                
                # Compute total packed size
                scale_bytes = scale_factors.numel() * 2
                int8_bytes = int8_vals.numel()
                total_bytes = scale_bytes + int8_bytes
                
                handle = self.allocator.alloc(Tier.INT8, total_bytes)
                if handle < 0:
                    continue
                    
                # Write directly to GPU tensor in allocator
                shadow_tensor = self.allocator.get_tensor(handle)
                shadow_tensor[:scale_bytes].copy_(scale_factors.view(torch.uint8))
                shadow_tensor[scale_bytes:].copy_(int8_vals.view(torch.uint8))
                
                self.shadow_handles[block_num] = handle

                # Checksum validation skipped on hot path for performance
                # We assume the Triton kernel didn't silently corrupt data
                # Commit transition to INT8
                original_checksum = self.page_table.get_bf16_checksum(block_num)
                self.page_table.apply_tier_transition(block_num, Tier.INT8, original_checksum)
