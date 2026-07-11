"""SGLang Integration Hooks (HiCache IO Interception).

These hooks intercept SGLang's native CPU offload mechanism (HiCache).
Instead of transferring massive BF16 tensors across the PCIe bus, we compress
the KV cache to INT8 on the GPU, and write the compact byte stream into the CPU
slots allocated by SGLang.
"""
from __future__ import annotations

import torch
import json
import os
import time
from typing import Any

from ashkv.codecs.int8 import _get_kernels
from ashkv.safety.circuit_breaker import CircuitBreakerRegistry


class SGLangHooks:
    """Manages the HiCache IO interception between SGLang's GPU and CPU memory pools."""

    def __init__(
        self,
        codec_name: str = "int8_default",
    ):
        self.codec_name = codec_name
        self.circuit_breaker = CircuitBreakerRegistry()
        
        # Using all layers by default for this integration, but can be configured later
        self.compressible_layers = None 
        
        self.stats = {
            "tokens_compressed": 0,
            "tokens_decompressed": 0,
            "bytes_saved": 0,
            "blocks_intercepted": 0,
            "last_flush_time": 0
        }
        self.stats_file = "/tmp/ashkv_stats.json"
        self._flush_stats()

    def _flush_stats(self):
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f)
        except Exception as e:
            print(f"[ASH-KV] Failed to flush stats: {e}")

    def demote_hook(
        self, 
        device_pool: Any, 
        host_pool: Any, 
        host_indices: torch.Tensor, 
        device_indices: torch.Tensor, 
        pool_type: str
    ) -> None:
        """GPU -> CPU Interceptor (Compression)
        
        Called by SGLang when evicting a KV node to CPU (write-through/write-back).
        We compress the BF16 tensor on the GPU and write INT8 into the CPU host_pool.
        """
        if pool_type != "MHA":
            # For now, only support MHA. Fallback to native copy for others.
            return
            
        num_tokens = len(device_indices)
        if num_tokens == 0:
            return
            
        encode_kernel, _ = _get_kernels()
        grid = (num_tokens,)
        
        head_num = device_pool.head_num
        head_dim = device_pool.head_dim
        layer_num = device_pool.layer_num
        
        if self.compressible_layers is None:
            self.compressible_layers = list(range(layer_num))
            
        scale_bytes = head_num * 2
        int8_bytes = head_num * head_dim
        total_bytes = scale_bytes + int8_bytes
        slot_bytes = head_num * head_dim * 2 # Standard BF16 slot size
        
        device = device_pool.k_buffer[0].device
        
        # Temporary GPU tensors
        int8_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.int8, device=device)
        scales_gpu = torch.empty((num_tokens, head_num), dtype=torch.float16, device=device)
            
        print(f"[ASH-KV] DEMOTE: Intercepting backup_from_device_all_layer. Compressing {num_tokens} tokens for MHA...")
        
        for layer_idx in self.compressible_layers:
            # 1. Get GPU BF16 tensor (K and V)
            k_gpu = device_pool.k_buffer[layer_idx][device_indices]
            v_gpu = device_pool.v_buffer[layer_idx][device_indices]
            
            # 2. Compress K
            encode_kernel[grid](
                k_gpu.view(num_tokens, -1),
                int8_gpu.view(num_tokens, -1),
                scales_gpu.view(num_tokens, -1),
                num_tokens,
                head_num * head_dim
            )
            
            # Pack K to CPU
            k_packed = torch.zeros((num_tokens, slot_bytes), dtype=torch.uint8, device="cpu", pin_memory=torch.cuda.is_available())
            k_packed[:, :scale_bytes] = scales_gpu.view(torch.uint8).view(num_tokens, -1).cpu()
            k_packed[:, scale_bytes:total_bytes] = int8_gpu.view(torch.uint8).view(num_tokens, -1).cpu()
            k_padded_bf16 = k_packed.view(torch.bfloat16).view(num_tokens, head_num, head_dim)
            
            # Write K to CPU pool
            host_pool.k_buffer[host_indices, layer_idx] = k_padded_bf16
            
            # 3. Compress V
            encode_kernel[grid](
                v_gpu.view(num_tokens, -1),
                int8_gpu.view(num_tokens, -1),
                scales_gpu.view(num_tokens, -1),
                num_tokens,
                head_num * head_dim
            )
            
            # Pack V to CPU
            v_packed = torch.zeros((num_tokens, slot_bytes), dtype=torch.uint8, device="cpu", pin_memory=torch.cuda.is_available())
            v_packed[:, :scale_bytes] = scales_gpu.view(torch.uint8).view(num_tokens, -1).cpu()
            v_packed[:, scale_bytes:total_bytes] = int8_gpu.view(torch.uint8).view(num_tokens, -1).cpu()
            v_padded_bf16 = v_packed.view(torch.bfloat16).view(num_tokens, head_num, head_dim)
            
            # Write V to CPU pool
            host_pool.v_buffer[host_indices, layer_idx] = v_padded_bf16

        self.stats["blocks_intercepted"] += 1
        self.stats["tokens_compressed"] += (num_tokens * len(self.compressible_layers))
        bf16_bytes = num_tokens * len(self.compressible_layers) * head_num * head_dim * 2 * 2
        int8_bytes = num_tokens * len(self.compressible_layers) * (head_num * head_dim + scale_bytes) * 2
        self.stats["bytes_saved"] += (bf16_bytes - int8_bytes)
        
        if time.time() - self.stats["last_flush_time"] > 1.0:
            self._flush_stats()
            self.stats["last_flush_time"] = time.time()

    def promote_hook(
        self, 
        device_pool: Any, 
        host_pool: Any, 
        host_indices: torch.Tensor, 
        device_indices: torch.Tensor, 
        layer_id: int, 
        pool_type: str
    ) -> None:
        """CPU -> GPU Interceptor (Decompression)
        
        Called by SGLang when a prefix matches an offloaded node.
        We read the INT8+scales from the CPU host_pool and decode directly into the GPU device_pool.
        Note: SGLang loads layer-by-layer, so this is called per-layer.
        """
        if pool_type != "MHA":
            return
            
        num_tokens = len(device_indices)
        if num_tokens == 0:
            return
            
        _, decode_kernel = _get_kernels()
        grid = (num_tokens,)
        
        head_num = device_pool.head_num
        head_dim = device_pool.head_dim
        
        scale_bytes = head_num * 2
        
        print(f"[ASH-KV] PROMOTE: Intercepting load_to_device_per_layer. Decompressing {num_tokens} tokens for MHA layer {layer_id}...")
        
        # 1. Read K from CPU pool
        host_k_bf16 = host_pool.k_buffer[host_indices, layer_id].contiguous()
        host_k_bytes = host_k_bf16.view(num_tokens, -1).contiguous().view(torch.uint8).view(num_tokens, -1)
        k_scales_cpu = host_k_bytes[:, :scale_bytes].contiguous().view(torch.float16)
        k_int8_cpu = host_k_bytes[:, scale_bytes : scale_bytes + head_num * head_dim].contiguous().view(torch.int8)
        
        # 2. Read V from CPU pool
        host_v_bf16 = host_pool.v_buffer[host_indices, layer_id].contiguous()
        host_v_bytes = host_v_bf16.view(num_tokens, -1).contiguous().view(torch.uint8).view(num_tokens, -1)
        v_scales_cpu = host_v_bytes[:, :scale_bytes].contiguous().view(torch.float16)
        v_int8_cpu = host_v_bytes[:, scale_bytes : scale_bytes + head_num * head_dim].contiguous().view(torch.int8)
        
        device = device_pool.k_buffer[0].device
        
        # 3. Move to GPU
        k_scales_gpu = k_scales_cpu.to(device, non_blocking=True)
        k_int8_gpu = k_int8_cpu.to(device, non_blocking=True)
        v_scales_gpu = v_scales_cpu.to(device, non_blocking=True)
        v_int8_gpu = v_int8_cpu.to(device, non_blocking=True)
        
        # 4. Decode directly into device_pool slots
        k_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.bfloat16, device=device)
        decode_kernel[grid](
            k_int8_gpu.view(num_tokens, -1),
            k_scales_gpu.view(num_tokens, -1),
            k_gpu.view(num_tokens, -1),
            num_tokens,
            head_num * head_dim
        )
        device_pool.k_buffer[layer_id][device_indices] = k_gpu
        
        v_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.bfloat16, device=device)
        decode_kernel[grid](
            v_int8_gpu.view(num_tokens, -1),
            v_scales_gpu.view(num_tokens, -1),
            v_gpu.view(num_tokens, -1),
            num_tokens,
            head_num * head_dim
        )
        device_pool.v_buffer[layer_id][device_indices] = v_gpu

        self.stats["tokens_decompressed"] += num_tokens
        
        if time.time() - self.stats["last_flush_time"] > 1.0:
            self._flush_stats()
            self.stats["last_flush_time"] = time.time()
