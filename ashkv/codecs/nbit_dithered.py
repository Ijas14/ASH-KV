"""Dithered N-Bit codec using group quantization, outlier isolation, and stochastic dithering.

This codec generalizes the experimental 2-bit pipeline to arbitrary bit widths (2, 4, 8)
to allow fine-grained trade-offs between Compression Ratio and Reasoning Fidelity.
"""
from __future__ import annotations

import struct
import hashlib


class DitheredNBitCodec:
    """N-Bit Codec using Stochastic Dithering and Group Scaling.
    
    Format:
    [num_tokens (I)] [num_outliers (I)]
    [scale_factors (bfloat16)]
    [outlier_indices (int32)]
    [outlier_values (bfloat16)]
    [nbit_packed_values (uint8)]
    """

    __slots__ = ("_encode_calls", "_decode_calls", "_hidden_dim", "_group_size", "_sigma", "_bits", "_num_states", "_half_states")

    def __init__(self, hidden_dim: int = 128, group_size: int = 32, sigma: float = 3.0, bits: int = 4) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(f"DitheredNBitCodec only supports 2, 4, or 8 bits. Got: {bits}")
            
        self._encode_calls = 0
        self._decode_calls = 0
        self._hidden_dim = hidden_dim
        self._group_size = group_size
        self._sigma = sigma
        
        self._bits = bits
        self._num_states = 2 ** self._bits
        self._half_states = (self._num_states - 1) / 2.0

    def checksum(self, raw_bytes: bytes) -> int:
        h = hashlib.sha256(raw_bytes).digest()
        return int.from_bytes(h[:8], "little", signed=False)

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        if not source_bytes:
            return struct.pack("<II", 0, 0)

        import torch
        import torch.nn.functional as F
        arr = torch.frombuffer(bytearray(source_bytes), dtype=torch.bfloat16)
        num_tokens = len(arr) // self._hidden_dim
        if num_tokens == 0:
            return struct.pack("<II", 0, 0)
        
        arr = arr[:num_tokens * self._hidden_dim].view(num_tokens, self._hidden_dim).float().cuda()
        
        num_groups = self._hidden_dim // self._group_size
        arr_grouped = arr.view(num_tokens, num_groups, self._group_size)
        
        # 1. Outlier Isolation (Sigma Rule per group)
        group_mean = arr_grouped.mean(dim=2, keepdim=True)
        group_std = arr_grouped.std(dim=2, keepdim=True) + 1e-8
        
        outlier_mask = torch.abs(arr_grouped - group_mean) > self._sigma * group_std
        outlier_mask_flat = outlier_mask.contiguous().view(-1)
        
        outlier_indices = torch.nonzero(outlier_mask_flat)[:, 0].int().cpu().numpy()
        outlier_values = arr.contiguous().view(-1)[outlier_mask_flat].bfloat16().view(torch.int16).cpu().numpy()
        num_outliers = len(outlier_indices)
        
        arr_inliers = arr_grouped.clone()
        arr_inliers[outlier_mask] = group_mean.expand_as(arr_inliers)[outlier_mask]
        
        # 2. Group Quantization
        abs_max = arr_inliers.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-8)
        delta = abs_max / self._half_states
        
        # 3. Stochastic Dithering (Dynamic Seed to prevent correlated noise)
        gen = torch.Generator(device=arr_inliers.device)
        gen.manual_seed(42 + self._encode_calls)
        noise = (torch.rand(arr_inliers.shape, generator=gen, device=arr_inliers.device, dtype=arr_inliers.dtype) - 0.5) * delta
        
        scaled = (arr_inliers + noise) / delta + self._half_states
        rounded = scaled.round().clamp(0, self._num_states - 1).to(torch.uint8)
        
        # 4. Dynamic Bit-Packing
        rounded_flat = rounded.contiguous().view(-1)
        
        if self._bits == 2:
            padding_len = (4 - (rounded_flat.size(0) % 4)) % 4
            if padding_len > 0:
                rounded_flat = F.pad(rounded_flat, (0, padding_len))
            rounded_cols = rounded_flat.view(-1, 4)
            packed = (rounded_cols[:, 0] << 6) | (rounded_cols[:, 1] << 4) | (rounded_cols[:, 2] << 2) | rounded_cols[:, 3]
            
        elif self._bits == 4:
            padding_len = (2 - (rounded_flat.size(0) % 2)) % 2
            if padding_len > 0:
                rounded_flat = F.pad(rounded_flat, (0, padding_len))
            rounded_cols = rounded_flat.view(-1, 2)
            packed = (rounded_cols[:, 0] << 4) | rounded_cols[:, 1]
            
        elif self._bits == 8:
            packed = rounded_flat

        header = struct.pack("<II", num_tokens, num_outliers)
        scales_bytes = delta.bfloat16().view(torch.int16).cpu().numpy().tobytes()
        packed_bytes = packed.cpu().numpy().tobytes()
        
        return header + scales_bytes + outlier_indices.tobytes() + outlier_values.tobytes() + packed_bytes

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        if len(target_bytes) < 8:
            return b""
            
        num_tokens, num_outliers = struct.unpack("<II", target_bytes[:8])
        if num_tokens == 0:
            return b""
            
        num_groups = self._hidden_dim // self._group_size
        
        offset = 8
        scales_size = num_tokens * num_groups * 2
        import torch
        scales = torch.frombuffer(bytearray(target_bytes[offset:offset+scales_size]), dtype=torch.bfloat16).cuda().float()
        offset += scales_size
        
        if num_outliers > 0:
            outlier_indices_size = num_outliers * 4
            outlier_indices = torch.frombuffer(bytearray(target_bytes[offset:offset+outlier_indices_size]), dtype=torch.int32).cuda().long()
            offset += outlier_indices_size
            
            outlier_values_size = num_outliers * 2
            outlier_values = torch.frombuffer(bytearray(target_bytes[offset:offset+outlier_values_size]), dtype=torch.bfloat16).cuda().float()
            offset += outlier_values_size
        
        packed = torch.frombuffer(bytearray(target_bytes[offset:]), dtype=torch.uint8).cuda()
        
        if self._bits == 2:
            unpacked = torch.empty((len(packed) * 4,), dtype=torch.uint8, device="cuda")
            unpacked[0::4] = (packed >> 6) & 0x03
            unpacked[1::4] = (packed >> 4) & 0x03
            unpacked[2::4] = (packed >> 2) & 0x03
            unpacked[3::4] = packed & 0x03
        elif self._bits == 4:
            unpacked = torch.empty((len(packed) * 2,), dtype=torch.uint8, device="cuda")
            unpacked[0::2] = (packed >> 4) & 0x0F
            unpacked[1::2] = packed & 0x0F
        elif self._bits == 8:
            unpacked = packed.clone()
        
        # Remove any padding added during encode
        original_len = num_tokens * num_groups * self._group_size
        unpacked = unpacked[:original_len]
        
        unpacked = unpacked.view(num_tokens, num_groups, self._group_size).float()
        
        delta = scales.view(num_tokens, num_groups, 1)
        dequantized = (unpacked - self._half_states) * delta
        
        dequantized_flat = dequantized.contiguous().view(-1)
        
        if num_outliers > 0:
            dequantized_flat[outlier_indices] = outlier_values
            
        return dequantized_flat.bfloat16().view(torch.int16).cpu().numpy().tobytes()
