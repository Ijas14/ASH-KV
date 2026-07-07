"""INT8 codec for Ampere GPUs (A100) using Triton.

INT8 on A100 requires per-token scaling factors to avoid quality
collapse. The format is:
    [num_tokens (uint32)] [scale_factors (float16 * num_tokens)] [int8_values]

The Triton kernel does:
1. Compute per-token abs_max from the BF16 source
2. Quantize: int8 = round(bf16 * 127 / abs_max)
3. Pack scale factors + int8 values
4. On decode: bf16 = int8 * abs_max / 127

Registration: codecs.int8.INT8Codec is registered as "int8_default"
when this module is imported.
"""
from __future__ import annotations

import struct

from .checksum import checksum


_encode_kernel = None
_decode_kernel = None


def _get_kernels():
    global _encode_kernel, _decode_kernel
    if _encode_kernel is not None:
        return _encode_kernel, _decode_kernel

    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['hidden_dim']
    )
    @triton.jit
    def _int8_encode_kernel(
        bf16_ptr,        # *float16, shape (num_tokens, hidden_dim)
        int8_ptr,        # *int8, shape (num_tokens, hidden_dim)
        scale_ptr,       # *float16, shape (num_tokens,)
        num_tokens,
        hidden_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        # Load BF16 values
        vals = tl.load(
            bf16_ptr + token_id * hidden_dim + offsets,
            mask=mask,
            other=0.0,
        )

        # Compute abs_max
        abs_max = tl.max(tl.abs(vals), axis=0)
        abs_max = tl.maximum(abs_max, 1e-8)

        # Store scale factor
        tl.store(scale_ptr + token_id, abs_max)

        # Quantize
        scaled = vals * 127.0 / abs_max
        rounded = scaled + tl.where(scaled > 0.0, 0.5, -0.5)
        int8_vals = tl.cast(rounded, tl.int8)
        tl.store(int8_ptr + token_id * hidden_dim + offsets, int8_vals, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['hidden_dim']
    )
    @triton.jit
    def _int8_decode_kernel(
        int8_ptr,        # *int8, shape (num_tokens, hidden_dim)
        scale_ptr,       # *float16, shape (num_tokens,)
        bf16_ptr,        # *float16, shape (num_tokens, hidden_dim)
        num_tokens,
        hidden_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        int8_vals = tl.load(
            int8_ptr + token_id * hidden_dim + offsets,
            mask=mask,
            other=0,
        )
        scale = tl.load(scale_ptr + token_id)

        # Dequantize
        bf16_vals = tl.cast(int8_vals, tl.float32) * tl.cast(scale, tl.float32) / 127.0
        tl.store(bf16_ptr + token_id * hidden_dim + offsets, tl.cast(bf16_vals, tl.float16), mask=mask)

    _encode_kernel = _int8_encode_kernel
    _decode_kernel = _int8_decode_kernel
    return _encode_kernel, _decode_kernel


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return max(16, n)


class INT8Codec:
    """INT8 codec with per-token scaling using Triton.

    Correctness invariant: decode(encode(x)) reproduces x closely
    enough that checksum(decode(encode(x))) == checksum(x).
    """

    __slots__ = ("_encode_calls", "_decode_calls", "_hidden_dim")

    def __init__(self, hidden_dim: int = 128) -> None:
        self._encode_calls = 0
        self._decode_calls = 0
        self._hidden_dim = hidden_dim

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1

        if not source_bytes:
            return struct.pack("<I", 0)

        import torch
        encode_kernel, _ = _get_kernels()

        arr = torch.frombuffer(bytearray(source_bytes), dtype=torch.float16)
        num_tokens = len(arr) // self._hidden_dim
        if num_tokens == 0:
            return struct.pack("<I", 0)

        arr = arr[:num_tokens * self._hidden_dim].view(num_tokens, self._hidden_dim)
        arr_gpu = arr.cuda()

        scale_factors = torch.empty((num_tokens,), dtype=torch.float16, device="cuda")
        int8_vals = torch.empty((num_tokens, self._hidden_dim), dtype=torch.int8, device="cuda")

        # The autotuner handles BLOCK_SIZE now.
        encode_kernel[grid](
            arr_gpu,
            int8_vals,
            scale_factors,
            num_tokens,
            self._hidden_dim,
        )

        header = struct.pack("<I", num_tokens)
        
        # Note: v1 CPU<->GPU overhead is acceptable.
        return header + scale_factors.cpu().numpy().tobytes() + int8_vals.cpu().numpy().tobytes()

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1

        if len(target_bytes) < 4:
            return b""

        num_tokens = struct.unpack("<I", target_bytes[:4])[0]
        if num_tokens == 0:
            return b""

        import torch
        _, decode_kernel = _get_kernels()

        scale_size = num_tokens * 2
        int8_size = num_tokens * self._hidden_dim

        scale_factors = torch.frombuffer(
            bytearray(target_bytes[4:4 + scale_size]), dtype=torch.float16
        ).cuda()
        
        int8_vals = torch.frombuffer(
            bytearray(target_bytes[4 + scale_size:4 + scale_size + int8_size]),
            dtype=torch.int8,
        ).view(num_tokens, self._hidden_dim).cuda()

        bf16_vals = torch.empty((num_tokens, self._hidden_dim), dtype=torch.float16, device="cuda")

        # The autotuner handles BLOCK_SIZE now.
        decode_kernel[grid](
            int8_vals,
            scale_factors,
            bf16_vals,
            num_tokens,
            self._hidden_dim,
        )

        return bf16_vals.cpu().numpy().tobytes()

    def checksum(self, raw_bytes: bytes) -> int:
        """Stable checksum."""
        return checksum(raw_bytes)
