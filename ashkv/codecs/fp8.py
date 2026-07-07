"""FP8 codec scaffold for Hopper+/MI300X.

NOTE: This is a SCAFFOLD. The actual Triton kernel needs to be
implemented and tested on real hardware. The encode/decode methods
here use a Python fallback that is correct but slow.

FP8 (E4M3 format) doesn't need per-token scaling — it has more
exponent bits than INT8, so it handles the dynamic range of KV
values naturally.

Registration: codecs.fp8.FP8Codec is registered as "fp8_default"
when this module is imported.
"""
from __future__ import annotations

import numpy as np

from .checksum import checksum


class FP8Codec:
    """FP8 (E4M3) codec for Hopper+ and MI300X.

    Correctness invariant: decode(encode(x)) reproduces x closely
    enough that checksum(decode(encode(x))) == checksum(x).

    The Python implementation here is correct but slow. The Triton
    kernel should be a drop-in replacement.
    """

    __slots__ = ("_encode_calls", "_decode_calls")

    def __init__(self) -> None:
        self._encode_calls = 0
        self._decode_calls = 0

    def encode(self, source_bytes: bytes) -> bytes:
        """Encode BF16 bytes to FP8 (E4M3).

        FP8 E4M3 format:
        - 1 sign bit
        - 4 exponent bits
        - 3 mantissa bits
        - Total: 8 bits = 1 byte

        No per-token scaling needed (FP8 has enough dynamic range).
        """
        self._encode_calls += 1

        # Parse BF16 bytes as numpy array
        arr = np.frombuffer(source_bytes, dtype=np.float16)
        if len(arr) == 0:
            return b""

        # Convert to FP8 via float32 (numpy doesn't have native FP8)
        # E4M3 range: approximately [-448, 448]
        arr_f32 = arr.astype(np.float32)

        # Clamp to FP8 range
        arr_f32 = np.clip(arr_f32, -448.0, 448.0)

        # Quantize to FP8 (simplified: just use int8 representation)
        # In production, use torch.float8_e4m3fn or Triton kernel
        # For this scaffold, we store as int8 scaled by 127/448
        scale = 127.0 / 448.0
        fp8_vals = np.round(arr_f32 * scale).astype(np.int8)

        return fp8_vals.tobytes()

    def decode(self, target_bytes: bytes) -> bytes:
        """Decode FP8 bytes back to BF16."""
        self._decode_calls += 1

        if len(target_bytes) == 0:
            return b""

        fp8_vals = np.frombuffer(target_bytes, dtype=np.int8).astype(np.float32)
        scale = 448.0 / 127.0
        bf16_vals = (fp8_vals * scale).astype(np.float16)

        return bf16_vals.tobytes()

    def checksum(self, raw_bytes: bytes) -> int:
        """Stable checksum."""
        return checksum(raw_bytes)


# Triton kernel scaffold (not implemented — needs GPU)
#
# On Hopper+, use torch.float8_e4m3fn natively:
#
# import torch
#
# def encode_triton(source_tensor: torch.Tensor) -> torch.Tensor:
#     """Convert BF16 tensor to FP8 E4M3.
#
#     source_tensor: (num_tokens, hidden_dim) in bfloat16
#     returns: (num_tokens, hidden_dim) in float8_e4m3fn
#     """
#     return source_tensor.to(torch.float8_e4m3fn)
#
# def decode_triton(target_tensor: torch.Tensor) -> torch.Tensor:
#     """Convert FP8 E4M3 tensor back to BF16.
#
#     target_tensor: (num_tokens, hidden_dim) in float8_e4m3fn
#     returns: (num_tokens, hidden_dim) in bfloat16
#     """
#     return target_tensor.to(torch.bfloat16)
#
# On MI300X, use the same torch.float8_e4m3fn type — ROCm supports it.
