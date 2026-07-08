"""INT4 codec scaffold for the cold tier.

NOTE: This is a SCAFFOLD. The actual Triton kernel needs to be
implemented and tested on real hardware.

INT4 packs two values per byte. Like INT8, it needs per-token
scaling to avoid quality collapse, but the scaling is per-group
(group_size = 32 or 64 values share one scale factor).
"""
from __future__ import annotations

import struct

import numpy as np

from .checksum import checksum


class INT4Codec:
    """INT4 codec with per-group scaling.

    INT4 packs two 4-bit values into one byte. Group quantization
    uses one scale factor per GROUP_SIZE values (default 64).

    Format:
        [num_groups (uint32)]
        [scale_factors (float16 * num_groups)]
        [packed_int4 (ceil(num_values / 2) bytes)]
    """

    __slots__ = ("_encode_calls", "_decode_calls", "_group_size")

    def __init__(self, group_size: int = 64) -> None:
        self._encode_calls = 0
        self._decode_calls = 0
        self._group_size = group_size

    def encode(self, source_bytes: bytes) -> bytes:
        """Encode BF16 bytes to INT4 with per-group scaling."""
        self._encode_calls += 1

        arr = np.frombuffer(source_bytes, dtype=np.float16)
        if len(arr) == 0:
            return struct.pack("<I", 0)

        gs = self._group_size
        # Pad to multiple of group_size
        pad_len = (gs - len(arr) % gs) % gs
        arr = np.pad(arr, (0, pad_len))
        num_values = len(arr)
        num_groups = num_values // gs

        arr = arr.reshape(num_groups, gs)

        # Per-group abs_max
        abs_max = np.max(np.abs(arr), axis=1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)

        # Quantize to INT4 range [-8, 7]
        int4_vals = np.round(arr * 7.0 / abs_max).astype(np.int8)
        int4_vals = np.clip(int4_vals, -8, 7)

        # Pack two INT4 values per byte
        flat = int4_vals.flatten()
        # Convert to unsigned [0, 15]
        flat_u = (flat + 8).astype(np.uint8)
        # Pack: high nibble = even index, low nibble = odd index
        packed = np.zeros(len(flat_u) // 2, dtype=np.uint8)
        packed = (flat_u[::2] << 4) | flat_u[1::2]

        scale_factors = abs_max.flatten().astype(np.float16)

        header = struct.pack("<II", num_groups, num_values)
        return header + scale_factors.tobytes() + packed.tobytes()

    def decode(self, target_bytes: bytes) -> bytes:
        """Decode INT4 bytes back to BF16."""
        self._decode_calls += 1

        if len(target_bytes) < 8:
            return b""

        num_groups, num_values = struct.unpack("<II", target_bytes[:8])
        if num_groups == 0:
            return b""

        gs = self._group_size
        scale_size = num_groups * 2  # float16
        packed_size = num_values // 2

        scale_factors = np.frombuffer(
            target_bytes[8:8 + scale_size], dtype=np.float16
        ).reshape(num_groups, 1)
        packed = np.frombuffer(
            target_bytes[8 + scale_size:8 + scale_size + packed_size],
            dtype=np.uint8,
        )

        # Unpack nibbles
        high = (packed >> 4).astype(np.int8) - 8
        low = (packed & 0x0F).astype(np.int8) - 8
        flat = np.empty(len(packed) * 2, dtype=np.int8)
        flat[::2] = high
        flat[1::2] = low

        int4_vals = flat.reshape(num_groups, gs)
        bf16_vals = int4_vals.astype(np.float16) * scale_factors / 7.0

        return bf16_vals.flatten()[:num_values].tobytes()

    def checksum(self, raw_bytes: bytes) -> int:
        """Stable checksum."""
        return checksum(raw_bytes)
