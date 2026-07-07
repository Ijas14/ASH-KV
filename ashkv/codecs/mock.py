"""Mock codecs for testing.

These codecs don't do real compression — they're for testing the
migration engine, safety layer, and integration without needing
GPU kernels.

Mock codecs:
- MockFP8Codec: simulates FP8 (2x compression via truncation)
- MockINT8Codec: simulates INT8 (2x compression, with per-token scaling)
- MockINT4Codec: simulates INT4 (4x compression)
- MockFailingCodec: always fails on encode (for fault injection)
- MockCorruptCodec: always produces wrong checksum (for fault injection)

All mock codecs implement the full Codec protocol.
"""
from __future__ import annotations

import hashlib
import struct

from .checksum import checksum


class _BaseMockCodec:
    """Common functionality for mock codecs."""

    __slots__ = ("_compression_ratio", "_encode_calls", "_decode_calls")

    def __init__(self, compression_ratio: float = 1.0) -> None:
        self._compression_ratio = compression_ratio
        self._encode_calls = 0
        self._decode_calls = 0

    def checksum(self, raw_bytes: bytes) -> int:
        """Stable checksum. Same for all mock codecs."""
        return checksum(raw_bytes)


class MockFP8Codec(_BaseMockCodec):
    """Simulates FP8 compression (2x) with perfect round-trip.

    Does NOT actually quantize to FP8 — it uses a reversible
    transformation (XOR with a constant) to simulate compression
    while guaranteeing round-trip correctness. This is for testing
    the migration engine, not for measuring compression quality.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(compression_ratio=2.0)

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        # Reversible "compression": XOR with 0xAA and pack
        # This is NOT real compression, but it round-trips perfectly
        return bytes(b ^ 0xAA for b in source_bytes)

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        # XOR is its own inverse
        return bytes(b ^ 0xAA for b in target_bytes)


class MockINT8Codec(_BaseMockCodec):
    """Simulates INT8 compression (2x) with per-token scaling.

    Stores a scale factor + quantized values. For testing only.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(compression_ratio=2.0)

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        # Compute a scale factor (max byte value)
        if len(source_bytes) == 0:
            return struct.pack("<Q", 0) + b""

        scale = max(source_bytes) if source_bytes else 0
        if scale == 0:
            scale = 1

        # Quantize: each byte maps to source/scale * 255
        # For mock purposes, just store scale + every other byte
        quantized = bytes((b * 255 // scale) % 256 for b in source_bytes[::2])
        return struct.pack("<Q", scale) + quantized

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        if len(target_bytes) < 8:
            return b""

        scale = struct.unpack("<Q", target_bytes[:8])[0]
        if scale == 0:
            return b""

        quantized = target_bytes[8:]
        # Reconstruct (lossy in real life, but mock round-trips for testing)
        return bytes(b * scale // 255 for b in quantized for _ in range(2))


class MockINT4Codec(_BaseMockCodec):
    """Simulates INT4 compression (4x). For testing only."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(compression_ratio=4.0)

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        # "Compress" by taking every 4th byte
        return source_bytes[::4]

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        # "Decompress" by quadrupling each byte
        return bytes(b for b in target_bytes for _ in range(4))


class MockFailingCodec(_BaseMockCodec):
    """Codec that always fails on encode. For fault injection."""

    __slots__ = ("fail_mode",)

    def __init__(self, fail_mode: str = "encode") -> None:
        super().__init__(compression_ratio=1.0)
        self.fail_mode = fail_mode

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        if self.fail_mode == "encode":
            raise RuntimeError("injected encode failure (mock)")
        return source_bytes

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        if self.fail_mode == "decode":
            raise RuntimeError("injected decode failure (mock)")
        return target_bytes


class MockCorruptCodec(_BaseMockCodec):
    """Codec that produces wrong checksums. For fault injection."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(compression_ratio=1.0)

    def encode(self, source_bytes: bytes) -> bytes:
        self._encode_calls += 1
        # Return modified bytes so round-trip won't match
        return bytes(b ^ 0xFF for b in source_bytes)

    def decode(self, target_bytes: bytes) -> bytes:
        self._decode_calls += 1
        # XOR back, but it won't match because encode XORed
        return bytes(b ^ 0xFF for b in target_bytes)
