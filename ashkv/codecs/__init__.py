"""Codec package — implements the Codec protocol.

This package contains:
- bf16.py: identity codec (BF16 → BF16) for fallback
- fp8.py: FP8 codec scaffold (Hopper+/MI300X)
- int8.py: INT8 codec scaffold (Ampere/A100)
- int4.py: INT4 codec scaffold (cold tier, all hardware)
- checksum.py: shared checksum utility
- mock.py: mock codecs for testing

Real Triton kernels need to be implemented and tested on hardware.
The scaffolds here use Python fallbacks that are correct but slow.
"""
from __future__ import annotations

from .bf16 import BF16Codec
from .checksum import checksum
from .fp8 import FP8Codec
from .int4 import INT4Codec
from .int8 import INT8Codec
from .mock import (
    MockCorruptCodec,
    MockFailingCodec,
    MockFP8Codec,
    MockINT4Codec,
    MockINT8Codec,
)

__all__ = [
    "BF16Codec",
    "FP8Codec",
    "INT8Codec",
    "INT4Codec",
    "checksum",
    "MockFP8Codec",
    "MockINT8Codec",
    "MockINT4Codec",
    "MockFailingCodec",
    "MockCorruptCodec",
]
