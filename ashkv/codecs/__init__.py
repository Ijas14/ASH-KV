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
from .int2_dithered import DitheredINT2Codec
from .nbit_dithered import DitheredNBitCodec
from .mock import (
    MockCorruptCodec,
    MockFailingCodec,
    MockFP8Codec,
    MockINT4Codec,
    MockINT8Codec,
)
from ashkv.compiler.registry import codec_registry

# Register actual codecs
codec_registry.register("bf16", BF16Codec())
codec_registry.register("int8_default", INT8Codec())
codec_registry.register("int4_default", INT4Codec())
codec_registry.register("int2_dithered", DitheredINT2Codec())
codec_registry.register("nbit_dithered", DitheredNBitCodec())
codec_registry.register("fp8_default", FP8Codec())

# Register mock codecs for testing
codec_registry.register("mock_int8", MockINT8Codec())
codec_registry.register("mock_fp8", MockFP8Codec())
codec_registry.register("mock_int4", MockINT4Codec())
codec_registry.register("mock_corrupt", MockCorruptCodec())
codec_registry.register("mock_fail", MockFailingCodec())

__all__ = [
    "BF16Codec",
    "FP8Codec",
    "INT8Codec",
    "INT4Codec",
    "DitheredINT2Codec",
    "DitheredNBitCodec",
    "checksum",
    "MockFP8Codec",
    "MockINT8Codec",
    "MockINT4Codec",
    "MockFailingCodec",
    "MockCorruptCodec",
]
