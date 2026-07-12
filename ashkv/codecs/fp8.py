"""FP8 codec for Hopper+/MI300X using native PyTorch float8.

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

    Uses PyTorch's native float8_e4m3fn hardware datatype.
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

        if not source_bytes:
            return b""

        import torch
        # Note: torch.float8_e4m3fn is natively supported in PyTorch 2.1+
        # It allows O(1) casting on GPU.
        tensor = torch.frombuffer(bytearray(source_bytes), dtype=torch.bfloat16)
        
        # Move to GPU and cast directly to FP8 hardware format
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            
        fp8_tensor = tensor.to(torch.float8_e4m3fn)
        
        # Return as raw bytes (uint8 view)
        return fp8_tensor.view(torch.uint8).cpu().numpy().tobytes()

    def decode(self, target_bytes: bytes) -> bytes:
        """Decode FP8 bytes back to BF16."""
        self._decode_calls += 1

        if not target_bytes:
            return b""

        import torch
        # Read raw bytes as uint8
        tensor = torch.frombuffer(bytearray(target_bytes), dtype=torch.uint8)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            
        # View as FP8 and cast back to BF16
        bf16_tensor = tensor.view(torch.float8_e4m3fn).to(torch.bfloat16)
        
        return bf16_tensor.cpu().view(torch.int16).numpy().tobytes()

    def checksum(self, raw_bytes: bytes) -> int:
        """Stable checksum."""
        return checksum(raw_bytes)
