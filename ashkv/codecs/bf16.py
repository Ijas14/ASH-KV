"""BF16 identity codec.

The simplest codec: BF16 → BF16. Used for:
- The fallback tier (page-level recovery to BF16)
- Testing the migration engine without real compression
- As a reference implementation of the Codec protocol

This codec does no compression — it's a passthrough. But it still
implements the full protocol including checksum verification.
"""
from __future__ import annotations

import hashlib


class BF16Codec:
    """Identity codec for the BF16 tier.

    encode() returns the input unchanged.
    decode() returns the input unchanged.
    checksum() computes a stable hash of the bytes.

    Stateless and thread-safe.
    """

    __slots__ = ()

    def encode(self, source_bytes: bytes) -> bytes:
        """Return source bytes unchanged. BF16 is the reference format."""
        return source_bytes

    def decode(self, target_bytes: bytes) -> bytes:
        """Return target bytes unchanged."""
        return target_bytes

    def checksum(self, raw_bytes: bytes) -> int:
        """Compute a stable 64-bit checksum of the bytes.

        Uses SHA-256 truncated to 64 bits. This is deterministic and
        collision-resistant, which is what we need for integrity
        verification.
        """
        h = hashlib.sha256(raw_bytes).digest()
        return int.from_bytes(h[:8], "little", signed=False)
