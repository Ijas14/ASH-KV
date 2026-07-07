"""Checksum utility for codecs.

Provides a fast, deterministic checksum function that all codecs
can use for integrity verification.

The checksum must be:
- Deterministic: same bytes → same int, always
- Fast: called on every migration verify step
- Collision-resistant: low probability of two different byte
  sequences producing the same checksum

We use xxhash if available (very fast), falling back to hashlib
(standard library, always available).
"""
from __future__ import annotations

import hashlib

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False


def checksum(raw_bytes: bytes) -> int:
    """Compute a 64-bit checksum of the bytes.

    Uses xxhash if available (faster), otherwise falls back to
    SHA-256 truncated to 64 bits.

    Deterministic. Never raises.
    """
    if _HAS_XXHASH:
        return xxhash.xxh64(raw_bytes).intdigest()
    h = hashlib.sha256(raw_bytes).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def checksum_pair(raw_bytes: bytes) -> tuple[int, int]:
    """Compute two independent 64-bit checksums.

    Used for extra safety in critical paths. Returns (checksum_a, checksum_b)
    where checksum_a uses xxhash/SHA and checksum_b uses a different algorithm.
    """
    if _HAS_XXHASH:
        a = xxhash.xxh64(raw_bytes).intdigest()
    else:
        h = hashlib.sha256(raw_bytes).digest()
        a = int.from_bytes(h[:8], "little", signed=False)

    # Second checksum: MD5 truncated (different algorithm, independent collisions)
    h2 = hashlib.md5(raw_bytes).digest()
    b = int.from_bytes(h2[:8], "little", signed=False)
    return (a, b)
