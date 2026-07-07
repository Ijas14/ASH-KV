"""Protocols (interfaces) for codecs and allocators.

Split 1 defines these. Split 2 implements Codec. Split 3 implements
Allocator. Neither protocol carries any runtime assumption beyond
the signatures — no fields, no base class, no behavior.

A codec or allocator that satisfies these signatures is valid. A
codec or allocator that needs more (e.g., a "tier" parameter on
encode) is a contract change, not an extension.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .tiers import Tier


@runtime_checkable
class Codec(Protocol):
    """Encode/decode a page's bytes between two representations.

    Implementations are stateless. The same codec instance may be
    called concurrently from multiple threads.

    Contract:
    - encode(source_bytes) returns the compressed representation.
    - decode(target_bytes) returns bytes equivalent to source_bytes
      within the codec's documented tolerance.
    - checksum(raw_bytes) computes an integer checksum over
      BF16-equivalent bytes. The same bytes always produce the same
      checksum.

    The migration engine verifies integrity by computing
    checksum(decode(encode(x))) and comparing against the page's
    bf16_checksum. Mismatch => CORRUPT, page quarantined.
    """

    def encode(self, source_bytes: bytes) -> bytes: ...

    def decode(self, target_bytes: bytes) -> bytes: ...

    def checksum(self, raw_bytes: bytes) -> int: ...


@runtime_checkable
class Allocator(Protocol):
    """Memory allocator with pressure reporting.

    The allocator owns bytes. The controller sees only `pressure`.

    Contract:
    - alloc(tier, size_bytes) returns an opaque integer handle.
      Returns -1 on failure (never raises).
    - free(handle) releases the buffer. No-op on invalid handle.
    - read(handle) returns the bytes at that handle.
      Returns b"" on invalid handle (never raises).
    - write(handle, data) writes bytes. No-op on invalid handle.
    - pressure() returns a scalar in [0, 1].
      0.0 = empty, 1.0 = full.

    The controller is allowed to see ONLY the pressure() scalar.
    No other allocator state crosses the boundary. If Split 3 finds
    itself needing two scalars (e.g., HBM pressure + CPU pressure),
    that is a contract change requiring coordinated review.
    """

    def alloc(self, tier: Tier, size_bytes: int) -> int: ...

    def free(self, handle: int) -> None: ...

    def read(self, handle: int) -> bytes: ...

    def write(self, handle: int, data: bytes) -> None: ...

    def pressure(self) -> float: ...
