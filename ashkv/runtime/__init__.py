"""Runtime kernel — the hot path.

This package imports ONLY contracts/ and stdlib + numpy. It never
imports compiler/, codecs/, config/, safety/, or sglang_integration/.

If you add an import here that is not contracts/ or numpy/stdlib,
the dependency-direction test will fail.
"""
from __future__ import annotations

from .allocator import MockAllocator
from .controller import desired_tiers
from .migrate import CodecTable, migrate
from .score import score_vectorized

__all__ = [
    "score_vectorized",
    "desired_tiers",
    "migrate",
    "CodecTable",
    "MockAllocator",
]
