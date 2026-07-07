"""Codec registry.

Plugin codecs register here at import time. After startup, the
registry is FROZEN — any further registration attempt raises
RuntimeError. This prevents dynamic codec loading during inference.

The compiler queries the registry to build the codec table that
the runtime uses.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..contracts.protocols import Codec
from ..contracts.tiers import Tier


class CodecRegistry:
    """Frozen-after-startup codec registry.

    Usage:
        # At plugin import time (cold path):
        codec_registry.register("my_fp8", MyFP8Codec())

        # After all plugins loaded (cold path):
        codec_registry.freeze()

        # Any later attempt:
        codec_registry.register("late", ...)  # raises RuntimeError
    """

    __slots__ = ("_by_name", "_frozen")

    def __init__(self) -> None:
        self._by_name: Dict[str, Codec] = {}
        self._frozen: bool = False

    def register(self, name: str, codec: Codec) -> None:
        """Register a codec by name. Raises if frozen or duplicate."""
        if self._frozen:
            raise RuntimeError(
                f"CodecRegistry is frozen; cannot register '{name}'. "
                f"All registrations must happen at startup."
            )
        if name in self._by_name:
            raise ValueError(f"Codec '{name}' already registered")
        self._by_name[name] = codec

    def get(self, name: str) -> Codec | None:
        return self._by_name.get(name)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name.keys())


# Singleton registry. Import this; do not instantiate your own.
codec_registry = CodecRegistry()


def build_codec_table(
    config: object,
    registry: CodecRegistry = codec_registry,
) -> Dict[Tuple[int, int], Codec]:
    """Resolve a config + registry into a flat (from, to) -> Codec dict.

    The config object is expected to expose which named codecs to use
    for which tier transitions. For now, we use a simple convention:
    the integration layer passes a dict of {(from_tier, to_tier): name}
    and we resolve names to Codecs.

    This function is called by build_runtime() (or by the integration
    layer directly) at startup.
    """
    # Default codec table: empty. The integration layer is responsible
    # for filling this in based on what codecs are registered and what
    # the config requests.
    return {}


# Re-exported for type-checking in compiler/runtime_builder.py
__all__ = ["codec_registry", "CodecRegistry", "build_codec_table"]
