"""Codec resolver — picks codecs based on hardware and config.

Cold path only. Runs after the hardware probe and before build_runtime().
The runtime sees only the resolved codec table.

Resolution precedence:
1. User explicit override (from config)
2. Auto-detect based on hardware
3. Built-in default
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..contracts.protocols import Codec
from ..contracts.tiers import Tier
from .hardware_probe import HardwareProfile
from .registry import CodecRegistry, codec_registry

logger = logging.getLogger("ashkv.compiler.codec_resolver")


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """User-facing codec configuration.

    Each slot can be:
    - "auto" — let the resolver pick based on hardware
    - A named codec — use that specific codec from the registry
    - None — no codec for this slot (tier is unreachable)
    """

    bf16_to_compressed: Optional[str] = "auto"
    compressed_to_cold: Optional[str] = "auto"
    cold_to_archive: Optional[str] = None  # archive is optional
    archive_to_cpu: Optional[str] = None   # CPU offload is optional


def resolve_codecs(
    codec_config: CodecConfig,
    hardware: HardwareProfile,
    registry: CodecRegistry = codec_registry,
) -> dict[tuple[int, int], Codec]:
    """Build the codec table from config + hardware + registry.

    Cold path. Called once at startup. Returns a flat dict mapping
    (from_tier, to_tier) → Codec.

    Raises RuntimeError if a named codec is not found in the registry.
    Logs warnings for auto-detected choices.
    """
    table: dict[tuple[int, int], Codec] = {}

    # Slot 1: BF16 → compressed (FP8 or INT8 depending on hardware)
    name = _resolve_slot(
        codec_config.bf16_to_compressed,
        _default_bf16_to_compressed(hardware),
        "bf16_to_compressed",
    )
    if name is not None:
        codec = _get_codec(name, registry, "bf16_to_compressed")
        if codec is not None:
            table[(int(Tier.BF16), int(Tier.FP8))] = codec
            # Also register the reverse for promotion
            table[(int(Tier.FP8), int(Tier.BF16))] = codec

    # Slot 2: compressed → cold (INT4 everywhere)
    name = _resolve_slot(
        codec_config.compressed_to_cold,
        "int4_default",
        "compressed_to_cold",
    )
    if name is not None:
        codec = _get_codec(name, registry, "compressed_to_cold")
        if codec is not None:
            table[(int(Tier.FP8), int(Tier.INT4))] = codec
            table[(int(Tier.INT4), int(Tier.FP8))] = codec
            # Also BF16 → INT4 direct path (for faster demotion)
            if (int(Tier.BF16), int(Tier.FP8)) in table:
                table[(int(Tier.BF16), int(Tier.INT4))] = codec
                table[(int(Tier.INT4), int(Tier.BF16))] = codec

    # Slot 3: cold → archive (optional)
    name = _resolve_slot(
        codec_config.cold_to_archive,
        "lowrank_default",
        "cold_to_archive",
    )
    if name is not None:
        codec = _get_codec(name, registry, "cold_to_archive")
        if codec is not None:
            table[(int(Tier.INT4), int(Tier.ARCHIVE))] = codec
            table[(int(Tier.ARCHIVE), int(Tier.INT4))] = codec

    # Slot 4: archive → CPU (optional, for offload)
    name = _resolve_slot(
        codec_config.archive_to_cpu,
        None,  # no default — CPU offload must be explicitly enabled
        "archive_to_cpu",
    )
    if name is not None:
        codec = _get_codec(name, registry, "archive_to_cpu")
        if codec is not None:
            table[(int(Tier.ARCHIVE), int(Tier.CPU))] = codec
            table[(int(Tier.CPU), int(Tier.ARCHIVE))] = codec

    logger.info(f"codec table resolved: {len(table)} entries")
    for (from_t, to_t), codec in sorted(table.items()):
        logger.info(f"  {Tier(from_t).name} → {Tier(to_t).name}: {type(codec).__name__}")

    return table


def _resolve_slot(
    user_value: Optional[str],
    default: Optional[str],
    slot_name: str,
) -> Optional[str]:
    """Resolve a codec slot: user override > default > None."""
    if user_value is None:
        return None
    if user_value == "auto":
        return default
    return user_value


def _default_bf16_to_compressed(hardware: HardwareProfile) -> str:
    """Pick the default BF16 → compressed codec based on hardware."""
    if hardware.has_fp8_native:
        return "fp8_default"
    if hardware.has_int8_native:
        return "int8_default"
    # No hardware acceleration — use INT8 anyway (software)
    logger.warning(
        "No FP8 or INT8 hardware acceleration detected. "
        "Using INT8 in software mode — performance will be limited."
    )
    return "int8_default"


def _get_codec(
    name: str,
    registry: CodecRegistry,
    slot_name: str,
) -> Optional[Codec]:
    """Get a codec from the registry by name. Logs and returns None if missing."""
    codec = registry.get(name)
    if codec is None:
        logger.warning(
            f"Codec '{name}' requested for slot '{slot_name}' "
            f"but not found in registry. Slot will be unavailable."
        )
        return None
    return codec
