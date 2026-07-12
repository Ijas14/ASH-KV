"""Hardware probe — detects GPU capabilities at startup.

Cold path only. Called once at startup before codec resolution.
The runtime never calls this — it sees only the resolved codec table.

The probe is conservative: if uncertain, it falls back to the safer
codec (INT8 over FP8) and logs a warning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("ashkv.compiler.hardware_probe")


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Detected hardware capabilities.

    Consumed by the codec resolver to pick appropriate codecs.
    Never crosses into runtime/.
    """

    gpu_name: str
    compute_capability: tuple[int, int]
    vram_bytes: int
    has_fp8_native: bool
    has_int8_native: bool
    has_bf16_native: bool
    is_amd: bool


def probe_hardware(device_id: int = 0) -> HardwareProfile:
    """Probe the GPU and return a HardwareProfile.

    Cold path. Called once at startup. Never called from runtime.

    Falls back gracefully if CUDA/ROCm is not available — returns
    a CPU-only profile that has no hardware acceleration.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA/ROCm not available — returning CPU-only profile")
            return _cpu_profile()
    except ImportError:
        logger.warning("PyTorch not installed — returning CPU-only profile")
        return _cpu_profile()

    try:
        import torch
        props = torch.cuda.get_device_properties(device_id)
        name = props.name
        cc = (props.major, props.minor)
        vram = props.total_mem

        # Detect AMD vs NVIDIA
        is_amd = "AMD" in name.upper() or "Instinct" in name.upper()

        # Determine capabilities
        if is_amd:
            # MI300X (CDNA3) and later have native FP8
            # MI250 (CDNA2) does not
            has_fp8 = "MI300" in name or "MI325" in name or "MI350" in name
            has_int8 = True  # all modern AMD GPUs support INT8
            has_bf16 = True  # CDNA2+ supports BF16
        else:
            # NVIDIA
            # Hopper (cc 9.0) and later have native FP8
            # Ampere (cc 8.0/8.6) does NOT have native FP8
            has_fp8 = cc >= (9, 0)
            has_int8 = cc >= (7, 5)  # Turing+
            has_bf16 = cc >= (8, 0)  # Ampere+

        profile = HardwareProfile(
            gpu_name=name,
            compute_capability=cc,
            vram_bytes=vram,
            has_fp8_native=has_fp8,
            has_int8_native=has_int8,
            has_bf16_native=has_bf16,
            is_amd=is_amd,
        )

        logger.info(f"hardware probe: {name} (cc {cc[0]}.{cc[1]})")
        logger.info(f"  has_fp8_native: {has_fp8}")
        logger.info(f"  has_int8_native: {has_int8}")
        logger.info(f"  has_bf16_native: {has_bf16}")

        if not has_fp8 and has_int8:
            logger.info("  → will use INT8 for compressed tier (no native FP8)")

        if has_int8:
            _warmup_triton()

        return profile

    except Exception as e:
        logger.warning(f"hardware probe failed: {e} — returning CPU-only profile")
        return _cpu_profile()


def _warmup_triton() -> None:
    """Warm up Triton compiler for common head dimensions to prevent JIT latency spikes."""
    try:
        from ashkv.codecs.int8 import INT8Codec
        for hidden_dim in [64, 128]:
            codec = INT8Codec(hidden_dim=hidden_dim)
            dummy_bf16 = (b'\x00' * 2) * hidden_dim
            compressed = codec.encode(dummy_bf16)
            codec.decode(compressed)
        logger.info("Triton JIT warmup complete (AOT emulation).")
    except Exception as e:
        logger.warning(f"Triton warmup failed: {e}")


def _cpu_profile() -> HardwareProfile:
    """Fallback profile when no GPU is detected."""
    return HardwareProfile(
        gpu_name="CPU",
        compute_capability=(0, 0),
        vram_bytes=0,
        has_fp8_native=False,
        has_int8_native=False,
        has_bf16_native=False,
        is_amd=False,
    )
