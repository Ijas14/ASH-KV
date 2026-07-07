"""Compiler package — cold path.

Turns ASHKVConfig into a CompiledRuntime. Imports contracts, codecs,
plugins, config. Never imported by runtime/.
"""
from __future__ import annotations

from .codec_resolver import CodecConfig, resolve_codecs
from .hardware_probe import HardwareProfile, probe_hardware
from .registry import CodecRegistry, codec_registry, build_codec_table
from .runtime_builder import (
    CompiledRuntime,
    build_runtime,
    compile_controller,
    compile_migrate,
    compile_score,
    compile_telemetry,
)

__all__ = [
    "CompiledRuntime",
    "build_runtime",
    "compile_score",
    "compile_controller",
    "compile_migrate",
    "compile_telemetry",
    "CodecRegistry",
    "codec_registry",
    "build_codec_table",
    "HardwareProfile",
    "probe_hardware",
    "CodecConfig",
    "resolve_codecs",
]
