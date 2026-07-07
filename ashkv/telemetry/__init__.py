"""Telemetry layer for ASH-KV.

Exports counters and Prometheus integration. The hot path interacts
with telemetry via the compiled closure (real or no-op). The cold
path (exporter) reads counters on a scrape interval.
"""
from __future__ import annotations

from .counters import Counters, counters
from .prometheus import PrometheusExporter

__all__ = ["Counters", "counters", "PrometheusExporter"]
