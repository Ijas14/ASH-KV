"""Prometheus exporter for ASH-KV counters.

Lazy-imports prometheus_client. If the library is not installed,
the exporter is a no-op. This keeps prometheus_client as an optional
dependency — ASH-KV runs fine without it.

Cold path only. The exporter is scraped by Prometheus on an interval,
not called from the hot path.
"""
from __future__ import annotations

from typing import Optional

from .counters import Counters, counters as default_counters


class PrometheusExporter:
    """Exports ASH-KV counters to Prometheus.

    Usage:
        exporter = PrometheusExporter(counters)
        exporter.start(port=9100)  # starts HTTP server
        # ... ASH-KV runs ...
        exporter.stop()
    """

    def __init__(self, counters: Counters = default_counters) -> None:
        self._counters = counters
        self._started = False
        self._registry = None
        self._metrics: dict = {}

    def start(self, port: int = 9100) -> bool:
        """Start the Prometheus HTTP exporter.

        Returns True if started, False if prometheus_client is not
        installed or already started. Never raises.
        """
        if self._started:
            return True

        try:
            from prometheus_client import (
                CollectorRegistry,
                Counter as PromCounter,
                Gauge as PromGauge,
                start_http_server,
            )
        except ImportError:
            return False

        try:
            self._registry = CollectorRegistry()

            # Migration counters
            self._metrics["migrations_ok"] = PromCounter(
                "ashkv_migrations_ok_total",
                "Total successful migrations",
                registry=self._registry,
            )
            self._metrics["migrations_failure"] = PromCounter(
                "ashkv_migrations_failure_total",
                "Total failed migrations",
                registry=self._registry,
            )
            self._metrics["migrations_corrupt"] = PromCounter(
                "ashkv_migrations_corrupt_total",
                "Total corrupt migrations (checksum mismatch)",
                registry=self._registry,
            )

            # Tier gauges
            for tier_name in ("bf16", "fp8", "int4", "archive", "cpu", "disk"):
                self._metrics[f"pages_in_{tier_name}"] = PromGauge(
                    f"ashkv_pages_in_{tier_name}",
                    f"Pages currently in {tier_name.upper()} tier",
                    registry=self._registry,
                )

            # Pressure gauges
            for state_name in ("normal", "elevated", "critical", "saturated"):
                self._metrics[f"pressure_{state_name}"] = PromCounter(
                    f"ashkv_pressure_{state_name}_total",
                    f"Decode steps at {state_name} pressure",
                    registry=self._registry,
                )

            # Fallback counters
            self._metrics["bf16_recoveries"] = PromCounter(
                "ashkv_bf16_recoveries_total",
                "Total BF16 page recoveries",
                registry=self._registry,
            )
            self._metrics["breakers_tripped"] = PromCounter(
                "ashkv_breakers_tripped_total",
                "Total circuit breaker trips",
                registry=self._registry,
            )

            start_http_server(port, registry=self._registry)
            self._started = True
            return True
        except Exception:
            self._started = False
            self._registry = None
            self._metrics = {}
            return False

    def update(self) -> None:
        """Push current counter values to Prometheus.

        Call this on a scrape interval (e.g., every 5 seconds).
        Never raises.
        """
        if not self._started:
            return

        try:
            snap = self._counters.snapshot()
            for name, value in snap.items():
                metric = self._metrics.get(name)
                if metric is None:
                    continue
                # For Counter metrics, we need to set the value.
                # prometheus_client Counter only increments, so we
                # track the delta. For Gauge, we set directly.
                from prometheus_client import Counter as PromCounter, Gauge as PromGauge
                if isinstance(metric, PromGauge):
                    metric.set(value)
                elif isinstance(metric, PromCounter):
                    # Counter only goes up. We track the last value
                    # and increment by the delta.
                    last = getattr(metric, "_ashkv_last", 0)
                    delta = value - last
                    if delta > 0:
                        metric.inc(delta)
                    metric._ashkv_last = value  # type: ignore[attr-defined]
        except Exception:
            pass  # Never let exporter errors affect the system

    def stop(self) -> None:
        """Stop the exporter. For cleanup."""
        self._started = False
        self._registry = None
        self._metrics = {}
