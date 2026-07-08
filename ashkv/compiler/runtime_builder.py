"""Compiler: turn ASHKVConfig into a specialized runtime.

This is the cold path. It runs once at startup (and on config reload).
The hot path receives the closures produced here and never touches
config directly.

The compiler does:
1. Resolve the codec registry into a flat dict.
2. Build score_fn as a closure over weights.
3. Build controller_fn as a closure over thresholds.
4. Build telemetry_fn as either real or no-op.
5. Assemble a Runtime object.

Principle: nothing the compiler produces depends on config at call
time. The closures capture values, not references to the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..contracts.config import ASHKVConfig
from ..contracts.page import PageTable
from ..contracts.protocols import Allocator
from ..contracts.results import MigrationResult
from ..contracts.tiers import Tier
from ..runtime.controller import desired_tiers
from ..runtime.migrate import CodecTable, migrate
from ..runtime.score import score_vectorized


# Type aliases for clarity
ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ControllerFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
MigrateFn = Callable[[int, Tier, PageTable, Allocator, CodecTable, Callable, Callable], MigrationResult]
TelemetryFn = Callable[[MigrationResult], None]


@dataclass(slots=True, frozen=True)
class CompiledRuntime:
    """The output of the compiler. Immutable after construction.

    The hot path holds a reference to this object and calls its
    function members. No member is ever re-bound; reload creates a
    new CompiledRuntime and atomic-swaps the reference.
    """

    config: ASHKVConfig
    score_fn: ScoreFn
    controller_fn: ControllerFn
    migrate_fn: MigrateFn
    codec_table: CodecTable
    telemetry_fn: TelemetryFn


def compile_score(config: ASHKVConfig) -> ScoreFn:
    """Build a score function closure over the config's weights.

    The returned function takes (T, S, N, P) arrays and returns R.
    It does NOT read config at call time.
    """
    w_T = config.w_T
    w_S = config.w_S
    w_N = config.w_N
    w_P = config.w_P

    def score_fn(
        T: np.ndarray,
        S: np.ndarray,
        N: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        return score_vectorized(T, S, N, P, w_T, w_S, w_N, w_P)

    return score_fn


def compile_controller(config: ASHKVConfig, codec_table: CodecTable) -> ControllerFn:
    """Build a controller function closure over thresholds and topology.

    The returned function takes (R, current_tiers, pressure) and
    returns target_tiers. It does NOT read config at call time.
    """
    theta_high = config.theta_high
    theta_low = config.theta_low
    delta = config.delta
    p_emergency = config.p_emergency

    num_tiers = len(Tier)
    next_colder = np.arange(num_tiers, dtype=np.int8)
    next_hotter = np.arange(num_tiers, dtype=np.int8)
    
    # Default sequential fallback
    for i in range(num_tiers - 1):
        next_colder[i] = i + 1
    for i in range(1, num_tiers):
        next_hotter[i] = i - 1

    # Override with actual reachable tiers from codec_table
    for src in Tier:
        colder_targets = [tgt for (s, tgt) in codec_table.keys() if s == src and int(tgt) > int(src)]
        if colder_targets:
            next_colder[int(src)] = int(min(colder_targets, key=int))
            
        hotter_targets = [tgt for (s, tgt) in codec_table.keys() if s == src and int(tgt) < int(src)]
        if hotter_targets:
            next_hotter[int(src)] = int(max(hotter_targets, key=int))

    def controller_fn(
        R: np.ndarray,
        current_tiers: np.ndarray,
        pressure: float,
    ) -> np.ndarray:
        return desired_tiers(
            R, current_tiers, pressure,
            theta_high, theta_low, delta, p_emergency,
            next_colder, next_hotter
        )

    return controller_fn


def compile_migrate() -> MigrateFn:
    """The migrate function is already stateless — just bind it.

    We could add closure capture here for circuit breakers, logging,
    etc., but the contract says migrate() is pure: it does what it's
    told and reports the outcome. Safety layers wrap it from outside.
    """
    return migrate


def compile_telemetry(enabled: bool) -> TelemetryFn:
    """Build a telemetry function. Either real (records) or no-op.

    This is the no-op binding pattern: the hot path calls telemetry_fn(result)
    unconditionally; whether anything is recorded depends on which
    function was bound at compile time.
    """
    if not enabled:
        def noop_telemetry(_result: MigrationResult) -> None:
            pass
        return noop_telemetry

    # Real telemetry: increment counters. The counters object is
    # owned by the integration layer; here we just hold a reference.
    # For now, we keep a simple internal counter dict.
    counters: dict[str, int] = {
        "migrations_ok": 0,
        "migrations_failure": 0,
        "migrations_skipped": 0,
        "migrations_corrupt": 0,
        "migrations_fallback": 0,
    }

    def real_telemetry(result: MigrationResult) -> None:
        # Map status to counter name. Single branch, one function call.
        key = {
            # MigrationStatus.OK
            0: "migrations_ok",
            # MigrationStatus.SKIPPED
            1: "migrations_skipped",
            # MigrationStatus.FAILURE
            2: "migrations_failure",
            # MigrationStatus.FALLBACK
            3: "migrations_fallback",
            # MigrationStatus.CORRUPT
            4: "migrations_corrupt",
        }.get(int(result.status), "migrations_failure")
        counters[key] = counters.get(key, 0) + 1

    # Stash the counters on the function object for the integration
    # layer to read. This is the one allowed bit of "state" — but
    # it's cold-path state (read by the metrics exporter, not by the
    # hot path).
    real_telemetry.counters = counters  # type: ignore[attr-defined]
    return real_telemetry


def build_runtime(
    config: ASHKVConfig,
    codec_table: CodecTable,
    telemetry_enabled: bool = False,
) -> CompiledRuntime:
    """Assemble a CompiledRuntime from config and a codec table.

    This is the entry point. The integration layer calls this once
    at startup, holds the returned CompiledRuntime, and passes its
    function members to the hot path.
    """
    return CompiledRuntime(
        config=config,
        score_fn=compile_score(config),
        controller_fn=compile_controller(config, codec_table),
        migrate_fn=compile_migrate(),
        codec_table=codec_table,
        telemetry_fn=compile_telemetry(telemetry_enabled),
    )
