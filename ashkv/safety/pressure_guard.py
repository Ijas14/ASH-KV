"""Pressure guard — maps a pressure scalar to a PressureState.

This is the safety layer that sits between the allocator (which
computes pressure) and the controller (which reacts to it). The
guard classifies pressure into four states, each with a defined
response.

The guard is stateless in the sense that it doesn't store pressure
history. It's a pure function from (pressure, config) → state.
The controller and the integration layer use the state to decide
how to react.
"""
from __future__ import annotations

from ..contracts.config import ASHKVConfig
from ..contracts.results import (
    PRESSURE_ELEVATED_THRESHOLD,
    PRESSURE_SATURATED_THRESHOLD,
    PressureState,
)


def classify_pressure(pressure: float, config: ASHKVConfig) -> PressureState:
    """Classify a pressure scalar into a PressureState.

    Pure function. Never raises. Invalid pressure values (NaN, < 0,
    > 1) are clamped to the nearest valid state.

    State boundaries:
        NORMAL     pressure < 0.85
        ELEVATED   0.85 <= pressure < p_emergency
        CRITICAL   p_emergency <= pressure < 0.99
        SATURATED  pressure >= 0.99
    """
    # Clamp defensively
    if pressure < 0.0:
        pressure = 0.0
    elif pressure > 1.0:
        pressure = 1.0
    # NaN check
    if pressure != pressure:  # NaN
        pressure = 1.0  # treat as worst case

    if pressure < PRESSURE_ELEVATED_THRESHOLD:
        return PressureState.NORMAL
    if pressure < config.p_emergency:
        return PressureState.ELEVATED
    if pressure < PRESSURE_SATURATED_THRESHOLD:
        return PressureState.CRITICAL
    return PressureState.SATURATED


def should_admit_new_request(state: PressureState) -> bool:
    """Whether the system should admit a new request under this state.

    NORMAL and ELEVATED: yes.
    CRITICAL: no (system is struggling, don't add load).
    SATURATED: absolutely no.
    """
    if state == PressureState.NORMAL:
        return True
    if state == PressureState.ELEVATED:
        return True
    return False


def should_demote_aggressively(state: PressureState) -> bool:
    """Whether the controller should demote pages more aggressively.

    NORMAL: no.
    ELEVATED: yes (start freeing memory proactively).
    CRITICAL: yes (free memory urgently).
    SATURATED: yes (free everything possible).
    """
    return state != PressureState.NORMAL


def should_offload_to_cpu(state: PressureState) -> bool:
    """Whether the system should start offloading cold pages to CPU.

    Only under CRITICAL or SATURATED. Under ELEVATED, GPU-tier
    compression is sufficient.
    """
    return state in (PressureState.CRITICAL, PressureState.SATURATED)
