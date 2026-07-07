"""ASHKVConfig — the 8-number policy surface.

This is the ONLY configuration the controller sees. Adding a 9th
parameter requires a design review and a coordinated contract change
across all three splits.

Principle: this object is consumed only by the compiler at startup.
The hot path receives closures derived from this object, never the
object itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ASHKVConfig:
    """Frozen configuration. Constructed once at startup, never mutated.

    Eight numbers. That is the entire tunable surface of ASH-KV.
    """

    # --- Score weights (must sum to 1.0, enforced at validation) ---
    w_T: float = 0.7   # temporal decay
    w_S: float = 0.1   # saliency (attention-derived)
    w_N: float = 0.1   # novelty (key-delta or residual energy)
    w_P: float = 0.1   # prefix affinity (shared-prefix reuse probability)

    # --- Controller thresholds (score-domain, in [0, 1]) ---
    theta_high: float = 0.72   # R >= theta_high + delta  =>  BF16
    theta_low: float = 0.33    # R <= theta_low - delta    =>  INT4 or colder

    # --- Hysteresis offset (score-domain) ---
    # Demotion requires R <= theta_low - delta.
    # Promotion requires R >= theta_high + delta.
    # This prevents flapping.
    delta: float = 0.04

    # --- Emergency pressure threshold ---
    # Allocator pressure >= p_emergency triggers CRITICAL state.
    p_emergency: float = 0.95

    def __post_init__(self) -> None:
        """Validate at construction. Cold path only. Raises on invalid."""
        w_sum = self.w_T + self.w_S + self.w_N + self.w_P
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Score weights must sum to 1.0, got {w_sum!r} "
                f"(w_T={self.w_T}, w_S={self.w_S}, w_N={self.w_N}, w_P={self.w_P})"
            )
        for name, val in (
            ("w_T", self.w_T), ("w_S", self.w_S),
            ("w_N", self.w_N), ("w_P", self.w_P),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val!r}")

        if not 0.0 <= self.theta_high <= 1.0:
            raise ValueError(f"theta_high must be in [0, 1], got {self.theta_high!r}")
        if not 0.0 <= self.theta_low <= 1.0:
            raise ValueError(f"theta_low must be in [0, 1], got {self.theta_low!r}")
        if self.theta_low >= self.theta_high:
            raise ValueError(
                f"theta_low ({self.theta_low!r}) must be strictly less than "
                f"theta_high ({self.theta_high!r})"
            )

        if self.delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {self.delta!r}")

        if not 0.0 < self.p_emergency < 1.0:
            raise ValueError(
                f"p_emergency must be in (0, 1), got {self.p_emergency!r}"
            )

    def as_dict(self) -> dict[str, float]:
        """Serialize to a plain dict. For logging, A/B bookkeeping, etc."""
        return {
            "w_T": self.w_T, "w_S": self.w_S,
            "w_N": self.w_N, "w_P": self.w_P,
            "theta_high": self.theta_high, "theta_low": self.theta_low,
            "delta": self.delta, "p_emergency": self.p_emergency,
        }
