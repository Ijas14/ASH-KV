"""Circuit breaker for codecs.

Tracks per-codec failures and trips when the failure count exceeds a
threshold within a time window. A tripped breaker prevents that codec
from being dispatched for the duration of the window.

This is the first line of defense against a bad kernel crashing the
server. Instead of repeatedly trying a failing codec, the breaker
short-circuits: pages that would have used the codec stay on their
current tier.

Cold-path state, hot-path reads. The breaker is checked by the
migration engine before dispatching to a codec.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class CircuitBreaker:
    """Per-codec failure tracker.

    Not thread-safe. Callers must hold the GIL and serialize hot-path
    operations (which they do, since the controller runs single-threaded
    per decode step).
    """

    threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 60.0

    _failures: list[float] = field(default_factory=list, repr=False)
    _tripped: bool = False
    _tripped_at: float = 0.0

    @property
    def is_tripped(self) -> bool:
        """True if the breaker is currently tripped.

        Automatically resets after cooldown_seconds have elapsed since
        the trip. Never raises.
        """
        if not self._tripped:
            return False
        if time.monotonic() - self._tripped_at >= self.cooldown_seconds:
            # Cooldown elapsed — reset
            self._tripped = False
            self._failures.clear()
            return False
        return True

    def record_failure(self) -> None:
        """Record a failure. May trip the breaker.

        Never raises. Prunes old failures outside the window.
        """
        now = time.monotonic()

        # Prune failures outside the window
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t >= cutoff]

        # Record this failure
        self._failures.append(now)

        # Check threshold
        if len(self._failures) >= self.threshold and not self._tripped:
            self._tripped = True
            self._tripped_at = now

    def record_success(self) -> None:
        """Record a successful operation.

        Does NOT reset the breaker (hysteresis — once tripped, stays
        tripped for the cooldown). But it does prune old failures,
        which helps the breaker recover faster after cooldown.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t >= cutoff]

    def reset(self) -> None:
        """Manually reset the breaker. Use for testing or admin ops."""
        self._tripped = False
        self._failures.clear()
        self._tripped_at = 0.0

    @property
    def failure_count(self) -> int:
        """Current failure count within the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        return sum(1 for t in self._failures if t >= cutoff)


@dataclass(slots=True)
class CircuitBreakerRegistry:
    """Manages breakers for all codecs.

    Keyed by codec name (string). The migration engine looks up the
    breaker for a codec before dispatching.
    """

    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict, repr=False)

    def get_or_create(self, codec_name: str) -> CircuitBreaker:
        """Get the breaker for a codec, creating it if needed."""
        if codec_name not in self._breakers:
            self._breakers[codec_name] = CircuitBreaker()
        return self._breakers[codec_name]

    def is_codec_available(self, codec_name: str) -> bool:
        """True if the codec's breaker is not tripped.

        Never raises. Unknown codecs are considered available (no
        breaker = no failures).
        """
        breaker = self._breakers.get(codec_name)
        if breaker is None:
            return True
        return not breaker.is_tripped

    def record_codec_failure(self, codec_name: str) -> None:
        """Record a failure for a codec. May trip its breaker."""
        breaker = self.get_or_create(codec_name)
        breaker.record_failure()

    def record_codec_success(self, codec_name: str) -> None:
        """Record a success for a codec."""
        breaker = self._breakers.get(codec_name)
        if breaker is not None:
            breaker.record_success()

    def all_breakers(self) -> dict[str, CircuitBreaker]:
        """Return a snapshot of all breakers. For telemetry."""
        return dict(self._breakers)

    def reset_all(self) -> None:
        """Reset all breakers. For testing or admin ops."""
        for breaker in self._breakers.values():
            breaker.reset()
