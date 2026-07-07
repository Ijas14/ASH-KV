"""Hot-path score computation.

Pure vectorized numpy. No config access. No allocator access.
No codec access. No exceptions.

The score function is built once at startup by the compiler and
bound to a closure that captures the weights. The hot path calls
that closure; it never reads config.
"""
from __future__ import annotations

import numpy as np


def score_vectorized(
    T: np.ndarray,
    S: np.ndarray,
    N: np.ndarray,
    P: np.ndarray,
    w_T: float,
    w_S: float,
    w_N: float,
    w_P: float,
) -> np.ndarray:
    """Compute the fidelity score R for an array of pages.

    All inputs are float32 numpy arrays of the same length.
    Returns a float32 array of scores in [0, 1] (assuming inputs
    are themselves in [0, 1] — that's the integration layer's job).

    Never raises. If inputs are mismatched in length, returns an
    array of zeros (the safe default — nothing migrates).
    """
    n = len(T)
    if not (len(S) == len(N) == len(P) == n):
        return np.zeros(0, dtype=np.float32)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # All numpy, no Python loops. One pass.
    R = (
        np.float32(w_T) * T
        + np.float32(w_S) * S
        + np.float32(w_N) * N
        + np.float32(w_P) * P
    )

    # Clamp to [0, 1] defensively. Inputs SHOULD be in [0, 1] but
    # we don't trust them.
    np.clip(R, np.float32(0.0), np.float32(1.0), out=R)
    return R
