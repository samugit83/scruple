"""Sampling weights and inverse-probability weighting (§8.5).

The gold sample combines a uniform stratum with an optional enriched stratum
that oversamples likely-positive items so rare codes have enough positive
instances to say anything about. Validity is preserved because the inclusion
probability of every sampled item is known by construction; every estimator in
this package therefore accepts weights and uses IPW.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def normalise_weights(weights: Sequence[float] | FloatArray | None, n: int) -> FloatArray:
    """Coerce ``weights`` to a validated float array of length ``n``.

    ``None`` means an unweighted (uniform random) sample, which is the default
    and the case the guarantee of §8.6 rests on.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have length {n}, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must not be negative")
    return w


def weights_from_inclusion(inclusion: Sequence[float] | FloatArray) -> FloatArray:
    """Convert known inclusion probabilities to IPW weights (1 / pi)."""
    pi = np.asarray(inclusion, dtype=np.float64)
    if pi.size and (np.any(pi <= 0.0) or np.any(pi > 1.0)):
        raise ValueError("inclusion probability must lie in (0, 1]")
    return 1.0 / pi


def effective_sample_size(weights: FloatArray) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``.

    Equals the raw count exactly for uniform weights of any scale, and shrinks
    as the weights become unequal. §8.6 uses it so that the binomial tail test
    is not handed more evidence than an enriched sample actually contains.
    """
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return total * total / float(np.square(weights).sum())


def weighted_mean(values: FloatArray, weights: FloatArray) -> float | None:
    """IPW mean of ``values``; ``None`` when the total weight is zero."""
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return float(np.dot(values, weights) / total)
