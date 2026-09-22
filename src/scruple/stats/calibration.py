"""Calibration diagnostics (§8.3).

These are diagnostics, not the guarantee. The guarantee comes from the
class-conditional threshold procedure in ``thresholds.py``; what lives here
tells the researcher *why* a code behaved the way it did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .agreement import _as_binary
from .types import Estimate, ReasonCode
from .weights import FloatArray, normalise_weights, weighted_mean

DEFAULT_BINS = 10


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram (§8.3).

    ``mean_predicted`` and ``observed_rate`` are ``None`` for an empty bin
    rather than 0.0, so a plot cannot show a point where there is no data.
    """

    index: int
    lo: float
    hi: float
    count: int
    weight: float
    mean_predicted: float | None
    observed_rate: float | None


def _as_probabilities(p: Sequence[float] | FloatArray) -> FloatArray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("probabilities must be finite")
    if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return arr


def _checked_pair(
    p: Sequence[float] | FloatArray,
    y: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    probs = _as_probabilities(p)
    labels = _as_binary(y, "outcome")
    if probs.shape != labels.shape:
        raise ValueError(
            f"probabilities and outcomes must have the same length, "
            f"got {probs.size} and {labels.size}"
        )
    return probs, labels, normalise_weights(weights, probs.size)


def brier_score(
    p: Sequence[float] | FloatArray,
    y: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None = None,
) -> Estimate:
    """``mean((p_i - y_i)^2)``. Lower is better; 0.25 is the always-0.5 baseline."""
    probs, labels, w = _checked_pair(p, y, weights)
    if probs.size == 0:
        return Estimate(value=None, n=0, reason=ReasonCode.EMPTY_INPUT)
    value = weighted_mean(np.square(probs - labels), w)
    if value is None:
        return Estimate(value=None, n=probs.size, reason=ReasonCode.ZERO_WEIGHT)
    return Estimate(value=value, n=probs.size)


def reliability_diagram(
    p: Sequence[float] | FloatArray,
    y: Sequence[int] | Sequence[float] | FloatArray,
    bins: int = DEFAULT_BINS,
    weights: Sequence[float] | FloatArray | None = None,
) -> list[ReliabilityBin]:
    """Per-bin mean predicted probability, observed positive rate and count."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    probs, labels, w = _checked_pair(p, y, weights)

    edges = np.linspace(0.0, 1.0, bins + 1)
    # floor(1.0 * bins) == bins would fall off the end, so clip the top bin.
    assignment = np.clip(np.floor(probs * bins).astype(int), 0, bins - 1)

    out: list[ReliabilityBin] = []
    for m in range(bins):
        mask = assignment == m
        count = int(mask.sum())
        bin_w = w[mask]
        total_w = float(bin_w.sum())
        mean_p = weighted_mean(probs[mask], bin_w) if count else None
        rate = weighted_mean(labels[mask], bin_w) if count else None
        out.append(
            ReliabilityBin(
                index=m,
                lo=float(edges[m]),
                hi=float(edges[m + 1]),
                count=count,
                weight=total_w,
                mean_predicted=mean_p,
                observed_rate=rate,
            )
        )
    return out


def expected_calibration_error(
    p: Sequence[float] | FloatArray,
    y: Sequence[int] | Sequence[float] | FloatArray,
    bins: int = DEFAULT_BINS,
    weights: Sequence[float] | FloatArray | None = None,
) -> Estimate:
    """``sum_m (n_m/n) * |acc_m - conf_m|`` over equal-width bins (§8.3)."""
    probs, _, w = _checked_pair(p, y, weights)
    if probs.size == 0:
        return Estimate(value=None, n=0, reason=ReasonCode.EMPTY_INPUT)
    total_w = float(w.sum())
    if total_w <= 0.0:
        return Estimate(value=None, n=probs.size, reason=ReasonCode.ZERO_WEIGHT)

    ece = 0.0
    for b in reliability_diagram(p, y, bins, weights):
        if b.mean_predicted is not None and b.observed_rate is not None:
            ece += (b.weight / total_w) * abs(b.observed_rate - b.mean_predicted)
    return Estimate(value=ece, n=probs.size)
