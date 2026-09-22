"""Stratified bootstrap confidence intervals (§8.1).

Resampling is stratified by the gold label. On a rare code an unstratified
bootstrap routinely draws a resample containing no positives at all, which
makes kappa undefined and the resulting interval meaningless; since most
research codes are rare (§4), stratification is not optional here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .agreement import _as_binary
from .types import Estimate
from .weights import FloatArray, normalise_weights

MIN_RESAMPLES = 2000
"""The count §8.1 commits to in the report. Callers may pass fewer in tests."""

# Below this share of usable resamples the percentile interval is being built
# from too few draws to mean anything, so we report no interval at all.
_MIN_DEFINED_SHARE = 0.5

Statistic = Callable[[FloatArray, FloatArray, FloatArray], Estimate]


def stratified_bootstrap_ci(
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pred: Sequence[int] | Sequence[float] | FloatArray,
    *,
    statistic: Statistic,
    weights: Sequence[float] | FloatArray | None = None,
    n_resamples: int = MIN_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for ``statistic``, stratified by gold label.

    Returns ``None`` when there is nothing to resample, or when too many
    resamples are degenerate for an interval to be trustworthy.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")

    g = _as_binary(gold, "gold")
    p = _as_binary(pred, "pred")
    w = normalise_weights(weights, g.size)
    if g.size == 0:
        return None

    positives = np.flatnonzero(g == 1.0)
    negatives = np.flatnonzero(g == 0.0)
    rng = np.random.default_rng(seed)

    values: list[float] = []
    for _ in range(n_resamples):
        parts = [
            rng.choice(stratum, size=stratum.size, replace=True)
            for stratum in (positives, negatives)
            if stratum.size
        ]
        idx = np.concatenate(parts)
        est = statistic(g[idx], p[idx], w[idx])
        if est.value is not None:
            values.append(est.value)

    if len(values) < max(1, int(_MIN_DEFINED_SHARE * n_resamples)):
        return None

    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(values), [tail, 1.0 - tail])
    return (float(lo), float(hi))
