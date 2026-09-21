"""Selective performance and the headline labour curve (plan §8.7, §8.8).

Two different things live here and must not be confused.

``selective_curve`` is a **diagnostic**: kappa computed on the accepted subset
only. It is reported with prevalence beside it because it can *fall* as
coverage drops even when the coder is behaving perfectly -- confident subsets
skew toward one class, which inflates chance agreement and compresses kappa
(§8.7). Treating it as the headline would make a working model look broken and
could fire the kill criterion of §3 on a false negative.

``labour_curve`` is the **headline** (§8.8): the reliability of the finished
dataset against the human labour needed to reach it. Abstained items receive
the gold label, because a human coded them; accepted items receive the model's
decision; kappa is then computed over *every* item. Plotted against the random
baseline, the vertical gap between the two curves is the product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .agreement import _as_binary, cohens_kappa
from .bootstrap import stratified_bootstrap_ci
from .calibration import _as_probabilities
from .thresholds import ThresholdPair, default_grid, order_by_conservatism
from .types import Estimate, ReasonCode
from .weights import BoolArray, FloatArray, normalise_weights, weighted_mean

DEFAULT_DECISION_THRESHOLD = 0.5
"""Where a coder with no abstention band cuts. Used for the zero-labour point
and for the unreviewed remainder of the random baseline."""


@dataclass(frozen=True)
class SelectivePoint:
    """Diagnostic performance on the accepted subset at one band (§8.7)."""

    pair: ThresholdPair
    coverage: float
    n_accepted: int
    kappa: Estimate
    accuracy: float | None
    prevalence: float | None


@dataclass(frozen=True)
class LabourPoint:
    """One point of the §8.8 hero chart."""

    labour: float
    coverage: float
    kappa: Estimate
    pair: ThresholdPair | None


def _prepare(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    p = _as_probabilities(probs)
    g = _as_binary(gold, "gold")
    if p.shape != g.shape:
        raise ValueError(f"probs and gold must have the same length, got {p.size} and {g.size}")
    return p, g, normalise_weights(weights, p.size)


def _decide(probs: FloatArray, pair: ThresholdPair | None) -> tuple[FloatArray, BoolArray]:
    """Decisions and accepted mask; ``pair is None`` means decide everything."""
    if pair is None:
        decisions = (probs >= DEFAULT_DECISION_THRESHOLD).astype(np.float64)
        return decisions, np.ones(probs.size, dtype=bool)
    return pair.decide_all(probs)


def final_reliability(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pair: ThresholdPair | None,
    *,
    weights: Sequence[float] | FloatArray | None = None,
    labour_offset: float = 0.0,
    n_resamples: int = 0,
    seed: int = 0,
) -> LabourPoint:
    """Reliability of the delivered dataset at one band (§8.8).

    ``labour_offset`` adds the fixed cost of collecting the gold sample, which
    is real human labour and must not be quietly left off the x-axis.
    """
    p, g, w = _prepare(probs, gold, weights)
    decisions, accepted = _decide(p, pair)

    # The simulation of §8.8: a human coded the abstained items, so those carry
    # the gold label and are correct by construction.
    final = np.where(accepted, decisions, g)

    coverage = weighted_mean(accepted.astype(np.float64), w)
    coverage = 0.0 if coverage is None else coverage
    kappa = cohens_kappa(g, final, w)
    if n_resamples:
        kappa = kappa.with_ci(
            stratified_bootstrap_ci(
                g, final, statistic=cohens_kappa, weights=w, n_resamples=n_resamples, seed=seed
            )
        )

    return LabourPoint(
        labour=labour_offset + (1.0 - coverage),
        coverage=coverage,
        kappa=kappa,
        pair=pair,
    )


def selective_curve(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    *,
    weights: Sequence[float] | FloatArray | None = None,
    grid: Sequence[ThresholdPair] | None = None,
) -> tuple[SelectivePoint, ...]:
    """Diagnostic kappa on the accepted subset, most conservative band first.

    Always read alongside ``prevalence``: a rising accuracy with a falling
    kappa is the §8.7 paradox, not a broken model.
    """
    p, g, w = _prepare(probs, gold, weights)
    bands = order_by_conservatism(list(grid)) if grid is not None else default_grid()

    points: list[SelectivePoint] = []
    for pair in bands:
        decisions, accepted = pair.decide_all(p)
        n_accepted = int(accepted.sum())
        coverage = weighted_mean(accepted.astype(np.float64), w)
        coverage = 0.0 if coverage is None else coverage

        if n_accepted == 0:
            points.append(
                SelectivePoint(
                    pair=pair,
                    coverage=coverage,
                    n_accepted=0,
                    kappa=Estimate(value=None, n=0, reason=ReasonCode.ALL_ABSTAINED),
                    accuracy=None,
                    prevalence=None,
                )
            )
            continue

        sub_w = w[accepted]
        points.append(
            SelectivePoint(
                pair=pair,
                coverage=coverage,
                n_accepted=n_accepted,
                kappa=cohens_kappa(g[accepted], decisions[accepted], sub_w),
                accuracy=weighted_mean(
                    (decisions[accepted] == g[accepted]).astype(np.float64), sub_w
                ),
                prevalence=weighted_mean(g[accepted], sub_w),
            )
        )
    return tuple(points)


def labour_curve(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    *,
    weights: Sequence[float] | FloatArray | None = None,
    grid: Sequence[ThresholdPair] | None = None,
    labour_offset: float = 0.0,
    n_resamples: int = 0,
    seed: int = 0,
) -> tuple[LabourPoint, ...]:
    """The §8.8 headline curve, from zero review to the widest band in the grid."""
    bands = order_by_conservatism(list(grid)) if grid is not None else default_grid()
    # Least conservative first, so labour rises left to right, and start from
    # the no-review point where the model decides everything.
    ordered: list[ThresholdPair | None] = [None, *reversed(bands)]
    return tuple(
        final_reliability(
            probs,
            gold,
            pair,
            weights=weights,
            labour_offset=labour_offset,
            n_resamples=n_resamples,
            seed=seed,
        )
        for pair in ordered
    )


def random_baseline_curve(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    labour_fractions: Sequence[float],
    *,
    weights: Sequence[float] | FloatArray | None = None,
    labour_offset: float = 0.0,
    n_resamples: int = 100,
    seed: int = 0,
) -> tuple[LabourPoint, ...]:
    """The naive strategy: code everything, then hand-check a random share.

    This is what a researcher would do without calibrated probabilities, and it
    is the line the calibrated curve has to beat for the project to be worth
    building (§3).
    """
    p, g, w = _prepare(probs, gold, weights)
    fractions = np.asarray(labour_fractions, dtype=np.float64)
    if fractions.size and (fractions.min() < 0.0 or fractions.max() > 1.0):
        raise ValueError("labour fractions must lie in [0, 1]")

    model_decisions = (p >= DEFAULT_DECISION_THRESHOLD).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = p.size

    points: list[LabourPoint] = []
    for fraction in fractions:
        n_reviewed = round(fraction * n)
        values: list[float] = []
        for _ in range(n_resamples):
            reviewed = rng.choice(n, size=n_reviewed, replace=False)
            final = model_decisions.copy()
            final[reviewed] = g[reviewed]
            est = cohens_kappa(g, final, w)
            if est.value is not None:
                values.append(est.value)

        kappa = (
            Estimate(value=float(np.mean(values)), n=n)
            if values
            else Estimate(value=None, n=n, reason=ReasonCode.DEGENERATE_SINGLE_CLASS)
        )
        if values:
            kappa = kappa.with_ci(
                (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
            )
        points.append(
            LabourPoint(
                labour=labour_offset + float(fraction),
                coverage=1.0 - float(fraction),
                kappa=kappa,
                pair=None,
            )
        )
    return tuple(points)
