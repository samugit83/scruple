"""Inter-rater agreement statistics (plan §8.1, §8.2).

Cohen's kappa and Krippendorff's alpha are implemented independently, from
their own definitions, because §14.3 uses their agreement on binary two-coder
data as a cross-check. Deriving one from the other would make that check
vacuous.

Both accept sampling weights so that the enriched gold strata of §8.5 can be
combined with the uniform stratum by inverse-probability weighting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import Confusion, Estimate, ReasonCode
from .weights import FloatArray, normalise_weights

# Tolerance for "chance agreement is indistinguishable from certainty". Below
# this the kappa denominator is numerical noise rather than signal.
_DEGENERATE_TOL = 1e-12


def _as_binary(labels: Sequence[int] | Sequence[float] | FloatArray, name: str) -> FloatArray:
    arr = np.asarray(labels, dtype=np.float64)
    if arr.size and not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"{name} labels must be 0 or 1")
    return arr


def confusion(
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pred: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None = None,
) -> Confusion:
    """Build the 2x2 table of §8.1, accumulating weights rather than counts."""
    g = _as_binary(gold, "gold")
    p = _as_binary(pred, "pred")
    if g.shape != p.shape:
        raise ValueError(f"gold and pred must have the same length, got {g.size} and {p.size}")
    w = normalise_weights(weights, g.size)
    return Confusion(
        a=float(w[(g == 1.0) & (p == 1.0)].sum()),
        b=float(w[(g == 1.0) & (p == 0.0)].sum()),
        c=float(w[(g == 0.0) & (p == 1.0)].sum()),
        d=float(w[(g == 0.0) & (p == 0.0)].sum()),
    )


def observed_agreement(cm: Confusion) -> float | None:
    """``p_o``: the raw proportion of items the two coders agreed on."""
    if cm.n <= 0:
        return None
    return cm.agreements / cm.n


def cohens_kappa_from_confusion(cm: Confusion, *, n_items: int | None = None) -> Estimate:
    """Cohen's kappa for two coders on a binary code (§8.1).

    ``n_items`` is the raw item count for reporting; it differs from ``cm.n``
    whenever weights are in play, and the raw count is what a reader needs in
    order to judge how much evidence there is.
    """
    n = cm.n
    reported_n = n_items if n_items is not None else round(n)
    if n <= 0:
        return Estimate(value=None, n=reported_n, reason=ReasonCode.EMPTY_INPUT)

    p_o = cm.agreements / n
    p_e = (cm.gold_positive / n) * (cm.pred_positive / n) + (cm.gold_negative / n) * (
        cm.pred_negative / n
    )

    # §8.1: handle this explicitly rather than by dividing by zero. p_e == 1
    # means both coders put every item in the same single class, so there is no
    # agreement beyond chance to measure.
    if 1.0 - p_e <= _DEGENERATE_TOL:
        return Estimate(value=None, n=reported_n, reason=ReasonCode.DEGENERATE_SINGLE_CLASS)

    return Estimate(value=(p_o - p_e) / (1.0 - p_e), n=reported_n)


def cohens_kappa(
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pred: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None = None,
) -> Estimate:
    """Cohen's kappa from label vectors."""
    cm = confusion(gold, pred, weights)
    return cohens_kappa_from_confusion(cm, n_items=len(gold))


def krippendorff_alpha_from_confusion(cm: Confusion, *, n_items: int | None = None) -> Estimate:
    """Krippendorff's alpha for nominal binary data, two coders, no missing values.

    Derived from the coincidence matrix rather than from kappa. For two coders
    each unit contributes two ordered pairs, giving

        o_11 = 2a, o_00 = 2d, o_10 = o_01 = b + c
        n_1  = 2a + b + c,    n_0  = 2d + b + c,    n_total = 2N

        D_o = 2(b + c)
        D_e = 2 * n_1 * n_0 / (n_total - 1)
        alpha = 1 - D_o / D_e = 1 - (b + c)(2N - 1) / (n_1 * n_0)
    """
    n = cm.n
    reported_n = n_items if n_items is not None else round(n)
    n_total = 2.0 * n
    if n_total <= 1.0:
        # Fewer than one pairable value: there is nothing to build a chance
        # model from. In practice this is the empty-input case.
        return Estimate(value=None, n=reported_n, reason=ReasonCode.EMPTY_INPUT)

    n_one = 2.0 * cm.a + cm.errors
    n_zero = 2.0 * cm.d + cm.errors
    d_e = 2.0 * n_one * n_zero / (n_total - 1.0)
    if d_e <= _DEGENERATE_TOL:
        return Estimate(value=None, n=reported_n, reason=ReasonCode.DEGENERATE_SINGLE_CLASS)

    d_o = 2.0 * cm.errors
    return Estimate(value=1.0 - d_o / d_e, n=reported_n)


def krippendorff_alpha(
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pred: Sequence[int] | Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray | None = None,
) -> Estimate:
    """Krippendorff's alpha from label vectors."""
    cm = confusion(gold, pred, weights)
    return krippendorff_alpha_from_confusion(cm, n_items=len(gold))
