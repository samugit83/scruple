"""Class-conditional (Mondrian) threshold selection (plan §8.6).

The guarantee this fits is

    P(decision wrong | accepted, gold = positive) <= alpha
    P(decision wrong | accepted, gold = negative) <= alpha

each holding with confidence 1 - delta, per code.

It is emphatically *not* a marginal guarantee. On a code with 3% prevalence, a
coder that answers "no" to everything has a 3% marginal error rate at 100%
coverage, so a marginal formulation certifies it at alpha = 5% while its kappa
is 0. Splitting the risk by true class removes that degeneracy, because the
all-negative coder is wrong on 100% of the accepted positive class.

Multiplicity: the grid is scanned in full and corrected with Bonferroni over
its candidates. The plan's §8.6 originally specified a fixed-sequence walk from
the most conservative band, which needs no within-code correction, and that
turned out to be unsound here -- see `_WHY_NOT_FIXED_SEQUENCE` below.

Across codes there is deliberately still no family-wise correction: each code is
a separate claim, reported separately, exactly as researchers already report
per-code kappa (§8.6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import binom

from .agreement import _as_binary
from .calibration import _as_probabilities
from .types import Estimate, ReasonCode, Verdict
from .weights import (
    BoolArray,
    FloatArray,
    effective_sample_size,
    normalise_weights,
    weighted_mean,
)

_WHY_NOT_FIXED_SEQUENCE = """
Fixed-sequence testing from the most conservative band assumes that band is the
easiest to certify, so that once it fails every wider-coverage band fails too.
That holds for *marginal* risk. It is false for *class-conditional* risk, which
is what this module controls, and it fails hardest on exactly the rare codes
most research uses (§4). Two mechanisms, either of which is enough:

  * No power. A high t_hi accepts only a handful of gold positives, and a
    handful of clean items is not evidence of controlled risk: with
    alpha = delta = 0.05, zero errors out of six gives p = 0.74.

  * Adverse selection. Narrowing the band drops moderately-confident correct
    decisions while keeping the confidently-wrong positives, because a low score
    is an accepted "no". The error *share* of the positive class can therefore
    rise as coverage falls.

Measured on a simulated coder that leaks 1-2% of its positives into the
confident-negative cluster, the most conservative band failed at every
prevalence from 0.20 down to 0.02 while the full-coverage band passed:
fixed-sequence certified 0% of trials, a corrected full scan ~47%.

The two classes also move in opposite directions as the band changes, so no
ordering of the grid is monotone for both at once, and no fixed-sequence walk in
either direction is sound. Hence the full scan with a Bonferroni correction over
the grid. Sample-splitting (choose on one half, test on the other) measured
slightly more powerful, ~60% versus ~47%, but makes the selected threshold
depend on a second random seed; determinism won, because §6 requires every
number in the report to be reproducible from files on disk.

tests/statistical/test_guarantee.py::test_the_most_conservative_band_fails_where_wider_bands_pass
pins the finding so the fixed-sequence walk cannot be reintroduced quietly.
"""

MIN_POSITIVE_INSTANCES = 15
"""§8.5 hard gate. Below this, `check` refuses to certify a code at all.
Reporting a confident-looking kappa built on nine items is the most likely way
this project produces a false result."""

PUBLICATION_KAPPA = 0.70
"""The bar journals broadly expect (§4)."""

COMFORTABLE_KAPPA = 0.75
"""Above the bar with room for sampling error; below it the code is 'weak'."""

# Integerising a weighted count for the binomial tail must never round in the
# direction that makes the evidence look stronger than it is.
_TOL = 1e-9


@dataclass(frozen=True)
class ThresholdPair:
    """An abstention band. Decide 1 at or above ``t_hi``, 0 at or below ``t_lo``."""

    t_lo: float
    t_hi: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_lo <= 1.0 or not 0.0 <= self.t_hi <= 1.0:
            raise ValueError(f"thresholds must lie in [0, 1], got {self}")
        if self.t_lo >= self.t_hi:
            raise ValueError(f"t_lo must be strictly below t_hi, got {self}")

    @property
    def band_width(self) -> float:
        """Width of the abstention band; the measure of how conservative it is."""
        return self.t_hi - self.t_lo

    def decide(self, p: float) -> int | None:
        """Decision for one probability, or ``None`` to abstain."""
        if p >= self.t_hi:
            return 1
        if p <= self.t_lo:
            return 0
        return None

    def decide_all(self, probs: FloatArray) -> tuple[FloatArray, BoolArray]:
        """Vectorised decisions and the accepted mask.

        Decisions for abstained items are meaningless and must be read only
        through the mask.
        """
        decisions = (probs >= self.t_hi).astype(np.float64)
        accepted = (probs >= self.t_hi) | (probs <= self.t_lo)
        return decisions, accepted


DEFAULT_T_HI = (
    0.99,
    0.98,
    0.97,
    0.96,
    0.95,
    0.925,
    0.90,
    0.875,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
)
"""Candidate upper thresholds, spaced finely where confident decisions live.

§8.6 illustrated a 45-point grid in 0.01 steps. Since every candidate is now
tested and Bonferroni-corrected, each extra point is paid for by every other
point: on the same simulated coder the 45-point grid certified ~27% of trials
against ~47% for these 15, at indistinguishable coverage. Finer resolution than
this buys nothing a researcher can act on."""


def default_grid(t_hi_values: Sequence[float] | None = None) -> tuple[ThresholdPair, ...]:
    """The symmetric grid of §8.6 step 1, spanning ``t_hi`` from 0.99 to 0.55."""
    if t_hi_values is None:
        t_hi_values = DEFAULT_T_HI
    return order_by_conservatism(
        [ThresholdPair(t_lo=round(1.0 - t, 10), t_hi=t) for t in t_hi_values]
    )


def order_by_conservatism(pairs: Sequence[ThresholdPair]) -> tuple[ThresholdPair, ...]:
    """Order most conservative (widest band) first, using the grid alone.

    Selection no longer depends on this order -- the whole grid is scanned -- but
    a stable, data-independent order keeps the candidate table in the validation
    report reproducible run to run.
    """
    return tuple(sorted(pairs, key=lambda p: (-p.band_width, -p.t_hi, p.t_lo)))


@dataclass(frozen=True)
class ClassRisk:
    """Risk control evidence for one true class at one candidate band."""

    gold_label: int
    n_accepted: int
    n_effective: float
    errors: float
    risk: float | None
    p_value: float
    passed: bool


@dataclass(frozen=True)
class Candidate:
    """One evaluated point of the grid."""

    pair: ThresholdPair
    coverage: float
    n_accepted: int
    positive: ClassRisk
    negative: ClassRisk

    @property
    def valid(self) -> bool:
        """§8.6 step 3: a band is valid only if *both* classes pass."""
        return self.positive.passed and self.negative.passed


@dataclass(frozen=True)
class ThresholdSelection:
    """Outcome of fitting one code's thresholds on the calibration split."""

    selected: ThresholdPair | None
    coverage: float | None
    verdict: Verdict
    reason: ReasonCode | None
    n_gold: int
    n_gold_positive: int
    alpha: float
    delta: float
    exact: bool
    candidates: tuple[Candidate, ...]
    delta_per_candidate: float = 0.0
    held_out_kappa: Estimate | None = None

    @property
    def message(self) -> str:
        """Plain-language explanation for `scruple check` (§12)."""
        if self.reason is ReasonCode.INSUFFICIENT_EVIDENCE:
            return (
                f"only {self.n_gold_positive} positive instances in the gold sample "
                f"(at least {MIN_POSITIVE_INSTANCES} are needed before any statistic means "
                "anything). Collect more with `scruple gold --enrich <code>`."
            )
        if self.reason is ReasonCode.NO_VALID_THRESHOLD:
            return (
                "could not be automated at any threshold: no band controlled the error rate "
                f"within each class at alpha={self.alpha:g}. Check whether your two coders "
                "agree on this code -- where humans disagree, the definition is the problem."
            )
        if self.reason is ReasonCode.BELOW_PUBLICATION_BAR:
            kappa = self.held_out_kappa.value if self.held_out_kappa else None
            shown = f"{kappa:.2f}" if kappa is not None else "undefined"
            return (
                f"held-out kappa {shown} is below the {PUBLICATION_KAPPA:g} publication bar "
                "even where the thresholds do decide."
            )
        return ""


def class_risk(
    errors: FloatArray,
    weights: FloatArray,
    *,
    gold_label: int,
    alpha: float,
    delta: float,
) -> ClassRisk:
    """Binomial tail test for one class (§8.6 step 3).

    ``H0: risk > alpha``; the p-value is ``P(Binom(n, alpha) <= E)``, so few
    observed errors give a small p-value and reject H0.

    Under uniform weights this is exact. Under the enriched sampling of §8.5 the
    weighted error count is integerised against Kish's effective sample size,
    which is an approximation; ``ThresholdSelection.exact`` records which case
    applies so the report can state it rather than imply exactness.
    """
    n_raw = int(errors.size)
    n_eff = effective_sample_size(weights)
    risk = weighted_mean(errors, weights)

    n_int = math.floor(n_eff + _TOL)
    if n_int < 1 or risk is None:
        # No accepted items in this class: zero evidence is not evidence of
        # safety. Refusing here is what stops "accept nothing from the positive
        # class" being a free pass.
        return ClassRisk(
            gold_label=gold_label,
            n_accepted=n_raw,
            n_effective=n_eff,
            errors=0.0,
            risk=risk,
            p_value=1.0,
            passed=False,
        )

    e_int = math.ceil(risk * n_int - _TOL)
    p_value = float(binom.cdf(e_int, n_int, alpha))
    return ClassRisk(
        gold_label=gold_label,
        n_accepted=n_raw,
        n_effective=n_eff,
        errors=float(np.dot(errors, weights)) if weights.size else 0.0,
        risk=risk,
        p_value=p_value,
        passed=p_value <= delta,
    )


def evaluate_candidate(
    probs: FloatArray,
    gold: FloatArray,
    weights: FloatArray,
    pair: ThresholdPair,
    *,
    alpha: float,
    delta: float,
) -> Candidate:
    """Evaluate one band: coverage plus a risk test within each true class."""
    decisions, accepted = pair.decide_all(probs)
    total_w = float(weights.sum())
    covered_w = float(weights[accepted].sum())

    risks = {}
    for label in (1.0, 0.0):
        mask = accepted & (gold == label)
        risks[label] = class_risk(
            (decisions[mask] != label).astype(np.float64),
            weights[mask],
            gold_label=int(label),
            alpha=alpha,
            delta=delta,
        )

    return Candidate(
        pair=pair,
        coverage=covered_w / total_w if total_w > 0 else 0.0,
        n_accepted=int(accepted.sum()),
        positive=risks[1.0],
        negative=risks[0.0],
    )


def realised_class_risk(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    pair: ThresholdPair,
    *,
    weights: Sequence[float] | FloatArray | None = None,
) -> dict[int, float | None]:
    """Observed error rate within each true class on already-accepted items.

    This is the quantity the §8.6 guarantee bounds. The report states it on the
    test split so a reader can check the claim rather than take it on trust, and
    §14.6 uses it to verify the guarantee holds on fresh draws.

    A class with no accepted items has no observed risk, and reports ``None``
    rather than a flattering 0.0.
    """
    p = _as_probabilities(probs)
    g = _as_binary(gold, "gold")
    if p.shape != g.shape:
        raise ValueError(f"probs and gold must have the same length, got {p.size} and {g.size}")
    w = normalise_weights(weights, p.size)

    decisions, accepted = pair.decide_all(p)
    out: dict[int, float | None] = {}
    for label in (1, 0):
        mask = accepted & (g == float(label))
        out[label] = weighted_mean((decisions[mask] != float(label)).astype(np.float64), w[mask])
    return out


def fit_thresholds(
    probs: Sequence[float] | FloatArray,
    gold: Sequence[int] | Sequence[float] | FloatArray,
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
    weights: Sequence[float] | FloatArray | None = None,
    grid: Sequence[ThresholdPair] | None = None,
    min_positives: int = MIN_POSITIVE_INSTANCES,
) -> ThresholdSelection:
    """Fit an abstention band on the **calibration split only** (§8.6).

    Every candidate in the grid is tested for class-conditional risk control at
    ``delta / len(grid)``, and the valid band with the highest coverage is
    selected. See ``_WHY_NOT_FIXED_SEQUENCE`` for why the grid is scanned in full
    rather than walked in order.

    Never pass the test split here. ``stats.sealed.Sealed`` exists so that a
    caller cannot do so by accident.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}")

    p = _as_probabilities(probs)
    g = _as_binary(gold, "gold")
    if p.shape != g.shape:
        raise ValueError(f"probs and gold must have the same length, got {p.size} and {g.size}")
    w = normalise_weights(weights, p.size)

    candidates_grid = order_by_conservatism(list(grid)) if grid is not None else default_grid()
    if not candidates_grid:
        raise ValueError("threshold grid must not be empty")

    # Uniform weights make the binomial test exact; anything else makes it an
    # approximation that the report has to disclose.
    exact = bool(p.size == 0 or np.allclose(w, w.flat[0]))
    n_positive = int((g == 1.0).sum())

    # Bonferroni over the grid. Every candidate is tested at delta/|grid|, so the
    # chance that *any* band with true risk above alpha is declared valid stays
    # at delta, whichever band we then choose among those that passed.
    delta_per_candidate = delta / len(candidates_grid)

    if n_positive < min_positives:
        return ThresholdSelection(
            selected=None,
            coverage=None,
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reason=ReasonCode.INSUFFICIENT_EVIDENCE,
            n_gold=int(g.size),
            n_gold_positive=n_positive,
            alpha=alpha,
            delta=delta,
            exact=exact,
            candidates=(),
            delta_per_candidate=delta_per_candidate,
        )

    evaluated = tuple(
        evaluate_candidate(p, g, w, pair, alpha=alpha, delta=delta_per_candidate)
        for pair in candidates_grid
    )
    passing = [c for c in evaluated if c.valid]

    if not passing:
        return ThresholdSelection(
            selected=None,
            coverage=None,
            verdict=Verdict.NOT_AUTOMATABLE,
            reason=ReasonCode.NO_VALID_THRESHOLD,
            n_gold=int(g.size),
            n_gold_positive=n_positive,
            alpha=alpha,
            delta=delta,
            exact=exact,
            candidates=evaluated,
            delta_per_candidate=delta_per_candidate,
        )

    # Among valid bands, take the one that sends fewest items to a human. Ties go
    # to the wider band: equal coverage means the extra width holds no
    # calibration items either way, and the wider band is the safer bet on data
    # this fit has not seen.
    best = max(passing, key=lambda c: (c.coverage, c.pair.band_width))

    return ThresholdSelection(
        selected=best.pair,
        coverage=best.coverage,
        verdict=Verdict.OK,
        reason=None,
        n_gold=int(g.size),
        n_gold_positive=n_positive,
        alpha=alpha,
        delta=delta,
        exact=exact,
        candidates=evaluated,
        delta_per_candidate=delta_per_candidate,
    )


def grade(selection: ThresholdSelection, held_out_kappa: Estimate) -> ThresholdSelection:
    """Refine a passing selection against held-out kappa, producing the §12 verdict.

    Controlling per-class error on the accepted subset does not by itself make
    the delivered dataset publishable, so the verdict shown to the researcher
    also depends on kappa measured on the test split.
    """
    if selection.verdict is not Verdict.OK:
        return replace(selection, held_out_kappa=held_out_kappa)

    if held_out_kappa.value is None:
        return replace(
            selection,
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reason=held_out_kappa.reason,
            held_out_kappa=held_out_kappa,
        )
    if held_out_kappa.value < PUBLICATION_KAPPA:
        return replace(
            selection,
            verdict=Verdict.NOT_AUTOMATABLE,
            reason=ReasonCode.BELOW_PUBLICATION_BAR,
            held_out_kappa=held_out_kappa,
        )
    if held_out_kappa.value < COMFORTABLE_KAPPA:
        return replace(selection, verdict=Verdict.WEAK, held_out_kappa=held_out_kappa)
    return replace(selection, held_out_kappa=held_out_kappa)
