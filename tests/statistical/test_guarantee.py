"""Statistical validity simulation for the §8.6 guarantee (plan §14.6).

This is the most important test in the repository. It does not check that the
code runs; it checks that the claim the code makes about the world is true.

Method: simulate a coder whose error probability is a known function of the
probability it emits, fit thresholds on one simulated calibration sample, then
measure the realised class-conditional error on a **fresh** sample. Repeat, and
assert the guarantee holds in at least (1 - delta) of trials.

These are slow. §14.13 runs them on every change to stats/ and nightly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scruple.stats.thresholds import (
    ThresholdPair,
    fit_thresholds,
    order_by_conservatism,
    realised_class_risk,
)

pytestmark = pytest.mark.statistical

ALPHA = 0.05
DELTA = 0.05
N_TRIALS = 1000
N_FRESH = 20_000
PREVALENCES = [0.50, 0.20, 0.05, 0.02]

# A coarser grid than the default keeps 1000 trials tractable. Grid size changes
# the Bonferroni correction and so the coverage achieved, never the validity of
# the guarantee, which is what this file tests.
SIM_GRID = order_by_conservatism(
    [
        ThresholdPair(t_lo=round(1 - t, 10), t_hi=t)
        for t in (0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
    ]
)


def _coder(
    rng: np.random.Generator,
    n: int,
    prevalence: float,
    *,
    leak: float,
    false_positive_rate: float = 0.01,
    c_hi: float = 25.0,
    c_lo: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A calibrated coder of controllable, prevalence-independent quality.

    Probabilities come from a two-component mixture -- a confident cluster and a
    near-zero one -- and outcomes are drawn as ``y ~ Bernoulli(p)``, so the
    emitted probability means exactly what it says.

    Quality is parameterised by the two class-conditional error rates, not by an
    absolute probability level, because those are the quantities §8.6 controls
    and they are what "equally good" means across prevalences. Fixing the
    confident cluster at, say, 0.93 instead would give a coder with 7% negative-
    class error at prevalence 0.50 and 0.4% at 0.05 -- the same coder looking
    very different to the procedure for reasons that have nothing to do with it.

    * ``leak`` is the share of true positives sitting in the near-zero cluster:
      positives the coder confidently and wrongly calls "no". It is the floor on
      P(wrong | accepted, gold = positive), because a low score is an accepted
      "no" at every band.
    * ``false_positive_rate`` is the target P(wrong | accepted, gold = negative).
    """
    mass_hi = (1.0 - leak) * prevalence
    # Solve for the confident cluster's mean that yields the requested
    # false-positive rate at this prevalence.
    ratio = false_positive_rate * (1.0 - prevalence) / mass_hi
    m_hi = 1.0 / (1.0 + ratio)
    w = mass_hi / m_hi
    m_lo = (leak * prevalence) / (1.0 - w)
    is_hi = rng.random(n) < w
    p = np.where(
        is_hi,
        rng.beta(m_hi * c_hi, (1.0 - m_hi) * c_hi, n),
        rng.beta(max(m_lo * c_lo, 1e-4), (1.0 - m_lo) * c_lo, n),
    )
    return p, (rng.random(n) < p).astype(np.float64)


def _good(rng: np.random.Generator, n: int, prevalence: float):
    """A coder worth automating: about 1% error in each class, well inside alpha."""
    return _coder(rng, n, prevalence, leak=0.01, false_positive_rate=0.01)


def _overconfident(rng: np.random.Generator, n: int, prevalence: float, gamma: float = 3.0):
    """Miscalibrated: outcomes follow q, but the coder states a sharpened p.

    This is the documented LLM failure mode -- stated confidence systematically
    exceeding observed frequency (§0.3).
    """
    q, y = _good(rng, n, prevalence)
    sharpened = q**gamma / (q**gamma + (1.0 - q) ** gamma)
    return np.clip(sharpened, 0.0, 1.0), y


def _always_negative(rng: np.random.Generator, n: int, prevalence: float):
    """A coder that answers "no" to everything, and says so confidently."""
    y = (rng.random(n) < prevalence).astype(np.float64)
    return np.full(n, 0.01), y


def _calibration_size(prevalence: float) -> int:
    """Enough calibration items that certification is arithmetically possible.

    With alpha = delta = 0.05 a class needs at least ceil(ln 0.05 / ln 0.95) = 59
    accepted items before *zero* observed errors is evidence of controlled risk,
    and the Bonferroni correction over the grid pushes that higher. Sizing by
    prevalence keeps the rare-prevalence arms meaningful rather than vacuously
    refused -- which is itself the §8.5 lesson, tested separately below.
    """
    return max(1000, math.ceil(300 / prevalence))


def _guarantee_holds(probs, gold, pair) -> bool:
    """Both class-conditional risks within alpha on fresh data."""
    risk = realised_class_risk(probs, gold, pair)
    return all(r is None or r <= ALPHA for r in risk.values())


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_class_conditional_risk_is_controlled_on_fresh_data(prevalence: float) -> None:
    """The guarantee itself, at every prevalence including the rare ones.

    Rare prevalences are mandatory here: they are exactly where the original
    marginal formulation of §8.6 failed.
    """
    rng = np.random.default_rng(int(prevalence * 10_000))
    n_cal = _calibration_size(prevalence)

    held = 0
    certified = 0
    for _ in range(N_TRIALS):
        cal_p, cal_y = _good(rng, n_cal, prevalence)
        selection = fit_thresholds(cal_p, cal_y, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
        if selection.selected is None:
            # Nothing was auto-coded, so no claim was made and none can be wrong.
            held += 1
            continue
        certified += 1
        fresh_p, fresh_y = _good(rng, N_FRESH, prevalence)
        held += _guarantee_holds(fresh_p, fresh_y, selection.selected)

    coverage = held / N_TRIALS
    tolerance = 3.0 * math.sqrt(0.95 * 0.05 / N_TRIALS)
    assert coverage >= (1.0 - DELTA) - tolerance, (
        f"class-conditional risk exceeded alpha in {(1 - coverage):.1%} of trials "
        f"at prevalence {prevalence}; the guarantee promises at most {DELTA:.0%}"
    )
    # Guard against a vacuous pass: the procedure must actually certify
    # sometimes, or this test proves only that refusing everything is safe.
    assert certified > N_TRIALS * 0.25, (
        f"only {certified}/{N_TRIALS} trials certified at prevalence {prevalence}; "
        "the coverage assertion above would be vacuous"
    )


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_a_coder_leaking_more_than_alpha_of_positives_is_refused(prevalence: float) -> None:
    """The guarantee has teeth: a plausible-looking coder is rejected.

    Leaking 15% of true positives into the confident-negative cluster gives a
    coder that looks excellent on accuracy -- it is right about 97% of the time
    at 2% prevalence -- and whose positive-class error is 15%, three times alpha.
    It must never be certified.
    """
    rng = np.random.default_rng(int(prevalence * 555))
    n_cal = _calibration_size(prevalence)
    for _ in range(100):
        probs, gold = _coder(rng, n_cal, prevalence, leak=0.15, false_positive_rate=0.01)
        selection = fit_thresholds(probs, gold, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
        assert selection.verdict.usable is False


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_negative_control_an_all_negative_coder_is_never_certified(prevalence: float) -> None:
    """The regression test for the §8.6 degeneracy.

    A coder that always predicts negative achieves a marginal error rate equal
    to the prevalence -- 2% on a rare code -- and a marginal guarantee at
    alpha = 5% would certify it at 100% coverage with kappa 0. The
    class-conditional formulation must reject it every single time, because it
    is wrong on 100% of the accepted positive class.

    If this test ever passes such a coder, the guarantee is broken and nothing
    else in the repository matters.
    """
    rng = np.random.default_rng(int(prevalence * 777))
    n_cal = _calibration_size(prevalence)

    for _ in range(200):
        probs, gold = _always_negative(rng, n_cal, prevalence)
        selection = fit_thresholds(probs, gold, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
        assert selection.verdict.usable is False, (
            f"an all-negative coder was certified at prevalence {prevalence} "
            f"with coverage {selection.coverage}"
        )


@pytest.mark.parametrize("prevalence", [0.20, 0.05])
def test_miscalibrated_control_does_not_break_risk_control(prevalence: float) -> None:
    """An overconfident coder must never be certified on the strength of its own confidence.

    The plan (§14.6) expected such a coder to be rejected or given conservative
    thresholds. Measurement says something sharper and more reassuring: the
    procedure is simply indifferent to overconfidence, because it tests
    *observed* error counts rather than trusting the stated probability. A
    monotone sharpening preserves the ranking of items, so it only slides where
    the grid cuts -- the accepted set at some band is exactly the set an honest
    coder would produce at a different band, with identical errors.

    So the guarantee holds, and for the reason that matters: it never relies on
    the probabilities meaning what they say. Calibration is what makes a *useful*
    band reachable on a fixed grid; the empirical test is what makes the band
    valid. What does defeat the procedure is probabilities that rank items badly,
    covered by `test_a_random_coder_is_never_certified` and
    `test_a_coder_leaking_more_than_alpha_of_positives_is_refused`.

    These two prevalences are the ones where the overconfident coder is certified
    often enough for the assertion to bite; prevalence 0.50 is covered by
    `test_overconfidence_is_never_rewarded`, where it is rejected outright.
    """
    rng = np.random.default_rng(int(prevalence * 31_337))
    n_cal = _calibration_size(prevalence)
    trials = 200

    held = 0
    certified = 0
    for _ in range(trials):
        cal_p, cal_y = _overconfident(rng, n_cal, prevalence)
        selection = fit_thresholds(cal_p, cal_y, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
        if selection.selected is None:
            held += 1
            continue
        certified += 1
        fresh_p, fresh_y = _overconfident(rng, N_FRESH, prevalence)
        held += _guarantee_holds(fresh_p, fresh_y, selection.selected)

    tolerance = 3.0 * math.sqrt(0.95 * 0.05 / trials)
    assert held / trials >= (1.0 - DELTA) - tolerance
    assert certified > trials * 0.5, "vacuous: too little was certified to test"


def test_overconfidence_is_never_rewarded() -> None:
    """Overstating confidence must not win a coder more automation than honesty.

    It cannot, and the reason is worth stating: a monotone sharpening preserves
    the ranking of items, so at any band it accepts exactly the set an honest
    coder would accept at some other band, with identical errors. The procedure
    tests observed error counts, not stated probabilities, so it is simply
    indifferent.

    An earlier version of this test asserted something stronger -- that at
    balanced prevalence the overconfident coder is rejected outright. That was
    real but was an artifact of the simulated coder's shape rather than of
    overconfidence: it fixed the confident cluster at a single probability, so
    at balanced prevalence the coder had 7% negative-class error and needed an
    interior band that sharpening then destroyed. Parameterising the coder by
    its class-conditional error rates removes the artifact, and the honest
    claim is the one asserted here.
    """
    trials = 120
    for prevalence in (0.50, 0.20, 0.05):
        n_cal = _calibration_size(prevalence)
        measured: list[float] = []
        for generator in (_good, _overconfident):
            rng = np.random.default_rng(777)
            certified = 0
            for _ in range(trials):
                probs, gold = generator(rng, n_cal, prevalence)
                selection = fit_thresholds(probs, gold, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
                certified += selection.selected is not None
            measured.append(certified / trials)
        honest, over = measured
        assert honest > 0.20, "the honest coder should be certified here, or this proves nothing"
        assert over <= honest + 0.05, (
            f"overconfidence was rewarded at prevalence {prevalence}: "
            f"{over:.0%} certified versus {honest:.0%} for the honest coder"
        )


@pytest.mark.parametrize("prevalence", [0.50, 0.20, 0.05])
def test_the_overconfident_control_really_is_miscalibrated(prevalence: float) -> None:
    """Guard on the tests above: if the sharpening stopped biting, they prove nothing."""
    from scruple.stats.calibration import expected_calibration_error

    rng = np.random.default_rng(int(prevalence * 4242))
    honest_p, honest_y = _good(rng, 40_000, prevalence)
    over_p, over_y = _overconfident(rng, 40_000, prevalence)

    honest_ece = expected_calibration_error(honest_p, honest_y).value
    over_ece = expected_calibration_error(over_p, over_y).value
    assert honest_ece is not None and over_ece is not None
    assert over_ece > honest_ece * 2.0


def test_a_random_coder_is_never_certified() -> None:
    """Probabilities carrying no signal must yield no automation at any band."""
    rng = np.random.default_rng(4242)
    for _ in range(200):
        n = 2000
        gold = (rng.random(n) < 0.3).astype(np.float64)
        probs = rng.random(n)  # independent of gold
        selection = fit_thresholds(probs, gold, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
        assert selection.verdict.usable is False


@pytest.mark.parametrize("prevalence", [0.20, 0.05, 0.02])
def test_the_most_conservative_band_fails_where_wider_bands_pass(prevalence: float) -> None:
    """Why the grid is scanned rather than walked (`_WHY_NOT_FIXED_SEQUENCE`).

    §8.6 originally specified a fixed-sequence walk from the most conservative
    band, stopping at the first failure. That is only sound if the most
    conservative band is the easiest to certify. It is the hardest, for two
    reasons that both bite on rare codes:

    * **No power.** A high ``t_hi`` accepts only a handful of gold positives, and
      a handful of clean items is not evidence of controlled risk -- with
      alpha = delta = 0.05, zero errors out of six gives p = 0.74.
    * **Adverse selection.** Narrowing drops moderately-confident correct
      decisions while keeping confidently-wrong positives, because a low score is
      an accepted "no". So the error *share* of the positive class can rise as
      coverage falls.

    Either way the first band in the walk fails, so a fixed-sequence walk
    declares these codes NOT_AUTOMATABLE while a corrected full scan certifies
    them at full coverage. That is the difference between a usable tool and one
    that refuses every rare code -- which is most research codes (§4).
    """
    rng = np.random.default_rng(int(prevalence * 2024))
    n = _calibration_size(prevalence)
    probs, gold = _good(rng, n, prevalence)

    # Evaluate through fit_thresholds so the candidates carry the real
    # per-candidate confidence levels rather than a flat approximation of them.
    selection = fit_thresholds(probs, gold, alpha=ALPHA, delta=DELTA, grid=SIM_GRID)
    candidates = selection.candidates
    most_conservative = max(candidates, key=lambda c: c.pair.band_width)
    chosen = next(c for c in candidates if c.pair == selection.selected)

    assert most_conservative.valid is False, (
        "the most conservative band passed; if that becomes the norm, revisit "
        "whether fixed-sequence testing is safe after all"
    )
    assert selection.selected is not None, "a wider-coverage band should still certify"
    assert chosen.coverage > most_conservative.coverage
    # And the mechanism is visible: far fewer positives to learn from.
    assert most_conservative.positive.n_accepted < chosen.positive.n_accepted


def test_bootstrap_intervals_achieve_nominal_coverage() -> None:
    """§14.3: at least 90% of nominal 95% intervals contain the true value."""
    from scruple.stats.agreement import cohens_kappa
    from scruple.stats.bootstrap import stratified_bootstrap_ci

    rng = np.random.default_rng(8080)
    n, prevalence, accuracy = 400, 0.3, 0.9

    # The estimand is kappa for this coder in the large-sample limit.
    big_gold = (rng.random(400_000) < prevalence).astype(np.float64)
    big_pred = np.where(rng.random(400_000) > accuracy, 1 - big_gold, big_gold)
    truth = cohens_kappa(big_gold, big_pred).value
    assert truth is not None

    contained = 0
    trials = 300
    for i in range(trials):
        gold = (rng.random(n) < prevalence).astype(np.float64)
        pred = np.where(rng.random(n) > accuracy, 1 - gold, gold)
        ci = stratified_bootstrap_ci(gold, pred, statistic=cohens_kappa, n_resamples=400, seed=i)
        if ci is not None and ci[0] <= truth <= ci[1]:
            contained += 1

    assert contained / trials >= 0.90
