"""Unit tests for selective and labour curves (plan §8.7, §8.8)."""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats.agreement import cohens_kappa
from scruple.stats.selective import (
    final_reliability,
    labour_curve,
    random_baseline_curve,
    selective_curve,
)
from scruple.stats.thresholds import ThresholdPair, default_grid


def _calibrated(
    n: int, seed: int, a: float = 0.35, b: float = 1.4
) -> tuple[np.ndarray, np.ndarray]:
    """A perfectly calibrated coder: y ~ Bernoulli(p), so p means what it says."""
    rng = np.random.default_rng(seed)
    p = rng.beta(a, b, n)
    y = (rng.random(n) < p).astype(float)
    return p, y


class TestFinalReliability:
    def test_abstained_items_are_correct_by_construction(self) -> None:
        # §8.8: a human coded them, so they carry the gold label.
        probs = np.array([0.5, 0.5, 0.5])
        gold = np.array([1.0, 0.0, 1.0])
        point = final_reliability(probs, gold, ThresholdPair(0.1, 0.9))
        assert point.labour == pytest.approx(1.0)
        assert point.coverage == pytest.approx(0.0)
        assert point.kappa.value == pytest.approx(1.0)

    def test_full_coverage_reproduces_the_raw_model_kappa(self) -> None:
        probs, gold = _calibrated(2000, seed=1)
        point = final_reliability(probs, gold, None)
        raw = cohens_kappa(gold, (probs >= 0.5).astype(float))
        assert point.labour == pytest.approx(0.0)
        assert point.coverage == pytest.approx(1.0)
        assert point.kappa.value == pytest.approx(raw.value)

    def test_labour_is_the_abstained_fraction(self) -> None:
        probs = np.array([0.99, 0.5, 0.5, 0.01])
        gold = np.array([1.0, 1.0, 0.0, 0.0])
        point = final_reliability(probs, gold, ThresholdPair(0.1, 0.9))
        assert point.labour == pytest.approx(0.5)

    def test_labour_offset_accounts_for_the_gold_sample(self) -> None:
        # The gold sample is real human labour too, and the report must not
        # quietly leave it off the x-axis.
        probs = np.array([0.99, 0.5, 0.5, 0.01])
        gold = np.array([1.0, 1.0, 0.0, 0.0])
        point = final_reliability(probs, gold, ThresholdPair(0.1, 0.9), labour_offset=0.06)
        assert point.labour == pytest.approx(0.56)

    def test_reviewing_more_never_lowers_reliability_much(self) -> None:
        probs, gold = _calibrated(4000, seed=2)
        none_reviewed = final_reliability(probs, gold, None).kappa.value
        some_reviewed = final_reliability(probs, gold, ThresholdPair(0.2, 0.8)).kappa.value
        assert none_reviewed is not None and some_reviewed is not None
        assert some_reviewed > none_reviewed

    def test_weights_are_honoured(self) -> None:
        probs = np.array([0.99, 0.5])
        gold = np.array([1.0, 0.0])
        point = final_reliability(
            probs, gold, ThresholdPair(0.1, 0.9), weights=np.array([3.0, 1.0])
        )
        assert point.labour == pytest.approx(0.25)

    def test_bootstrap_ci_is_attached_when_requested(self) -> None:
        probs, gold = _calibrated(500, seed=3)
        point = final_reliability(probs, gold, ThresholdPair(0.1, 0.9), n_resamples=200, seed=3)
        assert point.kappa.ci is not None
        assert point.kappa.ci[0] <= point.kappa.value <= point.kappa.ci[1]  # type: ignore[operator]


class TestSelectiveCurve:
    def test_reports_prevalence_alongside_kappa(self) -> None:
        # §8.7: selective kappa without prevalence beside it is misleading.
        probs, gold = _calibrated(3000, seed=4)
        curve = selective_curve(probs, gold)
        assert all(pt.prevalence is not None for pt in curve if pt.n_accepted > 0)

    def test_coverage_is_monotone_non_increasing_as_the_band_widens(self) -> None:
        probs, gold = _calibrated(3000, seed=5)
        curve = selective_curve(probs, gold)  # ordered most conservative first
        coverages = [pt.coverage for pt in curve]
        assert coverages == sorted(coverages)

    def test_is_a_diagnostic_not_the_headline(self) -> None:
        """Selective kappa can fall as coverage drops even on a good coder.

        This reproduces the §8.7 paradox on generated data: restricting to the
        confident subset raises accuracy and lowers kappa, because the subset
        skews toward one class and inflates chance agreement.
        """
        probs, gold = _calibrated(20_000, seed=6, a=0.2, b=1.8)
        curve = selective_curve(probs, gold)
        confident = curve[0]
        full = curve[-1]
        assert confident.accuracy is not None and full.accuracy is not None
        assert confident.accuracy > full.accuracy
        assert confident.kappa.value is not None and full.kappa.value is not None
        assert confident.kappa.value < full.kappa.value
        # And the reason is visible in the prevalence column.
        assert confident.prevalence is not None and full.prevalence is not None
        assert confident.prevalence < full.prevalence

    def test_empty_accepted_subset_yields_an_undefined_kappa(self) -> None:
        probs = np.full(10, 0.5)
        gold = np.tile([1.0, 0.0], 5)
        curve = selective_curve(probs, gold, grid=[ThresholdPair(0.1, 0.9)])
        assert curve[0].n_accepted == 0
        assert curve[0].kappa.value is None
        assert curve[0].prevalence is None
        assert curve[0].accuracy is None


class TestLabourCurve:
    def test_spans_from_no_review_to_full_review(self) -> None:
        probs, gold = _calibrated(2000, seed=7)
        curve = labour_curve(probs, gold)
        assert curve[0].labour == pytest.approx(0.0)
        assert curve[0].pair is None
        assert curve[-1].labour > curve[0].labour

    def test_labour_is_non_decreasing_along_the_curve(self) -> None:
        probs, gold = _calibrated(2000, seed=8)
        labours = [pt.labour for pt in labour_curve(probs, gold)]
        assert labours == sorted(labours)

    def test_kappa_rises_with_labour_for_a_calibrated_coder(self) -> None:
        probs, gold = _calibrated(8000, seed=9)
        curve = labour_curve(probs, gold)
        first = curve[0].kappa.value
        last = curve[-1].kappa.value
        assert first is not None and last is not None
        assert last > first

    def test_a_custom_grid_is_honoured(self) -> None:
        probs, gold = _calibrated(1000, seed=10)
        curve = labour_curve(probs, gold, grid=[ThresholdPair(0.2, 0.8)])
        assert len(curve) == 2  # the no-review point plus the one band

    def test_default_grid_length(self) -> None:
        probs, gold = _calibrated(1000, seed=11)
        assert len(labour_curve(probs, gold)) == len(default_grid()) + 1


class TestRandomBaseline:
    def test_zero_labour_matches_the_model_alone(self) -> None:
        probs, gold = _calibrated(2000, seed=12)
        base = random_baseline_curve(probs, gold, [0.0], seed=12, n_resamples=20)
        model_only = final_reliability(probs, gold, None).kappa.value
        assert base[0].kappa.value == pytest.approx(model_only)

    def test_full_labour_reaches_perfect_agreement(self) -> None:
        probs, gold = _calibrated(500, seed=13)
        base = random_baseline_curve(probs, gold, [1.0], seed=13, n_resamples=20)
        assert base[0].kappa.value == pytest.approx(1.0)

    def test_kappa_rises_with_labour(self) -> None:
        probs, gold = _calibrated(3000, seed=14)
        base = random_baseline_curve(probs, gold, [0.0, 0.25, 0.5, 0.75], seed=14, n_resamples=40)
        values = [pt.kappa.value for pt in base]
        assert all(v is not None for v in values)
        assert values == sorted(values)  # type: ignore[type-var]

    def test_calibrated_selection_beats_random_at_matched_labour(self) -> None:
        """This gap is the product (§8.8).

        At the same number of items handed to a human, choosing them by
        calibrated uncertainty must deliver a more reliable dataset than
        choosing them at random. If these curves ever converge on real data, §3
        says stop building.
        """
        probs, gold = _calibrated(20_000, seed=15)
        curve = [pt for pt in labour_curve(probs, gold) if 0.05 < pt.labour < 0.6]
        assert curve, "expected the grid to produce mid-range labour points"
        baseline = random_baseline_curve(
            probs, gold, [pt.labour for pt in curve], seed=15, n_resamples=40
        )
        for calibrated_pt, random_pt in zip(curve, baseline, strict=True):
            assert calibrated_pt.kappa.value is not None
            assert random_pt.kappa.value is not None
            assert calibrated_pt.kappa.value > random_pt.kappa.value

    def test_labour_fractions_are_validated(self) -> None:
        probs, gold = _calibrated(100, seed=16)
        with pytest.raises(ValueError, match="labour"):
            random_baseline_curve(probs, gold, [1.5], seed=16, n_resamples=5)

    def test_is_deterministic_given_a_seed(self) -> None:
        probs, gold = _calibrated(800, seed=17)
        a = random_baseline_curve(probs, gold, [0.3], seed=99, n_resamples=25)
        b = random_baseline_curve(probs, gold, [0.3], seed=99, n_resamples=25)
        assert a[0].kappa.value == b[0].kappa.value


class TestInputValidation:
    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            final_reliability(np.array([0.5, 0.5]), np.array([1.0]), None)

    def test_selective_curve_also_validates_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            selective_curve(np.array([0.5, 0.5]), np.array([1.0]))

    def test_random_baseline_also_validates_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            random_baseline_curve(np.array([0.5, 0.5]), np.array([1.0]), [0.5])
