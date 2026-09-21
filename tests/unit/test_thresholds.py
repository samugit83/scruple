"""Unit tests for class-conditional threshold selection (plan §8.6, §14.3).

The statistical-validity simulation lives in tests/statistical/; what is here
pins the mechanics: grid ordering, the binomial tail test, the fixed-sequence
walk, and the two hard gates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import binom

from scruple.stats.agreement import cohens_kappa
from scruple.stats.thresholds import (
    COMFORTABLE_KAPPA,
    DEFAULT_T_HI,
    MIN_POSITIVE_INSTANCES,
    PUBLICATION_KAPPA,
    ThresholdPair,
    class_risk,
    default_grid,
    evaluate_candidate,
    fit_thresholds,
    grade,
    order_by_conservatism,
    realised_class_risk,
)
from scruple.stats.types import Estimate, ReasonCode, Verdict


class TestThresholdPair:
    def test_decide_above_t_hi_is_positive(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).decide(0.95) == 1

    def test_decide_at_t_hi_is_positive(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).decide(0.9) == 1

    def test_decide_below_t_lo_is_negative(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).decide(0.05) == 0

    def test_decide_at_t_lo_is_negative(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).decide(0.1) == 0

    def test_decide_inside_the_band_abstains(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).decide(0.5) is None

    def test_band_width(self) -> None:
        assert ThresholdPair(t_lo=0.1, t_hi=0.9).band_width == pytest.approx(0.8)

    def test_decide_all_returns_decisions_and_an_accepted_mask(self) -> None:
        pair = ThresholdPair(t_lo=0.1, t_hi=0.9)
        decisions, accepted = pair.decide_all(np.array([0.95, 0.5, 0.05]))
        assert accepted.tolist() == [True, False, True]
        assert decisions[0] == 1
        assert decisions[2] == 0

    @pytest.mark.parametrize(
        ("lo", "hi", "problem"),
        [
            (0.9, 0.1, "strictly below"),
            (0.5, 0.5, "strictly below"),
            (-0.1, 0.9, r"\[0, 1\]"),
            (0.1, 1.1, r"\[0, 1\]"),
        ],
    )
    def test_invalid_pairs_are_rejected(self, lo: float, hi: float, problem: str) -> None:
        with pytest.raises(ValueError, match=problem):
            ThresholdPair(t_lo=lo, t_hi=hi)


class TestGrid:
    def test_default_grid_spans_the_documented_range(self) -> None:
        grid = default_grid()
        t_his = [p.t_hi for p in grid]
        assert max(t_his) == pytest.approx(0.99)
        assert min(t_his) == pytest.approx(0.55)

    def test_default_grid_is_coarse_enough_to_keep_the_correction_cheap(self) -> None:
        # Bonferroni over the grid means resolution is not free (see
        # DEFAULT_T_HI): a 45-point grid roughly halved the certification rate.
        assert len(default_grid()) == len(DEFAULT_T_HI) == 15

    def test_default_grid_is_symmetric(self) -> None:
        assert all(p.t_lo == pytest.approx(1.0 - p.t_hi) for p in default_grid())

    def test_default_grid_is_ordered_most_conservative_first(self) -> None:
        grid = default_grid()
        assert grid[0].t_hi == pytest.approx(0.99)
        assert grid[-1].t_hi == pytest.approx(0.55)
        widths = [p.band_width for p in grid]
        assert widths == sorted(widths, reverse=True)

    def test_ordering_is_independent_of_the_data(self) -> None:
        """Fixed-sequence testing requires an order fixed before the data is seen.

        Ordering by observed coverage would make the sequence depend on the very
        calibration sample the tests are run against, which quietly breaks the
        error control §8.6 claims. Band width is a property of the grid alone.
        """
        pairs = [
            ThresholdPair(t_lo=0.2, t_hi=0.8),
            ThresholdPair(t_lo=0.05, t_hi=0.95),
            ThresholdPair(t_lo=0.4, t_hi=0.6),
        ]
        ordered = order_by_conservatism(pairs)
        assert [p.t_hi for p in ordered] == [0.95, 0.8, 0.6]

    def test_asymmetric_pairs_are_permitted_and_ordered_by_width(self) -> None:
        pairs = [
            ThresholdPair(t_lo=0.1, t_hi=0.6),  # width 0.5
            ThresholdPair(t_lo=0.3, t_hi=0.99),  # width 0.69
        ]
        assert order_by_conservatism(pairs)[0].t_hi == pytest.approx(0.99)


class TestClassRisk:
    def test_p_value_matches_the_binomial_tail(self) -> None:
        # §8.6 step 3: p = P(Binom(n_c, alpha) <= E_c)
        errors = np.array([1.0] * 7 + [0.0] * 300)
        r = class_risk(errors, np.ones(307), gold_label=1, alpha=0.05, delta=0.05)
        assert r.p_value == pytest.approx(binom.cdf(7, 307, 0.05))
        assert r.risk == pytest.approx(7 / 307)

    def test_zero_errors_in_a_large_sample_passes(self) -> None:
        r = class_risk(np.zeros(200), np.ones(200), gold_label=1, alpha=0.05, delta=0.05)
        assert r.p_value == pytest.approx(0.95**200)
        assert r.passed is True

    def test_zero_errors_in_a_small_sample_does_not_pass(self) -> None:
        # 0.95**20 = 0.358: twenty clean items are not evidence of 5% risk.
        r = class_risk(np.zeros(20), np.ones(20), gold_label=1, alpha=0.05, delta=0.05)
        assert r.passed is False

    def test_observing_exactly_alpha_error_rate_does_not_pass(self) -> None:
        # Seeing 5% errors is not evidence that the risk is at most 5%.
        errors = np.array([1.0] * 5 + [0.0] * 95)
        r = class_risk(errors, np.ones(100), gold_label=0, alpha=0.05, delta=0.05)
        assert r.passed is False

    def test_no_accepted_items_in_the_class_cannot_pass(self) -> None:
        # Zero evidence is not evidence of safety, and accepting nothing from a
        # class would otherwise be a free pass (§8.6's degeneracy).
        r = class_risk(np.array([]), np.array([]), gold_label=1, alpha=0.05, delta=0.05)
        assert r.passed is False
        assert r.risk is None
        assert r.p_value == 1.0

    def test_effective_sample_size_is_used_under_weighting(self) -> None:
        # Unequal weights carry less evidence than their raw count suggests.
        w = np.array([1.0] * 100 + [50.0])
        r = class_risk(np.zeros(101), w, gold_label=1, alpha=0.05, delta=0.05)
        assert r.n_accepted == 101
        assert r.n_effective < 101

    def test_uniform_weights_reproduce_the_unweighted_p_value_exactly(self) -> None:
        errors = np.array([1.0] * 3 + [0.0] * 97)
        a = class_risk(errors, np.ones(100), gold_label=1, alpha=0.05, delta=0.05)
        b = class_risk(errors, np.full(100, 2.5), gold_label=1, alpha=0.05, delta=0.05)
        assert a.p_value == b.p_value


class TestEvaluateCandidate:
    def test_splits_risk_by_true_class(self) -> None:
        probs = np.array([0.99] * 10 + [0.01] * 10)
        gold = np.array([1] * 9 + [0] + [0] * 9 + [1])
        cand = evaluate_candidate(
            probs, gold, np.ones(20), ThresholdPair(0.05, 0.95), alpha=0.05, delta=0.05
        )
        # One gold-positive was decided negative; one gold-negative decided positive.
        assert cand.positive.n_accepted == 10
        assert cand.positive.errors == pytest.approx(1.0)
        assert cand.negative.n_accepted == 10
        assert cand.negative.errors == pytest.approx(1.0)

    def test_coverage_is_the_weighted_accepted_fraction(self) -> None:
        probs = np.array([0.99, 0.5, 0.01, 0.5])
        gold = np.array([1, 1, 0, 0])
        cand = evaluate_candidate(
            probs, gold, np.ones(4), ThresholdPair(0.05, 0.95), alpha=0.05, delta=0.05
        )
        assert cand.coverage == pytest.approx(0.5)
        assert cand.n_accepted == 2


def _clean_coder(
    n_pos: int, n_neg: int, err_pos: int, err_neg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Confident coder: `err_*` of each class land on the wrong side of the band."""
    probs = np.concatenate(
        [
            np.full(n_pos - err_pos, 0.99),  # gold positive, decided positive
            np.full(err_pos, 0.01),  # gold positive, decided negative (error)
            np.full(n_neg - err_neg, 0.01),  # gold negative, decided negative
            np.full(err_neg, 0.99),  # gold negative, decided positive (error)
        ]
    )
    gold = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return probs, gold


class TestFitThresholds:
    def test_a_confident_accurate_coder_is_certified(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold)
        assert sel.verdict is Verdict.OK
        assert sel.selected is not None
        assert sel.coverage == pytest.approx(1.0)

    def test_the_highest_coverage_passing_pair_is_selected(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold)
        best = max(c.coverage for c in sel.candidates if c.valid)
        assert sel.coverage == pytest.approx(best)

    def test_ties_on_coverage_go_to_the_wider_band(self) -> None:
        # Every band here accepts the same items, so the extra width costs
        # nothing and is the safer choice on data the fit has not seen.
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold)
        assert sel.selected == default_grid()[0]

    def test_fewer_than_fifteen_positives_is_refused(self) -> None:
        # §8.5 hard gate: reporting a confident-looking kappa on nine items is
        # the most likely way this project produces a false result.
        probs, gold = _clean_coder(n_pos=11, n_neg=600, err_pos=0, err_neg=0)
        sel = fit_thresholds(probs, gold)
        assert sel.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert sel.reason is ReasonCode.INSUFFICIENT_EVIDENCE
        assert sel.n_gold_positive == 11
        assert "11" in sel.message

    def test_the_gate_is_exactly_fifteen(self) -> None:
        assert MIN_POSITIVE_INSTANCES == 15
        probs, gold = _clean_coder(n_pos=15, n_neg=600, err_pos=0, err_neg=0)
        assert fit_thresholds(probs, gold).verdict is not Verdict.INSUFFICIENT_EVIDENCE

    def test_the_gate_counts_raw_items_not_weighted_ones(self) -> None:
        # Ten items reweighted to look like a hundred are still ten items read.
        probs, gold = _clean_coder(n_pos=10, n_neg=600, err_pos=0, err_neg=0)
        w = np.where(gold == 1, 10.0, 1.0)
        assert fit_thresholds(probs, gold, weights=w).verdict is Verdict.INSUFFICIENT_EVIDENCE

    def test_an_unreliable_coder_is_not_automatable(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=90, err_neg=180)
        sel = fit_thresholds(probs, gold)
        assert sel.verdict is Verdict.NOT_AUTOMATABLE
        assert sel.reason is ReasonCode.NO_VALID_THRESHOLD
        assert sel.selected is None

    @pytest.mark.parametrize("prevalence", [0.50, 0.20, 0.05, 0.02])
    def test_negative_control_an_all_negative_coder_is_never_certified(
        self, prevalence: float
    ) -> None:
        """The single most important assertion in the repository (§14.6).

        A coder that always says "no" achieves a small *marginal* error rate on a
        rare code and would be certified by a marginal guarantee at full coverage
        with kappa 0. The class-conditional formulation must reject it at every
        prevalence. If this ever passes, the guarantee is broken and nothing else
        in the repo matters.
        """
        rng = np.random.default_rng(int(prevalence * 1000))
        n = 4000
        gold = (rng.random(n) < prevalence).astype(float)
        probs = np.full(n, 0.01)  # always "no", stated with confidence
        sel = fit_thresholds(probs, gold)
        assert sel.verdict.usable is False

    def test_an_all_positive_coder_is_also_never_certified(self) -> None:
        rng = np.random.default_rng(99)
        gold = (rng.random(4000) < 0.05).astype(float)
        assert fit_thresholds(np.full(4000, 0.99), gold).verdict.usable is False

    def test_the_scan_does_not_stop_at_the_first_failing_band(self) -> None:
        """Regression guard for the finding in `_WHY_NOT_FIXED_SEQUENCE`.

        Class-conditional positive risk is *not* monotone in coverage. Ten true
        positives scored 0.005 are accepted as a confident "no" at every band, so
        they are errors at every band; the 600 correctly-scored positives are
        only accepted once t_hi drops to 0.93. The most conservative bands
        therefore hold nothing but the errors and fail, while the full-coverage
        bands pass.

        A fixed-sequence walk from the most conservative band would stop at the
        first failure and declare this code NOT_AUTOMATABLE, discarding a coder
        that genuinely controls risk at 100% coverage.
        """
        probs = np.concatenate(
            [
                np.full(600, 0.93),  # positives the coder scored correctly
                np.full(10, 0.005),  # positives it confidently got wrong
                np.full(1200, 0.005),  # correctly scored negatives
                np.full(20, 0.93),  # negatives it confidently got wrong
            ]
        )
        gold = np.concatenate([np.ones(610), np.zeros(1220)])

        sel = fit_thresholds(probs, gold)
        assert sel.verdict is Verdict.OK
        assert sel.coverage == pytest.approx(1.0)

        by_t_hi = {c.pair.t_hi: c for c in sel.candidates}
        assert by_t_hi[0.99].valid is False, "the conservative band should fail here"
        assert by_t_hi[0.90].valid is True, "a wider-coverage band should pass"
        assert by_t_hi[0.99].positive.risk == pytest.approx(1.0)
        assert by_t_hi[0.90].positive.risk is not None
        assert by_t_hi[0.90].positive.risk < by_t_hi[0.99].positive.risk

    def test_every_candidate_is_evaluated_for_the_report(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold)
        assert len(sel.candidates) == len(default_grid())
        assert all(c.valid for c in sel.candidates)

    def test_the_grid_correction_is_bonferroni_over_its_candidates(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold, delta=0.05)
        assert sel.delta_per_candidate == pytest.approx(0.05 / len(default_grid()))

    def test_a_smaller_grid_gets_a_weaker_correction(self) -> None:
        # Each extra candidate is paid for by every other candidate.
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        small = fit_thresholds(probs, gold, grid=[ThresholdPair(0.05, 0.95)])
        assert small.delta_per_candidate > fit_thresholds(probs, gold).delta_per_candidate

    def test_alpha_and_delta_are_recorded_for_the_report(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        sel = fit_thresholds(probs, gold, alpha=0.10, delta=0.01)
        assert (sel.alpha, sel.delta) == (0.10, 0.01)

    def test_a_stricter_alpha_cannot_increase_coverage(self) -> None:
        probs, gold = _clean_coder(n_pos=400, n_neg=800, err_pos=12, err_neg=10)
        loose = fit_thresholds(probs, gold, alpha=0.10)
        strict = fit_thresholds(probs, gold, alpha=0.01)
        assert (strict.coverage or 0.0) <= (loose.coverage or 0.0) + 1e-12

    def test_uniform_weights_reproduce_the_unweighted_selection_exactly(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        a = fit_thresholds(probs, gold)
        b = fit_thresholds(probs, gold, weights=np.ones(900))
        assert a.selected == b.selected
        assert a.coverage == b.coverage

    def test_uniform_weights_are_reported_as_an_exact_test(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        assert fit_thresholds(probs, gold).exact is True

    def test_non_uniform_weights_are_reported_as_approximate(self) -> None:
        # The binomial tail on a Kish effective sample size is an approximation,
        # and §8.9 requires the report to say so rather than imply exactness.
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        w = np.where(gold == 1, 4.0, 1.0)
        assert fit_thresholds(probs, gold, weights=w).exact is False

    def test_empty_input_is_insufficient_evidence(self) -> None:
        sel = fit_thresholds(np.array([]), np.array([]))
        assert sel.verdict is Verdict.INSUFFICIENT_EVIDENCE

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            fit_thresholds(np.array([0.5, 0.5]), np.array([1.0]))

    @pytest.mark.parametrize(
        ("alpha", "delta", "problem"),
        [(0.0, 0.05, "alpha"), (1.0, 0.05, "alpha"), (0.05, 0.0, "delta")],
    )
    def test_alpha_and_delta_are_validated(self, alpha: float, delta: float, problem: str) -> None:
        with pytest.raises(ValueError, match=problem):
            fit_thresholds(np.array([0.9]), np.array([1.0]), alpha=alpha, delta=delta)

    def test_a_custom_grid_is_honoured_and_reordered(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        grid = [ThresholdPair(0.4, 0.6), ThresholdPair(0.02, 0.98)]
        sel = fit_thresholds(probs, gold, grid=grid)
        assert sel.candidates[0].pair.t_hi == pytest.approx(0.98)

    def test_an_empty_grid_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="grid"):
            fit_thresholds(np.array([0.9]), np.array([1.0]), grid=[])


class TestGrade:
    def _certified(self) -> object:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        return fit_thresholds(probs, gold)

    def test_comfortably_above_the_bar_is_ok(self) -> None:
        sel = self._certified()
        assert grade(sel, Estimate(value=0.84, n=300)).verdict is Verdict.OK  # type: ignore[arg-type]

    def test_just_above_the_bar_is_weak(self) -> None:
        sel = self._certified()
        assert grade(sel, Estimate(value=0.71, n=300)).verdict is Verdict.WEAK  # type: ignore[arg-type]

    def test_below_the_publication_bar_is_not_automatable(self) -> None:
        sel = self._certified()
        graded = grade(sel, Estimate(value=0.41, n=300))  # type: ignore[arg-type]
        assert graded.verdict is Verdict.NOT_AUTOMATABLE
        assert graded.reason is ReasonCode.BELOW_PUBLICATION_BAR

    def test_the_boundaries_are_the_documented_constants(self) -> None:
        assert PUBLICATION_KAPPA == 0.70
        assert COMFORTABLE_KAPPA == 0.75
        sel = self._certified()
        assert grade(sel, Estimate(value=0.70, n=300)).verdict is Verdict.WEAK  # type: ignore[arg-type]
        assert grade(sel, Estimate(value=0.75, n=300)).verdict is Verdict.OK  # type: ignore[arg-type]

    def test_an_undefined_kappa_downgrades_to_insufficient_evidence(self) -> None:
        sel = self._certified()
        undefined = Estimate(value=None, n=5, reason=ReasonCode.DEGENERATE_SINGLE_CLASS)
        graded = grade(sel, undefined)  # type: ignore[arg-type]
        assert graded.verdict is Verdict.INSUFFICIENT_EVIDENCE

    def test_a_failed_selection_is_passed_through_unchanged(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=90, err_neg=180)
        sel = fit_thresholds(probs, gold)
        assert grade(sel, Estimate(value=0.9, n=300)).verdict is Verdict.NOT_AUTOMATABLE

    def test_grading_carries_the_kappa_for_the_report(self) -> None:
        sel = self._certified()
        graded = grade(sel, Estimate(value=0.84, n=300))  # type: ignore[arg-type]
        assert graded.held_out_kappa is not None
        assert graded.held_out_kappa.value == pytest.approx(0.84)


def test_kappa_of_the_accepted_subset_is_not_the_guarantee() -> None:
    """Sanity anchor tying §8.6 to §8.7: passing the guarantee says nothing
    directly about selective kappa, which is why §8.8 reports something else."""
    probs, gold = _clean_coder(n_pos=5, n_neg=495, err_pos=1, err_neg=2)
    pair = ThresholdPair(0.05, 0.95)
    decisions, accepted = pair.decide_all(probs)
    k = cohens_kappa(gold[accepted], decisions[accepted])
    assert k.value == pytest.approx(0.7242647, abs=1e-6)


class TestRealisedClassRisk:
    def test_splits_observed_error_by_true_class(self) -> None:
        probs = np.array([0.99] * 10 + [0.01] * 10)
        gold = np.array([1.0] * 9 + [0.0] + [0.0] * 9 + [1.0])
        risk = realised_class_risk(probs, gold, ThresholdPair(0.05, 0.95))
        assert risk[1] == pytest.approx(1 / 10)
        assert risk[0] == pytest.approx(1 / 10)

    def test_a_class_with_no_accepted_items_reports_none(self) -> None:
        # Never a flattering 0.0 where there is simply no evidence.
        probs = np.array([0.5, 0.99])
        gold = np.array([1.0, 0.0])
        risk = realised_class_risk(probs, gold, ThresholdPair(0.05, 0.95))
        assert risk[1] is None
        assert risk[0] == pytest.approx(1.0)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            realised_class_risk(np.array([0.9, 0.1]), np.array([1.0]), ThresholdPair(0.1, 0.9))

    def test_weights_are_honoured(self) -> None:
        probs = np.array([0.99, 0.99])
        gold = np.array([0.0, 0.0])
        risk = realised_class_risk(
            probs, gold, ThresholdPair(0.05, 0.95), weights=np.array([1.0, 3.0])
        )
        assert risk[0] == pytest.approx(1.0)


class TestSelectionMessage:
    """`scruple check` prints these, and they are the difference between a
    failure the researcher can act on and one they cannot (§12)."""

    def test_insufficient_evidence_names_the_count_and_the_remedy(self) -> None:
        probs, gold = _clean_coder(n_pos=9, n_neg=500, err_pos=0, err_neg=0)
        message = fit_thresholds(probs, gold).message
        assert "9 positive instances" in message
        assert "scruple gold --enrich" in message

    def test_not_automatable_points_at_the_codebook_not_the_model(self) -> None:
        # A code where two humans disagree is a definition problem, and saying so
        # turns a failure into a useful finding (§12).
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=90, err_neg=180)
        message = fit_thresholds(probs, gold).message
        assert "could not be automated" in message
        assert "coders" in message

    def test_below_publication_bar_quotes_the_kappa(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        graded = grade(fit_thresholds(probs, gold), Estimate(value=0.41, n=300))
        assert "0.41" in graded.message
        assert "0.7" in graded.message

    def test_below_publication_bar_copes_with_an_absent_kappa(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        broken = replace(
            fit_thresholds(probs, gold),
            verdict=Verdict.NOT_AUTOMATABLE,
            reason=ReasonCode.BELOW_PUBLICATION_BAR,
        )
        assert "undefined" in broken.message

    def test_a_passing_code_has_nothing_to_explain(self) -> None:
        probs, gold = _clean_coder(n_pos=300, n_neg=600, err_pos=5, err_neg=3)
        assert fit_thresholds(probs, gold).message == ""
