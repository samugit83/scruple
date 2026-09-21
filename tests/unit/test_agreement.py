"""Unit tests for agreement statistics (plan §8.1, §8.2, §14.3).

Every expected value below is hand-computed from the formulae in §8.1 and
recorded in the test, so a refactor cannot quietly change the mathematics.
"""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats.agreement import (
    cohens_kappa,
    cohens_kappa_from_confusion,
    confusion,
    krippendorff_alpha,
    krippendorff_alpha_from_confusion,
    observed_agreement,
)
from scruple.stats.types import Confusion, ReasonCode

# --------------------------------------------------------------------------
# The eight hand-computed confusion tables required by §14.3.
# --------------------------------------------------------------------------

PERFECT = Confusion(a=50, b=0, c=0, d=50)
#   p_o = 1.0 ; p_e = 0.5*0.5 + 0.5*0.5 = 0.5 ; k = (1-0.5)/(1-0.5) = 1.0

PERFECT_DISAGREEMENT = Confusion(a=0, b=50, c=50, d=0)
#   p_o = 0.0 ; p_e = 0.5*0.5 + 0.5*0.5 = 0.5 ; k = (0-0.5)/0.5 = -1.0

CHANCE = Confusion(a=25, b=25, c=25, d=25)
#   p_o = 0.5 ; p_e = 0.5*0.5 + 0.5*0.5 = 0.5 ; k = 0.0

BALANCED = Confusion(a=45, b=5, c=5, d=45)
#   p_o = 0.90 ; p_e = 0.5*0.5 + 0.5*0.5 = 0.5 ; k = 0.40/0.50 = 0.80

EXTREME_PREVALENCE = Confusion(a=1, b=1, c=1, d=97)
#   p_o = 0.98 ; p_e = 0.02*0.02 + 0.98*0.98 = 0.9608
#   k = (0.98 - 0.9608) / (1 - 0.9608) = 0.0192/0.0392 = 0.4897959...
#   98% accuracy, kappa below the 0.70 bar: the reason §8.7 exists.

SINGLE_CLASS = Confusion(a=0, b=0, c=0, d=100)
#   both coders said "no" to everything ; p_e = 1 ; kappa undefined

SINGLE_ITEM = Confusion(a=1, b=0, c=0, d=0)
#   n = 1, both positive ; p_e = 1 ; kappa undefined

SINGLE_ITEM_DISAGREE = Confusion(a=0, b=1, c=0, d=0)
#   n = 1, they disagree ; p_e = 0 ; kappa = 0.0 (defined, but meaningless)

# The §8.7 paradox pair. The confident subset is *more* accurate and scores
# *lower*. Asserting the inversion keeps a future refactor from "fixing" it.
PARADOX_CONFIDENT = Confusion(a=4, b=1, c=2, d=493)
#   n = 500, 5 gold positives ; p_o = 497/500 = 0.994
#   p_e = 0.01*0.012 + 0.99*0.988 = 0.97824 ; k = 0.01576/0.02176 = 0.7242647...
PARADOX_FULL = Confusion(a=170, b=30, c=40, d=760)
#   n = 1000, 200 gold positives ; p_o = 0.930
#   p_e = 0.2*0.21 + 0.8*0.79 = 0.674 ; k = 0.256/0.326 = 0.7852761...


class TestConfusionCounting:
    def test_counts_cells_from_label_vectors(self) -> None:
        gold = [1, 1, 0, 0, 1]
        pred = [1, 0, 1, 0, 1]
        cm = confusion(gold, pred)
        assert (cm.a, cm.b, cm.c, cm.d) == (2.0, 1.0, 1.0, 1.0)

    def test_weights_accumulate_instead_of_counts(self) -> None:
        cm = confusion([1, 1], [1, 1], weights=[2.5, 1.5])
        assert cm.a == pytest.approx(4.0)
        assert cm.n == pytest.approx(4.0)

    def test_empty_input_gives_an_empty_table(self) -> None:
        assert confusion([], []).n == 0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            confusion([1, 0], [1])

    @pytest.mark.parametrize("bad", [2, -1, 0.5])
    def test_non_binary_labels_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            confusion([1, bad], [1, 1])

    def test_observed_agreement(self) -> None:
        assert observed_agreement(PARADOX_FULL) == pytest.approx(0.930)
        assert observed_agreement(Confusion(0, 0, 0, 0)) is None


class TestCohensKappa:
    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            (PERFECT, 1.0),
            (PERFECT_DISAGREEMENT, -1.0),
            (CHANCE, 0.0),
            (BALANCED, 0.80),
            (EXTREME_PREVALENCE, 0.0192 / 0.0392),
            (SINGLE_ITEM_DISAGREE, 0.0),
            (PARADOX_CONFIDENT, 0.01576 / 0.02176),
            (PARADOX_FULL, 0.256 / 0.326),
        ],
    )
    def test_hand_computed_tables(self, table: Confusion, expected: float) -> None:
        est = cohens_kappa_from_confusion(table)
        assert est.value == pytest.approx(expected)

    @pytest.mark.parametrize("table", [SINGLE_CLASS, SINGLE_ITEM])
    def test_degenerate_single_class_returns_none_with_a_reason(self, table: Confusion) -> None:
        # §8.1: never divide by zero; never return NaN silently.
        est = cohens_kappa_from_confusion(table)
        assert est.value is None
        assert est.reason is ReasonCode.DEGENERATE_SINGLE_CLASS

    def test_all_positive_agreement_is_also_degenerate(self) -> None:
        est = cohens_kappa_from_confusion(Confusion(a=100, b=0, c=0, d=0))
        assert est.reason is ReasonCode.DEGENERATE_SINGLE_CLASS

    def test_empty_input_returns_none_with_a_reason(self) -> None:
        est = cohens_kappa_from_confusion(Confusion(0, 0, 0, 0))
        assert est.value is None
        assert est.reason is ReasonCode.EMPTY_INPUT

    def test_the_8_7_paradox_inversion_actually_occurs(self) -> None:
        """The higher-accuracy confident subset scores LOWER than full coverage.

        This is the documented "high agreement, low reliability" paradox, and it
        is why §8.8 demotes selective kappa to a diagnostic. If this assertion
        ever fails, someone has changed the kappa implementation and the kill
        criterion in §3 can no longer be trusted.
        """
        confident = cohens_kappa_from_confusion(PARADOX_CONFIDENT)
        full = cohens_kappa_from_confusion(PARADOX_FULL)
        assert observed_agreement(PARADOX_CONFIDENT) > observed_agreement(PARADOX_FULL)  # type: ignore[operator]
        assert confident.value is not None and full.value is not None
        assert confident.value < full.value

    def test_from_label_vectors(self) -> None:
        gold = [1] * 45 + [1] * 5 + [0] * 5 + [0] * 45
        pred = [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45
        assert cohens_kappa(gold, pred).value == pytest.approx(0.80)

    def test_n_is_the_raw_item_count_not_the_weighted_one(self) -> None:
        est = cohens_kappa([1, 0], [1, 0], weights=[10.0, 10.0])
        assert est.n == 2

    def test_identical_coders_always_score_one(self) -> None:
        rng = np.random.default_rng(7)
        labels = (rng.random(500) < 0.3).astype(int).tolist()
        assert cohens_kappa(labels, labels).value == pytest.approx(1.0)

    def test_independent_coders_score_about_zero(self) -> None:
        rng = np.random.default_rng(11)
        gold = (rng.random(20_000) < 0.3).astype(int).tolist()
        pred = (rng.random(20_000) < 0.3).astype(int).tolist()
        est = cohens_kappa(gold, pred)
        assert est.value == pytest.approx(0.0, abs=0.02)

    def test_uniform_weights_equal_the_unweighted_result_exactly(self) -> None:
        gold = [1, 1, 0, 0, 1, 0, 0, 1]
        pred = [1, 0, 0, 0, 1, 1, 0, 1]
        plain = cohens_kappa(gold, pred).value
        weighted = cohens_kappa(gold, pred, weights=[1.0] * 8).value
        assert plain == weighted

    def test_zero_total_weight_returns_none(self) -> None:
        est = cohens_kappa([1, 0], [1, 0], weights=[0.0, 0.0])
        assert est.reason is ReasonCode.EMPTY_INPUT


class TestKrippendorffAlpha:
    def test_hand_computed_balanced_table(self) -> None:
        # n_1 = 2a+b+c = 100 ; n_0 = 2d+b+c = 100 ; N = 100
        # alpha = 1 - (b+c)(2N-1)/(n_1 n_0) = 1 - 10*199/10000 = 0.801
        assert krippendorff_alpha_from_confusion(BALANCED).value == pytest.approx(0.801)

    def test_perfect_agreement_is_one(self) -> None:
        assert krippendorff_alpha_from_confusion(PERFECT).value == pytest.approx(1.0)

    def test_perfect_disagreement_is_negative(self) -> None:
        # n_1 = 100, n_0 = 100, N = 100 -> 1 - 100*199/10000 = -0.99
        assert krippendorff_alpha_from_confusion(PERFECT_DISAGREEMENT).value == pytest.approx(-0.99)

    def test_agrees_with_cohens_kappa_for_binary_two_coders(self) -> None:
        """§14.3's cross-check of two independently implemented statistics.

        alpha and kappa are not algebraically identical: alpha is Scott's pi with
        a small-sample correction, so it uses pooled marginals where kappa uses
        each coder's own. They coincide as N grows and as the coders' marginals
        converge, which is the regime a real gold sample sits in.
        """
        for table in (PERFECT, BALANCED, CHANCE, PARADOX_FULL, EXTREME_PREVALENCE):
            k = cohens_kappa_from_confusion(table).value
            a = krippendorff_alpha_from_confusion(table).value
            assert k is not None and a is not None
            assert a == pytest.approx(k, abs=0.01)

    def test_convergence_to_kappa_is_tight_at_large_n(self) -> None:
        big = Confusion(a=1700, b=300, c=400, d=7600)
        k = cohens_kappa_from_confusion(big).value
        a = krippendorff_alpha_from_confusion(big).value
        assert k is not None and a is not None
        assert a == pytest.approx(k, abs=1e-3)

    @pytest.mark.parametrize("table", [SINGLE_CLASS, Confusion(a=100, b=0, c=0, d=0)])
    def test_single_class_is_degenerate(self, table: Confusion) -> None:
        est = krippendorff_alpha_from_confusion(table)
        assert est.value is None
        assert est.reason is ReasonCode.DEGENERATE_SINGLE_CLASS

    def test_empty_input_is_reported(self) -> None:
        est = krippendorff_alpha_from_confusion(Confusion(0, 0, 0, 0))
        assert est.reason is ReasonCode.EMPTY_INPUT

    def test_single_disagreeing_item_gives_zero_like_kappa(self) -> None:
        # With one unit holding two differing values the pooled marginals are
        # exactly 50/50, so D_o == D_e and alpha is 0.0. Meaningless, but it is
        # the correct value of the formula and it matches kappa on this table.
        est = krippendorff_alpha_from_confusion(SINGLE_ITEM_DISAGREE)
        assert est.value == pytest.approx(0.0)
        assert cohens_kappa_from_confusion(SINGLE_ITEM_DISAGREE).value == pytest.approx(0.0)

    def test_from_label_vectors(self) -> None:
        gold = [1] * 45 + [1] * 5 + [0] * 5 + [0] * 45
        pred = [1] * 45 + [0] * 5 + [1] * 5 + [0] * 45
        assert krippendorff_alpha(gold, pred).value == pytest.approx(0.801)

    def test_uniform_weights_equal_the_unweighted_result_exactly(self) -> None:
        gold = [1, 1, 0, 0, 1, 0]
        pred = [1, 0, 0, 0, 1, 1]
        assert (
            krippendorff_alpha(gold, pred).value
            == krippendorff_alpha(gold, pred, weights=[1.0] * 6).value
        )
