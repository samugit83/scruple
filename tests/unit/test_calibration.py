"""Unit tests for calibration diagnostics (plan §8.3, §14.3)."""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)
from scruple.stats.types import ReasonCode


class TestBrierScore:
    def test_hand_computed(self) -> None:
        # ((0-0)^2 + (0.5-1)^2 + (1-1)^2) / 3 = 0.25/3
        assert brier_score([0.0, 0.5, 1.0], [0, 1, 1]).value == pytest.approx(0.25 / 3)

    def test_perfect_prediction_scores_zero(self) -> None:
        assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]).value == pytest.approx(0.0)

    def test_maximally_wrong_prediction_scores_one(self) -> None:
        assert brier_score([0.0, 1.0], [1, 0]).value == pytest.approx(1.0)

    def test_always_one_half_scores_a_quarter(self) -> None:
        assert brier_score([0.5] * 8, [1, 0, 1, 0, 1, 0, 1, 0]).value == pytest.approx(0.25)

    def test_empty_input(self) -> None:
        est = brier_score([], [])
        assert est.value is None
        assert est.reason is ReasonCode.EMPTY_INPUT

    def test_uniform_weights_equal_the_unweighted_result_exactly(self) -> None:
        p, y = [0.1, 0.8, 0.3, 0.9], [0, 1, 0, 1]
        assert brier_score(p, y).value == brier_score(p, y, weights=[1.0] * 4).value

    def test_weights_are_applied(self) -> None:
        # Weighting the wrong item 3x pulls the score toward it.
        est = brier_score([0.0, 1.0], [0, 0], weights=[1.0, 3.0])
        assert est.value == pytest.approx(0.75)

    def test_zero_total_weight(self) -> None:
        assert brier_score([0.5], [1], weights=[0.0]).reason is ReasonCode.ZERO_WEIGHT

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_probabilities_outside_the_unit_interval_are_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="probabilit"):
            brier_score([0.5, bad], [1, 0])

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            brier_score([0.5, 0.5], [1])


class TestExpectedCalibrationError:
    def test_hand_computed(self) -> None:
        # bin 0: n=5, conf=0.05, acc=0.00 -> |diff| 0.05, share 0.5
        # bin 9: n=5, conf=0.95, acc=1.00 -> |diff| 0.05, share 0.5
        # ECE = 0.5*0.05 + 0.5*0.05 = 0.05
        p = [0.05] * 5 + [0.95] * 5
        y = [0] * 5 + [1] * 5
        assert expected_calibration_error(p, y).value == pytest.approx(0.05)

    def test_a_perfectly_calibrated_set_is_near_zero(self) -> None:
        # §14.3 requires a perfectly calibrated synthetic set with ECE ~ 0.
        rng = np.random.default_rng(2026)
        p = rng.random(200_000)
        y = (rng.random(200_000) < p).astype(int)
        est = expected_calibration_error(p, y)
        assert est.value is not None
        assert est.value < 0.01

    def test_a_maximally_overconfident_set_is_near_one_half(self) -> None:
        # Claims certainty, is right half the time. This is the LLM failure mode
        # the whole project is built around (§0.3).
        rng = np.random.default_rng(2027)
        n = 50_000
        p = np.where(rng.random(n) < 0.5, 0.0, 1.0)
        y = rng.integers(0, 2, n)
        est = expected_calibration_error(p, y)
        assert est.value is not None
        assert est.value == pytest.approx(0.5, abs=0.02)

    def test_empty_input(self) -> None:
        assert expected_calibration_error([], []).reason is ReasonCode.EMPTY_INPUT

    def test_probability_of_exactly_one_lands_in_the_last_bin(self) -> None:
        # Guards the clip: floor(1.0 * 10) == 10 would be out of range.
        bins = reliability_diagram([1.0], [1], bins=10)
        assert bins[9].count == 1
        assert bins[9].observed_rate == pytest.approx(1.0)

    def test_bin_count_is_validated(self) -> None:
        with pytest.raises(ValueError, match="bins"):
            expected_calibration_error([0.5], [1], bins=0)

    def test_uniform_weights_equal_the_unweighted_result_exactly(self) -> None:
        p = [0.05] * 5 + [0.95] * 5
        y = [0] * 5 + [1] * 5
        assert (
            expected_calibration_error(p, y).value
            == expected_calibration_error(p, y, weights=[1.0] * 10).value
        )


class TestReliabilityDiagram:
    def test_shape_and_edges(self) -> None:
        bins = reliability_diagram([0.05] * 5 + [0.95] * 5, [0] * 5 + [1] * 5)
        assert len(bins) == 10
        assert bins[0].lo == pytest.approx(0.0)
        assert bins[0].hi == pytest.approx(0.1)
        assert bins[9].hi == pytest.approx(1.0)
        assert [b.index for b in bins] == list(range(10))

    def test_populated_bins_carry_their_statistics(self) -> None:
        bins = reliability_diagram([0.05] * 5 + [0.95] * 5, [0] * 5 + [1] * 5)
        assert bins[0].count == 5
        assert bins[0].mean_predicted == pytest.approx(0.05)
        assert bins[0].observed_rate == pytest.approx(0.0)
        assert bins[9].mean_predicted == pytest.approx(0.95)
        assert bins[9].observed_rate == pytest.approx(1.0)

    def test_empty_bins_report_no_statistics_rather_than_zero(self) -> None:
        # An empty bin with observed_rate 0.0 would be plotted as a real point
        # sitting on the floor of the diagram, which is a lie.
        bins = reliability_diagram([0.05] * 5, [0] * 5)
        assert bins[5].count == 0
        assert bins[5].mean_predicted is None
        assert bins[5].observed_rate is None

    def test_weighted_rates(self) -> None:
        bins = reliability_diagram([0.95, 0.95], [1, 0], weights=[3.0, 1.0])
        assert bins[9].count == 2
        assert bins[9].weight == pytest.approx(4.0)
        assert bins[9].observed_rate == pytest.approx(0.75)

    def test_empty_input_gives_empty_bins(self) -> None:
        bins = reliability_diagram([], [])
        assert len(bins) == 10
        assert all(b.count == 0 for b in bins)


class TestZeroWeightPaths:
    def test_ece_with_zero_total_weight(self) -> None:
        est = expected_calibration_error([0.5, 0.9], [1, 0], weights=[0.0, 0.0])
        assert est.value is None
        assert est.reason is ReasonCode.ZERO_WEIGHT
