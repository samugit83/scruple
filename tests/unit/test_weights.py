"""Unit tests for inverse-probability weighting helpers (plan §8.5)."""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats.weights import (
    effective_sample_size,
    normalise_weights,
    weighted_mean,
    weights_from_inclusion,
)


class TestNormaliseWeights:
    def test_none_becomes_unit_weights(self) -> None:
        w = normalise_weights(None, 4)
        assert np.array_equal(w, np.ones(4))

    def test_passthrough_preserves_values(self) -> None:
        w = normalise_weights([1.0, 2.0, 3.0], 3)
        assert np.array_equal(w, np.array([1.0, 2.0, 3.0]))

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="length"):
            normalise_weights([1.0, 2.0], 3)

    def test_negative_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            normalise_weights([1.0, -0.5], 2)

    def test_non_finite_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            normalise_weights([1.0, float("inf")], 2)

    def test_zero_weights_are_allowed(self) -> None:
        w = normalise_weights([0.0, 1.0], 2)
        assert np.array_equal(w, np.array([0.0, 1.0]))

    def test_empty_is_allowed(self) -> None:
        assert normalise_weights(None, 0).shape == (0,)


class TestWeightsFromInclusion:
    def test_inverse_of_the_inclusion_probability(self) -> None:
        w = weights_from_inclusion([0.5, 0.25, 1.0])
        assert np.allclose(w, [2.0, 4.0, 1.0])

    def test_uniform_inclusion_gives_uniform_weights(self) -> None:
        w = weights_from_inclusion([0.3, 0.3, 0.3])
        assert np.allclose(w, w[0])

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_inclusion_probability_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="inclusion probability"):
            weights_from_inclusion([0.5, bad])


class TestEffectiveSampleSize:
    def test_uniform_weights_give_the_raw_count(self) -> None:
        assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)

    def test_uniform_non_unit_weights_also_give_the_raw_count(self) -> None:
        # Kish's n_eff is scale-invariant.
        assert effective_sample_size(np.full(10, 3.7)) == pytest.approx(10.0)

    def test_unequal_weights_shrink_the_effective_count(self) -> None:
        w = np.array([1.0, 1.0, 1.0, 9.0])
        n_eff = effective_sample_size(w)
        assert n_eff < 4.0
        assert n_eff == pytest.approx(144.0 / 84.0)

    def test_all_zero_weights_give_zero(self) -> None:
        assert effective_sample_size(np.zeros(5)) == 0.0

    def test_empty_gives_zero(self) -> None:
        assert effective_sample_size(np.array([])) == 0.0


class TestWeightedMean:
    def test_uniform_weights_equal_the_plain_mean_exactly(self) -> None:
        # §14.3: with uniform weights, results equal the unweighted versions exactly.
        values = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        assert weighted_mean(values, np.ones(5)) == values.mean()

    def test_weights_shift_the_mean(self) -> None:
        values = np.array([0.0, 1.0])
        assert weighted_mean(values, np.array([3.0, 1.0])) == pytest.approx(0.25)

    def test_zero_total_weight_returns_none(self) -> None:
        assert weighted_mean(np.array([1.0, 0.0]), np.zeros(2)) is None

    def test_empty_returns_none(self) -> None:
        assert weighted_mean(np.array([]), np.array([])) is None

    def test_ipw_recovers_the_population_mean_under_biased_sampling(self) -> None:
        # §14.3: with known non-uniform inclusion probabilities, IPW recovers the
        # true population value. Positives are oversampled 5x, exactly as the
        # enriched stratum of §8.5 does.
        rng = np.random.default_rng(20260921)
        population = (rng.random(200_000) < 0.05).astype(float)
        truth = population.mean()

        pi = np.where(population == 1.0, 0.50, 0.10)
        sampled = rng.random(population.size) < pi
        values = population[sampled]
        w = weights_from_inclusion(pi[sampled])

        estimate = weighted_mean(values, w)
        assert estimate is not None
        assert estimate == pytest.approx(truth, abs=0.003)
        # The unweighted mean of the enriched sample is badly biased upward,
        # which is why the weighting is mandatory rather than optional.
        assert values.mean() > truth * 2
