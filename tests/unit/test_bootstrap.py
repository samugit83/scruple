"""Unit tests for the stratified bootstrap (plan §8.1, §14.3)."""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats.agreement import cohens_kappa
from scruple.stats.bootstrap import MIN_RESAMPLES, stratified_bootstrap_ci


def _sample(n: int, prevalence: float, agreement: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    gold = (rng.random(n) < prevalence).astype(int)
    flip = rng.random(n) > agreement
    pred = np.where(flip, 1 - gold, gold)
    return gold.tolist(), pred.tolist()


class TestStratifiedBootstrapCI:
    def test_default_resample_count_meets_the_documented_minimum(self) -> None:
        # §8.1 promises "at least 2000 resamples" in the report.
        assert MIN_RESAMPLES == 2000

    def test_interval_brackets_the_point_estimate(self) -> None:
        gold, pred = _sample(400, 0.3, 0.9, seed=1)
        point = cohens_kappa(gold, pred).value
        ci = stratified_bootstrap_ci(gold, pred, statistic=cohens_kappa, seed=1, n_resamples=500)
        assert ci is not None
        assert ci[0] <= point <= ci[1]

    def test_interval_narrows_as_the_sample_grows(self) -> None:
        small = stratified_bootstrap_ci(
            *_sample(100, 0.3, 0.9, seed=2), statistic=cohens_kappa, seed=2, n_resamples=500
        )
        large = stratified_bootstrap_ci(
            *_sample(4000, 0.3, 0.9, seed=2), statistic=cohens_kappa, seed=2, n_resamples=500
        )
        assert small is not None and large is not None
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_is_deterministic_given_a_seed(self) -> None:
        gold, pred = _sample(200, 0.3, 0.9, seed=3)
        a = stratified_bootstrap_ci(gold, pred, statistic=cohens_kappa, seed=7, n_resamples=300)
        b = stratified_bootstrap_ci(gold, pred, statistic=cohens_kappa, seed=7, n_resamples=300)
        assert a == b

    def test_stratification_preserves_the_gold_class_balance(self) -> None:
        # Unstratified resampling of a rare code routinely draws zero positives,
        # which makes kappa undefined and the interval meaningless (§8.5).
        gold = [1] * 5 + [0] * 195
        pred = [1] * 4 + [0] * 196
        ci = stratified_bootstrap_ci(gold, pred, statistic=cohens_kappa, seed=4, n_resamples=500)
        assert ci is not None

    def test_perfect_agreement_gives_a_degenerate_interval_at_one(self) -> None:
        gold = [1] * 50 + [0] * 50
        ci = stratified_bootstrap_ci(gold, gold, statistic=cohens_kappa, seed=5, n_resamples=300)
        assert ci == pytest.approx((1.0, 1.0))

    def test_empty_input_has_no_interval(self) -> None:
        assert stratified_bootstrap_ci([], [], statistic=cohens_kappa, seed=0) is None

    def test_all_degenerate_resamples_yield_no_interval(self) -> None:
        # Single-class gold and pred: every resample is degenerate, so there is
        # no interval to report and we must say so rather than invent one.
        gold = [0] * 40
        assert (
            stratified_bootstrap_ci(gold, gold, statistic=cohens_kappa, seed=6, n_resamples=200)
            is None
        )

    def test_weights_are_resampled_with_their_items(self) -> None:
        gold, pred = _sample(300, 0.3, 0.9, seed=8)
        w = [2.0 if g == 1 else 1.0 for g in gold]
        ci = stratified_bootstrap_ci(
            gold, pred, statistic=cohens_kappa, weights=w, seed=8, n_resamples=300
        )
        assert ci is not None
        assert ci[0] < ci[1]

    def test_confidence_level_is_validated(self) -> None:
        with pytest.raises(ValueError, match="level"):
            stratified_bootstrap_ci([1, 0], [1, 0], statistic=cohens_kappa, level=1.5)

    def test_resample_count_is_validated(self) -> None:
        with pytest.raises(ValueError, match="resamples"):
            stratified_bootstrap_ci([1, 0], [1, 0], statistic=cohens_kappa, n_resamples=0)

    def test_a_wider_level_gives_a_wider_interval(self) -> None:
        gold, pred = _sample(300, 0.3, 0.88, seed=9)
        narrow = stratified_bootstrap_ci(
            gold, pred, statistic=cohens_kappa, seed=9, n_resamples=500, level=0.80
        )
        wide = stratified_bootstrap_ci(
            gold, pred, statistic=cohens_kappa, seed=9, n_resamples=500, level=0.99
        )
        assert narrow is not None and wide is not None
        assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])
