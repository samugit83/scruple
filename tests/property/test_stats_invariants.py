"""Property-based invariants for the statistics layer (§14.5)."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from scruple.stats.agreement import cohens_kappa, cohens_kappa_from_confusion
from scruple.stats.calibration import expected_calibration_error
from scruple.stats.thresholds import ThresholdPair, default_grid
from scruple.stats.types import Confusion

labels = st.lists(st.sampled_from([0, 1]), min_size=1, max_size=200)
probabilities = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=200
)
cells = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)


@given(a=cells, b=cells, c=cells, d=cells)
def test_kappa_lies_within_minus_one_and_one(a: float, b: float, c: float, d: float) -> None:
    est = cohens_kappa_from_confusion(Confusion(a=a, b=b, c=c, d=d))
    if est.value is not None:
        assert -1.0 <= est.value <= 1.0 + 1e-9


@given(a=cells, b=cells, c=cells, d=cells)
def test_kappa_is_invariant_under_a_consistent_label_permutation(
    a: float, b: float, c: float, d: float
) -> None:
    """Relabelling yes<->no for *both* coders must not change agreement."""
    original = cohens_kappa_from_confusion(Confusion(a=a, b=b, c=c, d=d))
    # swapping 0<->1 for both coders maps a<->d and b<->c
    swapped = cohens_kappa_from_confusion(Confusion(a=d, b=c, c=b, d=a))
    if original.value is None:
        assert swapped.value is None
    else:
        assert swapped.value is not None
        assert abs(original.value - swapped.value) < 1e-9


@given(gold=labels)
def test_a_coder_identical_to_gold_scores_one(gold: list[int]) -> None:
    assume(len(set(gold)) == 2)
    est = cohens_kappa(gold, gold)
    assert est.value is not None
    assert abs(est.value - 1.0) < 1e-9


@given(half=st.lists(st.sampled_from([0, 1]), min_size=1, max_size=100))
def test_a_coder_inverted_from_gold_scores_minus_one_at_balance(half: list[int]) -> None:
    # Balance is built rather than filtered for: `assume(sum == len/2)` rejects
    # almost every draw and makes the test flaky under hypothesis' health checks.
    gold = half + [1 - x for x in half]
    inverted = [1 - g for g in gold]
    est = cohens_kappa(gold, inverted)
    assert est.value is not None
    assert abs(est.value + 1.0) < 1e-9


@given(probs=probabilities, data=st.data())
def test_ece_lies_within_zero_and_one(probs: list[float], data: st.DataObject) -> None:
    y = data.draw(st.lists(st.sampled_from([0, 1]), min_size=len(probs), max_size=len(probs)))
    est = expected_calibration_error(probs, y)
    assert est.value is not None
    assert 0.0 <= est.value <= 1.0 + 1e-9


@given(probs=probabilities)
@settings(max_examples=50)
def test_coverage_is_monotone_non_increasing_as_the_band_widens(probs: list[float]) -> None:
    p = np.asarray(probs)
    grid = default_grid()  # most conservative (widest band) first
    coverages = [float(pair.decide_all(p)[1].mean()) for pair in grid]
    assert coverages == sorted(coverages)


@given(probs=probabilities)
@settings(max_examples=50)
def test_a_wider_band_accepts_a_subset_of_what_a_narrower_band_accepts(
    probs: list[float],
) -> None:
    p = np.asarray(probs)
    wide = ThresholdPair(t_lo=0.05, t_hi=0.95)
    narrow = ThresholdPair(t_lo=0.30, t_hi=0.70)
    _, accepted_wide = wide.decide_all(p)
    _, accepted_narrow = narrow.decide_all(p)
    assert np.all(accepted_wide <= accepted_narrow)


@given(
    chunks=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=50
    )
)
def test_max_aggregation_is_at_least_every_chunk_score(chunks: list[float]) -> None:
    """§14.5 and §10.3: the aggregation rule must dominate its inputs, which is
    exactly why `max` over many passages is biased upward and calibration has to
    be fitted on the aggregate."""
    from scruple.corpus.chunking import aggregate

    aggregated = aggregate(chunks, rule="max")
    assert aggregated is not None
    assert all(aggregated >= c - 1e-12 for c in chunks)
