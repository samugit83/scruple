"""The synthetic generator must produce what it claims (§14.11).

A generator that quietly drifts makes every test built on it meaningless, so
its own properties are asserted here against known truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from scruple.stats import cohens_kappa, expected_calibration_error, realised_class_risk
from scruple.stats.thresholds import ThresholdPair
from tests.fixtures import SyntheticCoder, gold_records, synthetic_corpus, synthetic_gold


class TestSyntheticCoder:
    @pytest.mark.parametrize("prevalence", [0.50, 0.20, 0.05, 0.02])
    def test_prevalence_is_as_specified(self, prevalence: float) -> None:
        data = synthetic_gold(40_000, coder=SyntheticCoder(prevalence=prevalence), seed=1)
        assert data["gold"].mean() == pytest.approx(prevalence, abs=0.01)

    @pytest.mark.parametrize("prevalence", [0.50, 0.20, 0.05])
    def test_class_conditional_error_is_as_specified(self, prevalence: float) -> None:
        """The property that matters: quality is independent of prevalence."""
        coder = SyntheticCoder(prevalence=prevalence, leak=0.03, false_positive_rate=0.02)
        data = synthetic_gold(60_000, coder=coder, seed=2)

        risk = realised_class_risk(data["probabilities"], data["gold"], ThresholdPair(0.45, 0.55))
        assert risk[1] == pytest.approx(0.03, abs=0.01)
        assert risk[0] == pytest.approx(0.02, abs=0.01)

    def test_the_emitted_probabilities_are_calibrated(self) -> None:
        coder = SyntheticCoder(prevalence=0.2, leak=0.02, false_positive_rate=0.02)
        data = synthetic_gold(60_000, coder=coder, seed=3)
        ece = expected_calibration_error(data["probabilities"], data["gold"])
        assert ece.value is not None
        assert ece.value < 0.05

    def test_a_worse_coder_produces_a_lower_kappa(self) -> None:
        good = synthetic_gold(20_000, coder=SyntheticCoder(0.2, 0.01, 0.01), seed=4)
        poor = synthetic_gold(20_000, coder=SyntheticCoder(0.2, 0.25, 0.20), seed=4)

        def kappa(data: dict[str, np.ndarray]) -> float:
            value = cohens_kappa(data["gold"], (data["probabilities"] >= 0.5).astype(float)).value
            assert value is not None
            return value

        assert kappa(good) > kappa(poor)

    def test_is_reproducible_from_the_seed(self) -> None:
        a = synthetic_gold(500, coder=SyntheticCoder(0.2), seed=5)
        b = synthetic_gold(500, coder=SyntheticCoder(0.2), seed=5)
        assert np.array_equal(a["gold"], b["gold"])
        assert np.array_equal(a["probabilities"], b["probabilities"])

    def test_second_coder_agreement_is_as_specified(self) -> None:
        data = synthetic_gold(20_000, coder=SyntheticCoder(0.3), seed=6, second_coder_agreement=0.9)
        agreement = float((data["gold"] == data["second_coder"]).mean())
        assert agreement == pytest.approx(0.9, abs=0.01)

    def test_no_second_coder_unless_asked(self) -> None:
        assert "second_coder" not in synthetic_gold(100, coder=SyntheticCoder(0.2), seed=7)


class TestSyntheticCorpus:
    def test_ids_and_texts_line_up(self) -> None:
        ids, texts = synthetic_corpus(50, seed=8)
        assert len(ids) == len(texts) == 50
        assert len(set(ids)) == 50

    def test_is_reproducible(self) -> None:
        assert synthetic_corpus(20, seed=9) == synthetic_corpus(20, seed=9)


class TestGoldRecords:
    def test_records_are_blind_by_construction(self) -> None:
        ids, _ = synthetic_corpus(10, seed=10)
        data = synthetic_gold(10, coder=SyntheticCoder(0.3), seed=10)
        records = gold_records(ids, data["gold"], code_id="abc_code", split="calibration")

        assert len(records) == 10
        assert all(r.model_visible is False for r in records)
        assert all(r.split == "calibration" for r in records)

    def test_weights_follow_the_inclusion_probability(self) -> None:
        ids, _ = synthetic_corpus(4, seed=11)
        data = synthetic_gold(4, coder=SyntheticCoder(0.3), seed=11)
        records = gold_records(
            ids, data["gold"], code_id="abc_code", split="test", inclusion_probability=0.25
        )
        assert all(r.weight == pytest.approx(4.0) for r in records)

    def test_length_mismatch_is_refused(self) -> None:
        # `zip(..., strict=True)` catches a caller that lost track of which
        # labels belong to which items.
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            gold_records(["a", "b"], np.array([1.0]), code_id="abc_code", split="test")
