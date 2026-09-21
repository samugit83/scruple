"""The fitting path must not read the test split (plan §8.4, §14.4).

§14.13 blocks merges when these fail. The unit tests in tests/unit/test_sealed.py
check that a sealed payload refuses to be read; these check that the real
`check` pipeline never asks.
"""

from __future__ import annotations

import numpy as np
import pytest

from scruple.codebook import parse_codebook
from scruple.config import parse_config
from scruple.corpus import Split, assign_splits
from scruple.engine.check import run_check
from scruple.gold import UNIFORM, GoldStore
from scruple.stats import Sealed, SealedDataError, fit_thresholds

CODEBOOK = parse_codebook(
    "version: 1\ncodes:\n  access_barrier:\n    definition: A practical obstacle.\n"
)


def build_gold(tmp_path, n=800, prevalence=0.35, accuracy=0.97, seed=7):  # type: ignore[no-untyped-def]
    """A gold sample and matching probabilities, across both fitting splits."""
    rng = np.random.default_rng(seed)
    ids = [f"r{i:04d}" for i in range(n)]
    splits = assign_splits(ids, corpus_hash="h", seed=42)

    store = GoldStore(tmp_path / "gold.jsonl")
    scores: dict[str, dict[str, float | None]] = {}
    records = []
    for item_id in ids:
        split = splits.of(item_id)
        if split is Split.DEV:
            continue
        label = int(rng.random() < prevalence)
        correct = rng.random() < accuracy
        probability = (0.97 if label else 0.03) if correct else (0.03 if label else 0.97)
        scores[item_id] = {"access_barrier": probability}
        records.append(
            store.record(
                item_id=item_id,
                code_id="access_barrier",
                label=label,
                coder="coder_1",
                split=split.value,
                stratum=UNIFORM,
                inclusion_probability=1.0,
                sample_seed=42,
            )
        )
    store.append(records)
    return store, scores, splits


class TestTheFittingPathNeverUnsealsTestData:
    def test_check_does_not_unseal_the_test_split(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """§14.4: the threshold fitter receives a sealed object, and leaves it sealed.

        The seal records every unsealing. After a full check the log must still
        be empty: nothing in the fitting path looked.
        """
        store, scores, splits = build_gold(tmp_path)
        sealed: Sealed[tuple[str, ...]] = splits.sealed(Split.TEST)

        report = run_check(
            codebook=CODEBOOK,
            config=parse_config(""),
            splits=splits,
            gold=store,
            scores=scores,
            backend_name="stub",
            model_version="stub-1",
            corpus_hash="h",
            bootstrap_resamples=50,
            sealed_test=sealed,
        )
        assert report.codes, "the check should have run"
        assert sealed.unseal_log == (), "something in the fitting path read the test split"

    def test_a_sealed_payload_handed_to_the_fitter_raises_if_touched(self) -> None:
        sealed: Sealed[np.ndarray] = Sealed(np.array([1.0, 0.0]), label="test")
        with pytest.raises(SealedDataError):
            fit_thresholds(np.array([0.9, 0.1]), sealed)  # type: ignore[arg-type]

    def test_thresholds_are_fitted_on_calibration_alone(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Corrupting only the test split must not move the fitted band.

        If a threshold changes when test labels change, the fitter is reading
        them -- which is the failure this whole mechanism exists to prevent.
        """
        store, scores, splits = build_gold(tmp_path)
        config = parse_config("")

        def fit(gold_store: GoldStore) -> tuple[float, float] | None:
            report = run_check(
                codebook=CODEBOOK,
                config=config,
                splits=splits,
                gold=gold_store,
                scores=scores,
                backend_name="stub",
                model_version="stub-1",
                corpus_hash="h",
                bootstrap_resamples=0,
            )
            calibration = report.codes[0].calibration
            return (
                None
                if calibration.t_lo is None or calibration.t_hi is None
                else (calibration.t_lo, calibration.t_hi)
            )

        before = fit(store)

        # Flip every test label. Calibration is untouched.
        flipped = GoldStore(tmp_path / "gold_flipped.jsonl")
        flipped.append(
            [
                store.record(
                    item_id=r.item_id,
                    code_id=r.code_id,
                    label=1 - r.label if r.split == Split.TEST.value else r.label,
                    coder=r.coder,
                    split=r.split,
                    stratum=r.stratum,
                    inclusion_probability=r.inclusion_probability,
                    sample_seed=r.sample_seed,
                )
                for r in store.all_records()
            ]
        )
        assert fit(flipped) == before

    def test_reported_kappa_does_move_when_test_labels_change(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The complement of the test above: results *are* computed on test.

        Without this, a fitter that ignored the test split entirely would pass
        the previous assertion for the wrong reason.
        """
        store, scores, splits = build_gold(tmp_path)
        config = parse_config("")

        def kappa(gold_store: GoldStore) -> float | None:
            report = run_check(
                codebook=CODEBOOK,
                config=config,
                splits=splits,
                gold=gold_store,
                scores=scores,
                backend_name="stub",
                model_version="stub-1",
                corpus_hash="h",
                bootstrap_resamples=0,
            )
            estimate = report.codes[0].kappa
            return estimate.value if estimate else None

        before = kappa(store)
        flipped = GoldStore(tmp_path / "gold_flipped.jsonl")
        flipped.append(
            [
                store.record(
                    item_id=r.item_id,
                    code_id=r.code_id,
                    label=1 - r.label if r.split == Split.TEST.value else r.label,
                    coder=r.coder,
                    split=r.split,
                    stratum=r.stratum,
                    inclusion_probability=r.inclusion_probability,
                    sample_seed=r.sample_seed,
                )
                for r in store.all_records()
            ]
        )
        assert before is not None
        assert kappa(flipped) != before
