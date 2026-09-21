"""Split-integrity tests (plan §8.4, §14.4).

The threshold fitter must never be able to read test data. These tests are the
regression guard for that property; §14.13 blocks merges when they fail.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from scruple.stats.sealed import ALLOWED_PURPOSES, Sealed, SealedDataError


@pytest.fixture
def sealed() -> Sealed[dict[str, list[int]]]:
    return Sealed({"gold": [1, 0, 1], "probs": [0.9, 0.1, 0.8]}, label="test")


class TestSealedRefusesToBeRead:
    def test_attribute_access_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError, match="test"):
            sealed.gold  # type: ignore[attr-defined]  # noqa: B018  (the read is the test)

    def test_indexing_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            sealed["gold"]

    def test_len_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            len(sealed)

    def test_iteration_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            list(sealed)

    def test_containment_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            "gold" in sealed  # noqa: B015

    def test_truthiness_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        # `if test_data:` is the easiest accidental peek to write.
        with pytest.raises(SealedDataError):
            bool(sealed)

    def test_calling_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            sealed()  # type: ignore[operator]

    def test_pickling_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        # Serialising sealed data into a run artefact would leak it sideways.
        with pytest.raises(SealedDataError):
            pickle.dumps(sealed)

    def test_deepcopy_raises(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        with pytest.raises(SealedDataError):
            copy.deepcopy(sealed)

    def test_repr_reveals_nothing(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        text = repr(sealed)
        assert "Sealed" in text
        assert "test" in text
        assert "0.9" not in text
        assert "gold" not in text

    def test_label_is_readable_because_it_is_only_metadata(
        self, sealed: Sealed[dict[str, list[int]]]
    ) -> None:
        assert sealed.label == "test"


class TestUnsealing:
    def test_reporting_is_an_allowed_purpose(self, sealed: Sealed[dict[str, list[int]]]) -> None:
        payload = sealed.unseal(purpose="reporting")
        assert payload["gold"] == [1, 0, 1]

    def test_final_evaluation_is_an_allowed_purpose(
        self, sealed: Sealed[dict[str, list[int]]]
    ) -> None:
        assert sealed.unseal(purpose="final_evaluation") is not None

    @pytest.mark.parametrize("purpose", ["fitting", "calibration", "threshold_selection", ""])
    def test_fitting_purposes_are_refused(
        self, sealed: Sealed[dict[str, list[int]]], purpose: str
    ) -> None:
        # §8.4: fitting on test data is the failure this whole class prevents.
        with pytest.raises(SealedDataError, match="purpose"):
            sealed.unseal(purpose=purpose)

    def test_allowed_purposes_are_exactly_these_two(self) -> None:
        assert set(ALLOWED_PURPOSES) == {"reporting", "final_evaluation"}

    def test_every_unseal_is_logged_for_the_audit_trail(
        self, sealed: Sealed[dict[str, list[int]]]
    ) -> None:
        assert sealed.unseal_log == ()
        sealed.unseal(purpose="reporting")
        sealed.unseal(purpose="final_evaluation")
        assert sealed.unseal_log == ("reporting", "final_evaluation")

    def test_a_refused_unseal_is_not_logged_as_a_read(
        self, sealed: Sealed[dict[str, list[int]]]
    ) -> None:
        with pytest.raises(SealedDataError):
            sealed.unseal(purpose="fitting")
        assert sealed.unseal_log == ()


class TestProtocolProbes:
    def test_dunder_probes_fail_as_missing_attributes_not_as_sealing_errors(self) -> None:
        """`hasattr(obj, "__array__")` and friends must not look like a leak attempt.

        numpy, copy and pretty-printers all probe dunders speculatively. Raising
        SealedDataError there would turn an unrelated library's feature detection
        into a confusing failure, so those probes get a plain AttributeError.
        """
        sealed: Sealed[list[int]] = Sealed([1, 2, 3], label="test")
        assert hasattr(sealed, "__deepcopy_fallback__") is False
        assert hasattr(sealed, "__fspath__") is False
        # ...while ordinary attribute names still raise the sealing error.
        with pytest.raises(SealedDataError):
            sealed.values  # type: ignore[attr-defined]  # noqa: B018

    def test_numpy_conversion_refuses_rather_than_wrapping_the_object(self) -> None:
        """numpy falls back to a zero-dimensional object array when `__array__`
        raises AttributeError, so a quiet failure here would mean
        `np.asarray(sealed)` succeeding and the fitter running on nonsense."""
        import numpy as np

        sealed: Sealed[list[int]] = Sealed([1, 2, 3], label="test")
        with pytest.raises(SealedDataError, match="conversion"):
            np.asarray(sealed)

        # Built-in conversions look their dunder up on the type rather than the
        # instance, so __getattr__ is never consulted. They already fail hard,
        # which is all that is required -- nothing is silently wrapped.
        with pytest.raises(TypeError):
            float(sealed)  # type: ignore[arg-type]
