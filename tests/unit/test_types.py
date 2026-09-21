"""Unit tests for the shared result types of the statistics layer."""

from __future__ import annotations

import pytest

from scruple.stats.types import Confusion, Estimate, ReasonCode, Verdict


class TestEstimate:
    def test_defined_estimate_exposes_its_value(self) -> None:
        e = Estimate(value=0.84, n=300)
        assert e.defined is True
        assert e.value == 0.84
        assert e.reason is None
        assert e.ci is None

    def test_undefined_estimate_must_carry_a_reason(self) -> None:
        e = Estimate(value=None, n=0, reason=ReasonCode.EMPTY_INPUT)
        assert e.defined is False
        assert e.reason is ReasonCode.EMPTY_INPUT

    def test_undefined_without_a_reason_is_rejected(self) -> None:
        # A silent None is exactly the failure mode §14.3 forbids.
        with pytest.raises(ValueError, match="reason"):
            Estimate(value=None, n=0)

    def test_defined_with_a_reason_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            Estimate(value=0.5, n=10, reason=ReasonCode.EMPTY_INPUT)

    def test_nan_is_never_accepted_as_a_value(self) -> None:
        # §14.3: never return NaN silently.
        with pytest.raises(ValueError, match="NaN"):
            Estimate(value=float("nan"), n=10)

    def test_negative_n_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n"):
            Estimate(value=0.5, n=-1)

    def test_ci_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            Estimate(value=0.5, n=10, ci=(0.6, 0.4))

    def test_ci_is_accepted_when_ordered(self) -> None:
        e = Estimate(value=0.5, n=10, ci=(0.4, 0.6))
        assert e.ci == (0.4, 0.6)

    def test_with_ci_returns_a_copy_carrying_the_interval(self) -> None:
        e = Estimate(value=0.5, n=10)
        w = e.with_ci((0.4, 0.6))
        assert w.ci == (0.4, 0.6)
        assert w.value == 0.5
        assert e.ci is None  # original untouched: the type is frozen

    def test_with_ci_none_clears_the_interval(self) -> None:
        e = Estimate(value=0.5, n=10, ci=(0.4, 0.6))
        assert e.with_ci(None).ci is None

    def test_is_frozen(self) -> None:
        e = Estimate(value=0.5, n=10)
        with pytest.raises((AttributeError, TypeError)):
            e.value = 0.9  # type: ignore[misc]


class TestConfusion:
    def test_counts_and_totals(self) -> None:
        cm = Confusion(a=170, b=30, c=40, d=760)
        assert cm.n == 1000
        assert cm.gold_positive == 200
        assert cm.gold_negative == 800
        assert cm.pred_positive == 210
        assert cm.pred_negative == 790
        assert cm.errors == 70
        assert cm.agreements == 930

    def test_empty_confusion(self) -> None:
        cm = Confusion(a=0, b=0, c=0, d=0)
        assert cm.n == 0
        assert cm.errors == 0

    def test_negative_cells_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            Confusion(a=-1, b=0, c=0, d=0)

    def test_accepts_fractional_cells_from_weighting(self) -> None:
        # Weighted counts are sums of weights and need not be integers (§8.5).
        cm = Confusion(a=1.5, b=0.5, c=2.25, d=3.75)
        assert cm.n == pytest.approx(8.0)


class TestVerdict:
    def test_verdicts_are_the_four_check_outcomes(self) -> None:
        # These strings appear in `scruple check` output and in --json, which
        # §13.1 makes a public versioned interface. Changing them is breaking.
        assert {v.value for v in Verdict} == {
            "ok",
            "weak",
            "NOT_AUTOMATABLE",
            "INSUFFICIENT_EVIDENCE",
        }

    def test_usable_verdicts_are_the_automatable_ones(self) -> None:
        assert Verdict.OK.usable is True
        assert Verdict.WEAK.usable is True
        assert Verdict.NOT_AUTOMATABLE.usable is False
        assert Verdict.INSUFFICIENT_EVIDENCE.usable is False
