"""Unit tests for the hero chart (plan §8.8).

The chart is the Phase 0 deliverable, the README image and the core figure of
the paper, so its framing and code-selection logic are worth pinning: a chart
that undersells the result by accident is as much a defect as a wrong number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scruple.errors import ValidationError
from scruple.report.chart import draw_labour_chart, draw_reliability_diagram, most_informative
from scruple.stats import Estimate, ReasonCode, ReliabilityBin
from scruple.stats.selective import LabourPoint


def points(values: list[float], ci: bool = False) -> list[LabourPoint]:
    out = []
    for index, value in enumerate(values):
        labour = index / max(1, len(values) - 1)
        kappa = Estimate(value=value, n=100)
        if ci:
            kappa = kappa.with_ci((max(-1.0, value - 0.03), min(1.0, value + 0.03)))
        out.append(LabourPoint(labour=labour, coverage=1 - labour, kappa=kappa, pair=None))
    return out


class TestLabourChart:
    def test_writes_a_file(self, tmp_path: Path) -> None:
        path = draw_labour_chart(
            calibrated=points([0.5, 0.7, 0.85, 1.0]),
            baseline=points([0.5, 0.6, 0.75, 1.0]),
            path=tmp_path / "chart.png",
        )
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        path = draw_labour_chart(
            calibrated=points([0.5, 0.8, 1.0]),
            baseline=points([0.5, 0.7, 1.0]),
            path=tmp_path / "nested" / "deeper" / "chart.png",
        )
        assert path.exists()

    def test_a_confidence_band_is_drawn_when_every_point_has_one(self, tmp_path: Path) -> None:
        path = draw_labour_chart(
            calibrated=points([0.5, 0.8, 1.0], ci=True),
            baseline=points([0.5, 0.7, 1.0]),
            path=tmp_path / "band.png",
        )
        assert path.exists()

    def test_an_operating_point_is_annotated(self, tmp_path: Path) -> None:
        calibrated = points([0.5, 0.8, 1.0])
        path = draw_labour_chart(
            calibrated=calibrated,
            baseline=points([0.5, 0.7, 1.0]),
            path=tmp_path / "op.png",
            operating_point=calibrated[1],
        )
        assert path.exists()

    def test_an_operating_point_without_a_kappa_is_skipped(self, tmp_path: Path) -> None:
        undefined = LabourPoint(
            labour=0.5,
            coverage=0.5,
            kappa=Estimate(value=None, n=0, reason=ReasonCode.EMPTY_INPUT),
            pair=None,
        )
        path = draw_labour_chart(
            calibrated=points([0.5, 0.8, 1.0]),
            baseline=points([0.5, 0.7, 1.0]),
            path=tmp_path / "skip.png",
            operating_point=undefined,
        )
        assert path.exists()

    def test_a_curve_entirely_above_the_bar_still_shows_it(self, tmp_path: Path) -> None:
        """A backend that is already reliable must not produce a chart that is
        nine-tenths empty space below an unreachable reference line."""
        path = draw_labour_chart(
            calibrated=points([0.95, 0.98, 1.0]),
            baseline=points([0.95, 0.96, 1.0]),
            path=tmp_path / "high.png",
        )
        assert path.exists()

    def test_a_curve_spanning_the_bar_is_fine(self, tmp_path: Path) -> None:
        path = draw_labour_chart(
            calibrated=points([0.40, 0.70, 0.95]),
            baseline=points([0.40, 0.55, 0.95]),
            path=tmp_path / "span.png",
        )
        assert path.exists()

    def test_an_empty_baseline_is_tolerated(self, tmp_path: Path) -> None:
        # `compare` can produce a calibrated curve with no baseline computed.
        path = draw_labour_chart(
            calibrated=points([0.5, 0.8, 1.0]), baseline=[], path=tmp_path / "nobase.png"
        )
        assert path.exists()

    def test_nothing_plottable_is_refused_clearly(self, tmp_path: Path) -> None:
        undefined = [
            LabourPoint(
                labour=0.0,
                coverage=1.0,
                kappa=Estimate(value=None, n=0, reason=ReasonCode.EMPTY_INPUT),
                pair=None,
            )
        ]
        with pytest.raises(ValidationError, match="no usable points") as caught:
            draw_labour_chart(calibrated=undefined, baseline=[], path=tmp_path / "x.png")
        assert "gold sample size" in (caught.value.hint or "")


class TestMostInformative:
    class FakeCheck:
        def __init__(self, code_id: str, calibrated: list[float], baseline: list[float]) -> None:
            self.code_id = code_id
            self.labour = points(calibrated)
            self.baseline = points(baseline)

    def test_picks_the_code_with_the_largest_gap(self) -> None:
        """Charting an already-near-perfect code undersells the tool by accident."""
        easy = self.FakeCheck("easy", [0.97, 0.99, 1.0], [0.97, 0.98, 1.0])
        informative = self.FakeCheck("informative", [0.60, 0.92, 1.0], [0.60, 0.72, 1.0])
        assert most_informative([easy, informative]).code_id == "informative"

    def test_codes_without_curves_are_skipped(self) -> None:
        class NoCurves:
            def __init__(self) -> None:
                self.code_id = "none"
                self.labour: list[LabourPoint] = []
                self.baseline: list[LabourPoint] = []

        good = self.FakeCheck("good", [0.6, 0.9, 1.0], [0.6, 0.7, 1.0])
        assert most_informative([NoCurves(), good]).code_id == "good"

    def test_no_candidates_gives_none(self) -> None:
        assert most_informative([]) is None


class TestReliabilityDiagram:
    def bins(self) -> list[ReliabilityBin]:
        return [
            ReliabilityBin(
                index=i,
                lo=i / 10,
                hi=(i + 1) / 10,
                count=10,
                weight=10.0,
                mean_predicted=(i + 0.5) / 10,
                observed_rate=(i + 0.5) / 10,
            )
            for i in range(10)
        ]

    def test_writes_a_file(self, tmp_path: Path) -> None:
        path = draw_reliability_diagram(bins=self.bins(), path=tmp_path / "rel.png")
        assert path.exists()

    def test_empty_bins_are_skipped(self, tmp_path: Path) -> None:
        mixed = self.bins()
        mixed[3] = ReliabilityBin(
            index=3, lo=0.3, hi=0.4, count=0, weight=0.0, mean_predicted=None, observed_rate=None
        )
        assert draw_reliability_diagram(bins=mixed, path=tmp_path / "rel2.png").exists()

    def test_no_populated_bins_is_refused(self, tmp_path: Path) -> None:
        empty = [
            ReliabilityBin(
                index=0,
                lo=0.0,
                hi=0.1,
                count=0,
                weight=0.0,
                mean_predicted=None,
                observed_rate=None,
            )
        ]
        with pytest.raises(ValidationError, match="no populated bins"):
            draw_reliability_diagram(bins=empty, path=tmp_path / "rel3.png")
