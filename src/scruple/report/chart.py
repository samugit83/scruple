"""The hero chart (§8.8).

Reliability of the finished dataset against the human labour needed to reach
it, with a random-selection baseline. The vertical gap between the two curves
**is the product**, expressed in the only currency the user cares about: hours
saved at a fixed quality bar.

This chart is the Phase 0 deliverable, the README hero image, and the core
figure of the paper. If the two curves converge on real data, §3 says stop
building and publish the negative result.

matplotlib is an optional dependency (`scruple[chart]`), so this module is
imported lazily and says what to install if it is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import ValidationError
from ..stats import PUBLICATION_KAPPA
from ..stats.selective import LabourPoint

# Deliberately plain: this audience reads methods papers, and a chart that
# looks like marketing is read as marketing (§15).
CALIBRATED_COLOUR = "#1f4e79"
BASELINE_COLOUR = "#9a9a9a"
BAR_COLOUR = "#c0392b"


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")  # never needs a display; CI has none
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValidationError(
            "drawing the chart needs matplotlib",
            hint="Install scruple[chart], or pass --no-chart.",
        ) from exc
    return plt


def _series(points: Sequence[LabourPoint]) -> tuple[list[float], list[float]]:
    usable = [p for p in points if p.kappa.value is not None]
    return (
        [p.labour * 100 for p in usable],
        [p.kappa.value for p in usable],  # type: ignore[misc]
    )


def draw_labour_chart(
    *,
    calibrated: Sequence[LabourPoint],
    baseline: Sequence[LabourPoint],
    path: Path,
    title: str = "Reliability of the finished dataset against human labour",
    operating_point: LabourPoint | None = None,
    dpi: int = 150,
) -> Path:
    """Render the §8.8 chart to `path`."""
    plt = _require_matplotlib()

    calibrated_x, calibrated_y = _series(calibrated)
    baseline_x, baseline_y = _series(baseline)
    if not calibrated_x:
        raise ValidationError(
            "no usable points to plot",
            hint="Every point had an undefined kappa; check the gold sample size.",
        )

    figure, axes = plt.subplots(figsize=(7.5, 4.8))

    axes.plot(
        calibrated_x,
        calibrated_y,
        color=CALIBRATED_COLOUR,
        linewidth=2.2,
        label="calibrated abstention",
        zorder=3,
    )
    if baseline_x:
        axes.plot(
            baseline_x,
            baseline_y,
            color=BASELINE_COLOUR,
            linewidth=1.8,
            linestyle="--",
            label="random selection for review",
            zorder=2,
        )

    # Shaded bootstrap interval where one is available.
    lower = [p.kappa.ci[0] for p in calibrated if p.kappa.ci is not None]
    upper = [p.kappa.ci[1] for p in calibrated if p.kappa.ci is not None]
    if len(lower) == len(calibrated_x) and lower:
        axes.fill_between(
            calibrated_x, lower, upper, color=CALIBRATED_COLOUR, alpha=0.15, linewidth=0, zorder=1
        )

    # Frame the y-axis on the data rather than on the origin, so a backend that
    # is already reliable does not produce a chart that is nine-tenths empty.
    # The bar is drawn inside that frame when it falls there, and said in words
    # when the whole curve clears it.
    values = calibrated_y + baseline_y
    low, high = min(values), max(values)
    margin = max(0.02, (high - low) * 0.15)
    bottom, top = low - margin, min(1.02, high + margin)

    axes.axhline(PUBLICATION_KAPPA, color=BAR_COLOUR, linewidth=1.1, linestyle=":", zorder=1)
    if bottom <= PUBLICATION_KAPPA <= top:
        label = f"publication bar, kappa = {PUBLICATION_KAPPA:.2f}"
        label_y = PUBLICATION_KAPPA
    else:
        # Keep the bar just in view, so a reader can see how far above it the
        # whole curve sits rather than having to infer it from the axis.
        bottom = min(bottom, PUBLICATION_KAPPA - 0.01)
        label = f"publication bar, kappa = {PUBLICATION_KAPPA:.2f} — the whole curve clears it"
        label_y = PUBLICATION_KAPPA
    axes.set_ylim(bottom, top)
    axes.annotate(
        label,
        xy=(min(calibrated_x), label_y),
        xytext=(4, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color=BAR_COLOUR,
    )

    if operating_point is not None and operating_point.kappa.value is not None:
        axes.plot(
            [operating_point.labour * 100],
            [operating_point.kappa.value],
            marker="o",
            markersize=7,
            color=CALIBRATED_COLOUR,
            zorder=4,
        )
        axes.annotate(
            f"chosen: {operating_point.labour * 100:.0f}% coded by hand, "
            f"kappa = {operating_point.kappa.value:.2f}",
            xy=(operating_point.labour * 100, operating_point.kappa.value),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8,
        )

    axes.set_xlabel("share of the corpus a person must code (%)")
    axes.set_ylabel("Cohen's kappa of the finished dataset")
    axes.set_title(title, fontsize=11)
    axes.set_xlim(0, max(calibrated_x + baseline_x) * 1.02 or 100)
    axes.grid(True, alpha=0.2, linewidth=0.6)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    # "best" keeps the legend off the curves; the bar annotation is pinned left
    # so the two cannot collide whichever corner matplotlib chooses.
    axes.legend(frameon=False, fontsize=9, loc="best")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def most_informative(checks: Sequence[Any]) -> Any | None:
    """The code whose chart best shows what abstention buys.

    Picks the largest area between the calibrated curve and the random
    baseline. A code the backend already codes near-perfectly has almost no gap
    to show, and charting it would undersell the tool by accident.
    """
    best = None
    best_gap = -1.0
    for check in checks:
        if not check.labour or not check.baseline:
            continue
        gap = 0.0
        for calibrated, random_point in zip(check.labour, check.baseline, strict=False):
            if calibrated.kappa.value is not None and random_point.kappa.value is not None:
                gap += max(0.0, calibrated.kappa.value - random_point.kappa.value)
        if gap > best_gap:
            best, best_gap = check, gap
    return best


def draw_reliability_diagram(
    *, bins: Sequence[Any], path: Path, title: str = "Calibration", dpi: int = 150
) -> Path:
    """A reliability diagram: predicted probability against observed frequency."""
    plt = _require_matplotlib()
    populated = [b for b in bins if b.mean_predicted is not None]
    if not populated:
        raise ValidationError("no populated bins to plot")

    figure, axes = plt.subplots(figsize=(4.6, 4.4))
    axes.plot([0, 1], [0, 1], color=BASELINE_COLOUR, linestyle="--", linewidth=1.2)
    axes.plot(
        [b.mean_predicted for b in populated],
        [b.observed_rate for b in populated],
        marker="o",
        color=CALIBRATED_COLOUR,
        linewidth=1.8,
    )
    axes.set_xlabel("stated probability")
    axes.set_ylabel("observed frequency")
    axes.set_title(title, fontsize=11)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.grid(True, alpha=0.2, linewidth=0.6)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
