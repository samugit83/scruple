"""Validation report and charts (§0.5, §8.8)."""

from .chart import draw_labour_chart, draw_reliability_diagram, most_informative
from .methods import methods_paragraph
from .validation import ReportInputs, build_report, normalise_for_comparison

__all__ = [
    "ReportInputs",
    "build_report",
    "draw_labour_chart",
    "draw_reliability_diagram",
    "methods_paragraph",
    "most_informative",
    "normalise_for_comparison",
]
