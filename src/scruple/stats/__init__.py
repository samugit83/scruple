"""Statistics for scruple.

This package is the product, so it is deliberately self-contained: it depends
only on numpy and scipy, and it MUST NOT import from ``scruple.backends`` or
any other part of the package (plan §6, design rule 1). That keeps every claim
in the validation report testable in isolation against synthetic data with
analytically known answers.
"""

from .agreement import (
    cohens_kappa,
    cohens_kappa_from_confusion,
    confusion,
    krippendorff_alpha,
    krippendorff_alpha_from_confusion,
    observed_agreement,
)
from .bootstrap import MIN_RESAMPLES, stratified_bootstrap_ci
from .calibration import (
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)
from .sealed import ALLOWED_PURPOSES, Sealed, SealedDataError
from .selective import (
    LabourPoint,
    SelectivePoint,
    final_reliability,
    labour_curve,
    random_baseline_curve,
    selective_curve,
)
from .thresholds import (
    COMFORTABLE_KAPPA,
    DEFAULT_T_HI,
    MIN_POSITIVE_INSTANCES,
    PUBLICATION_KAPPA,
    Candidate,
    ClassRisk,
    ThresholdPair,
    ThresholdSelection,
    class_risk,
    default_grid,
    evaluate_candidate,
    fit_thresholds,
    grade,
    order_by_conservatism,
    realised_class_risk,
)
from .types import Confusion, Estimate, ReasonCode, Verdict
from .weights import (
    effective_sample_size,
    normalise_weights,
    weighted_mean,
    weights_from_inclusion,
)

__all__ = [
    "ALLOWED_PURPOSES",
    "COMFORTABLE_KAPPA",
    "DEFAULT_T_HI",
    "MIN_POSITIVE_INSTANCES",
    "MIN_RESAMPLES",
    "PUBLICATION_KAPPA",
    "Candidate",
    "ClassRisk",
    "Confusion",
    "Estimate",
    "LabourPoint",
    "ReasonCode",
    "ReliabilityBin",
    "Sealed",
    "SealedDataError",
    "SelectivePoint",
    "ThresholdPair",
    "ThresholdSelection",
    "Verdict",
    "brier_score",
    "class_risk",
    "cohens_kappa",
    "cohens_kappa_from_confusion",
    "confusion",
    "default_grid",
    "effective_sample_size",
    "evaluate_candidate",
    "expected_calibration_error",
    "final_reliability",
    "fit_thresholds",
    "grade",
    "krippendorff_alpha",
    "krippendorff_alpha_from_confusion",
    "labour_curve",
    "normalise_weights",
    "observed_agreement",
    "order_by_conservatism",
    "random_baseline_curve",
    "realised_class_risk",
    "reliability_diagram",
    "selective_curve",
    "stratified_bootstrap_ci",
    "weighted_mean",
    "weights_from_inclusion",
]
