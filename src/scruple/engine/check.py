"""`scruple check`: calibration, threshold fitting, and the per-code verdict.

This is the decision point (§12). It answers the only question the researcher
actually has: *which of my codes can this thing do, how much of the corpus will
it hand back to me, and how reliable is the result?*

Split discipline (§8.4) is structural here, not a matter of care:

* thresholds are fitted on the **calibration** split only;
* every reported number comes from the **test** split;
* the test split is handed to the fitting path sealed, so touching it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..codebook import Codebook
from ..config import ProjectConfig
from ..corpus.splits import Split, Splits
from ..gold import GoldStore
from ..stats import (
    MIN_POSITIVE_INSTANCES,
    Estimate,
    ReasonCode,
    Sealed,
    ThresholdSelection,
    Verdict,
    brier_score,
    cohens_kappa,
    expected_calibration_error,
    final_reliability,
    fit_thresholds,
    grade,
    krippendorff_alpha,
    labour_curve,
    random_baseline_curve,
    realised_class_risk,
    selective_curve,
    stratified_bootstrap_ci,
)
from ..stats.selective import LabourPoint, SelectivePoint
from .calibration import CalibrationRecord, CodeCalibration, new_record


@dataclass
class CodeCheck:
    """Everything `check` learned about one code."""

    code_id: str
    selection: ThresholdSelection
    calibration: CodeCalibration
    human_ceiling: Estimate | None = None
    brier: Estimate | None = None
    ece: Estimate | None = None
    alpha_agreement: Estimate | None = None
    labour: tuple[LabourPoint, ...] = ()
    baseline: tuple[LabourPoint, ...] = ()
    selective: tuple[SelectivePoint, ...] = ()
    n_calibration: int = 0
    n_test: int = 0
    n_test_positive: int = 0

    @property
    def verdict(self) -> Verdict:
        return Verdict(self.selection.verdict)

    @property
    def kappa(self) -> Estimate | None:
        return self.selection.held_out_kappa


@dataclass
class CheckReport:
    """The full outcome of a check, ready for the terminal table and the report."""

    codes: list[CodeCheck] = field(default_factory=list)
    record: CalibrationRecord | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[CodeCheck]:
        return [c for c in self.codes if c.verdict.usable]

    @property
    def refused(self) -> list[CodeCheck]:
        """Codes with no thresholds at all, which must be hand-coded throughout."""
        return [c for c in self.codes if not c.verdict.usable]

    @property
    def corpus_share_needing_review(self) -> float:
        """Share of the corpus a human must review **on the usable codes**.

        An item needs a person if *any* usable code abstains on it, so the
        per-code shares do not simply add -- this is the union, approximated by
        the maximum, which is the honest lower bound to quote.

        Read it with `refused`: this number says nothing about codes that were
        not certified at all, and quoting it alone would understate the work.
        """
        coverages = [c.selection.coverage for c in self.usable if c.selection.coverage is not None]
        if not coverages:
            return 1.0
        return 1.0 - min(coverages)


def _weighted_positives(labels: np.ndarray) -> int:
    return int((labels == 1.0).sum())


def check_code(
    code_id: str,
    *,
    gold: GoldStore,
    probabilities: dict[str, float | None],
    alpha: float,
    delta: float,
    bootstrap_resamples: int,
    seed: int,
    reference_coder: str | None = None,
) -> CodeCheck:
    """Fit one code's thresholds and measure the result on held-out data.

    Only the reference coder's judgements are used as ground truth. A second
    coder's overlapping subset exists to measure human-human agreement (§12),
    not to add rows: pooling both would double-count every double-coded item and
    let one rater's noise contaminate the standard the other is measured
    against.
    """
    cal_ids, cal_labels, cal_weights = gold.labels_for(
        code_id, split=Split.CALIBRATION.value, coder=reference_coder
    )
    test_ids, test_labels, test_weights = gold.labels_for(
        code_id, split=Split.TEST.value, coder=reference_coder
    )

    cal_probs = np.array(
        [probabilities.get(i) if probabilities.get(i) is not None else np.nan for i in cal_ids],
        dtype=np.float64,
    )
    test_probs = np.array(
        [probabilities.get(i) if probabilities.get(i) is not None else np.nan for i in test_ids],
        dtype=np.float64,
    )
    # An item the backend could not score carries no information about the
    # threshold and must not be silently treated as 0 (§10.2).
    cal_usable = ~np.isnan(cal_probs)
    test_usable = ~np.isnan(test_probs)
    cal_probs, cal_labels, cal_weights = (
        cal_probs[cal_usable],
        cal_labels[cal_usable],
        cal_weights[cal_usable],
    )
    test_probs, test_labels, test_weights = (
        test_probs[test_usable],
        test_labels[test_usable],
        test_weights[test_usable],
    )

    selection = fit_thresholds(cal_probs, cal_labels, alpha=alpha, delta=delta, weights=cal_weights)

    n_test_positive = _weighted_positives(test_labels)
    kappa: Estimate
    realised: dict[int, float | None] = {}
    labour: tuple[LabourPoint, ...] = ()
    baseline: tuple[LabourPoint, ...] = ()
    selective: tuple[SelectivePoint, ...] = ()

    if selection.selected is None:
        kappa = Estimate(value=None, n=int(test_labels.size), reason=ReasonCode.NO_VALID_THRESHOLD)
    elif n_test_positive < MIN_POSITIVE_INSTANCES:
        # §8.5: refuse to certify on too little evidence, whichever split is short.
        kappa = Estimate(
            value=None, n=int(test_labels.size), reason=ReasonCode.INSUFFICIENT_EVIDENCE
        )
    else:
        point = final_reliability(
            test_probs,
            test_labels,
            selection.selected,
            weights=test_weights,
            n_resamples=bootstrap_resamples,
            seed=seed,
        )
        kappa = point.kappa
        realised = realised_class_risk(
            test_probs, test_labels, selection.selected, weights=test_weights
        )
        labour = labour_curve(
            test_probs, test_labels, weights=test_weights, n_resamples=0, seed=seed
        )
        baseline = random_baseline_curve(
            test_probs,
            test_labels,
            [p.labour for p in labour],
            weights=test_weights,
            n_resamples=max(20, bootstrap_resamples // 20),
            seed=seed,
        )
        selective = selective_curve(test_probs, test_labels, weights=test_weights)

    graded = grade(selection, kappa)

    ceiling: Estimate | None = None
    first, second, shared = gold.double_coded(code_id)
    if shared:
        ceiling = cohens_kappa(first, second)
        if bootstrap_resamples:
            ceiling = ceiling.with_ci(
                stratified_bootstrap_ci(
                    first,
                    second,
                    statistic=cohens_kappa,
                    n_resamples=bootstrap_resamples,
                    seed=seed,
                )
            )

    return CodeCheck(
        code_id=code_id,
        selection=graded,
        calibration=CodeCalibration.from_selection(code_id, graded, realised=realised),
        human_ceiling=ceiling,
        brier=brier_score(test_probs, test_labels, weights=test_weights)
        if test_probs.size
        else None,
        ece=expected_calibration_error(test_probs, test_labels, weights=test_weights)
        if test_probs.size
        else None,
        alpha_agreement=krippendorff_alpha(first, second) if shared else None,
        labour=labour,
        baseline=baseline,
        selective=selective,
        n_calibration=int(cal_labels.size),
        n_test=int(test_labels.size),
        n_test_positive=n_test_positive,
    )


def run_check(
    *,
    codebook: Codebook,
    config: ProjectConfig,
    splits: Splits,
    gold: GoldStore,
    scores: dict[str, dict[str, float | None]],
    backend_name: str,
    model_version: str,
    corpus_hash: str,
    alpha: float | None = None,
    delta: float | None = None,
    bootstrap_resamples: int = 2000,
    sealed_test: Sealed[tuple[str, ...]] | None = None,
) -> CheckReport:
    """Check every code and build the calibration record `run` will honour.

    `sealed_test` is accepted and deliberately never unsealed: it is the
    §14.4 regression guard, asserting in the type system what §8.4 asks for in
    prose -- the fitting path cannot read the test split.
    """
    del sealed_test  # fitting must never look at it

    resolved_alpha = config.thresholds.alpha if alpha is None else alpha
    resolved_delta = config.thresholds.delta if delta is None else delta

    report = CheckReport()
    if not gold.all_records():
        report.warnings.append("no gold sample yet -- run `scruple gold` before `scruple check`.")
        return report

    # The coder who did the most work is the reference; any others are second
    # raters contributing the human ceiling only.
    reference_coder = gold.primary_coder()

    exact = True
    delta_per_candidate = resolved_delta
    for code in codebook:
        per_code = {item_id: item_scores.get(code.id) for item_id, item_scores in scores.items()}
        checked = check_code(
            code.id,
            gold=gold,
            probabilities=per_code,
            alpha=resolved_alpha,
            delta=resolved_delta,
            bootstrap_resamples=bootstrap_resamples,
            seed=config.splits.seed,
            reference_coder=reference_coder,
        )
        report.codes.append(checked)
        exact = exact and checked.selection.exact
        delta_per_candidate = checked.selection.delta_per_candidate or delta_per_candidate

    record = new_record(
        backend=backend_name,
        model_version=model_version,
        codebook_hash=codebook.hash,
        code_hashes=codebook.code_hashes,
        corpus_hash=corpus_hash,
        splits_seed=splits.seed,
        chunking={
            "max_tokens": config.chunking.max_tokens,
            "overlap_tokens": config.chunking.overlap_tokens,
            "aggregation": config.chunking.aggregation,
        },
        alpha=resolved_alpha,
        delta=resolved_delta,
        delta_per_candidate=delta_per_candidate,
        gold_n=len({r.item_id for r in gold.all_records()}),
        gold_n_double_coded=len(
            {r.item_id for r in gold.all_records() if r.coder != gold.primary_coder()}
        ),
        exact_guarantee=exact,
    )
    record.codes = {c.code_id: c.calibration for c in report.codes}
    record.human_ceiling = {
        c.code_id: (c.human_ceiling.value if c.human_ceiling else None) for c in report.codes
    }
    report.record = record

    _add_diagnostic_warnings(report)
    return report


def _add_diagnostic_warnings(report: CheckReport) -> None:
    """Turn failures into something the researcher can act on (§12).

    When a code fails, say why, and check human-human agreement on it first. A
    code where humans disagree is a codebook problem, not a model problem --
    saying so turns a failure into a useful finding.
    """
    for checked in report.codes:
        if checked.verdict.usable:
            continue
        ceiling = checked.human_ceiling
        if (
            checked.verdict is Verdict.NOT_AUTOMATABLE
            and ceiling is not None
            and ceiling.value is not None
            and ceiling.value < 0.70
        ):
            report.warnings.append(
                f"{checked.code_id} could not be automated. Your two coders agreed only at "
                f"kappa={ceiling.value:.2f} on this code, which suggests the definition is "
                "underspecified. Consider rewriting it."
            )
        elif checked.selection.message:
            report.warnings.append(f"{checked.code_id} {checked.selection.message}")
