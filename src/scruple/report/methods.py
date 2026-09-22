"""The methods paragraph (§0.5, §15).

A paragraph the researcher can paste into a paper, stating what was done and
what was assumed, in the register a methods section uses. Getting this right
matters more than it looks: if describing scruple is hard, researchers will
describe it vaguely instead, and a vague description is indistinguishable from
the "AI coding" category this project is trying not to be in (§0.6).
"""

from __future__ import annotations

from ..engine.calibration import CalibrationRecord
from ..stats import Verdict


def methods_paragraph(
    record: CalibrationRecord,
    *,
    corpus_size: int,
    gold_n: int,
    reviewed: int,
    kappa_range: tuple[float, float] | None,
    scruple_version: str,
) -> str:
    """Compose the paste-ready paragraph."""
    refused = [c for c in record.codes.values() if not Verdict(c.verdict).usable]

    kappa_clause = (
        f"Cohen's kappa for the delivered dataset ranged from {kappa_range[0]:.2f} to "
        f"{kappa_range[1]:.2f} across codes"
        if kappa_range
        else "Cohen's kappa could not be estimated for any code"
    )

    guarantee_clause = (
        "exact, since the gold sample was drawn uniformly at random"
        if record.exact_guarantee
        else (
            "approximate, since rare codes were supplemented by an enriched stratum and the "
            "risk test was applied to an effective sample size"
        )
    )

    refused_clause = (
        f" {len(refused)} of {len(record.codes)} codes did not meet the criterion at any "
        "threshold and were coded entirely by hand."
        if refused
        else ""
    )

    return (
        f"Open-ended responses (n = {corpus_size:,}) were coded deductively against an "
        f"author-written codebook of {len(record.codes)} binary codes using scruple "
        f"{scruple_version}, with {record.backend} ({record.model_version}) providing "
        "calibrated probabilities. A random sample of "
        f"{gold_n:,} responses was hand-coded blind, without sight of model output, and "
        "partitioned into disjoint calibration and test sets fixed in advance "
        f"(seed {record.splits_seed}). Per-code decision thresholds were selected on the "
        "calibration set so that the class-conditional error rate, computed separately "
        "within each true class, was at most "
        f"{record.alpha:.0%} with {1 - record.delta:.0%} confidence; candidate thresholds "
        "were tested in full with a Bonferroni correction across the grid. This guarantee "
        f"is {guarantee_clause}. It is stated per code and marginally over items, and it "
        "applies only to the subset the model decided, not to items routed to a human. "
        f"Responses whose probability fell inside the resulting abstention band "
        f"(n = {reviewed:,}) were coded by hand. Reliability was then assessed on the "
        "held-out test set over the dataset as delivered -- model-decided and "
        f"human-coded items together. {kappa_clause}.{refused_clause} Full agreement "
        "statistics, calibration diagnostics, the fitted thresholds and complete "
        "provenance are reported in the accompanying validation report."
    )
