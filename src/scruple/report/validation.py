"""`validation_report.md` (§0.5, §8.9).

This file is the actual product. Everything else scruple writes, a competitor
also writes; this is the part that lets a reviewer decide whether to believe the
numbers, and it is written for a methodologist reading sceptically.

Requirements it discharges:

* §8.2 -- kappa and alpha reported together, and the report says so where they
  disagree substantially, because that usually means skewed prevalence.
* §8.6 -- the guarantee is stated as per-code and marginal over items, with the
  exchangeability assumption in plain language.
* §8.7 -- selective kappa appears as a diagnostic, with prevalence beside it.
* §8.9 -- reliability of the mixed-provenance dataset is defined, with the
  per-source breakdown, and the guarantee's scope stated explicitly.
* §10.3 -- the aggregation rule is stated.
* §14.9 -- byte-stable modulo timestamps, paths and durations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..engine.apply import AppliedCoding
from ..engine.calibration import CalibrationRecord
from ..engine.check import CheckReport, CodeCheck
from ..engine.manifest import Manifest
from ..stats import PUBLICATION_KAPPA, Estimate, Verdict
from .methods import methods_paragraph

VOLATILE_MARK = "<!-- volatile -->"
"""Lines carrying a timestamp, path, duration or traffic counter are marked so
§14.9's golden tests can normalise them without normalising anything that
matters. A counter is volatile because a warm cache legitimately makes it zero;
the substance -- hashes, seeds, item counts, every statistic -- is not."""

ALPHA_KAPPA_DIVERGENCE = 0.10
"""Above this gap, §8.2 requires the report to point it out."""


def _number(estimate: Estimate | None, digits: int = 2) -> str:
    if estimate is None or estimate.value is None:
        return "--"
    text = f"{estimate.value:.{digits}f}"
    if estimate.ci is not None:
        text += f" [{estimate.ci[0]:.{digits}f}, {estimate.ci[1]:.{digits}f}]"
    return text


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


@dataclass
class ReportInputs:
    """Everything the report needs, gathered once."""

    check: CheckReport
    record: CalibrationRecord
    manifest: Manifest
    applied: AppliedCoding | None
    corpus_size: int
    reviewed: int
    scruple_version: str
    chart_path: str | None = None


def _verdict_table(codes: Sequence[CodeCheck]) -> list[str]:
    lines = [
        "| code | kappa (held-out) | auto-coded | needs you | n gold (test) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for checked in codes:
        coverage = checked.selection.coverage
        lines.append(
            f"| `{checked.code_id}` | {_number(checked.kappa)} | {_percent(coverage)} | "
            f"{_percent(None if coverage is None else 1 - coverage)} | "
            f"{checked.n_test} ({checked.n_test_positive} positive) | "
            f"{Verdict(checked.selection.verdict).value} |"
        )
    return lines


def _threshold_table(codes: Sequence[CodeCheck]) -> list[str]:
    lines = [
        "| code | t_lo | t_hi | observed error, positive class | observed error, negative class |",
        "|---|---|---|---|---|",
    ]
    for checked in codes:
        calibration = checked.calibration
        if calibration.pair is None:
            lines.append(f"| `{checked.code_id}` | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| `{checked.code_id}` | {calibration.t_lo:.2f} | {calibration.t_hi:.2f} | "
            f"{_percent(calibration.realised_risk_positive)} | "
            f"{_percent(calibration.realised_risk_negative)} |"
        )
    return lines


def _diagnostics_table(codes: Sequence[CodeCheck]) -> list[str]:
    lines = [
        "| code | Brier | ECE | Krippendorff alpha (human-human) | human-human kappa |",
        "|---|---|---|---|---|",
    ]
    for checked in codes:
        lines.append(
            f"| `{checked.code_id}` | {_number(checked.brier, 3)} | {_number(checked.ece, 3)} | "
            f"{_number(checked.alpha_agreement)} | {_number(checked.human_ceiling)} |"
        )
    return lines


def _selective_table(checked: CodeCheck, limit: int = 6) -> list[str]:
    lines = [
        f"#### `{checked.code_id}`",
        "",
        "| coverage | selective kappa | accuracy | prevalence in accepted subset |",
        "|---|---|---|---|",
    ]
    points = list(checked.selective)
    step = max(1, len(points) // limit)
    for point in points[::step]:
        lines.append(
            f"| {_percent(point.coverage)} | {_number(point.kappa)} | "
            f"{_percent(point.accuracy)} | {_percent(point.prevalence)} |"
        )
    return [*lines, ""]


def _divergence_notes(codes: Sequence[CodeCheck]) -> list[str]:
    notes = []
    for checked in codes:
        kappa, alpha = checked.human_ceiling, checked.alpha_agreement
        if (
            kappa is not None
            and alpha is not None
            and kappa.value is not None
            and alpha.value is not None
            and abs(kappa.value - alpha.value) > ALPHA_KAPPA_DIVERGENCE
        ):
            notes.append(
                f"- `{checked.code_id}`: kappa {kappa.value:.2f} against alpha "
                f"{alpha.value:.2f}. A gap this size usually means skewed prevalence; "
                "read both alongside the prevalence column above."
            )
    return notes


def build_report(inputs: ReportInputs) -> str:
    """Render the full validation report as Markdown."""
    check, record, manifest = inputs.check, inputs.record, inputs.manifest
    usable = check.usable
    kappas = [c.kappa.value for c in usable if c.kappa is not None and c.kappa.value is not None]
    kappa_range = (min(kappas), max(kappas)) if kappas else None

    lines: list[str] = [
        "# Validation report",
        "",
        f"{VOLATILE_MARK} Generated {manifest.created_at} by scruple {inputs.scruple_version}, "
        f"run `{manifest.run_id}`.",
        "",
        "## What this reports",
        "",
        "This describes the dataset as delivered: items the model decided and items a "
        "person decided, together. It is not a claim about the model in isolation.",
        "",
        f"- **{len(usable)} of {len(check.codes)} codes** are usable.",
        f"- On those codes, **{_percent(check.corpus_share_needing_review)} of the corpus** "
        "needs a person, taking the code that abstains most.",
        f"- **{len(check.refused)} code(s)** were not certified at all and must be coded by "
        "hand from end to end. That is the tool working correctly, not a gap to route around.",
        f"- Reliability bar used throughout: kappa >= {PUBLICATION_KAPPA:.2f}.",
        "",
        "## Per-code results",
        "",
        *_verdict_table(check.codes),
        "",
        "`kappa (held-out)` is Cohen's kappa over the **whole** test split under the "
        "delivered-dataset simulation: items inside the abstention band take the human "
        "label, because a person coded them; items outside it take the model's decision. "
        "Intervals are 95% bootstrap percentiles, stratified by the gold label.",
        "",
        "## The guarantee, and exactly what it covers",
        "",
        f"For each code marked usable, the decision thresholds were fitted so that within "
        f"**each true class separately**, the error rate among accepted items is at most "
        f"{record.alpha:.0%}, with {1 - record.delta:.0%} confidence.",
        "",
        "Read the scope carefully:",
        "",
        "- The guarantee is **per code**, and **marginal over items**. It is not a "
        "simultaneous statement across codes, and no family-wise correction is applied "
        "across them: each code is a separate claim, reported separately.",
        "- It covers **only the model-decided subset**. Items routed to a human are not "
        "covered by it, and do not need to be.",
        "- It is class-conditional on purpose. A marginal error guarantee would certify a "
        'coder that answers "no" to everything on a rare code, at full coverage, with '
        "kappa 0.",
        "- Candidate thresholds were tested in full and corrected with Bonferroni across "
        f"the grid (per-candidate level {record.delta_per_candidate:.4f}).",
        "",
        "**Assumption.** The guarantee holds if the gold sample is a random sample of the "
        "corpus, with known inclusion probabilities. It was drawn with a recorded seed "
        f"({record.splits_seed}); a hand-picked sample would break it silently. This "
        f"sample's risk test is **{'exact' if record.exact_guarantee else 'approximate'}**"
        + (
            "."
            if record.exact_guarantee
            else ", because rare codes were supplemented by an enriched stratum and the "
            "binomial test was applied to an effective sample size rather than a raw count."
        ),
        "",
        "## Fitted thresholds",
        "",
        *_threshold_table(check.codes),
        "",
        "Observed error rates are measured on the test split, so a reader can check the "
        "guarantee rather than take it on trust.",
        "",
        "## Calibration diagnostics",
        "",
        *_diagnostics_table(check.codes),
        "",
        "Brier and ECE describe the probabilities; they are diagnostics, not the "
        "guarantee. The human-human columns are the ceiling: no automated coder should be "
        "expected to beat the agreement two trained people reach on the same code.",
        "",
    ]

    divergences = _divergence_notes(check.codes)
    if divergences:
        lines += [
            "### Where kappa and alpha disagree",
            "",
            *divergences,
            "",
        ]

    if inputs.applied is not None:
        counts = inputs.applied.counts_by_source()
        total = sum(counts.values()) or 1
        lines += [
            "## Provenance of the delivered dataset",
            "",
            "| source | cells | share |",
            "|---|---|---|",
            *[
                f"| {name} | {count:,} | {count / total * 100:.1f}% |"
                for name, count in counts.items()
            ],
            "",
            "A cell is one (item, code) judgement. `auto` is the only subset the guarantee "
            "above covers.",
            "",
        ]

    if any(c.selective for c in usable):
        lines += [
            "## Selective performance (diagnostic only)",
            "",
            "Selective kappa can **fall** as coverage drops even when the coder is behaving "
            "perfectly: confident subsets skew toward one class, which raises chance "
            "agreement and compresses kappa. Read it with the prevalence column beside it, "
            "and do not read it as the headline -- the headline is the held-out kappa above.",
            "",
        ]
        for checked in usable:
            if checked.selective:
                lines += _selective_table(checked)

    if inputs.chart_path:
        lines += [
            "## Reliability against human labour",
            "",
            f"![Final reliability against human labour]({inputs.chart_path})",
            "",
            "Reliability of the finished dataset against the share of the corpus a person "
            "must code, with a random-selection baseline. The gap between the two lines is "
            "what calibrated abstention buys, in the only currency that matters here: "
            "hours at a fixed quality bar.",
            "",
        ]

    if check.warnings:
        lines += ["## What needs your attention", "", *[f"- {w}" for w in check.warnings], ""]

    lines += [
        "## Provenance",
        "",
        "| field | value |",
        "|---|---|",
        f"| backend | `{manifest.backend}` |",
        f"| model version | `{manifest.model_version}` |",
        f"| codebook hash | `{manifest.codebook_hash[:16]}` |",
        f"| corpus hash | `{manifest.corpus_hash[:16]}` |",
        f"| corpus file hash | `{manifest.corpus_file_hash[:16]}` |",
        f"| split seed | `{manifest.splits_seed}` |",
        f"| chunking | max_tokens={manifest.chunking.get('max_tokens')}, "
        f"overlap={manifest.chunking.get('overlap_tokens')}, "
        f"aggregation=`{manifest.chunking.get('aggregation')}` |",
        f"| items | {manifest.item_count:,} |",
        f"| passages scored | {manifest.unit_count:,} |",
        f"| failures | {manifest.failures:,} |",
        f"| scruple version | `{manifest.scruple_version}` |",
        # Traffic counters describe this execution rather than the finding: a
        # warm cache legitimately makes them zero. They are recorded in the run
        # manifest either way.
        f"{VOLATILE_MARK} | backend calls | {manifest.calls:,} |",
        f"{VOLATILE_MARK} | input tokens | {manifest.input_tokens:,} |",
        f"{VOLATILE_MARK} | corpus path | `{manifest.corpus_path}` |",
        f"{VOLATILE_MARK} | wall clock | {manifest.wall_clock_seconds:.1f}s |",
        f"{VOLATILE_MARK} | python | `{manifest.python_version}` on `{manifest.platform}` |",
        "",
        "Chunking settings and the aggregation rule are part of the calibration: "
        "changing either invalidates the thresholds above, and `scruple run` refuses to "
        "proceed until `scruple check` is re-run.",
        "",
        "## Per-code hashes",
        "",
        "| code | hash |",
        "|---|---|",
        *[f"| `{cid}` | `{h[:16]}` |" for cid, h in sorted(manifest.code_hashes.items())],
        "",
        "## Methods paragraph",
        "",
        "Ready to paste into a paper, and to edit -- it is a starting point, not a "
        "substitute for your own account of what you did.",
        "",
        "> "
        + methods_paragraph(
            record,
            corpus_size=inputs.corpus_size,
            gold_n=record.gold_n,
            reviewed=inputs.reviewed,
            kappa_range=kappa_range,
            scruple_version=inputs.scruple_version,
        ),
        "",
        "## Limitations",
        "",
        "- Deductive coding only. scruple applies the codebook you wrote; it does not "
        "discover categories.",
        "- Codes marked `NOT_AUTOMATABLE` or `INSUFFICIENT_EVIDENCE` were not coded "
        "automatically at all. That is the tool working correctly, not a failure to "
        "route around.",
        "- A gold sample below about 200 items gives intervals too wide to support "
        "strong claims, whatever the point estimates say.",
        "- The automated coder is a coder, not ground truth. Report it as a coder.",
        "",
    ]
    return "\n".join(lines) + "\n"


def normalise_for_comparison(report: str) -> str:
    """Drop volatile lines, so §14.9's golden test compares what matters."""
    return "\n".join(line for line in report.splitlines() if VOLATILE_MARK not in line)
