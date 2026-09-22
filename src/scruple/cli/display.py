"""Terminal rendering and the --json envelope (§12, §13.1).

`--json` is a public, versioned interface: the R package of Phase 2 wraps the
CLI as a subprocess contract rather than reimplementing the statistics, so
breaking this output is a breaking change.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..engine.check import CheckReport
from ..errors import ExitCode, ScrupleError
from ..stats import Estimate, Verdict

VERDICT_STYLE = {
    Verdict.OK: "green",
    Verdict.WEAK: "yellow",
    Verdict.NOT_AUTOMATABLE: "red",
    Verdict.INSUFFICIENT_EVIDENCE: "red",
}


def envelope(command: str, data: dict[str, Any]) -> str:
    """The success envelope. Shape is part of the contract; keep it stable."""
    return json.dumps(
        {"scruple": __version__, "command": command, "ok": True, "data": data},
        indent=2,
        sort_keys=True,
        default=str,
    )


def error_envelope(command: str, error: ScrupleError) -> str:
    return json.dumps(
        {
            "scruple": __version__,
            "command": command,
            "ok": False,
            "error": {
                "code": ExitCode(error.exit_code).name,
                "exit_code": int(error.exit_code),
                "message": error.message,
                "hint": error.hint,
            },
        },
        indent=2,
        sort_keys=True,
    )


def show_error(console: Console, error: ScrupleError) -> None:
    console.print(Text(f"error: {error.message}", style="bold red"))
    if error.hint:
        console.print(Text(f"  {error.hint}", style="dim"))


def _number(estimate: Estimate | None, digits: int = 2) -> str:
    if estimate is None or estimate.value is None:
        return "--"
    return f"{estimate.value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def check_table(report: CheckReport, console: Console) -> None:
    """The output of `scruple check` (§12) -- designed carefully; it is the product."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("code")
    table.add_column("kappa (held-out)", justify="right")
    table.add_column("auto-code", justify="right")
    table.add_column("needs you", justify="right")
    table.add_column("verdict")

    for checked in report.codes:
        verdict = Verdict(checked.selection.verdict)
        coverage = checked.selection.coverage
        detail = ""
        if verdict is Verdict.INSUFFICIENT_EVIDENCE:
            detail = f" ({checked.selection.n_gold_positive} positives)"
        table.add_row(
            checked.code_id,
            _number(checked.kappa),
            _percent(coverage),
            _percent(None if coverage is None else 1 - coverage),
            Text(verdict.value + detail, style=VERDICT_STYLE[verdict]),
        )

    console.print()
    console.print(table)
    console.print("─" * 74, style="dim")

    usable = len(report.usable)
    refused = len(report.codes) - usable
    summary = (
        f"{usable} of {len(report.codes)} codes usable · "
        f"{_percent(report.corpus_share_needing_review)} of items abstained on those"
    )
    if refused:
        # Saying only "0% needs you" while four codes are uncertified would be a
        # lie by omission: those have to be coded by hand from end to end.
        summary += f" · {refused} code(s) to code by hand throughout"
    console.print(summary)
    if report.record is not None:
        record = report.record
        console.print(
            Text(
                f"guarantee: class-conditional error <= {record.alpha:.0%} per class, "
                f"confidence {1 - record.delta:.0%}, per code"
                + ("" if record.exact_guarantee else " (approximate: enriched sampling)"),
                style="dim",
            )
        )

    # When a code fails, say why, and point at the codebook rather than the
    # model where that is the honest answer (§12).
    for warning in report.warnings:
        console.print()
        console.print(Text(f"  {warning}", style="yellow"))


def check_json(report: CheckReport) -> dict[str, Any]:
    """Machine-readable check results. Part of the §13.1 contract."""
    return {
        "codes": [
            {
                "code_id": c.code_id,
                "verdict": Verdict(c.selection.verdict).value,
                "reason": c.selection.reason.value if c.selection.reason else None,
                "kappa": c.kappa.value if c.kappa else None,
                "kappa_ci": list(c.kappa.ci) if c.kappa and c.kappa.ci else None,
                "coverage": c.selection.coverage,
                "t_lo": c.calibration.t_lo,
                "t_hi": c.calibration.t_hi,
                "n_calibration": c.n_calibration,
                "n_test": c.n_test,
                "n_test_positive": c.n_test_positive,
                "brier": c.brier.value if c.brier else None,
                "ece": c.ece.value if c.ece else None,
                "human_kappa": c.human_ceiling.value if c.human_ceiling else None,
                "message": c.selection.message,
            }
            for c in report.codes
        ],
        "usable": [c.code_id for c in report.usable],
        "corpus_share_needing_review": report.corpus_share_needing_review,
        "warnings": report.warnings,
        "alpha": report.record.alpha if report.record else None,
        "delta": report.record.delta if report.record else None,
        "exact_guarantee": report.record.exact_guarantee if report.record else None,
    }
