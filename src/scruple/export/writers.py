"""Writing the three output files (§0.5, §7.4).

    coded.csv              the data, coded. Every competitor has this.
    abstentions.csv        the items it refused to code, with probabilities.
    validation_report.md   the agreement statistics and the guarantee.

Original columns are carried through unchanged, because the researcher's
analysis is built on them and a coded file they have to re-join by hand is a
coded file they will not use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..corpus import Corpus
from ..engine.apply import AppliedCoding, Source
from ..errors import ValidationError

FORMATS = ("csv", "parquet", "stata", "spss")


def coded_rows(corpus: Corpus, applied: AppliedCoding) -> list[dict[str, Any]]:
    """Original columns, plus `<code>`, `<code>_p` and `<code>_src` (§7.4)."""
    rows: list[dict[str, Any]] = []
    for item in corpus:
        row: dict[str, Any] = dict(item.row)
        row.setdefault("scruple_item_id", item.id)
        per_code = applied.decisions.get(item.id, {})
        for code_id in applied.codes:
            decision = per_code.get(code_id)
            row[code_id] = None if decision is None or decision.label is None else decision.label
            row[f"{code_id}_p"] = None if decision is None else decision.probability
            row[f"{code_id}_src"] = (
                Source.ABSTAIN_UNREVIEWED.value if decision is None else decision.source.value
            )
        rows.append(row)
    return rows


def abstention_rows(corpus: Corpus, applied: AppliedCoding) -> list[dict[str, Any]]:
    """One row per (item, code) the engine declined, with the text and probability.

    Ordered by probability distance from the middle of the band, so the most
    tractable cases come first: a reviewer who runs out of time has still
    cleared the easiest part of the backlog.
    """
    by_id = corpus.by_id()
    rows = []
    for decision in applied.abstentions():
        item = by_id.get(decision.item_id)
        rows.append(
            {
                "item_id": decision.item_id,
                "code_id": decision.code_id,
                "probability": decision.probability,
                "text": item.text if item else "",
            }
        )

    def distance_from_the_middle(row: dict[str, Any]) -> float:
        probability = row["probability"]
        return abs((0.5 if probability is None else float(probability)) - 0.5)

    rows.sort(key=distance_from_the_middle, reverse=True)
    return rows


def _frame(rows: list[dict[str, Any]], *, integer_columns: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    import pandas as pd

    frame = pd.DataFrame(rows)
    # §7.4 specifies 0/1/NA for a code column. Plain float columns would write
    # "1.0" and "0.0", which is not what a researcher then reads into R or
    # Stata as a binary variable. The nullable integer dtype keeps missing
    # values missing without turning the column into floats.
    for column in integer_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("Int64")
    return frame


def write_table(
    rows: list[dict[str, Any]],
    path: Path,
    fmt: str = "csv",
    *,
    integer_columns: tuple[str, ...] = (),
) -> Path:
    """Write a table in the requested format."""
    if fmt not in FORMATS:
        raise ValidationError(
            f"unknown export format {fmt!r}", hint=f"Supported: {', '.join(FORMATS)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _frame(rows, integer_columns=integer_columns)

    if fmt == "csv":
        frame.to_csv(path, index=False)
        return path
    if fmt == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise ValidationError(
                "parquet export needs pyarrow", hint="Install scruple[parquet]."
            ) from exc
        return path
    if fmt == "stata":
        # Stata rejects names Python is happy with, and has no nullable integer
        # type, so make both legal rather than failing at the last step of a
        # long pipeline.
        renamed = frame.rename(columns=lambda c: str(c).replace(".", "_")[:32])
        for column in renamed.columns:
            if str(renamed[column].dtype) == "Int64":
                renamed[column] = renamed[column].astype("float64")
        renamed.to_stata(path, write_index=False, version=118)
        return path

    try:
        import pyreadstat
    except ImportError as exc:
        raise ValidationError(
            "SPSS export needs pyreadstat", hint="Install scruple[spss]."
        ) from exc
    pyreadstat.write_sav(frame, str(path))
    return path


SUFFIXES = {"csv": "csv", "parquet": "parquet", "stata": "dta", "spss": "sav"}


def default_filename(name: str, fmt: str) -> str:
    """The output filename for a format, refusing an unknown one clearly.

    Validated here as well as in `write_table` because the filename is computed
    first: an unknown format would otherwise surface as a raw KeyError rather
    than as something the user can act on.
    """
    if fmt not in SUFFIXES:
        raise ValidationError(
            f"unknown export format {fmt!r}", hint=f"Supported: {', '.join(FORMATS)}."
        )
    return f"{name}.{SUFFIXES[fmt]}"
